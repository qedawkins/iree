// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTilingInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-pcf-tile-to-pcf"

namespace mlir::iree_compiler::IREE::PCF {

namespace {

/// Returns true if the tile size is zero (meaning "don't tile this dim").
static bool isTileSizeZero(OpFoldResult tileSize) {
  if (auto attr = dyn_cast<Attribute>(tileSize)) {
    return cast<IntegerAttr>(attr).getInt() == 0;
  }
  return false;
}

/// Computes the static size from an OpFoldResult for building result types.
/// Returns ShapedType::kDynamic for dynamic values.
static int64_t getStaticSize(OpFoldResult size) {
  if (auto attr = dyn_cast<Attribute>(size)) {
    return cast<IntegerAttr>(attr).getInt();
  }
  return ShapedType::kDynamic;
}

/// Core implementation of tileToPCFLoop. Templated to support both LoopOp
/// and GenericOp, though only LoopOp is currently implemented.
template <typename OpTy>
static FailureOr<OpTy> tileToPCFImpl(RewriterBase &rewriter,
                                     PCFTilingOpInterface target,
                                     ScopeAttrInterface scope,
                                     ArrayRef<OpFoldResult> tileSizes) {
  // Only LoopOp is supported for now.
  if constexpr (!std::is_same_v<OpTy, LoopOp>) {
    return rewriter.notifyMatchFailure(
        target, "tileToPCFGeneric is not yet implemented");
  }

  Location loc = target.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);

  // Step 1: Get the iteration domain.
  SmallVector<Range> iterationDomain =
      cast<TilingInterface>(target.getOperation()).getIterationDomain(rewriter);
  int64_t numDims = iterationDomain.size();

  if (static_cast<int64_t>(tileSizes.size()) != numDims) {
    return rewriter.notifyMatchFailure(
        target,
        "number of tile sizes does not match iteration domain dimensions");
  }

  // Step 2: Determine which dimensions are tiled (non-zero tile size).
  SmallVector<int64_t> tiledDims;
  for (int64_t i = 0; i < numDims; ++i) {
    if (!isTileSizeZero(tileSizes[i])) {
      tiledDims.push_back(i);
    }
  }

  if (tiledDims.empty()) {
    return rewriter.notifyMatchFailure(target, "all tile sizes are zero");
  }

  // Step 3: Compute iteration counts for tiled dimensions.
  // count[i] = ceil(domain[i].size / tileSizes[i]).
  AffineExpr s0, s1;
  bindSymbols(rewriter.getContext(), s0, s1);
  AffineMap ceilDivMap =
      AffineMap::get(0, 2, s0.ceilDiv(s1), rewriter.getContext());

  SmallVector<Value> iterationCounts;
  for (int64_t dim : tiledDims) {
    OpFoldResult count = affine::makeComposedFoldedAffineApply(
        rewriter, loc, ceilDivMap,
        ArrayRef<OpFoldResult>{iterationDomain[dim].size, tileSizes[dim]});
    iterationCounts.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, count));
  }

  // Step 4: Classify operands into readonly (tileable non-DPS-init) and
  // readwrite (DPS init) categories.
  Operation *op = target.getOperation();
  SmallVector<unsigned> tileableIndices = target.getTileableOperandIndices();
  DenseSet<unsigned> tileableSet(tileableIndices.begin(),
                                 tileableIndices.end());

  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);

  // Build sets of DPS init operand indices for quick lookup.
  DenseSet<unsigned> dpsInitIndices;
  if (dpsOp) {
    for (OpOperand &init : dpsOp.getDpsInitsMutable()) {
      dpsInitIndices.insert(init.getOperandNumber());
    }
  }

  // Classify into readonly and readwrite.
  SmallVector<Value> readonlyInits;
  SmallVector<Value> readwriteInits;
  // Track the mapping from operand index to sref arg index.
  // -1 means not an sref arg (non-tileable).
  SmallVector<int64_t> operandToReadonlySrefIdx(op->getNumOperands(), -1);
  SmallVector<int64_t> operandToReadwriteSrefIdx(op->getNumOperands(), -1);

  for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    unsigned idx = static_cast<unsigned>(i);
    if (!tileableSet.contains(idx)) {
      // Non-tileable operand: not an sref arg.
      continue;
    }
    if (dpsInitIndices.contains(idx)) {
      // DPS init -> readwrite sref.
      operandToReadwriteSrefIdx[i] =
          static_cast<int64_t>(readwriteInits.size());
      readwriteInits.push_back(op->getOperand(i));
    } else {
      // Tileable non-DPS-init -> readonly sref.
      operandToReadonlySrefIdx[i] =
          static_cast<int64_t>(readonlyInits.size());
      readonlyInits.push_back(op->getOperand(i));
    }
  }

  // Step 5: Create the LoopOp.
  rewriter.setInsertionPoint(op);
  LoopOp loopOp = LoopOp::create(rewriter, loc, scope, iterationCounts,
                                  readonlyInits, readwriteInits);

  // Step 6: Inside the body, compute tile offsets/sizes from the loop's
  // ID args.
  {
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToStart(loopOp.getBody());

    ArrayRef<BlockArgument> readonlyRefs = loopOp.getReadonlyRefArgs();
    ArrayRef<BlockArgument> readwriteRefs = loopOp.getRegionRefArgs();
    ArrayRef<BlockArgument> idArgs = loopOp.getIdArgs();

    // Build affine maps for offset and size computation.
    // offset[i] = id[i] * tileSize[i].
    AffineExpr d0, d1;
    bindSymbols(rewriter.getContext(), d0, d1);
    AffineMap mulMap =
        AffineMap::get(0, 2, d0 * d1, rewriter.getContext());

    // size[i] = min(tileSize[i], domain[i].size - offset[i]).
    AffineExpr s2;
    bindSymbols(rewriter.getContext(), s0, s1, s2);
    AffineMap minMap =
        AffineMap::get(0, 3, {s0, s1 - s2}, rewriter.getContext());

    SmallVector<OpFoldResult> offsets(numDims);
    SmallVector<OpFoldResult> sizes(numDims);
    int64_t tiledDimIdx = 0;
    for (int64_t i = 0; i < numDims; ++i) {
      if (isTileSizeZero(tileSizes[i])) {
        // Untiled dimension: offset=0, size=domain size.
        offsets[i] = rewriter.getIndexAttr(0);
        sizes[i] = iterationDomain[i].size;
      } else {
        // Tiled dimension: compute offset and clamped size.
        OpFoldResult idVal = idArgs[tiledDimIdx];
        OpFoldResult offset = affine::makeComposedFoldedAffineApply(
            rewriter, loc, mulMap,
            ArrayRef<OpFoldResult>{idVal, tileSizes[i]});
        offsets[i] = offset;

        // Clamp size to handle boundary tiles.
        OpFoldResult size = affine::makeComposedFoldedAffineMin(
            rewriter, loc, minMap,
            ArrayRef<OpFoldResult>{tileSizes[i], iterationDomain[i].size,
                                   offset});
        sizes[i] = size;
        ++tiledDimIdx;
      }
    }

    // Step 7: Build DistributedOperandInfo per operand.
    SmallVector<DistributedOperandInfo> operandInfo;
    operandInfo.reserve(op->getNumOperands());
    for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
      if (!tileableSet.contains(static_cast<unsigned>(i))) {
        // Non-tileable: use the original value directly.
        operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
        continue;
      }

      // Determine which sref arg this operand maps to.
      Value sref;
      if (operandToReadonlySrefIdx[i] >= 0) {
        sref = readonlyRefs[operandToReadonlySrefIdx[i]];
      } else {
        assert(operandToReadwriteSrefIdx[i] >= 0 &&
               "tileable operand must be either readonly or readwrite");
        sref = readwriteRefs[operandToReadwriteSrefIdx[i]];
      }

      // Create ReadSliceOp to read a tile from the sref.
      auto srefType = cast<ShapedRefType>(sref.getType());
      int64_t srefRank = srefType.getRank();

      // Compute the tile offsets/sizes for this operand. For now, we use
      // the iteration domain offsets/sizes directly. This works for ops
      // where the operand shape matches the iteration domain (like linalg
      // ops with identity indexing maps). The getDistributedImplementation
      // method is responsible for computing operand-specific tile positions
      // from the iteration domain tile.
      SmallVector<OpFoldResult> readOffsets(offsets.begin(), offsets.end());
      SmallVector<OpFoldResult> readSizes(sizes.begin(), sizes.end());

      // Build result type for the read.
      SmallVector<int64_t> staticSizes;
      staticSizes.reserve(srefRank);
      for (int64_t dim = 0; dim < srefRank; ++dim) {
        if (dim < numDims) {
          staticSizes.push_back(getStaticSize(sizes[dim]));
        } else {
          staticSizes.push_back(srefType.getDimSize(dim));
        }
      }
      RankedTensorType resultType =
          RankedTensorType::get(staticSizes, srefType.getElementType());
      SmallVector<OpFoldResult> strides(srefRank, rewriter.getIndexAttr(1));

      // Pad offsets/sizes to match sref rank if needed.
      while (static_cast<int64_t>(readOffsets.size()) < srefRank) {
        readOffsets.push_back(rewriter.getIndexAttr(0));
      }
      while (static_cast<int64_t>(readSizes.size()) < srefRank) {
        readSizes.push_back(
            rewriter.getIndexAttr(srefType.getDimSize(readSizes.size())));
      }

      Value tile = ReadSliceOp::create(rewriter, loc, resultType, sref,
                                       readOffsets, readSizes, strides);
      operandInfo.push_back({tile, /*isTile=*/true});
    }

    // Step 8: Build DistributedResultInfo per result.
    SmallVector<DistributedResultInfo> resultInfo;
    int64_t numResults = op->getNumResults();
    resultInfo.reserve(numResults);
    for (int64_t i = 0; i < numResults; ++i) {
      resultInfo.push_back({readwriteRefs[i]});
    }

    // Step 9: Call getDistributedImplementation.
    FailureOr<TilingResult> tilingResult =
        target.getDistributedImplementation(rewriter, offsets, sizes,
                                            operandInfo, resultInfo);
    if (failed(tilingResult)) {
      // Clean up the loop op on failure.
      rewriter.eraseOp(loopOp);
      return rewriter.notifyMatchFailure(
          target, "failed to get distributed implementation");
    }

    // Step 10: Handle returned tiled values. For results where destSref is
    // set and tiledValues[i] is non-null, write via pcf.write_slice.
    for (int64_t i = 0; i < numResults; ++i) {
      Value tiledValue = tilingResult->tiledValues[i];
      if (!tiledValue) {
        // The implementation already wrote to the sref (scatter-like).
        continue;
      }
      if (!resultInfo[i].destSref) {
        // No dest sref - shouldn't happen in this context but be safe.
        continue;
      }

      // Write the tile to the sref. Compute write position from offsets/sizes.
      // Use the same approach as the distributed implementation for computing
      // result tile position - typically the implementation handles this itself
      // when destSref is set. But if it returns a tile value, we write it.
      int64_t resultRank =
          cast<ShapedType>(tiledValue.getType()).getRank();
      SmallVector<OpFoldResult> writeOffsets(offsets.begin(), offsets.end());
      SmallVector<OpFoldResult> writeSizes(sizes.begin(), sizes.end());
      SmallVector<OpFoldResult> writeStrides(resultRank,
                                             rewriter.getIndexAttr(1));

      // Pad to result rank if needed.
      while (static_cast<int64_t>(writeOffsets.size()) < resultRank) {
        writeOffsets.push_back(rewriter.getIndexAttr(0));
      }
      while (static_cast<int64_t>(writeSizes.size()) < resultRank) {
        writeSizes.push_back(rewriter.getIndexAttr(
            cast<ShapedType>(tiledValue.getType())
                .getDimSize(writeSizes.size())));
      }
      writeOffsets.resize(resultRank);
      writeSizes.resize(resultRank);

      WriteSliceOp::create(rewriter, loc, tiledValue, resultInfo[i].destSref,
                           writeOffsets, writeSizes, writeStrides);
    }
  }

  // Step 11: Replace original op results with the PCF op results.
  rewriter.replaceOp(op, loopOp.getResults());

  if constexpr (std::is_same_v<OpTy, LoopOp>) {
    return loopOp;
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API: tileToPCFLoop
//===----------------------------------------------------------------------===//

FailureOr<LoopOp> tileToPCFLoop(RewriterBase &rewriter,
                                 PCFTilingOpInterface target,
                                 ScopeAttrInterface scope,
                                 ArrayRef<OpFoldResult> tileSizes) {
  return tileToPCFImpl<LoopOp>(rewriter, target, scope, tileSizes);
}

//===----------------------------------------------------------------------===//
// Public API: tileToPCFGeneric
//===----------------------------------------------------------------------===//

FailureOr<GenericOp> tileToPCFGeneric(RewriterBase &rewriter,
                                       PCFTilingOpInterface target,
                                       ScopeAttrInterface scope,
                                       ArrayRef<OpFoldResult> tileSizes) {
  return tileToPCFImpl<GenericOp>(rewriter, target, scope, tileSizes);
}

} // namespace mlir::iree_compiler::IREE::PCF
