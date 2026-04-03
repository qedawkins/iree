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

/// Core implementation of tileToPCFLoop. Templated to support both LoopOp
/// and GenericOp, though only LoopOp is currently implemented.
template <typename OpTy>
static FailureOr<OpTy>
tileToPCFImpl(RewriterBase &rewriter, PCFTilingOpInterface target,
              ScopeAttrInterface scope, ArrayRef<OpFoldResult> tileSizes) {
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
    // Only shaped (tensor) operands can become sref args. Scalar operands
    // (e.g. the fill value in linalg.fill) are passed through directly.
    if (!isa<ShapedType>(op->getOperand(i).getType())) {
      continue;
    }
    if (dpsInitIndices.contains(idx)) {
      // DPS init -> readwrite sref.
      operandToReadwriteSrefIdx[i] =
          static_cast<int64_t>(readwriteInits.size());
      readwriteInits.push_back(op->getOperand(i));
    } else {
      // Tileable non-DPS-init -> readonly sref.
      operandToReadonlySrefIdx[i] = static_cast<int64_t>(readonlyInits.size());
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
    AffineMap mulMap = AffineMap::get(0, 2, d0 * d1, rewriter.getContext());

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
            rewriter, loc, mulMap, ArrayRef<OpFoldResult>{idVal, tileSizes[i]});
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

      // Scalar tileable operands (e.g. fill value) are passed through
      // directly; they don't have sref args.
      if (operandToReadonlySrefIdx[i] < 0 && operandToReadwriteSrefIdx[i] < 0) {
        operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
        continue;
      }

      // Pass the sref block arg directly. The distributed implementation
      // will read the appropriate tile based on the indexing map.
      Value sref;
      if (operandToReadonlySrefIdx[i] >= 0) {
        sref = readonlyRefs[operandToReadonlySrefIdx[i]];
      } else {
        sref = readwriteRefs[operandToReadwriteSrefIdx[i]];
      }
      operandInfo.push_back({sref, /*isTile=*/false});
    }

    // Step 8: Build DistributedResultInfo per result. Map each result to
    // its readwrite sref via the DPS init operand number, not result index.
    SmallVector<DistributedResultInfo> resultInfo;
    int64_t numResults = op->getNumResults();
    resultInfo.reserve(numResults);
    if (dpsOp) {
      for (OpOperand &init : dpsOp.getDpsInitsMutable()) {
        int64_t rwIdx = operandToReadwriteSrefIdx[init.getOperandNumber()];
        assert(rwIdx >= 0 &&
               "DPS init should have been mapped to readwrite sref");
        resultInfo.push_back({readwriteRefs[rwIdx]});
      }
    } else {
      for (int64_t i = 0; i < numResults; ++i) {
        resultInfo.push_back({readwriteRefs[i]});
      }
    }

    // Step 9: Check feasibility, then call getDistributedImplementation.
    if (failed(target.canDistribute(offsets, sizes, operandInfo, resultInfo))) {
      // Clean up the loop op on failure.
      rewriter.eraseOp(loopOp);
      return rewriter.notifyMatchFailure(target, "canDistribute check failed");
    }
    FailureOr<TilingResult> tilingResult = target.getDistributedImplementation(
        rewriter, offsets, sizes, operandInfo, resultInfo);
    if (failed(tilingResult)) {
      // Clean up the loop op on failure.
      rewriter.eraseOp(loopOp);
      return rewriter.notifyMatchFailure(
          target, "getDistributedImplementation failed unexpectedly");
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

      // Compute the result tile position using the TilingInterface.
      // This correctly handles non-identity output indexing maps.
      SmallVector<OpFoldResult> writeOffsets;
      SmallVector<OpFoldResult> writeSizes;
      TilingInterface tilingIface =
          cast<TilingInterface>(target.getOperation());
      LogicalResult posResult = tilingIface.getResultTilePosition(
          rewriter, i, offsets, sizes, writeOffsets, writeSizes);
      if (failed(posResult)) {
        // Fallback: use iteration domain offsets truncated to result rank.
        int64_t resultRank = cast<ShapedType>(tiledValue.getType()).getRank();
        writeOffsets.assign(offsets.begin(),
                            offsets.begin() +
                                std::min<int64_t>(offsets.size(), resultRank));
        writeSizes.assign(sizes.begin(),
                          sizes.begin() +
                              std::min<int64_t>(sizes.size(), resultRank));
        while (static_cast<int64_t>(writeOffsets.size()) < resultRank) {
          writeOffsets.push_back(rewriter.getIndexAttr(0));
        }
        while (static_cast<int64_t>(writeSizes.size()) < resultRank) {
          writeSizes.push_back(
              rewriter.getIndexAttr(cast<ShapedType>(tiledValue.getType())
                                        .getDimSize(writeSizes.size())));
        }
      }
      int64_t resultRank = cast<ShapedType>(tiledValue.getType()).getRank();
      SmallVector<OpFoldResult> writeStrides(resultRank,
                                             rewriter.getIndexAttr(1));
      WriteSliceOp::create(rewriter, loc, tiledValue, resultInfo[i].destSref,
                           writeOffsets, writeSizes, writeStrides);
    }

    // Step 10b: Create the pcf.return terminator.
    ReturnOp::create(rewriter, loc);
  }

  // Step 11: Replace original op results with the PCF op results.
  if (op->getNumResults() == 0) {
    rewriter.eraseOp(op);
  } else {
    rewriter.replaceOp(op, loopOp.getResults());
  }

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
