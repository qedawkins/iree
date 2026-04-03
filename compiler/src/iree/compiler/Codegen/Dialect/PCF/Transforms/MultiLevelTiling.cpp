// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTilingInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-pcf-multi-level-tiling"

namespace mlir::iree_compiler::IREE::PCF {

/// Returns true if the OpFoldResult is a constant zero.
static bool isZeroTileSize(OpFoldResult ofr) {
  std::optional<int64_t> val = getConstantIntValue(ofr);
  return val && *val == 0;
}

/// Computes tile offsets and sizes for a set of tiled dimensions given
/// worker IDs and tile sizes. Only dimensions with non-zero tile sizes
/// are tiled; the corresponding worker ID is used.
///
/// `workerIds` has one entry per tiled dimension (in order).
/// `tileSizes` has one entry per iteration domain dimension.
/// `domainSizes` has one entry per iteration domain dimension.
static void computeTileOffsetsAndSizes(OpBuilder &b, Location loc,
                                       ArrayRef<OpFoldResult> tileSizes,
                                       ArrayRef<Range> iterDomain,
                                       ArrayRef<BlockArgument> workerIds,
                                       SmallVectorImpl<OpFoldResult> &offsets,
                                       SmallVectorImpl<OpFoldResult> &sizes) {
  int64_t numDims = tileSizes.size();
  offsets.resize(numDims);
  sizes.resize(numDims);

  AffineExpr s0, s1, s2;
  bindSymbols(b.getContext(), s0, s1, s2);
  AffineMap minMap3 = AffineMap::get(0, 3, {s0, s1 - s2}, b.getContext());

  // Add domain offset: offset = domainOffset + id * tileSize.
  AffineMap addMulMap = AffineMap::get(0, 3, s0 + s1 * s2, b.getContext());

  int64_t tiledIdx = 0;
  for (int64_t i = 0; i < numDims; ++i) {
    if (isZeroTileSize(tileSizes[i])) {
      offsets[i] = iterDomain[i].offset;
      sizes[i] = iterDomain[i].size;
    } else {
      OpFoldResult idVal = workerIds[tiledIdx];
      OpFoldResult offset = affine::makeComposedFoldedAffineApply(
          b, loc, addMulMap, {iterDomain[i].offset, idVal, tileSizes[i]});
      offsets[i] = offset;
      OpFoldResult size = affine::makeComposedFoldedAffineMin(
          b, loc, minMap3, {tileSizes[i], iterDomain[i].size, offset});
      sizes[i] = size;
      ++tiledIdx;
    }
  }
}

/// Counts the number of non-zero tile sizes.
static int64_t countTiledDims(ArrayRef<OpFoldResult> tileSizes) {
  int64_t count = 0;
  for (OpFoldResult ts : tileSizes) {
    if (!isZeroTileSize(ts)) {
      ++count;
    }
  }
  return count;
}

/// Returns the promotion attribute for the given operand index.
/// If promotionTypes is set, uses the corresponding entry.
/// Otherwise falls back to DerivedThreadConfig.
static IREE::GPU::PromotionAttr
getPromotionAttr(MLIRContext *ctx, const MultiLevelTilingParams &params,
                 unsigned operandIdx) {
  // Find the position of operandIdx in operandsToPromote.
  for (int64_t i = 0, e = params.operandsToPromote.size(); i < e; ++i) {
    if (params.operandsToPromote[i] == operandIdx) {
      if (i < static_cast<int64_t>(params.promotionTypes.size())) {
        return cast<IREE::GPU::PromotionAttr>(params.promotionTypes[i]);
      }
      break;
    }
  }
  // Default.
  return cast<IREE::GPU::PromotionAttr>(
      IREE::GPU::DerivedThreadConfigAttr::get(ctx));
}

/// Builds DistributedOperandInfo for all operands, handling promotion and
/// sref routing. Used by both MMA and regular tiling paths.
static SmallVector<DistributedOperandInfo>
buildOperandInfo(RewriterBase &rewriter, Location loc, Operation *op,
                 const DenseSet<unsigned> &tileableSet,
                 const DenseSet<unsigned> &dpsInitIndices,
                 ArrayRef<int64_t> operandToReadonlyIdx,
                 ArrayRef<int64_t> operandToReadwriteIdx,
                 ArrayRef<BlockArgument> sgReadonlyRefs, ValueRange iterArgs,
                 const MultiLevelTilingParams &params) {
  SmallVector<DistributedOperandInfo> operandInfo;
  for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    unsigned idx = static_cast<unsigned>(i);
    if (!tileableSet.contains(idx) ||
        !isa<ShapedType>(op->getOperand(i).getType())) {
      operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
      continue;
    }

    if (dpsInitIndices.contains(idx)) {
      // DPS init: use the iter_arg from the reduction loop.
      int64_t rwIdx = operandToReadwriteIdx[i];
      if (rwIdx >= 0 && rwIdx < static_cast<int64_t>(iterArgs.size())) {
        operandInfo.push_back({iterArgs[rwIdx], /*isTile=*/true});
      } else {
        operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
      }
    } else {
      // Input: use the readonly sref from outer generic.
      int64_t roIdx = operandToReadonlyIdx[i];
      if (roIdx >= 0) {
        Value sref = sgReadonlyRefs[roIdx];

        // Check if this operand should be promoted.
        if (llvm::is_contained(params.operandsToPromote, idx)) {
          ShapedRefType srefType = cast<ShapedRefType>(sref.getType());
          int64_t rank = srefType.getRank();
          SmallVector<Attribute> symbolNames;
          for (int64_t d = 0; d < rank; ++d) {
            symbolNames.push_back(rewriter.getStringAttr(
                "operand_" + std::to_string(i) + "_dim_" + std::to_string(d)));
          }
          ShapedRefType promotedType = ShapedRefType::get(
              rewriter.getContext(), srefType.getShape(),
              srefType.getElementType(),
              cast<ScopeAttrInterface>(params.subgroup.scope));
          IREE::GPU::PromotionAttr promotion =
              getPromotionAttr(rewriter.getContext(), params, idx);
          Value promoted = IREE::GPU::PromoteOperandOp::create(
              rewriter, loc, promotedType, promotion, sref,
              rewriter.getArrayAttr(symbolNames));
          operandInfo.push_back({promoted, /*isTile=*/false});
        } else {
          operandInfo.push_back({sref, /*isTile=*/false});
        }
      } else {
        operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
      }
    }
  }
  return operandInfo;
}

/// Callback invoked on each TilingResult inside the reduction loop body.
/// Can be used to attach attributes (e.g., mma_kind) to the tiled ops.
using PostTilingCallback =
    function_ref<void(RewriterBase &rewriter, TilingResult &tiledResult)>;

/// Builds a reduction loop (scf.for) over the first reduction dimension,
/// calling getDistributedImplementation inside the loop body. The
/// |postTilingCb| callback is invoked on the tiled result before yielding.
///
/// Populates |loopResults| with the scf.for results. Emits
/// emitReductionWriteback after the loop if |reductionDims| is non-empty.
///
/// Returns failure if getDistributedImplementation fails.
static FailureOr<SmallVector<Value>> buildReductionLoop(
    RewriterBase &rewriter, Location loc, PCFTilingOpInterface target,
    Operation *op, ArrayRef<Range> iterDomain,
    ArrayRef<utils::IteratorType> iterTypes,
    ArrayRef<OpFoldResult> redTileSizes, ArrayRef<OpFoldResult> laneOffsets,
    ArrayRef<OpFoldResult> laneSizes, ArrayRef<int64_t> reductionDims,
    const DenseSet<unsigned> &tileableSet,
    const DenseSet<unsigned> &dpsInitIndices,
    ArrayRef<int64_t> operandToReadonlyIdx,
    ArrayRef<int64_t> operandToReadwriteIdx,
    ArrayRef<BlockArgument> sgReadonlyRefs,
    ArrayRef<BlockArgument> sgReadwriteRefs, ValueRange initValues,
    const MultiLevelTilingParams &params, PostTilingCallback postTilingCb) {
  int64_t redDim = reductionDims.front();
  Value lb =
      getValueOrCreateConstantIndexOp(rewriter, loc, rewriter.getIndexAttr(0));
  Value ub = getValueOrCreateConstantIndexOp(rewriter, loc,
                                             iterDomain[redDim].size);
  Value step =
      getValueOrCreateConstantIndexOp(rewriter, loc, redTileSizes[redDim]);

  auto forOp = scf::ForOp::create(rewriter, loc, lb, ub, step, initValues);
  {
    OpBuilder::InsertionGuard forGuard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());

    Value iv = forOp.getInductionVar();
    ValueRange iterArgs = forOp.getRegionIterArgs();

    // Compute tile offsets including reduction.
    SmallVector<OpFoldResult> tileOffsets(laneOffsets);
    SmallVector<OpFoldResult> tileSizes(laneSizes);
    tileOffsets[redDim] = iv;
    tileSizes[redDim] = redTileSizes[redDim];

    // Build operand info using shared helper.
    SmallVector<DistributedOperandInfo> operandInfo =
        buildOperandInfo(rewriter, loc, op, tileableSet, dpsInitIndices,
                         operandToReadonlyIdx, operandToReadwriteIdx,
                         sgReadonlyRefs, iterArgs, params);

    // Inside the reduction loop, results are returned as tiles (not written
    // to srefs) since they become iter_args.
    SmallVector<DistributedResultInfo> resultInfo;
    for (int64_t i = 0, e = op->getNumResults(); i < e; ++i) {
      resultInfo.push_back({Value()});
    }

    FailureOr<TilingResult> tiledResult =
        target.getDistributedImplementation(rewriter, tileOffsets, tileSizes,
                                            operandInfo, resultInfo);
    if (failed(tiledResult)) {
      return rewriter.notifyMatchFailure(
          target, "getDistributedImplementation failed");
    }

    // Apply post-tiling modifications (e.g., attaching mma_kind attribute).
    if (postTilingCb) {
      postTilingCb(rewriter, *tiledResult);
    }

    scf::YieldOp::create(rewriter, loc, tiledResult->tiledValues);
  }

  SmallVector<Value> loopResults(forOp.getResults().begin(),
                                 forOp.getResults().end());

  // Emit writeback after the reduction loop.
  target.emitReductionWriteback(rewriter, loopResults, sgReadwriteRefs,
                                laneOffsets, laneSizes, params);

  return loopResults;
}

FailureOr<GenericOp>
applyMultiLevelTiling(RewriterBase &rewriter, PCFTilingOpInterface target,
                      const MultiLevelTilingParams &params) {
  Location loc = target.getLoc();
  Operation *op = target.getOperation();

  // Get iteration domain and iterator types.
  auto tilingIface = cast<TilingInterface>(op);
  SmallVector<Range> iterDomain = tilingIface.getIterationDomain(rewriter);
  SmallVector<utils::IteratorType> iterTypes =
      tilingIface.getLoopIteratorTypes();
  int64_t numDims = iterDomain.size();

  // Classify operands: tileable inputs → readonly, DPS inits → readwrite.
  SmallVector<unsigned> tileableIndices = target.getTileableOperandIndices();
  DenseSet<unsigned> tileableSet(tileableIndices.begin(),
                                 tileableIndices.end());
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);

  DenseSet<unsigned> dpsInitIndices;
  if (dpsOp) {
    for (OpOperand &init : dpsOp.getDpsInitsMutable()) {
      dpsInitIndices.insert(init.getOperandNumber());
    }
  }

  SmallVector<Value> readonlyInits;
  SmallVector<Value> readwriteInits;
  SmallVector<int64_t> operandToReadonlyIdx(op->getNumOperands(), -1);
  SmallVector<int64_t> operandToReadwriteIdx(op->getNumOperands(), -1);

  for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    unsigned idx = static_cast<unsigned>(i);
    if (!tileableSet.contains(idx)) {
      continue;
    }
    if (!isa<ShapedType>(op->getOperand(i).getType())) {
      continue;
    }
    if (dpsInitIndices.contains(idx)) {
      operandToReadwriteIdx[i] = static_cast<int64_t>(readwriteInits.size());
      readwriteInits.push_back(op->getOperand(i));
    } else {
      operandToReadonlyIdx[i] = static_cast<int64_t>(readonlyInits.size());
      readonlyInits.push_back(op->getOperand(i));
    }
  }

  // Pad tile sizes to iteration domain size.
  SmallVector<OpFoldResult> sgTileSizes(numDims, rewriter.getIndexAttr(0));
  for (int64_t i = 0,
               e = std::min(numDims, static_cast<int64_t>(
                                         params.subgroup.tileSizes.size()));
       i < e; ++i) {
    sgTileSizes[i] = params.subgroup.tileSizes[i];
  }
  SmallVector<OpFoldResult> laneTileSizes(numDims, rewriter.getIndexAttr(0));
  for (int64_t i = 0, e = std::min(numDims, static_cast<int64_t>(
                                                params.lane.tileSizes.size()));
       i < e; ++i) {
    laneTileSizes[i] = params.lane.tileSizes[i];
  }
  SmallVector<OpFoldResult> redTileSizes(numDims, rewriter.getIndexAttr(0));
  for (int64_t i = 0,
               e = std::min(numDims, static_cast<int64_t>(
                                         params.reductionTileSizes.size()));
       i < e; ++i) {
    redTileSizes[i] = params.reductionTileSizes[i];
  }

  // Count subgroup-tiled dimensions (non-zero subgroup tiles on parallel dims).
  int64_t numSgIterators = countTiledDims(sgTileSizes);
  if (numSgIterators == 0) {
    return rewriter.notifyMatchFailure(target, "no subgroup-tiled dimensions");
  }

  // Count lane-tiled dimensions.
  int64_t numLaneIterators = countTiledDims(laneTileSizes);

  // === Create outer pcf.generic (subgroup scope) ===
  rewriter.setInsertionPoint(op);
  auto sgScope = cast<ScopeAttrInterface>(params.subgroup.scope);
  GenericOp outerGeneric = GenericOp::create(
      rewriter, loc, sgScope, readonlyInits, readwriteInits, numSgIterators);

  // === Populate initializer with promotion symbols ===
  if (!params.operandsToPromote.empty()) {
    OpBuilder::InsertionGuard initGuard(rewriter);
    Region &initRegion = outerGeneric.getInitializer();
    Block *initBlock = rewriter.createBlock(&initRegion);
    rewriter.setInsertionPointToStart(initBlock);

    // For each promoted operand, define symbols for the tile sizes of each
    // dimension. The tile size is determined by the indexing map: parallel
    // dims use subgroup tile sizes, reduction dims use reduction tile sizes.
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    for (unsigned promIdx : params.operandsToPromote) {
      if (!linalgOp) {
        break;
      }
      OpOperand &opOperand = op->getOpOperand(promIdx);
      AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&opOperand);

      for (auto [d, expr] : llvm::enumerate(indexingMap.getResults())) {
        std::string symName =
            "operand_" + std::to_string(promIdx) + "_dim_" + std::to_string(d);
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        OpFoldResult tileSize;
        if (dimExpr) {
          unsigned pos = dimExpr.getPosition();
          // Use subgroup tile for parallel dims, reduction tile for
          // reduction dims.
          if (iterTypes[pos] == utils::IteratorType::reduction) {
            tileSize = (pos < redTileSizes.size()) ? redTileSizes[pos]
                                                   : rewriter.getIndexAttr(0);
          } else {
            tileSize = (pos < sgTileSizes.size()) ? sgTileSizes[pos]
                                                  : rewriter.getIndexAttr(0);
          }
        } else {
          // Non-simple indexing — use 0 as placeholder.
          tileSize = rewriter.getIndexAttr(0);
        }
        Value tileSizeVal =
            getValueOrCreateConstantIndexOp(rewriter, loc, tileSize);
        IndexSymbolOp::create(rewriter, loc, rewriter.getStringAttr(symName),
                              tileSizeVal);
      }
    }

    // Yield nothing from the initializer.
    YieldOp::create(rewriter, loc, ValueRange{});
  }

  // === Build outer body ===
  {
    OpBuilder::InsertionGuard outerGuard(rewriter);
    rewriter.setInsertionPointToStart(&outerGeneric.getRegion().front());

    ArrayRef<BlockArgument> sgReadonlyRefs = outerGeneric.getReadonlyRefArgs();
    ArrayRef<BlockArgument> sgReadwriteRefs = outerGeneric.getRegionRefArgs();
    ArrayRef<BlockArgument> sgIdArgs = outerGeneric.getIdArgs();

    // Compute subgroup tile offsets/sizes.
    SmallVector<OpFoldResult> sgOffsets, sgSizes;
    computeTileOffsetsAndSizes(rewriter, loc, sgTileSizes, iterDomain, sgIdArgs,
                               sgOffsets, sgSizes);

    // Build a sub-domain for the lane level (subgroup-relative).
    SmallVector<Range> sgDomain(numDims);
    for (int64_t i = 0; i < numDims; ++i) {
      sgDomain[i].offset = sgOffsets[i];
      sgDomain[i].size = sgSizes[i];
      sgDomain[i].stride = rewriter.getIndexAttr(1);
    }

    // === Create inner pcf.generic (lane scope) ===
    // No inits/results — captures outer srefs directly.
    auto laneScope = cast<ScopeAttrInterface>(params.lane.scope);
    GenericOp innerGeneric =
        numLaneIterators > 0
            ? GenericOp::create(rewriter, loc, laneScope, numLaneIterators)
            : GenericOp::create(rewriter, loc, laneScope, /*numIterators=*/1);

    {
      OpBuilder::InsertionGuard innerGuard(rewriter);
      rewriter.setInsertionPointToStart(&innerGeneric.getRegion().front());

      ArrayRef<BlockArgument> laneIdArgs = innerGeneric.getIdArgs();

      // Compute lane tile offsets/sizes (relative to subgroup tile).
      SmallVector<OpFoldResult> laneOffsets, laneSizes;
      if (numLaneIterators > 0) {
        computeTileOffsetsAndSizes(rewriter, loc, laneTileSizes, sgDomain,
                                   laneIdArgs, laneOffsets, laneSizes);
      } else {
        // No lane tiling — use full subgroup tile.
        laneOffsets = llvm::to_vector(sgOffsets);
        laneSizes = llvm::to_vector(sgSizes);
      }

      // Emit reduction init and determine reduction dimensions. Both MMA
      // and regular paths share the same reduction loop structure.
      SmallVector<Value> initValues = target.emitReductionInit(
          rewriter, sgReadwriteRefs, laneOffsets, laneSizes, params);

      SmallVector<int64_t> reductionDims;
      for (int64_t i = 0; i < numDims; ++i) {
        if (iterTypes[i] == utils::IteratorType::reduction &&
            !isZeroTileSize(redTileSizes[i])) {
          reductionDims.push_back(i);
        }
      }

      if (!reductionDims.empty()) {
        if (!params.mmaKind && reductionDims.size() > 1) {
          return rewriter.notifyMatchFailure(
              target, "multiple reduction dimensions not yet supported");
        }

        // Post-tiling callback: attach mma_kind attribute for MMA path.
        // The lambda must outlive the function_ref, so declare it in the
        // same scope where buildReductionLoop is called.
        Attribute mmaKind = params.mmaKind;
        auto mmaCallback = [mmaKind](RewriterBase & /*rewriter*/,
                                     TilingResult &tiledResult) {
          for (Operation *tiledOp : tiledResult.tiledOps) {
            tiledOp->setAttr("mma_kind", mmaKind);
          }
        };
        PostTilingCallback postTilingCb =
            mmaKind ? PostTilingCallback(mmaCallback) : nullptr;

        FailureOr<SmallVector<Value>> loopResults = buildReductionLoop(
            rewriter, loc, target, op, iterDomain, iterTypes, redTileSizes,
            laneOffsets, laneSizes, reductionDims, tileableSet, dpsInitIndices,
            operandToReadonlyIdx, operandToReadwriteIdx, sgReadonlyRefs,
            sgReadwriteRefs, initValues, params, postTilingCb);
        if (failed(loopResults)) {
          return failure();
        }
      } else {
        // No reduction -- call getDistributedImplementation directly.
        SmallVector<DistributedOperandInfo> operandInfo;
        for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
          unsigned idx = static_cast<unsigned>(i);
          if (!tileableSet.contains(idx) ||
              !isa<ShapedType>(op->getOperand(i).getType())) {
            operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
            continue;
          }
          if (dpsInitIndices.contains(idx)) {
            int64_t rwIdx = operandToReadwriteIdx[i];
            assert(rwIdx >= 0 &&
                   "DPS init should have been mapped to readwrite sref");
            operandInfo.push_back({sgReadwriteRefs[rwIdx], /*isTile=*/false});
          } else {
            int64_t roIdx = operandToReadonlyIdx[i];
            assert(roIdx >= 0 &&
                   "tileable input should have been mapped to readonly sref");
            operandInfo.push_back({sgReadonlyRefs[roIdx], /*isTile=*/false});
          }
        }
        // Build result info: map each result to its readwrite sref via the
        // DPS init operand, not via result index directly.
        SmallVector<DistributedResultInfo> resultInfo;
        if (dpsOp) {
          for (auto [i, init] :
               llvm::enumerate(dpsOp.getDpsInitsMutable())) {
            int64_t rwIdx =
                operandToReadwriteIdx[init.getOperandNumber()];
            assert(rwIdx >= 0 &&
                   "DPS init should have been mapped to readwrite sref");
            resultInfo.push_back({sgReadwriteRefs[rwIdx]});
          }
        } else {
          for (int64_t i = 0, e = op->getNumResults(); i < e; ++i) {
            resultInfo.push_back({sgReadwriteRefs[i]});
          }
        }
        FailureOr<TilingResult> tiledResult =
            target.getDistributedImplementation(
                rewriter, laneOffsets, laneSizes, operandInfo, resultInfo);
        if (failed(tiledResult)) {
          return rewriter.notifyMatchFailure(
              target, "getDistributedImplementation failed");
        }
      }

      // Create pcf.return for inner generic.
      ReturnOp::create(rewriter, loc);
    }

    // Create pcf.return for outer generic.
    ReturnOp::create(rewriter, loc);
  }

  // Replace original op with outer generic's results.
  rewriter.replaceOp(op, outerGeneric.getResults());

  return outerGeneric;
}

} // namespace mlir::iree_compiler::IREE::PCF
