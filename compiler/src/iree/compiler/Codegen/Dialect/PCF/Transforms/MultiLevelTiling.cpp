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

/// Returns true if the OpFoldResult is zero.
static bool isZeroTileSize(OpFoldResult ofr) {
  if (auto attr = dyn_cast<Attribute>(ofr)) {
    return cast<IntegerAttr>(attr).getInt() == 0;
  }
  return false;
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
  bindSymbols(b.getContext(), s0, s1);
  AffineMap mulMap = AffineMap::get(0, 2, s0 * s1, b.getContext());

  bindSymbols(b.getContext(), s0, s1, s2);
  AffineMap minMap3 = AffineMap::get(0, 3, {s0, s1 - s2}, b.getContext());

  int64_t tiledIdx = 0;
  for (int64_t i = 0; i < numDims; ++i) {
    if (isZeroTileSize(tileSizes[i])) {
      offsets[i] = b.getIndexAttr(0);
      sizes[i] = iterDomain[i].size;
    } else {
      OpFoldResult idVal = workerIds[tiledIdx];
      OpFoldResult offset = affine::makeComposedFoldedAffineApply(
          b, loc, mulMap, {idVal, tileSizes[i]});
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
        laneOffsets = SmallVector<OpFoldResult>(sgOffsets);
        laneSizes = SmallVector<OpFoldResult>(sgSizes);
      }

      // Branch on MMA vs regular tiling for the inner body.
      if (params.mmaKind) {
        // === MMA path: use InnerTileDescAttrInterface ===
        auto mmaDesc =
            cast<IREE::Codegen::InnerTileDescAttrInterface>(params.mmaKind);

        // Get undistributed tile types for operands and accumulator.
        SmallVector<VectorType> undistTileTypes;
        mmaDesc.getUndistributedTileTypes(undistTileTypes);

        // The last undistributed type is the accumulator.
        // For matmul: [MxKxf16, KxNxf16, MxNxf32]
        (void)undistTileTypes;
        (void)mmaDesc;

        // Iter arg types are the accumulator types (as tensors for now,
        // will be converted to vectors by inner_tiled conversion).
        // TODO: Determine whether iter args should be tensors or vectors.
        // For now, use the PCFTilingOpInterface path for init/writeback.
        SmallVector<Value> initValues = target.emitReductionInit(
            rewriter, sgReadwriteRefs, laneOffsets, laneSizes, params);

        // Determine reduction loop bounds.
        SmallVector<int64_t> reductionDims;
        for (int64_t i = 0; i < numDims; ++i) {
          if (iterTypes[i] == utils::IteratorType::reduction &&
              !isZeroTileSize(redTileSizes[i])) {
            reductionDims.push_back(i);
          }
        }

        SmallVector<Value> loopResults;
        if (!reductionDims.empty()) {
          int64_t redDim = reductionDims.front();
          Value lb = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                     rewriter.getIndexAttr(0));
          Value ub = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                     iterDomain[redDim].size);
          Value step = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                       redTileSizes[redDim]);

          auto forOp =
              scf::ForOp::create(rewriter, loc, lb, ub, step, initValues);
          {
            OpBuilder::InsertionGuard forGuard(rewriter);
            rewriter.setInsertionPointToStart(forOp.getBody());
            Value iv = forOp.getInductionVar();
            ValueRange iterArgs = forOp.getRegionIterArgs();

            SmallVector<OpFoldResult> tileOffsets(laneOffsets);
            SmallVector<OpFoldResult> tileSizes(laneSizes);
            tileOffsets[redDim] = iv;
            tileSizes[redDim] = redTileSizes[redDim];

            // Build operand info — same as regular path but with MMA
            // annotation. For now, use the same operand handling.
            SmallVector<DistributedOperandInfo> operandInfo;
            for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
              unsigned idx = static_cast<unsigned>(i);
              if (!tileableSet.contains(idx) ||
                  !isa<ShapedType>(op->getOperand(i).getType())) {
                operandInfo.push_back({op->getOperand(i), false});
                continue;
              }
              if (dpsInitIndices.contains(idx)) {
                int64_t rwIdx = operandToReadwriteIdx[i];
                if (rwIdx >= 0 &&
                    rwIdx < static_cast<int64_t>(iterArgs.size())) {
                  operandInfo.push_back({iterArgs[rwIdx], true});
                } else {
                  operandInfo.push_back({op->getOperand(i), false});
                }
              } else {
                int64_t roIdx = operandToReadonlyIdx[i];
                if (roIdx >= 0) {
                  Value sref = sgReadonlyRefs[roIdx];
                  if (llvm::is_contained(params.operandsToPromote, idx)) {
                    ShapedRefType srefType =
                        cast<ShapedRefType>(sref.getType());
                    int64_t rank = srefType.getRank();
                    SmallVector<Attribute> symbolNames;
                    for (int64_t d = 0; d < rank; ++d) {
                      symbolNames.push_back(rewriter.getStringAttr(
                          "operand_" + std::to_string(i) + "_dim_" +
                          std::to_string(d)));
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
                    operandInfo.push_back({promoted, false});
                  } else {
                    operandInfo.push_back({sref, false});
                  }
                } else {
                  operandInfo.push_back({op->getOperand(i), false});
                }
              }
            }

            // For MMA path, call getDistributedImplementation with the
            // MMA kind set. The implementation should create the inner_tiled
            // op. For now, fall back to the regular implementation and let
            // the inner_tiled conversion happen in a subsequent pass.
            // TODO: Create iree_codegen.inner_tiled directly here.
            SmallVector<DistributedResultInfo> resultInfo;
            for (int64_t i = 0, e = op->getNumResults(); i < e; ++i) {
              resultInfo.push_back({Value()});
            }
            FailureOr<TilingResult> tiledResult =
                target.getDistributedImplementation(
                    rewriter, tileOffsets, tileSizes, operandInfo, resultInfo);
            if (failed(tiledResult)) {
              return rewriter.notifyMatchFailure(
                  target, "MMA: getDistributedImplementation failed");
            }
            // Attach the mma_kind as an attribute on the tiled op for later
            // conversion.
            for (Operation *tiledOp : tiledResult->tiledOps) {
              tiledOp->setAttr("mma_kind", params.mmaKind);
            }
            scf::YieldOp::create(rewriter, loc, tiledResult->tiledValues);
          }
          loopResults = SmallVector<Value>(forOp.getResults());
        }

        // Writeback.
        if (!reductionDims.empty()) {
          target.emitReductionWriteback(rewriter, loopResults, sgReadwriteRefs,
                                        laneOffsets, laneSizes, params);
        }

        ReturnOp::create(rewriter, loc);
      } else {
        // === Regular path (non-MMA): use PCFTilingOpInterface ===
        SmallVector<Value> initValues = target.emitReductionInit(
            rewriter, sgReadwriteRefs, laneOffsets, laneSizes, params);

        // === Determine reduction loop bounds ===
        SmallVector<int64_t> reductionDims;
        for (int64_t i = 0; i < numDims; ++i) {
          if (iterTypes[i] == utils::IteratorType::reduction &&
              !isZeroTileSize(redTileSizes[i])) {
            reductionDims.push_back(i);
          }
        }

        SmallVector<Value> loopResults;
        if (!reductionDims.empty()) {
          // For simplicity, handle one reduction dim. Multiple reduction dims
          // would need nested loops.
          int64_t redDim = reductionDims.front();
          Value lb = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                     rewriter.getIndexAttr(0));
          Value ub = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                     iterDomain[redDim].size);
          Value step = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                       redTileSizes[redDim]);

          auto forOp =
              scf::ForOp::create(rewriter, loc, lb, ub, step, initValues);

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

            // === Build DistributedOperandInfo ===
            SmallVector<DistributedOperandInfo> operandInfo;
            for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
              unsigned idx = static_cast<unsigned>(i);
              if (!tileableSet.contains(idx) ||
                  !isa<ShapedType>(op->getOperand(i).getType())) {
                // Non-tileable or scalar: pass through.
                operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
                continue;
              }

              if (dpsInitIndices.contains(idx)) {
                // DPS init: use the iter_arg from the reduction loop.
                int64_t rwIdx = operandToReadwriteIdx[i];
                if (rwIdx >= 0 &&
                    rwIdx < static_cast<int64_t>(iterArgs.size())) {
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
                    // Create promote_operand op. Build symbol names for each
                    // dimension of the sref.
                    ShapedRefType srefType =
                        cast<ShapedRefType>(sref.getType());
                    int64_t rank = srefType.getRank();
                    SmallVector<Attribute> symbolNames;
                    for (int64_t d = 0; d < rank; ++d) {
                      std::string name = "operand_" + std::to_string(i) +
                                         "_dim_" + std::to_string(d);
                      symbolNames.push_back(rewriter.getStringAttr(name));
                    }
                    ArrayAttr symbols = rewriter.getArrayAttr(symbolNames);

                    // The promoted sref has the same shape but the subgroup
                    // scope (inner scope for shared memory).
                    ShapedRefType promotedType = ShapedRefType::get(
                        rewriter.getContext(), srefType.getShape(),
                        srefType.getElementType(),
                        cast<ScopeAttrInterface>(params.subgroup.scope));

                    // Get the promotion attribute for this operand.
                    IREE::GPU::PromotionAttr promotion =
                        getPromotionAttr(rewriter.getContext(), params, idx);

                    Value promoted = IREE::GPU::PromoteOperandOp::create(
                        rewriter, loc, promotedType, promotion, sref, symbols);
                    operandInfo.push_back({promoted, /*isTile=*/false});
                  } else {
                    operandInfo.push_back({sref, /*isTile=*/false});
                  }
                } else {
                  operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
                }
              }
            }

            // === Build DistributedResultInfo ===
            // Inside the reduction loop, results should be returned as tiles
            // (not written to srefs) since they become iter_args.
            SmallVector<DistributedResultInfo> resultInfo;
            for (int64_t i = 0, e = op->getNumResults(); i < e; ++i) {
              resultInfo.push_back({Value()}); // null destSref — return tile.
            }

            // === Call getDistributedImplementation ===
            FailureOr<TilingResult> tiledResult =
                target.getDistributedImplementation(
                    rewriter, tileOffsets, tileSizes, operandInfo, resultInfo);
            if (failed(tiledResult)) {
              return rewriter.notifyMatchFailure(
                  target, "getDistributedImplementation failed");
            }

            // Yield tiled values as loop results.
            scf::YieldOp::create(rewriter, loc, tiledResult->tiledValues);
          }

          loopResults = SmallVector<Value>(forOp.getResults());
        } else {
          // No reduction — just call getDistributedImplementation directly.
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
              operandInfo.push_back({sgReadwriteRefs[rwIdx], /*isTile=*/false});
            } else {
              int64_t roIdx = operandToReadonlyIdx[i];
              operandInfo.push_back({sgReadonlyRefs[roIdx], /*isTile=*/false});
            }
          }
          SmallVector<DistributedResultInfo> resultInfo;
          for (int64_t i = 0, e = op->getNumResults(); i < e; ++i) {
            resultInfo.push_back({sgReadwriteRefs[i]});
          }
          FailureOr<TilingResult> tiledResult =
              target.getDistributedImplementation(
                  rewriter, laneOffsets, laneSizes, operandInfo, resultInfo);
          if (failed(tiledResult)) {
            return rewriter.notifyMatchFailure(
                target, "getDistributedImplementation failed");
          }
          loopResults = SmallVector<Value>(tiledResult->tiledValues);
        }

        // === Emit writeback ===
        if (!reductionDims.empty()) {
          target.emitReductionWriteback(rewriter, loopResults, sgReadwriteRefs,
                                        laneOffsets, laneSizes, params);
        }

        // Create pcf.return for inner generic.
        ReturnOp::create(rewriter, loc);
      } // end else (regular path)
    }

    // Create pcf.return for outer generic.
    ReturnOp::create(rewriter, loc);
  }

  // Replace original op with outer generic's results.
  rewriter.replaceOp(op, outerGeneric.getResults());

  return outerGeneric;
}

} // namespace mlir::iree_compiler::IREE::PCF
