// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LLVMGPUStreamKTile.cpp - Stream-K tiling transformation -----------===//
//
// Implements the Stream-K tiling transformation for GPU targets. Given a
// TilingInterface op with a `streamed_reduction` tiling level in its lowering
// config, this pass linearizes the output_tiles x k_tiles work space and
// distributes it across workgroups using a pcf.generic.
//
// The produced IR structure:
//   1. pcf.generic scope(#workgroup_scope) with initializer (scratch + counter)
//   2. Work range computation per workgroup (items_per_wg, my_start, my_end).
//   3. Outer scf.for over output tiles this workgroup touches.
//   4. Inner scf.for accumulating k-tile partials via TilingInterface.
//   5. pcf.stream_k_recombine per output tile with combiner + writeback.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-llvmgpu-stream-k-tile"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUSTREAMKTILEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Find a linalg op with `streamed_reduction` in its lowering config.
/// Stream-K tiling currently only supports linalg ops (for DPS init access).
static linalg::LinalgOp findStreamKTarget(Operation *funcOp) {
  linalg::LinalgOp target;
  funcOp->walk([&](linalg::LinalgOp op) {
    if (target) {
      return WalkResult::interrupt();
    }
    IREE::Codegen::LoweringConfigAttrInterface config = getLoweringConfig(op);
    if (!config) {
      return WalkResult::advance();
    }
    if (config.hasTilingLevel(
            llvm::to_underlying(IREE::GPU::TilingLevel::StreamedReduction))) {
      target = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return target;
}

/// Compute ceil(a, b) for index values.
static Value ceilDiv(OpBuilder &builder, Location loc, Value a, Value b) {
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value aMinusOne = arith::SubIOp::create(builder, loc, a, one,
                                          arith::IntegerOverflowFlags::nsw);
  Value sum = arith::AddIOp::create(builder, loc, aMinusOne, b,
                                    arith::IntegerOverflowFlags::nsw);
  return arith::DivUIOp::create(builder, loc, sum, b);
}

/// Classify iteration dimensions into parallel and reduction.
/// Returns {parallelDimIndices, reductionDimIndices}.
static std::pair<SmallVector<unsigned>, SmallVector<unsigned>>
classifyDims(TilingInterface target) {
  SmallVector<utils::IteratorType> iterTypes = target.getLoopIteratorTypes();
  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  for (auto [i, iterType] : llvm::enumerate(iterTypes)) {
    if (iterType == utils::IteratorType::parallel) {
      parallelDims.push_back(i);
    } else {
      reductionDims.push_back(i);
    }
  }
  return {parallelDims, reductionDims};
}

/// Extract a static integer value from a Range's size field.
static std::optional<int64_t> getStaticRangeSize(OpFoldResult size) {
  if (auto attr = dyn_cast<Attribute>(size)) {
    return cast<IntegerAttr>(attr).getInt();
  }
  if (auto val = dyn_cast<Value>(size)) {
    if (auto cstOp =
            dyn_cast_if_present<arith::ConstantIndexOp>(val.getDefiningOp())) {
      return cstOp.value();
    }
  }
  return std::nullopt;
}

/// Decompose a linear index into per-dimension coordinates.
/// dimSizes are ordered outermost-to-innermost; innermost varies fastest.
static SmallVector<Value> decomposeLinearIndex(OpBuilder &builder, Location loc,
                                               Value linearIdx,
                                               ArrayRef<int64_t> dimSizes) {
  SmallVector<Value> coords;
  Value remaining = linearIdx;
  for (int64_t i = dimSizes.size() - 1; i >= 0; --i) {
    Value dimSize =
        arith::ConstantIndexOp::create(builder, loc, dimSizes[i]);
    if (i > 0) {
      Value coord =
          arith::RemUIOp::create(builder, loc, remaining, dimSize);
      coords.push_back(coord);
      remaining = arith::DivUIOp::create(builder, loc, remaining, dimSize);
    } else {
      coords.push_back(remaining);
    }
  }
  std::reverse(coords.begin(), coords.end());
  return coords;
}

/// Compute max(a, b) for index values.
static Value indexMax(OpBuilder &builder, Location loc, Value a, Value b) {
  Value cond =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt, a, b);
  return arith::SelectOp::create(builder, loc, cond, a, b);
}

/// Compute min(a, b) for index values.
static Value indexMin(OpBuilder &builder, Location loc, Value a, Value b) {
  Value cond =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult, a, b);
  return arith::SelectOp::create(builder, loc, cond, a, b);
}

struct LLVMGPUStreamKTilePass final
    : impl::LLVMGPUStreamKTilePassBase<LLVMGPUStreamKTilePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::PCF::PCFDialect, IREE::Codegen::IREECodegenDialect,
                    arith::ArithDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};

void LLVMGPUStreamKTilePass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  MLIRContext *ctx = &getContext();
  Location loc = funcOp.getLoc();

  // === Step 1: Find the target linalg op. ===
  linalg::LinalgOp target = findStreamKTarget(funcOp);
  if (!target) {
    return; // No Stream-K op in this function.
  }

  // Validate: target must have exactly one result feeding a store.
  if (target->getNumResults() != 1) {
    target.emitOpError("Stream-K tiling requires exactly one result");
    return signalPassFailure();
  }

  IREE::Codegen::LoweringConfigAttrInterface configIface =
      getLoweringConfig(target);

  // Build a lowering config for the inner tiled op by stripping the levels
  // consumed by this pass (workgroup, streamed_reduction) and keeping
  // everything else (subgroup, reduction, mma_kind, promote_operands, etc.).
  IREE::GPU::LoweringConfigAttr innerLoweringConfig;
  if (auto gpuConfig =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(target)) {
    StringRef wgName =
        getTilingLevelName(IREE::GPU::TilingLevel::Workgroup);
    StringRef streamName =
        getTilingLevelName(IREE::GPU::TilingLevel::StreamedReduction);
    SmallVector<NamedAttribute> innerAttrs;
    for (NamedAttribute attr : gpuConfig.getAttributes()) {
      StringRef name = attr.getName().getValue();
      if (name != wgName && name != streamName) {
        innerAttrs.push_back(attr);
      }
    }
    innerLoweringConfig = IREE::GPU::LoweringConfigAttr::get(
        ctx, DictionaryAttr::get(ctx, innerAttrs));
  }

  // === Step 2: Get tile sizes. ===
  unsigned wgLevel = llvm::to_underlying(IREE::GPU::TilingLevel::Workgroup);
  unsigned streamLevel =
      llvm::to_underlying(IREE::GPU::TilingLevel::StreamedReduction);

  SmallVector<int64_t> wgTileSizes =
      configIface.getStaticTilingLevelSizes(wgLevel, target);
  SmallVector<int64_t> streamTileSizes =
      configIface.getStaticTilingLevelSizes(streamLevel, target);

  // === Step 3: Classify dims and build merged tile sizes. ===
  TilingInterface tilingTarget = cast<TilingInterface>(target.getOperation());
  auto [parallelDims_, reductionDims_] = classifyDims(tilingTarget);
  // Copy out of structured binding so we can capture in lambdas (C++17).
  SmallVector<unsigned> parallelDims = parallelDims_;
  SmallVector<unsigned> reductionDims = reductionDims_;
  unsigned numDims = tilingTarget.getLoopIteratorTypes().size();
  SmallVector<int64_t> tileSizes(numDims, 0);
  for (unsigned d : parallelDims) {
    if (d < wgTileSizes.size()) {
      tileSizes[d] = wgTileSizes[d];
    }
  }
  for (unsigned d : reductionDims) {
    if (d < streamTileSizes.size()) {
      tileSizes[d] = streamTileSizes[d];
    }
  }

  // === Step 4: Get iteration domain and require static shapes. ===
  OpBuilder builder(ctx);
  builder.setInsertionPoint(target);
  SmallVector<Range> iterDomain = tilingTarget.getIterationDomain(builder);

  SmallVector<int64_t> dimSizes;
  for (Range &range : iterDomain) {
    std::optional<int64_t> staticSize = getStaticRangeSize(range.size);
    if (!staticSize) {
      target.emitOpError("Stream-K tiling requires static iteration domain");
      return signalPassFailure();
    }
    dimSizes.push_back(*staticSize);
  }

  // === Step 5: Compute tile counts, outputTiles, kTilesPerOut, totalWork. ===
  SmallVector<int64_t> tileCounts;
  for (auto [dimSize, tileSize] : llvm::zip(dimSizes, tileSizes)) {
    if (tileSize == 0) {
      tileCounts.push_back(1);
    } else {
      tileCounts.push_back((dimSize + tileSize - 1) / tileSize);
    }
  }

  int64_t outputTiles = 1;
  for (unsigned d : parallelDims) {
    outputTiles *= tileCounts[d];
  }
  int64_t kTilesPerOut = 1;
  for (unsigned d : reductionDims) {
    kTilesPerOut *= tileCounts[d];
  }
  int64_t totalWork = outputTiles * kTilesPerOut;

  SmallVector<int64_t> parallelTileCounts;
  for (unsigned d : parallelDims) {
    parallelTileCounts.push_back(tileCounts[d]);
  }

  SmallVector<int64_t> reductionTileCounts;
  for (unsigned d : reductionDims) {
    reductionTileCounts.push_back(tileCounts[d]);
  }

  // === Step 6: Find output chain. ===
  Operation *storeOp = nullptr;
  Value targetResult = target->getResult(0);
  for (Operation *user : targetResult.getUsers()) {
    if (isa<IREE::Codegen::StoreToBufferOp>(user) ||
        isa<IREE::TensorExt::DispatchTensorStoreOp>(user)) {
      storeOp = user;
      break;
    }
  }
  if (!storeOp) {
    target.emitOpError("could not find dispatch tensor store for output");
    return signalPassFailure();
  }

  // Find the fill op (optional) feeding into the target's DPS output init.
  Value outInit = target.getDpsInits()[0];
  auto fillOp = outInit.getDefiningOp<linalg::FillOp>();
  Value outTensor;
  if (fillOp) {
    outTensor = fillOp.getOutputs()[0]; // tensor.empty() result.
  } else {
    outTensor = outInit;
  }

  auto outTensorType = cast<RankedTensorType>(outTensor.getType());
  Type elemType = outTensorType.getElementType();

  // === Step 7: Get CU count from GPU target. ===
  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(funcOp);
  int64_t cuCount = 512; // Conservative default.
  if (gpuTarget) {
    if (auto chip = gpuTarget.getChip()) {
      cuCount = chip.getWgpCount();
    }
  }

  // === Step 8: Compute scratch + counter types. ===
  IREE::PCF::ScopeAttrInterface wgScope =
      cast<IREE::PCF::ScopeAttrInterface>(
          IREE::Codegen::WorkgroupScopeAttr::get(ctx, /*linearize=*/true));

  SmallVector<int64_t> tileShape;
  for (unsigned d : parallelDims) {
    tileShape.push_back(tileSizes[d] == 0 ? dimSizes[d] : tileSizes[d]);
  }

  int64_t maxNumInGroup = std::min(cuCount, kTilesPerOut);
  SmallVector<int64_t> scratchShape;
  // Scratch is partitioned per output tile: each tile gets maxNumInGroup
  // partial-tile slots.  Total rows = outputTiles * maxNumInGroup * tileM.
  scratchShape.push_back(outputTiles * maxNumInGroup * tileShape[0]);
  for (int64_t i = 1, e = tileShape.size(); i < e; ++i) {
    scratchShape.push_back(tileShape[i]);
  }
  auto scratchSrefType =
      IREE::PCF::ShapedRefType::get(ctx, scratchShape, elemType, wgScope);
  // Counter is a 1D array with one entry per output tile.  Each entry tracks
  // how many workgroups have contributed to that tile (for recombine).
  auto counterSrefType = IREE::PCF::ShapedRefType::get(
      ctx, {outputTiles}, builder.getI32Type(), wgScope);

  // === Step 9: Create workgroup count hint. ===
  IREE::Codegen::WorkgroupCountHintOp::create(
      builder, loc, ArrayRef<OpFoldResult>{builder.getIndexAttr(cuCount)});

  // === Step 10: Create pcf.generic with initializer. ===
  auto genericOp = IREE::PCF::GenericOp::create(
      builder, loc,
      /*resultTypes=*/TypeRange{outTensorType},
      /*scope=*/wgScope,
      /*inits=*/ValueRange{outTensor},
      /*dynamicSizes=*/ValueRange{},
      /*isTied=*/ArrayRef<bool>{true},
      /*numIterators=*/1);

  // Populate initializer region with scratch + counter allocs.
  {
    Region &initRegion = genericOp.getInitializer();
    Block *initBlock = builder.createBlock(&initRegion);
    builder.setInsertionPointToStart(initBlock);
    Value scratchAlloc =
        IREE::PCF::AllocOp::create(builder, loc, scratchSrefType).getResult();
    Value counterAlloc =
        IREE::PCF::AllocOp::create(builder, loc, counterSrefType).getResult();
    IREE::PCF::YieldOp::create(builder, loc,
                                ValueRange{scratchAlloc, counterAlloc});
  }

  // Add leading args (from initializer yields) to the execute region.
  Block &execBlock = genericOp.getRegion().front();
  execBlock.insertArgument(0u, scratchSrefType, loc);
  execBlock.insertArgument(1u, counterSrefType, loc);
  genericOp.setNumLeadingArgs(2);

  // Block args: [scratch, counter, out_sref, wg_id, wg_count]
  Value scratchArg = genericOp.getLeadingArgs()[0];
  Value counterArg = genericOp.getLeadingArgs()[1];
  Value outRef = genericOp.getRegionRefArgs()[0];
  Value wgId = genericOp.getIdArgs()[0];
  Value wgCount = genericOp.getCountArgs()[0];

  // === Step 11: Work range computation inside execute region. ===
  builder.setInsertionPointToEnd(&execBlock);

  Value cTotalWork = arith::ConstantIndexOp::create(builder, loc, totalWork);
  Value cKTilesPerOut =
      arith::ConstantIndexOp::create(builder, loc, kTilesPerOut);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

  // items_per_wg = ceildiv(totalWork, wg_count)
  Value itemsPerWg = ceilDiv(builder, loc, cTotalWork, wgCount);

  // my_start = wg_id * items_per_wg
  Value myStart = arith::MulIOp::create(builder, loc, wgId, itemsPerWg,
                                        arith::IntegerOverflowFlags::nsw);

  // my_end = min(my_start + items_per_wg, totalWork)
  Value myStartPlusItems = arith::AddIOp::create(
      builder, loc, myStart, itemsPerWg, arith::IntegerOverflowFlags::nsw);
  Value myEnd = indexMin(builder, loc, myStartPlusItems, cTotalWork);

  // first_out = my_start / kTilesPerOut
  Value firstOut =
      arith::DivUIOp::create(builder, loc, myStart, cKTilesPerOut);

  // last_out = (my_end - 1) / kTilesPerOut  (safe: my_end >= 1 always)
  Value myEndMinus1 = arith::SubIOp::create(builder, loc, myEnd, c1,
                                             arith::IntegerOverflowFlags::nsw);
  Value lastOut =
      arith::DivUIOp::create(builder, loc, myEndMinus1, cKTilesPerOut);
  Value lastOutPlus1 = arith::AddIOp::create(builder, loc, lastOut, c1,
                                              arith::IntegerOverflowFlags::nsw);

  // Tile sizes for getTiledImplementation (static, reused across iterations).
  SmallVector<OpFoldResult> tileSizesOFR;
  for (unsigned d = 0; d < numDims; ++d) {
    int64_t ts = tileSizes[d] == 0 ? dimSizes[d] : tileSizes[d];
    tileSizesOFR.push_back(builder.getIndexAttr(ts));
  }

  // === Step 12: Outer scf.for over output tiles. ===
  auto outerFor = scf::ForOp::create(builder, loc, firstOut, lastOutPlus1, c1);
  {
    OpBuilder::InsertionGuard outerGuard(builder);
    builder.setInsertionPointToStart(outerFor.getBody());
    Value outIdx = outerFor.getInductionVar();

    // Compute k-range for this output tile within our work range.
    Value outStartLinear = arith::MulIOp::create(
        builder, loc, outIdx, cKTilesPerOut,
        arith::IntegerOverflowFlags::nsw);
    Value outEndLinear = arith::AddIOp::create(
        builder, loc, outStartLinear, cKTilesPerOut,
        arith::IntegerOverflowFlags::nsw);

    // k_start = max(my_start, outStartLinear) - outStartLinear
    Value kStartRaw = indexMax(builder, loc, myStart, outStartLinear);
    Value kStart = arith::SubIOp::create(builder, loc, kStartRaw,
                                          outStartLinear,
                                          arith::IntegerOverflowFlags::nsw);

    // k_end = min(my_end, outEndLinear) - outStartLinear
    Value kEndRaw = indexMin(builder, loc, myEnd, outEndLinear);
    Value kEnd = arith::SubIOp::create(builder, loc, kEndRaw, outStartLinear,
                                       arith::IntegerOverflowFlags::nsw);

    // Decompose out_idx into per-parallel-dim tile coordinates.
    SmallVector<Value> parallelTileCoords =
        decomposeLinearIndex(builder, loc, outIdx, parallelTileCounts);

    // Compute parallel offsets: coord * tile_size.
    SmallVector<Value> parallelOffsets;
    for (unsigned i = 0; i < parallelDims.size(); ++i) {
      unsigned d = parallelDims[i];
      if (tileSizes[d] == 0) {
        parallelOffsets.push_back(
            arith::ConstantIndexOp::create(builder, loc, 0));
      } else {
        Value ts =
            arith::ConstantIndexOp::create(builder, loc, tileSizes[d]);
        parallelOffsets.push_back(arith::MulIOp::create(
            builder, loc, parallelTileCoords[i], ts,
            arith::IntegerOverflowFlags::nsw));
      }
    }

    // Create zero-filled tile for accumulator init.
    Value zero = arith::ConstantOp::create(builder, loc,
                                           builder.getZeroAttr(elemType));
    Value emptyTile =
        tensor::EmptyOp::create(builder, loc, tileShape, elemType);
    Value zeroTile =
        linalg::FillOp::create(builder, loc, zero, emptyTile).getResult(0);

    // === Step 13: Inner scf.for accumulating k-tile partials. ===
    auto innerFor = scf::ForOp::create(
        builder, loc, kStart, kEnd, c1,
        /*iterArgs=*/ValueRange{zeroTile},
        [&](OpBuilder &ib, Location il, Value k, ValueRange iterArgs) {
          Value acc = iterArgs[0];

          // Decompose k into per-reduction-dim tile coordinates.
          SmallVector<Value> reductionTileCoords =
              decomposeLinearIndex(ib, il, k, reductionTileCounts);

          // Merge parallel + reduction offsets into full offset array.
          SmallVector<OpFoldResult> tileOffsets(numDims);
          unsigned pIdx = 0;
          unsigned rIdx = 0;
          for (unsigned d = 0; d < numDims; ++d) {
            if (tileSizes[d] == 0) {
              tileOffsets[d] =
                  arith::ConstantIndexOp::create(ib, il, 0).getResult();
            } else {
              bool isParallel = llvm::is_contained(parallelDims, d);
              if (isParallel) {
                tileOffsets[d] = parallelOffsets[pIdx++];
              } else {
                Value ts =
                    arith::ConstantIndexOp::create(ib, il, tileSizes[d]);
                tileOffsets[d] =
                    arith::MulIOp::create(ib, il, reductionTileCoords[rIdx++],
                                          ts,
                                          arith::IntegerOverflowFlags::nsw)
                        .getResult();
              }
            }
          }

          // Tile computation using TilingInterface.
          FailureOr<TilingResult> tilingResult =
              tilingTarget.getTiledImplementation(ib, tileOffsets, tileSizesOFR);
          // Note: if tiling fails at this point it's a programming error in the
          // pass setup, not a user-visible error. We assert instead of
          // signalPassFailure since we're inside a lambda.
          assert(succeeded(tilingResult) &&
                 "getTiledImplementation failed inside Stream-K inner loop");

          // The tiled op uses a slice of the original fill as output init.
          // Replace it with the iter_arg accumulator so the matmul accumulates
          // directly: result = acc + A_slice @ B_slice.
          Operation *tiledOp = tilingResult->tiledOps.front();
          if (auto tiledLinalgOp = dyn_cast<linalg::LinalgOp>(tiledOp)) {
            Value oldOutInit = tiledLinalgOp.getDpsInits()[0];
            tiledLinalgOp.getDpsInitsMutable()[0].set(acc);
            // Propagate inner lowering config (with workgroup and
            // streamed_reduction stripped) so thread-level tiling passes
            // pick up this op.
            if (innerLoweringConfig) {
              setLoweringConfig(tiledOp, innerLoweringConfig);
            } else {
              tiledOp->removeAttr("lowering_config");
            }
            // Erase dead extract_slice of the fill result.
            if (auto sliceOp =
                    oldOutInit.getDefiningOp<tensor::ExtractSliceOp>()) {
              if (sliceOp->use_empty()) {
                sliceOp->erase();
              }
            }
          }

          Value newAcc = tilingResult->tiledValues.front();
          scf::YieldOp::create(ib, il, ValueRange{newAcc});
        });

    Value accum = innerFor.getResult(0);

    // === Step 14: Compute numInGroup and create recombine. ===
    Value firstLinear = arith::MulIOp::create(
        builder, loc, outIdx, cKTilesPerOut,
        arith::IntegerOverflowFlags::nsw);
    Value kTilesM1 = arith::SubIOp::create(
        builder, loc, cKTilesPerOut, c1, arith::IntegerOverflowFlags::nsw);
    Value lastLinear = arith::AddIOp::create(
        builder, loc, firstLinear, kTilesM1,
        arith::IntegerOverflowFlags::nsw);
    Value firstWg =
        arith::DivUIOp::create(builder, loc, firstLinear, itemsPerWg);
    Value lastWg =
        arith::DivUIOp::create(builder, loc, lastLinear, itemsPerWg);
    Value numInGroup = arith::AddIOp::create(
        builder, loc,
        arith::SubIOp::create(builder, loc, lastWg, firstWg,
                              arith::IntegerOverflowFlags::nsw),
        c1, arith::IntegerOverflowFlags::nsw);

    // Build output tile offsets/sizes/strides for recombine.
    SmallVector<OpFoldResult> outTileOffsets;
    SmallVector<OpFoldResult> outTileSizes;
    SmallVector<OpFoldResult> outTileStrides;
    for (unsigned i = 0; i < parallelDims.size(); ++i) {
      outTileOffsets.push_back(parallelOffsets[i]);
      unsigned d = parallelDims[i];
      int64_t ts = tileSizes[d] == 0 ? dimSizes[d] : tileSizes[d];
      outTileSizes.push_back(builder.getI64IntegerAttr(ts));
      outTileStrides.push_back(builder.getI64IntegerAttr(1));
    }

    // Create pcf.stream_k_recombine.
    auto recombineOp = IREE::PCF::StreamKRecombineOp::create(
        builder, loc, accum,
        /*dest=*/outRef, outTileOffsets, outTileSizes, outTileStrides,
        /*scratch=*/scratchArg,
        /*counter=*/counterArg,
        /*counter_index=*/outIdx,
        /*numInGroup=*/numInGroup);

    // Populate combiner region: element-wise addition.
    {
      Block &combinerBlock = recombineOp.getCombiner().front();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&combinerBlock);

      Value lhs = combinerBlock.getArgument(0);
      Value rhs = combinerBlock.getArgument(1);
      Value sum;
      if (isa<FloatType>(elemType)) {
        sum = arith::AddFOp::create(builder, loc, lhs, rhs);
      } else {
        sum = arith::AddIOp::create(builder, loc, lhs, rhs);
      }
      IREE::PCF::YieldOp::create(builder, loc, sum);
    }

    // Populate writeback region: write final tile to output sref.
    {
      Block &writebackBlock = recombineOp.getWriteback().front();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&writebackBlock);

      Value finalTile = writebackBlock.getArgument(0);
      IREE::PCF::WriteSliceOp::create(builder, loc, finalTile, outRef,
                                      outTileOffsets, outTileSizes,
                                      outTileStrides);
      IREE::PCF::YieldOp::create(builder, loc, ValueRange{});
    }

    // Outer loop yield is auto-created by SingleBlockImplicitTerminator.
  }

  // === Step 15: Add pcf.return terminator. ===
  IREE::PCF::ReturnOp::create(builder, loc);

  // === Step 16: Replace original ops with generic result. ===
  targetResult.replaceAllUsesWith(genericOp.getResult(0));

  // Erase the original target and fill (in reverse order of def-use).
  target->erase();
  if (fillOp && fillOp->use_empty()) {
    fillOp->erase();
  }
}

} // namespace

} // namespace mlir::iree_compiler
