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
// distributes it across workgroups using a pcf.loop.
//
// The produced IR structure:
//   1. pcf.loop scope(#workgroup_scope) count(%total_work)
//   2. Decode: divmod chain decomposes linear index.
//   3. Group size: arithmetic for num_in_group.
//   4. Slice inputs using TilingInterface.
//   5. Compute partial tile.
//   6. pcf.stream_k_recombine with combiner + writeback.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
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
  Value aMinusOne = arith::SubIOp::create(builder, loc, a, one);
  Value sum = arith::AddIOp::create(builder, loc, aMinusOne, b);
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
  auto [parallelDims, reductionDims] = classifyDims(tilingTarget);
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

  // === Step 6: Find output chain. ===
  // Look for: target -> store_to_buffer (or dispatch.tensor.store).
  // At this point in the pipeline, dispatch tensors have been converted to
  // hal.interface.binding.subspan + iree_codegen.store_to_buffer.
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

  // === Step 7: Allocate scratch + counter srefs. ===
  IREE::PCF::ScopeAttrInterface wgScope =
      cast<IREE::PCF::ScopeAttrInterface>(
          IREE::Codegen::WorkgroupScopeAttr::get(ctx, /*linearize=*/true));

  // Per-output-tile scratch: [kTilesPerOut * tile_dim0, tile_dim1, ...].
  SmallVector<int64_t> tileShape;
  for (unsigned d : parallelDims) {
    tileShape.push_back(tileSizes[d] == 0 ? dimSizes[d] : tileSizes[d]);
  }

  SmallVector<int64_t> scratchShape;
  scratchShape.push_back(kTilesPerOut * tileShape[0]);
  for (int64_t i = 1, e = tileShape.size(); i < e; ++i) {
    scratchShape.push_back(tileShape[i]);
  }
  auto scratchSrefType =
      IREE::PCF::ShapedRefType::get(ctx, scratchShape, elemType, wgScope);
  Value scratch =
      IREE::PCF::AllocOp::create(builder, loc, scratchSrefType).getResult();

  // Counter: rank-0 i32 sref.
  auto counterSrefType =
      IREE::PCF::ShapedRefType::get(ctx, {}, builder.getI32Type(), wgScope);
  Value counter =
      IREE::PCF::AllocOp::create(builder, loc, counterSrefType).getResult();

  // === Step 8: Create pcf.loop with output as init. ===
  Value cTotalWork = arith::ConstantIndexOp::create(builder, loc, totalWork);

  auto loopOp = IREE::PCF::LoopOp::create(builder, loc, wgScope,
                                           /*count=*/ValueRange{cTotalWork},
                                           /*inits=*/ValueRange{outTensor});

  // Get the loop body block and its arguments.
  Block &body = loopOp.getRegion().front();
  Value outRef = loopOp.getRegionRefArgs()[0]; // Output sref.
  Value workIdx = loopOp.getIdArgs()[0];       // Linear work index.

  // Set insertion point inside the loop body.
  builder.setInsertionPointToEnd(&body);

  // === Step 9: Decode arithmetic. ===
  // out_idx = work_idx / k_tiles_per_out.
  // k_idx   = work_idx % k_tiles_per_out.
  Value cKTilesPerOut =
      arith::ConstantIndexOp::create(builder, loc, kTilesPerOut);
  Value outIdx =
      arith::DivUIOp::create(builder, loc, workIdx, cKTilesPerOut);
  Value kIdx = arith::RemUIOp::create(builder, loc, workIdx, cKTilesPerOut);

  // Decompose out_idx into per-parallel-dimension tile coordinates.
  // Innermost dimension varies fastest.
  SmallVector<Value> parallelTileCoords;
  Value remaining = outIdx;
  for (int64_t i = parallelDims.size() - 1; i >= 0; --i) {
    Value tileCount =
        arith::ConstantIndexOp::create(builder, loc, parallelTileCounts[i]);
    if (i > 0) {
      Value coord =
          arith::RemUIOp::create(builder, loc, remaining, tileCount);
      parallelTileCoords.push_back(coord);
      remaining = arith::DivUIOp::create(builder, loc, remaining, tileCount);
    } else {
      parallelTileCoords.push_back(remaining);
    }
  }
  std::reverse(parallelTileCoords.begin(), parallelTileCoords.end());

  // Decompose k_idx into per-reduction-dimension coordinates.
  SmallVector<int64_t> reductionTileCounts;
  for (unsigned d : reductionDims) {
    reductionTileCounts.push_back(tileCounts[d]);
  }
  SmallVector<Value> reductionTileCoords;
  remaining = kIdx;
  for (int64_t i = reductionDims.size() - 1; i >= 0; --i) {
    Value tileCount =
        arith::ConstantIndexOp::create(builder, loc, reductionTileCounts[i]);
    if (i > 0) {
      Value coord =
          arith::RemUIOp::create(builder, loc, remaining, tileCount);
      reductionTileCoords.push_back(coord);
      remaining = arith::DivUIOp::create(builder, loc, remaining, tileCount);
    } else {
      reductionTileCoords.push_back(remaining);
    }
  }
  std::reverse(reductionTileCoords.begin(), reductionTileCoords.end());

  // Compute per-dimension offsets: coord * tile_size.
  SmallVector<Value> offsets(numDims);
  {
    unsigned pIdx = 0;
    unsigned rIdx = 0;
    for (unsigned d = 0; d < numDims; ++d) {
      if (tileSizes[d] == 0) {
        offsets[d] = arith::ConstantIndexOp::create(builder, loc, 0);
      } else {
        Value ts = arith::ConstantIndexOp::create(builder, loc, tileSizes[d]);
        bool isParallel = llvm::is_contained(parallelDims, d);
        if (isParallel) {
          offsets[d] = arith::MulIOp::create(builder, loc,
                                             parallelTileCoords[pIdx++], ts);
        } else {
          offsets[d] = arith::MulIOp::create(builder, loc,
                                             reductionTileCoords[rIdx++], ts);
        }
      }
    }
  }

  // === Step 10: Group size arithmetic. ===
  // num_workgroups from hal.interface.workgroup.count[0].
  Value numWorkgroups =
      IREE::HAL::InterfaceWorkgroupCountOp::create(builder, loc, 0)
          .getResult();
  Value itemsPerWg = ceilDiv(builder, loc, cTotalWork, numWorkgroups);

  Value firstLinear =
      arith::MulIOp::create(builder, loc, outIdx, cKTilesPerOut);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
  Value kTilesM1 = arith::SubIOp::create(builder, loc, cKTilesPerOut, c1);
  Value lastLinear =
      arith::AddIOp::create(builder, loc, firstLinear, kTilesM1);
  Value firstWg =
      arith::DivUIOp::create(builder, loc, firstLinear, itemsPerWg);
  Value lastWg =
      arith::DivUIOp::create(builder, loc, lastLinear, itemsPerWg);
  Value numInGroup = arith::AddIOp::create(
      builder, loc, arith::SubIOp::create(builder, loc, lastWg, firstWg), c1);

  // === Step 11: Tile the computation using TilingInterface. ===
  SmallVector<OpFoldResult> tileOffsets;
  SmallVector<OpFoldResult> tileSizesOFR;
  for (unsigned d = 0; d < numDims; ++d) {
    tileOffsets.push_back(offsets[d]);
    int64_t ts = tileSizes[d] == 0 ? dimSizes[d] : tileSizes[d];
    tileSizesOFR.push_back(builder.getIndexAttr(ts));
  }

  FailureOr<TilingResult> tilingResult =
      tilingTarget.getTiledImplementation(builder, tileOffsets, tileSizesOFR);
  if (failed(tilingResult)) {
    tilingTarget.emitOpError("failed to tile with Stream-K parameters");
    return signalPassFailure();
  }

  Value partial = tilingResult->tiledValues.front();

  // === Step 12: Build output tile offsets/sizes/strides for recombine. ===
  SmallVector<OpFoldResult> outTileOffsets;
  SmallVector<OpFoldResult> outTileSizes;
  SmallVector<OpFoldResult> outTileStrides;
  for (unsigned d : parallelDims) {
    outTileOffsets.push_back(offsets[d]);
    int64_t ts = tileSizes[d] == 0 ? dimSizes[d] : tileSizes[d];
    outTileSizes.push_back(builder.getI64IntegerAttr(ts));
    outTileStrides.push_back(builder.getI64IntegerAttr(1));
  }

  // === Step 13: Create pcf.stream_k_recombine. ===
  // The builder auto-creates combiner (2 scalar args) and writeback (1 tensor
  // arg) regions. We populate them after creation.
  auto recombineOp = IREE::PCF::StreamKRecombineOp::create(
      builder, loc, partial,
      /*dest=*/outRef, outTileOffsets, outTileSizes, outTileStrides,
      /*scratch=*/scratch,
      /*counter=*/counter,
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

  // === Step 14: Add pcf.return terminator. ===
  builder.setInsertionPointToEnd(&body);
  IREE::PCF::ReturnOp::create(builder, loc);

  // === Step 15: Replace original ops with loop result. ===
  // The dispatch.tensor.store now stores the loop result.
  targetResult.replaceAllUsesWith(loopOp.getResult(0));

  // Erase the original target and fill (in reverse order of def-use).
  target->erase();
  if (fillOp && fillOp->use_empty()) {
    fillOp->erase();
  }
}

} // namespace

} // namespace mlir::iree_compiler
