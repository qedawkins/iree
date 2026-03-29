// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/TilingImplementations/RegisterAll.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "tile-and-distribute-to-workgroups-using-pcf"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TILEANDDISTRIBUTETOWORKGROUPSUSINGPCFPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Find the lowering config to use for getting the tile sizes.
struct TilingInfo {
  Operation *tilableOp;
  SmallVector<OpFoldResult> tileSizes;
};

static FailureOr<TilingInfo>
getTilingInfo(RewriterBase &rewriter, ArrayRef<Operation *> computeOps) {
  Operation *tilableOp = nullptr;
  for (Operation *op : llvm::reverse(computeOps)) {
    if (getLoweringConfig(op)) {
      if (!getLoweringConfig(op).hasWorkgroupTilingLevel()) {
        continue;
      }
      tilableOp = op;
      break;
    }
  }
  if (!tilableOp) {
    return TilingInfo{nullptr, {}};
  }

  IREE::Codegen::LoweringConfigAttrInterface tilableOpConfig =
      getLoweringConfig(tilableOp);
  if (!tilableOpConfig) {
    return tilableOp->emitOpError("unable to find configuration of root op");
  }
  SmallVector<OpFoldResult> tileSizes = llvm::map_to_vector(
      tilableOpConfig.getWorkgroupTileSizes(),
      [&](int64_t t) -> OpFoldResult { return rewriter.getIndexAttr(t); });

  // Set tile sizes for non-partitioned loops to zero.
  if (auto partitionableLoopsInterface =
          dyn_cast<PartitionableLoopsInterface>(tilableOp)) {
    SmallVector<unsigned> partitionableLoops =
        partitionableLoopsInterface.getPartitionableLoops(std::nullopt);
    llvm::SmallDenseSet<unsigned> partitionableLoopsSet(
        partitionableLoops.begin(), partitionableLoops.end());
    OpFoldResult zero = rewriter.getIndexAttr(0);
    for (unsigned loopId : llvm::seq<unsigned>(0, tileSizes.size())) {
      if (partitionableLoopsSet.count(loopId)) {
        continue;
      }
      tileSizes[loopId] = zero;
    }
  }

  // Set tile sizes for full tiles to zero to prevent single-trip loops.
  if (auto tilingInterfaceOp = dyn_cast<TilingInterface>(tilableOp)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(tilingInterfaceOp);
    SmallVector<Range> bounds = tilingInterfaceOp.getIterationDomain(rewriter);
    SmallVector<int64_t> staticLoopSizes;
    SmallVector<Value> d;
    for (Range bound : bounds) {
      dispatchIndexOpFoldResult(bound.size, d, staticLoopSizes);
    }
    SmallVector<int64_t> tileSizesInt = tilableOpConfig.getWorkgroupTileSizes();
    int numNonZero = tileSizesInt.size() - llvm::count(tileSizesInt, 0);
    OpFoldResult zero = rewriter.getIndexAttr(0);
    for (unsigned loopId :
         llvm::reverse(llvm::seq<unsigned>(0, tileSizesInt.size()))) {
      if (numNonZero <= 1) {
        break;
      }
      if (loopId < staticLoopSizes.size() &&
          staticLoopSizes[loopId] == tileSizesInt[loopId]) {
        tileSizes[loopId] = zero;
        --numNonZero;
      }
    }
  }

  return TilingInfo{tilableOp, tileSizes};
}

//===----------------------------------------------------------------------===//
// Fusion patterns.
//===----------------------------------------------------------------------===//

/// Pattern to fuse distributed consumers into pcf.loop.
struct FuseConsumerIntoLoopOp final
    : OpRewritePattern<IREE::PCF::LoopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    IREE::PCF::ConsumerFusionParams params;
    IREE::PCF::PCFTilingOpInterface fusionTarget;
    for (Operation *user : loopOp->getUsers()) {
      fusionTarget = dyn_cast<IREE::PCF::PCFTilingOpInterface>(user);
      if (!fusionTarget) {
        continue;
      }
      IREE::PCF::ConsumerFusionParams tempParams;
      if (succeeded(IREE::PCF::matchTilableConsumer(
              rewriter, loopOp, cast<TilingInterface>(*fusionTarget),
              tempParams))) {
        std::swap(params, tempParams);
        break;
      }
      fusionTarget = IREE::PCF::PCFTilingOpInterface();
    }
    if (!fusionTarget) {
      return failure();
    }
    IREE::PCF::fuseDistributedConsumer(rewriter, loopOp, fusionTarget, params);
    return success();
  }
};

/// Pattern to fuse distributed producers into pcf.loop.
struct FuseProducerIntoLoopOp final
    : OpRewritePattern<IREE::PCF::LoopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    IREE::PCF::DistributedProducerFusionParams params;
    if (failed(
            IREE::PCF::matchDistributedProducer(rewriter, loopOp, params))) {
      return failure();
    }
    IREE::PCF::fuseDistributedProducer(rewriter, loopOp, params);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass.
//===----------------------------------------------------------------------===//

struct TileAndDistributeToWorkgroupsUsingPCFPass final
    : impl::TileAndDistributeToWorkgroupsUsingPCFPassBase<
          TileAndDistributeToWorkgroupsUsingPCFPass> {
  using Base::Base;
  void runOnOperation() override;
};

void TileAndDistributeToWorkgroupsUsingPCFPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();

  // Ensure PCFTilingOpInterface external models are attached.
  IREE::PCF::attachAllDistributedTilingModels(context);

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  IRRewriter rewriter(context);

  FailureOr<TilingInfo> tilingInfo = getTilingInfo(rewriter, computeOps);
  if (failed(tilingInfo)) {
    return signalPassFailure();
  }

  Operation *tilableOp = tilingInfo->tilableOp;
  if (!tilableOp) {
    // No tileable op found. Nothing to do.
    return;
  }

  auto pcfTilingOp = dyn_cast<IREE::PCF::PCFTilingOpInterface>(tilableOp);
  if (!pcfTilingOp) {
    tilableOp->emitOpError("op does not implement PCFTilingOpInterface");
    return signalPassFailure();
  }

  // Use workgroup scope.
  IREE::PCF::ScopeAttrInterface scope = cast<IREE::PCF::ScopeAttrInterface>(
      IREE::Codegen::WorkgroupScopeAttr::get(context, /*linearize=*/true));

  // Tile the root op into a pcf.loop.
  rewriter.setInsertionPoint(tilableOp);
  FailureOr<IREE::PCF::LoopOp> loopOp = IREE::PCF::tileToPCFLoop(
      rewriter, pcfTilingOp, scope, tilingInfo->tileSizes);
  if (failed(loopOp)) {
    tilableOp->emitOpError("failed to tile to PCF loop");
    return signalPassFailure();
  }

  // Fuse producers greedily.
  {
    RewritePatternSet fusionPatterns(context);
    fusionPatterns.add<FuseProducerIntoLoopOp>(context);
    IREE::PCF::populatePCFDropUnusedResultPatterns(fusionPatterns);
    if (failed(
            applyPatternsGreedily(funcOp, std::move(fusionPatterns)))) {
      funcOp.emitOpError("producer fusion failed");
      return signalPassFailure();
    }
  }

  // Fuse consumers via walk.
  {
    IRRewriter irRewriter(context);
    SmallVector<IREE::PCF::LoopOp> loops;
    funcOp->walk([&](IREE::PCF::LoopOp loopOp) {
      loops.push_back(loopOp);
    });
    for (IREE::PCF::LoopOp loopOp : loops) {
      IREE::PCF::ConsumerFusionParams params;
      IREE::PCF::PCFTilingOpInterface fusionTarget;
      for (Operation *user : loopOp->getUsers()) {
        fusionTarget =
            dyn_cast<IREE::PCF::PCFTilingOpInterface>(user);
        if (!fusionTarget) {
          continue;
        }
        IREE::PCF::ConsumerFusionParams tempParams;
        if (succeeded(IREE::PCF::matchTilableConsumer(
                irRewriter, loopOp,
                cast<TilingInterface>(*fusionTarget), tempParams))) {
          std::swap(params, tempParams);
          break;
        }
        fusionTarget = IREE::PCF::PCFTilingOpInterface();
      }
      if (!fusionTarget) {
        continue;
      }
      IREE::PCF::fuseDistributedConsumer(irRewriter, loopOp,
                                         fusionTarget, params);
    }
  }
}

} // namespace
} // namespace mlir::iree_compiler
