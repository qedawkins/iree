// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-fuse-into-pcf-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUFUSEINTOPCFOPSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct GPUFuseIntoPCFOpsPass final
    : impl::GPUFuseIntoPCFOpsPassBase<GPUFuseIntoPCFOpsPass> {
  void runOnOperation() override;
};

//===---------------------------------------------------------------------===//
// Consumer fusion patterns (subgroup-scoped)
//===---------------------------------------------------------------------===//

struct FuseConsumerIntoSubgroupGeneric
    : public OpRewritePattern<IREE::PCF::GenericOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<IREE::GPU::SubgroupScopeAttr>(genericOp.getScope()))
      return failure();

    IREE::PCF::ConsumerFusionParams params;
    TilingInterface fusionTarget;
    for (Operation *user : genericOp->getUsers()) {
      fusionTarget = dyn_cast<TilingInterface>(user);
      IREE::PCF::ConsumerFusionParams tempParams;
      if (fusionTarget &&
          succeeded(IREE::PCF::matchTilableConsumer(rewriter, genericOp,
                                                    fusionTarget, tempParams))) {
        std::swap(params, tempParams);
        break;
      }
      fusionTarget = TilingInterface();
    }
    if (!fusionTarget)
      return failure();
    IREE::PCF::fuseTilableConsumer(rewriter, genericOp, fusionTarget, params);
    return success();
  }
};

struct FuseConsumerIntoSubgroupLoop
    : public OpRewritePattern<IREE::PCF::LoopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<IREE::GPU::SubgroupScopeAttr>(loopOp.getScope()))
      return failure();

    IREE::PCF::ConsumerFusionParams params;
    TilingInterface fusionTarget;
    for (Operation *user : loopOp->getUsers()) {
      fusionTarget = dyn_cast<TilingInterface>(user);
      IREE::PCF::ConsumerFusionParams tempParams;
      if (fusionTarget &&
          succeeded(IREE::PCF::matchTilableConsumer(rewriter, loopOp,
                                                    fusionTarget, tempParams))) {
        std::swap(params, tempParams);
        break;
      }
      fusionTarget = TilingInterface();
    }
    if (!fusionTarget)
      return failure();
    IREE::PCF::fuseTilableConsumer(rewriter, loopOp, fusionTarget, params);
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Producer fusion patterns (subgroup-scoped)
//===---------------------------------------------------------------------===//

struct FuseProducerIntoSubgroupGeneric
    : public OpRewritePattern<IREE::PCF::GenericOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<IREE::GPU::SubgroupScopeAttr>(genericOp.getScope()))
      return failure();

    IREE::PCF::ProducerFusionParams params;
    if (failed(IREE::PCF::matchTilableProducer(rewriter, genericOp, params)))
      return failure();
    IREE::PCF::fuseTilableProducer(rewriter, genericOp, params);
    return success();
  }
};

struct FuseProducerIntoSubgroupLoop
    : public OpRewritePattern<IREE::PCF::LoopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<IREE::GPU::SubgroupScopeAttr>(loopOp.getScope()))
      return failure();

    IREE::PCF::ProducerFusionParams params;
    if (failed(IREE::PCF::matchTilableProducer(rewriter, loopOp, params)))
      return failure();
    IREE::PCF::fuseTilableProducer(rewriter, loopOp, params);
    return success();
  }
};

void GPUFuseIntoPCFOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<FuseConsumerIntoSubgroupGeneric, FuseConsumerIntoSubgroupLoop,
               FuseProducerIntoSubgroupGeneric, FuseProducerIntoSubgroupLoop>(
      &getContext());
  IREE::PCF::populatePCFDropUnusedResultPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace
} // namespace mlir::iree_compiler
