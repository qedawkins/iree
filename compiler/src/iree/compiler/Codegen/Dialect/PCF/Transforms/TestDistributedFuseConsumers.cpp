// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTilingInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/TilingImplementations/RegisterAll.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_TESTDISTRIBUTEDFUSECONSUMERSPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

struct DistributedFuseIntoGenericOp final
    : OpRewritePattern<IREE::PCF::GenericOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    ConsumerFusionParams params;
    PCFTilingOpInterface fusionTarget;
    for (Operation *user : genericOp->getUsers()) {
      auto candidate = dyn_cast<PCFTilingOpInterface>(user);
      if (!candidate) {
        continue;
      }
      ConsumerFusionParams tempParams;
      if (succeeded(matchTilableConsumer(rewriter, genericOp,
                                         cast<TilingInterface>(*candidate),
                                         tempParams))) {
        std::swap(params, tempParams);
        fusionTarget = candidate;
        break;
      }
    }
    if (!fusionTarget) {
      return failure();
    }
    fuseDistributedConsumer(rewriter, genericOp, fusionTarget, params);
    return success();
  }
};

struct DistributedFuseIntoLoopOp final : OpRewritePattern<IREE::PCF::LoopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    ConsumerFusionParams params;
    PCFTilingOpInterface fusionTarget;
    for (Operation *user : loopOp->getUsers()) {
      auto candidate = dyn_cast<PCFTilingOpInterface>(user);
      if (!candidate) {
        continue;
      }
      ConsumerFusionParams tempParams;
      if (succeeded(matchTilableConsumer(rewriter, loopOp,
                                         cast<TilingInterface>(*candidate),
                                         tempParams))) {
        std::swap(params, tempParams);
        fusionTarget = candidate;
        break;
      }
    }
    if (!fusionTarget) {
      return failure();
    }
    fuseDistributedConsumer(rewriter, loopOp, fusionTarget, params);
    return success();
  }
};

struct TestDistributedFuseConsumersPass final
    : impl::TestDistributedFuseConsumersPassBase<
          TestDistributedFuseConsumersPass> {
  using TestDistributedFuseConsumersPassBase::
      TestDistributedFuseConsumersPassBase;
  void runOnOperation() override {
    attachAllDistributedTilingModels(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<DistributedFuseIntoGenericOp, DistributedFuseIntoLoopOp>(
        &getContext());
    populatePCFDropUnusedResultPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::PCF
