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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_TESTDISTRIBUTEDFUSEPRODUCERSPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

struct DistributedFuseProducerIntoGenericOp final
    : OpRewritePattern<IREE::PCF::GenericOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    DistributedProducerFusionParams params;
    if (failed(matchDistributedProducer(rewriter, genericOp, params))) {
      return failure();
    }
    fuseDistributedProducer(rewriter, genericOp, params);
    return success();
  }
};

struct DistributedFuseProducerIntoLoopOp final
    : OpRewritePattern<IREE::PCF::LoopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    DistributedProducerFusionParams params;
    if (failed(matchDistributedProducer(rewriter, loopOp, params))) {
      return failure();
    }
    fuseDistributedProducer(rewriter, loopOp, params);
    return success();
  }
};

struct TestDistributedFuseProducersPass final
    : impl::TestDistributedFuseProducersPassBase<
          TestDistributedFuseProducersPass> {
  using TestDistributedFuseProducersPassBase::
      TestDistributedFuseProducersPassBase;
  void runOnOperation() override {
    attachAllDistributedTilingModels(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<DistributedFuseProducerIntoGenericOp,
                 DistributedFuseProducerIntoLoopOp>(&getContext());
    populatePCFDropUnusedResultPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::PCF
