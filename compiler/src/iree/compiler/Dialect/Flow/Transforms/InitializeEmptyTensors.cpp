// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Returns a zero value attribute based on the `elementType`.
/// Returns failure, when the type is not handled.
static FailureOr<Attribute> getZero(OpBuilder &builder, Location loc,
                                    Type elementType) {
  if (auto intType = elementType.dyn_cast<IntegerType>()) {
    return builder.getIntegerAttr(intType, 0);
  }
  if (auto floatType = elementType.dyn_cast<FloatType>()) {
    return builder.getFloatAttr(floatType, 0.0);
  }
  return failure();
}

namespace {

/// Converts an tensor.empty() op to `flow.tensor.splat` op.
struct RewriteInitTensorToSplat : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::EmptyOp emptyTensorOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::all_of(emptyTensorOp->getUsers(), [](Operation *user) -> bool {
          return isa<linalg::LinalgOp, LinalgExt::LinalgExtOp, tensor::PackOp,
                     tensor::UnPackOp>(user);
        })) {
      return failure();
    }

    RankedTensorType resultType = emptyTensorOp.getType();
    Type elementType = resultType.getElementType();
    Location loc = emptyTensorOp.getLoc();
    FailureOr<Attribute> zero = getZero(rewriter, loc, elementType);
    if (failed(zero)) {
      return rewriter.notifyMatchFailure(
          emptyTensorOp, "unable to get zero value for element type");
    }
    Value value =
        rewriter.create<arith::ConstantOp>(loc, elementType, zero.value());
    rewriter.replaceOpWithNewOp<TensorSplatOp>(emptyTensorOp, resultType, value,
                                               emptyTensorOp.getDynamicSizes());
    return success();
  }
};

/// Converts an tensor.empty() op to `flow.tensor.empty` op.
struct RewriteInitTensorToEmpty : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::EmptyOp emptyTensorOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::all_of(emptyTensorOp->getUsers(), [](Operation *user) -> bool {
          return isa<linalg::LinalgOp, LinalgExt::LinalgExtOp, tensor::PackOp,
                     tensor::UnPackOp>(user);
        })) {
      return failure();
    }
    RankedTensorType resultType = emptyTensorOp.getType();
    rewriter.replaceOpWithNewOp<TensorEmptyOp>(emptyTensorOp, resultType,
                                               emptyTensorOp.getDynamicSizes());
    return success();
  }
};

/// Pass to invoke the pattern.
struct InitializeEmptyTensorsPass
    : public InitializeEmptyTensorsBase<InitializeEmptyTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, IREE::Flow::FlowDialect,
                    linalg::LinalgDialect>();
  }
  InitializeEmptyTensorsPass(bool zeroFill) { this->zeroFill = zeroFill; }
  InitializeEmptyTensorsPass(const InitializeEmptyTensorsPass &pass)
      : InitializeEmptyTensorsPass(pass.zeroFill) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    if (zeroFill) {
      patterns.insert<RewriteInitTensorToSplat>(context);
    } else {
      patterns.insert<RewriteInitTensorToEmpty>(context);
    }
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createInitializeEmptyTensorsPass(bool zeroFill) {
  return std::make_unique<InitializeEmptyTensorsPass>(zeroFill);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
