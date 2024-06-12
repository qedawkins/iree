// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

struct ConvertConvToIm2colPass
    : public ConvertConvToIm2colPassBase<ConvertConvToIm2colPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();

    RewritePatternSet patterns(ctx);
    linalg::populateConvertConv2DToImg2ColPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertConvToIm2colPass() {
  return std::make_unique<ConvertConvToIm2colPass>();
}

} // namespace mlir::iree_compiler
