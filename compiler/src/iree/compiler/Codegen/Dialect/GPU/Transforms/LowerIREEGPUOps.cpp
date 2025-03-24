// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_LOWERIREEGPUOPSPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct LowerIREEGPUOpsPass final
    : impl::LowerIREEGPUOpsPassBase<LowerIREEGPUOpsPass> {
  void runOnOperation() override;
};

constexpr StringLiteral kSwapName = "iree_gpu.swap_mfma";

struct SwapSetPrioWithMFMA : public OpRewritePattern<ROCDL::SetPrioOp> {
  using OpRewritePattern<ROCDL::SetPrioOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ROCDL::SetPrioOp setPrio,
                                PatternRewriter &rewriter) const override {
    if (!setPrio->hasAttr(kSwapName)) {
      return failure();
    }

    auto count = setPrio->getAttrOfType<IntegerAttr>(kSwapName);
    if (!count) {
      return failure();
    }

    int64_t remainingToSwap = count.getInt();
    rewriter.startOpModification(setPrio);
    setPrio->removeDiscardableAttr(kSwapName);

    Operation *current = setPrio;

    while (remainingToSwap > 0 && (current = current->getNextNode())) {
      if (isa<mlir::amdgpu::MFMAOp>(current)) {
        remainingToSwap--;
      }
    }
    if (current != setPrio && remainingToSwap != count.getInt()) {
      rewriter.moveOpAfter(setPrio, current);
    }
    return success();
  }
};
} // namespace

void LowerIREEGPUOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  {
    RewritePatternSet patterns(context);
    populateIREEGPULowerValueBarrierPatterns(patterns);
    populateIREEGPULowerMultiMmaPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  {
    RewritePatternSet patterns(context);
    patterns.add<SwapSetPrioWithMFMA>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
