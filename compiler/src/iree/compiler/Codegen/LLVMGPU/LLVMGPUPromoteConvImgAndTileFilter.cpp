// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cwchar>
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-promote-conv-img"

namespace mlir::iree_compiler {

namespace {
struct LLVMGPUPromoteConvImgAndTileFilterPass
    : public LLVMGPUPromoteConvImgAndTileFilterBase<
          LLVMGPUPromoteConvImgAndTileFilterPass> {
  void getDependentDialects(DialectRegistry &registry) const override {}
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();

    linalg::LinalgOp conv;
    auto found = funcOp->walk([&](linalg::LinalgOp op) {
      if (!linalg::isaConvolutionOpInterface(op)) {
        return WalkResult::advance();
      }
      if (linalg::inferConvolutionDims(op)->filterLoop.empty()) {
        return WalkResult::advance();
      }
      if (conv) {
        return WalkResult::interrupt();
      }
      conv = op;
      return WalkResult::advance();
    });
    if (found.wasInterrupted()) {
      LLVM_DEBUG(llvm::dbgs() << "skip, expect a single conv\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "candidate: " << conv << "\n");
    IRRewriter rewriter(ctx);
    SmallVector<int64_t> paddingDims = llvm::to_vector(llvm::seq(
        static_cast<int64_t>(0), static_cast<int64_t>(conv.getNumLoops())));
    SmallVector<bool> packPaddings = {1, 0, 0};
    SmallVector<int64_t> padToMultipleOf(paddingDims.size(), 1);
    SmallVector<Attribute> paddingValueAttributes;
    for (auto &operand : conv->getOpOperands()) {
      auto elemType = getElementTypeOrSelf(operand.get().getType());
      paddingValueAttributes.push_back(rewriter.getZeroAttr(elemType));
    }

    auto options =
        linalg::LinalgPaddingOptions()
            .setPaddingDimensions(paddingDims)
            .setPaddingValues(paddingValueAttributes)
            .setPadToMultipleOf(padToMultipleOf)
            .setPackPaddings(packPaddings)
            .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);
    linalg::LinalgOp paddedOp;
    SmallVector<Value> replacements;
    SmallVector<tensor::PadOp> newPadOps;
    if (failed(rewriteAsPaddedOp(rewriter, conv, options, paddedOp,
                                 replacements, newPadOps))) {
      LLVM_DEBUG(llvm::dbgs() << "failed to pad op " << conv << "\n");
      return signalPassFailure();
    }
    rewriter.replaceOp(conv, replacements);

    // tile filter
    {
      auto found = funcOp->walk([&](linalg::LinalgOp op) {
        if (!linalg::isaConvolutionOpInterface(op)) {
          return WalkResult::advance();
        }
        if (linalg::inferConvolutionDims(op)->filterLoop.empty()) {
          return WalkResult::advance();
        }
        conv = op;
        return WalkResult::advance();
      });
      if (found.wasInterrupted()) {
        return signalPassFailure();
      }
      FailureOr<IREE::Codegen::LoweringConfigAttr> loweringConfig =
          getLoweringConfig(conv);
      if (failed(loweringConfig)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "skip tiling because there are no lowering_config\n");
        return;
      }

      IRRewriter rewriter(ctx);
      SmallVector<OpFoldResult> tileSizes = llvm::map_to_vector(
          loweringConfig->getTileSizeVals(1), [&](int64_t val) -> OpFoldResult {
            return rewriter.getIndexAttr(val);
          });
      auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);
      FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
          rewriter, cast<TilingInterface>(conv.getOperation()), options);
      if (failed(tilingResult))
        return signalPassFailure();
      rewriter.replaceOp(conv, tilingResult->replacements);
    }

    // Canonicalize tiled ops.
    {
      RewritePatternSet patterns(ctx);
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
      ctx->getOrLoadDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUPromoteConvImgAndTileFilterPass() {
  return std::make_unique<LLVMGPUPromoteConvImgAndTileFilterPass>();
}

} // namespace mlir::iree_compiler
