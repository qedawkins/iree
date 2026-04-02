// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFNamespace.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-gpu-lower-promote-operand"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_LOWERPROMOTEOPERANDPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {

/// Lowers iree_gpu.promote_operand into pcf.alloc + distributed copy.
struct LowerPromoteOperandPattern final
    : OpRewritePattern<PromoteOperandOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(PromoteOperandOp promoteOp,
                                PatternRewriter &rewriter) const override {
    Location loc = promoteOp.getLoc();
    Value source = promoteOp.getSource();
    auto sourceType = cast<PCF::ShapedRefType>(source.getType());
    auto resultType = cast<PCF::ShapedRefType>(promoteOp.getType());
    int64_t rank = sourceType.getRank();

    // Resolve symbol values from the namespace.
    SmallVector<OpFoldResult> tileSizes;
    for (Attribute sym : promoteOp.getSymbols()) {
      StringAttr symName = cast<StringAttr>(sym);
      // Build a NamespacedSymbolAttr with just the leaf name.
      PCF::NamespacedSymbolAttr nsSym = PCF::NamespacedSymbolAttr::get(
          rewriter.getContext(), ArrayRef<StringAttr>{symName});
      std::optional<OpFoldResult> resolved =
          PCF::resolveNamespacedSymbol(promoteOp, nsSym);
      if (!resolved) {
        return rewriter.notifyMatchFailure(
            promoteOp, "failed to resolve symbol '" +
                           symName.getValue().str() + "'");
      }
      tileSizes.push_back(*resolved);
    }

    // Create the allocation at the result scope.
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (resultType.isDynamicDim(i)) {
        dynamicSizes.push_back(
            getValueOrCreateConstantIndexOp(rewriter, loc, tileSizes[i]));
      }
    }
    Value alloc = PCF::AllocOp::create(rewriter, loc, resultType, dynamicSizes);

    // Find the enclosing lane scope's worker IDs for distribution.
    // Walk up to find the pcf.generic that provides lane IDs.
    Value laneId;
    Value laneCount;
    Operation *parent = promoteOp->getParentOp();
    while (parent) {
      if (auto genericOp = dyn_cast<PCF::GenericOp>(parent)) {
        ArrayRef<BlockArgument> idArgs = genericOp.getIdArgs();
        ArrayRef<BlockArgument> countArgs = genericOp.getCountArgs();
        if (!idArgs.empty()) {
          // Use the first ID/count pair (linearized lane ID).
          laneId = idArgs.front();
          laneCount = countArgs.front();
          break;
        }
      }
      parent = parent->getParentOp();
    }

    if (!laneId) {
      return rewriter.notifyMatchFailure(
          promoteOp, "no enclosing pcf.generic with worker IDs");
    }

    // Emit the distributed copy using the promotion attribute.
    PromotionAttr promotion = promoteOp.getPromotion();
    promotion.emitDistributedCopy(rewriter, loc, source, alloc, tileSizes,
                                  laneId, laneCount);

    // Replace the promote_operand with the allocation.
    rewriter.replaceOp(promoteOp, alloc);
    return success();
  }
};

struct LowerPromoteOperandPass final
    : impl::LowerPromoteOperandPassBase<LowerPromoteOperandPass> {
  using LowerPromoteOperandPassBase::LowerPromoteOperandPassBase;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerPromoteOperandPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::GPU
