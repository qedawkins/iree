// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Template/IR/Template.h"
#include "iree/compiler/Codegen/Dialect/Template/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-template-inline-instances"

namespace mlir::iree_compiler::IREE::Template {

#define GEN_PASS_DEF_INLINETEMPLATEINSTANCESPASS
#include "iree/compiler/Codegen/Dialect/Template/Transforms/Passes.h.inc"

namespace {

/// Inline a single template.branch op by cloning the corresponding
/// implementation block at the branch site.
static LogicalResult inlineBranch(BranchOp branchOp, InstanceOp instanceOp,
                                  PatternRewriter &rewriter) {
  int64_t blockIndex = branchOp.getBlockIndex();
  Region &implRegion = instanceOp.getImplementations();

  // Find the target block.
  if (blockIndex < 0 ||
      static_cast<size_t>(blockIndex) >= implRegion.getBlocks().size()) {
    return branchOp.emitOpError("block index ")
           << blockIndex << " out of range (have "
           << implRegion.getBlocks().size() << " blocks)";
  }

  Block &targetBlock = *std::next(implRegion.begin(), blockIndex);

  // Set up value mapping from block arguments to branch arguments.
  IRMapping mapping;
  for (auto [blockArg, branchArg] :
       llvm::zip(targetBlock.getArguments(), branchOp.getArguments())) {
    mapping.map(blockArg, branchArg);
  }

  // Clone all operations from the block except the terminator.
  rewriter.setInsertionPoint(branchOp);
  for (Operation &op : targetBlock.without_terminator()) {
    rewriter.clone(op, mapping);
  }

  // Get the return values from the block's terminator.
  auto returnOp = cast<ReturnOp>(targetBlock.getTerminator());
  SmallVector<Value> returnValues;
  for (Value v : returnOp.getOperands()) {
    returnValues.push_back(mapping.lookupOrDefault(v));
  }

  // Replace the branch op results with the return values.
  rewriter.replaceOp(branchOp, returnValues);
  return success();
}

/// Pattern to inline template.instance ops.
struct InlineTemplateInstancePattern : public OpRewritePattern<InstanceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InstanceOp instanceOp,
                                PatternRewriter &rewriter) const override {
    // Collect all template.branch ops in the main region.
    SmallVector<BranchOp> branchOps;
    instanceOp.getMain().walk(
        [&](BranchOp branchOp) { branchOps.push_back(branchOp); });

    // Inline each branch op.
    for (BranchOp branchOp : branchOps) {
      if (failed(inlineBranch(branchOp, instanceOp, rewriter))) {
        return failure();
      }
    }

    // Now inline the main region content in place of the instance op.
    Block &mainBlock = instanceOp.getMain().front();

    // The main block has no block arguments - instance inputs are used directly
    // as SSA values within the region. No mapping is needed for cloning.
    IRMapping mapping;

    // Clone operations from main block to before the instance op.
    rewriter.setInsertionPoint(instanceOp);
    for (Operation &op : mainBlock.without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // Get return values from the terminator.
    auto returnOp = cast<ReturnOp>(mainBlock.getTerminator());
    SmallVector<Value> returnValues;
    for (Value v : returnOp.getOperands()) {
      returnValues.push_back(mapping.lookupOrDefault(v));
    }

    // Replace instance results with return values.
    rewriter.replaceOp(instanceOp, returnValues);
    return success();
  }
};

struct InlineTemplateInstancesPass final
    : impl::InlineTemplateInstancesPassBase<InlineTemplateInstancesPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InlineTemplateInstancePattern>(&getContext());

    // REQUIRED: walkAndApplyPatterns uses post-order traversal, meaning nested
    // template.instance ops are processed before their parents. This is
    // essential for correctness - inner instances must be fully inlined before
    // the outer instance's branches are inlined, otherwise we would miss
    // template.branch ops inside nested instances.
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Template
