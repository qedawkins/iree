// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUWRAPINSHAREDEXECUTORPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

using namespace IREE::GPU;
using namespace IREE::PCF;

namespace {

/// Wraps the body of a workgroup-scoped pcf.generic or pcf.loop op in a
/// pcf.shared_executor with thread scope. The shared_executor captures srefs
/// from the enclosing scope (since it is not IsolatedFromAbove) and provides
/// a threadgroup block argument for collective execution.
static LogicalResult wrapBodyInSharedExecutor(Operation *op, Region &bodyRegion,
                                              IRRewriter &rewriter) {
  Block &body = bodyRegion.front();
  Operation *terminator = body.getTerminator();

  // Collect all ops in the body except the terminator.
  SmallVector<Operation *> opsToMove;
  for (Operation &bodyOp : body.getOperations()) {
    if (&bodyOp == terminator) {
      continue;
    }
    opsToMove.push_back(&bodyOp);
  }

  // If the body is empty (only terminator), nothing to wrap.
  if (opsToMove.empty()) {
    return success();
  }

  // Create the shared_executor with thread scope just before the terminator.
  rewriter.setInsertionPoint(terminator);
  Attribute threadScopeAttr = ThreadScopeAttr::get(rewriter.getContext());
  IREE::PCF::ScopeAttrInterface threadScope =
      cast<IREE::PCF::ScopeAttrInterface>(threadScopeAttr);
  SharedExecutorOp sharedExec = SharedExecutorOp::create(
      rewriter, op->getLoc(), threadScope,
      /*readwriteInits=*/ValueRange());

  // Move all ops into the shared_executor's execute region, then add a
  // pcf.return terminator.
  Block &execBlock = sharedExec.getRegion().front();
  for (Operation *bodyOp : opsToMove) {
    bodyOp->moveBefore(&execBlock, execBlock.end());
  }

  // Add the pcf.return terminator to the shared_executor body.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&execBlock);
    ReturnOp::create(rewriter, op->getLoc());
  }

  return success();
}

struct GPUWrapInSharedExecutorPass final
    : impl::GPUWrapInSharedExecutorPassBase<GPUWrapInSharedExecutorPass> {
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    // Walk all pcf.generic and pcf.loop ops with workgroup scope.
    getOperation()->walk([&](Operation *op) {
      IREE::PCF::ScopeAttrInterface scope;
      Region *bodyRegion = nullptr;

      if (auto genericOp = dyn_cast<GenericOp>(op)) {
        scope = genericOp.getScope();
        bodyRegion = &genericOp.getRegion();
      } else if (auto loopOp = dyn_cast<LoopOp>(op)) {
        scope = loopOp.getScope();
        bodyRegion = &loopOp.getRegion();
      } else {
        return;
      }

      // Only wrap workgroup-scoped ops.
      if (!isa<IREE::Codegen::WorkgroupScopeAttr>(scope)) {
        return;
      }

      // Skip if the body already contains a shared_executor.
      Block &body = bodyRegion->front();
      bool alreadyWrapped = llvm::any_of(body.getOperations(),
                                         [](Operation &bodyOp) {
                                           return isa<SharedExecutorOp>(bodyOp);
                                         });
      if (alreadyWrapped) {
        return;
      }

      if (failed(wrapBodyInSharedExecutor(op, *bodyRegion, rewriter))) {
        op->emitError("failed to wrap body in shared_executor");
        return signalPassFailure();
      }
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
