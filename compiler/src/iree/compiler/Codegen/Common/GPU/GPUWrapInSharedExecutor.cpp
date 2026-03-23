// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
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

    // If no PCF ops were found (single-workgroup dispatch with no workgroup
    // tiling), wrap the function body directly. Keep binding ops
    // (hal.interface.binding.subspan, amdgpu.fat_raw_buffer_cast,
    // iree_codegen.load_from_buffer, constants) outside the shared_executor
    // and move everything else inside.
    FunctionOpInterface funcOp = getOperation();
    bool hasPCFOps = false;
    funcOp->walk([&](Operation *inner) {
      if (isa<GenericOp, LoopOp, SharedExecutorOp>(inner)) {
        hasPCFOps = true;
      }
    });
    if (hasPCFOps) {
      return;
    }

    Block &funcBody = funcOp.getFunctionBody().front();
    Operation *funcTerminator = funcBody.getTerminator();

    // Partition ops into "keep outside" and "move inside".
    SmallVector<Operation *> opsToMove;
    for (Operation &op : funcBody.getOperations()) {
      if (&op == funcTerminator) {
        continue;
      }
      // Keep binding-related ops and constants outside.
      StringRef dialect = op.getDialect()
                              ? op.getDialect()->getNamespace()
                              : "";
      if (dialect == "hal" || dialect == "amdgpu" ||
          isa<IREE::Codegen::LoadFromBufferOp,
              IREE::Codegen::WorkgroupCountHintOp>(op) ||
          op.hasTrait<OpTrait::ConstantLike>()) {
        continue;
      }
      opsToMove.push_back(&op);
    }

    if (opsToMove.empty()) {
      return;
    }

    // Create shared_executor before the first op to move.
    rewriter.setInsertionPoint(opsToMove.front());
    Attribute threadScopeAttr = ThreadScopeAttr::get(rewriter.getContext());
    IREE::PCF::ScopeAttrInterface threadScope =
        cast<IREE::PCF::ScopeAttrInterface>(threadScopeAttr);
    SharedExecutorOp sharedExec = SharedExecutorOp::create(
        rewriter, funcOp.getLoc(), threadScope,
        /*readwriteInits=*/ValueRange());

    Block &execBlock = sharedExec.getRegion().front();
    for (Operation *op : opsToMove) {
      op->moveBefore(&execBlock, execBlock.end());
    }

    // Add pcf.return terminator.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&execBlock);
      ReturnOp::create(rewriter, funcOp.getLoc());
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
