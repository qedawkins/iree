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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUWRAPINSHAREDEXECUTORPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

using namespace IREE::GPU;
using namespace IREE::PCF;

namespace {

/// Wraps the body of a region in a pcf.shared_executor with thread scope.
/// Tensor values used by ops in the body but defined outside the region are
/// captured as readonly sref inputs to the shared_executor. Inside the body,
/// pcf.read_slice ops produce full-sized tensor loads from the sref block
/// arguments, and uses of the original tensors are replaced.
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

  // Identify tensor values used by ops to move that are defined outside.
  // These will become readonly sref inputs to the shared_executor.
  SetVector<Value> capturedTensors;
  for (Operation *bodyOp : opsToMove) {
    for (Value operand : bodyOp->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType())) {
        continue;
      }
      // Check if defined outside the body region.
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (!bodyRegion.isAncestor(blockArg.getOwner()->getParent())) {
          capturedTensors.insert(operand);
        }
      } else if (!bodyRegion.isAncestor(
                     operand.getDefiningOp()->getParentRegion())) {
        capturedTensors.insert(operand);
      }
    }
  }

  // Create the shared_executor with readonly inputs.
  rewriter.setInsertionPoint(terminator);
  Attribute threadScopeAttr = ThreadScopeAttr::get(rewriter.getContext());
  IREE::PCF::ScopeAttrInterface threadScope =
      cast<IREE::PCF::ScopeAttrInterface>(threadScopeAttr);
  SharedExecutorOp sharedExec =
      SharedExecutorOp::create(rewriter, op->getLoc(), threadScope,
                               /*readonlyInits=*/capturedTensors.getArrayRef(),
                               /*readwriteInits=*/ValueRange());

  // Move all ops into the shared_executor's execute region.
  Block &execBlock = sharedExec.getRegion().front();
  for (Operation *bodyOp : opsToMove) {
    bodyOp->moveBefore(&execBlock, execBlock.end());
  }

  // Replace uses of captured tensors with pcf.read_slice from sref block args.
  ArrayRef<BlockArgument> readonlyRefs = sharedExec.getReadonlyRefArgs();
  SmallVector<Value> capturedList(capturedTensors.begin(),
                                  capturedTensors.end());
  assert(readonlyRefs.size() == capturedList.size());
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&execBlock);

    for (auto [tensor, srefArg] : llvm::zip_equal(capturedList, readonlyRefs)) {
      auto tensorType = cast<RankedTensorType>(tensor.getType());
      int64_t rank = tensorType.getRank();

      // Create pcf.read_slice with full size (identity slice).
      SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      for (int64_t i = 0; i < rank; ++i) {
        sizes.push_back(rewriter.getIndexAttr(tensorType.getDimSize(i)));
      }
      SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

      Value readSlice = ReadSliceOp::create(rewriter, op->getLoc(), tensorType,
                                            srefArg, offsets, sizes, strides);

      // Replace all uses of the original tensor within the execute region.
      tensor.replaceUsesWithIf(readSlice, [&](OpOperand &use) {
        return sharedExec.getRegion().isAncestor(
            use.getOwner()->getParentRegion());
      });
    }
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
      bool alreadyWrapped =
          llvm::any_of(body.getOperations(), [](Operation &bodyOp) {
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

    // If no PCF ops were found (single-workgroup dispatch), wrap the
    // function body directly. Keep binding infrastructure outside and
    // capture tensor results as readonly sref inputs.
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

    // Partition: binding/constant ops stay outside, everything else moves in.
    // Tensors produced by load_from_buffer become readonly sref inputs.
    SmallVector<Operation *> opsToKeep;
    SmallVector<Operation *> opsToMove;
    SetVector<Value> capturedTensors;

    for (Operation &op : funcBody.getOperations()) {
      if (&op == funcTerminator) {
        continue;
      }
      StringRef dialect =
          op.getDialect() ? op.getDialect()->getNamespace() : "";
      bool keep = dialect == "hal" || dialect == "amdgpu" ||
                  isa<IREE::Codegen::LoadFromBufferOp>(op) ||
                  isa<IREE::Codegen::WorkgroupCountHintOp>(op) ||
                  op.hasTrait<OpTrait::ConstantLike>();
      if (keep) {
        opsToKeep.push_back(&op);
        // If this produces a tensor, it may need to be captured.
        for (Value result : op.getResults()) {
          if (isa<RankedTensorType>(result.getType())) {
            capturedTensors.insert(result);
          }
        }
      } else {
        opsToMove.push_back(&op);
      }
    }

    if (opsToMove.empty()) {
      return;
    }

    // Filter capturedTensors to only those actually used by opsToMove.
    SetVector<Value> usedCaptures;
    DenseSet<Operation *> moveSet(opsToMove.begin(), opsToMove.end());
    for (Value tensor : capturedTensors) {
      for (OpOperand &use : tensor.getUses()) {
        if (moveSet.contains(use.getOwner())) {
          usedCaptures.insert(tensor);
          break;
        }
      }
    }

    // Create shared_executor with readonly inputs.
    rewriter.setInsertionPoint(opsToMove.front());
    Attribute threadScopeAttr = ThreadScopeAttr::get(rewriter.getContext());
    IREE::PCF::ScopeAttrInterface threadScope =
        cast<IREE::PCF::ScopeAttrInterface>(threadScopeAttr);
    SharedExecutorOp sharedExec =
        SharedExecutorOp::create(rewriter, funcOp.getLoc(), threadScope,
                                 /*readonlyInits=*/usedCaptures.getArrayRef(),
                                 /*readwriteInits=*/ValueRange());

    Block &execBlock = sharedExec.getRegion().front();
    for (Operation *op : opsToMove) {
      op->moveBefore(&execBlock, execBlock.end());
    }

    // Replace captured tensor uses with pcf.read_slice from sref block args.
    ArrayRef<BlockArgument> readonlyRefs = sharedExec.getReadonlyRefArgs();
    SmallVector<Value> captureList(usedCaptures.begin(), usedCaptures.end());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&execBlock);
      for (auto [tensor, srefArg] :
           llvm::zip_equal(captureList, readonlyRefs)) {
        auto tensorType = cast<RankedTensorType>(tensor.getType());
        int64_t rank = tensorType.getRank();
        SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        for (int64_t i = 0; i < rank; ++i) {
          sizes.push_back(rewriter.getIndexAttr(tensorType.getDimSize(i)));
        }
        SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
        Value readSlice =
            ReadSliceOp::create(rewriter, funcOp.getLoc(), tensorType, srefArg,
                                offsets, sizes, strides);
        tensor.replaceUsesWithIf(readSlice, [&](OpOperand &use) {
          return sharedExec.getRegion().isAncestor(
              use.getOwner()->getParentRegion());
        });
      }
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
