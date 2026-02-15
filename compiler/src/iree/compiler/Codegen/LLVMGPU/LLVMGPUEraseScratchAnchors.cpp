// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LLVMGPUEraseScratchAnchors.cpp ------------------------------------===//
//
// Erases dead dispatch.tensor.scratch anchor ops and their dead producer
// chain before bufferization. The host-side FormDispatchScratchAllocations
// pass inserts these anchors to retain the scratch binding, but on the
// device side the actual scratch binding is created later by
// AggregateScratchAllocations from iree_codegen.alloc_scratch ops.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUERASESCRATCHANCHORSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct LLVMGPUEraseScratchAnchorsPass final
    : impl::LLVMGPUEraseScratchAnchorsPassBase<
          LLVMGPUEraseScratchAnchorsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // Collect dead scratch anchor ops.
    SmallVector<IREE::TensorExt::DispatchTensorScratchOp> deadScratchOps;
    funcOp->walk(
        [&](IREE::TensorExt::DispatchTensorScratchOp scratchOp) {
          if (scratchOp.getResult().use_empty()) {
            deadScratchOps.push_back(scratchOp);
          }
        });

    // Erase each dead scratch op and its dead producer chain.
    for (IREE::TensorExt::DispatchTensorScratchOp scratchOp :
         deadScratchOps) {
      Value source = scratchOp.getSource();
      scratchOp->erase();
      // Walk up the producer chain erasing dead ops. This removes the
      // hal.interface.binding.subspan and any flow.dispatch.tie_shape ops
      // that are no longer needed.
      while (source) {
        Operation *producer = source.getDefiningOp();
        if (!producer || !producer->use_empty()) {
          break;
        }
        // Follow single operand if it exists for chain cleanup.
        source = producer->getNumOperands() > 0 ? producer->getOperand(0)
                                                 : Value();
        producer->erase();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
