// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPCFDistribution.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-gpu-distribute-shared-executor"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUDISTRIBUTESHAREDEXECUTORPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Returns true if |op| is nested inside a pcf.shared_executor with thread
/// scope. Walks the parent chain, not just the immediate parent.
static bool isInsideThreadScopedSharedExecutor(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto sharedExec = dyn_cast<IREE::PCF::SharedExecutorOp>(parent)) {
      if (isa<IREE::GPU::ThreadScopeAttr>(sharedExec.getScope())) {
        return true;
      }
    }
    parent = parent->getParentOp();
  }
  return false;
}

/// Clone vector-typed constants that were hoisted outside shared_executor
/// regions back to their use sites inside the regions. MLIR's constant
/// folding hoists constants to function scope, but vector distribution
/// only operates inside shared_executor regions. Without this, to_simt
/// ops on hoisted constants survive distribution and reach LLVM lowering.
static void sinkVectorConstantsIntoSharedExecutors(
    FunctionOpInterface funcOp) {
  SmallVector<arith::ConstantOp> constantsToProcess;
  funcOp.walk([&](arith::ConstantOp constOp) {
    if (!isa<VectorType>(constOp.getType())) {
      return;
    }
    // Check if any use is inside a shared_executor.
    bool hasSharedExecUse = false;
    for (OpOperand &use : constOp->getUses()) {
      if (isInsideThreadScopedSharedExecutor(use.getOwner())) {
        hasSharedExecUse = true;
        break;
      }
    }
    if (hasSharedExecUse) {
      constantsToProcess.push_back(constOp);
    }
  });

  for (arith::ConstantOp constOp : constantsToProcess) {
    // Clone the constant at each use site inside a shared_executor.
    SmallVector<OpOperand *> usesToReplace;
    for (OpOperand &use : constOp->getUses()) {
      if (isInsideThreadScopedSharedExecutor(use.getOwner())) {
        usesToReplace.push_back(&use);
      }
    }
    for (OpOperand *use : usesToReplace) {
      OpBuilder builder(use->getOwner());
      arith::ConstantOp cloned = cast<arith::ConstantOp>(
          builder.clone(*constOp.getOperation()));
      use->set(cloned.getResult());
    }
    // If the original constant has no remaining uses, erase it.
    if (constOp->use_empty()) {
      constOp->erase();
    }
  }
}

struct GPUDistributeSharedExecutorPass final
    : impl::GPUDistributeSharedExecutorPassBase<
          GPUDistributeSharedExecutorPass> {
  using GPUDistributeSharedExecutorPassBase::
      GPUDistributeSharedExecutorPassBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "GPUDistributeSharedExecutor running on: "
                            << funcOp.getName() << "\n");

    // Sink vector constants back into shared_executor regions so that
    // distribution can process them.
    sinkVectorConstantsIntoSharedExecutors(funcOp);

    // Read workgroup and subgroup sizes from the function's translation_info.
    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);

    // Create the distribution interface if sizes are available.
    std::unique_ptr<IREE::PCF::DistributionInterface> distInterface;
    if (workgroupSize && subgroupSize) {
      distInterface = std::make_unique<VectorDistributionImpl>(*subgroupSize,
                                                               *workgroupSize);
    }

    if (failed(IREE::PCF::distributeAndLowerSharedExecutors(
            funcOp, distInterface.get()))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
