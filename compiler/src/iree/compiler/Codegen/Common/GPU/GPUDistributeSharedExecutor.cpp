// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPCFDistribution.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-gpu-distribute-shared-executor"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUDISTRIBUTESHAREDEXECUTORPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct GPUDistributeSharedExecutorPass final
    : impl::GPUDistributeSharedExecutorPassBase<
          GPUDistributeSharedExecutorPass> {
  using GPUDistributeSharedExecutorPassBase::
      GPUDistributeSharedExecutorPassBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "GPUDistributeSharedExecutor running on: "
                            << funcOp.getName() << "\n");

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
