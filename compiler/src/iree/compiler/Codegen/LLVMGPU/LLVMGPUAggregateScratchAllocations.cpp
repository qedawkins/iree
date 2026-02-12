// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LLVMGPUAggregateScratchAllocations.cpp ----------------------------===//
//
// Collects all iree_codegen.alloc_scratch ops in a dispatch function and
// replaces them with views into a single scratch binding. The byte size is
// bump-allocated with 16-byte alignment, and the parent export's scratch_size
// region is populated with the total.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#define DEBUG_TYPE "iree-llvmgpu-aggregate-scratch-allocations"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUAGGREGATESCRATCHALLOCATIONSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Alignment in bytes for each scratch allocation slot.
static constexpr int64_t kScratchAlignment = 16;

struct LLVMGPUAggregateScratchAllocationsPass final
    : impl::LLVMGPUAggregateScratchAllocationsPassBase<
          LLVMGPUAggregateScratchAllocationsPass> {
  void runOnOperation() override;
};

void LLVMGPUAggregateScratchAllocationsPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  // Find the function with alloc_scratch ops. V1 asserts at most one.
  FunctionOpInterface targetFunc;
  SmallVector<IREE::Codegen::AllocScratchOp> allocOps;

  moduleOp->walk([&](FunctionOpInterface funcOp) {
    SmallVector<IREE::Codegen::AllocScratchOp> funcAllocOps;
    funcOp->walk([&](IREE::Codegen::AllocScratchOp op) {
      funcAllocOps.push_back(op);
    });
    if (funcAllocOps.empty()) {
      return;
    }
    if (targetFunc) {
      funcOp.emitOpError(
          "multiple functions with alloc_scratch ops not supported");
      return;
    }
    targetFunc = funcOp;
    allocOps = std::move(funcAllocOps);
  });

  if (!targetFunc) {
    return; // No alloc_scratch ops — pass is a no-op.
  }

  Location loc = targetFunc.getLoc();
  DataLayout dataLayout = DataLayout::closest(targetFunc);

  // Step 1: Compute byte sizes and assign bump-allocated offsets.
  SmallVector<int64_t> byteOffsets;
  int64_t currentOffset = 0;
  for (IREE::Codegen::AllocScratchOp allocOp : allocOps) {
    MemRefType memrefType = allocOp.getResultType();
    if (!memrefType.hasStaticShape()) {
      allocOp.emitOpError("dynamic scratch allocations not yet supported");
      return signalPassFailure();
    }

    int64_t elementBits =
        dataLayout.getTypeSizeInBits(memrefType.getElementType());
    int64_t numElements = memrefType.getNumElements();
    int64_t byteSize = numElements * elementBits / 8;

    currentOffset = llvm::alignTo(currentOffset, kScratchAlignment);
    byteOffsets.push_back(currentOffset);
    currentOffset += byteSize;
  }
  int64_t totalBytes = llvm::alignTo(currentOffset, kScratchAlignment);

  // Step 2: Find pipeline layout from existing binding subspan ops.
  IREE::HAL::PipelineLayoutAttr pipelineLayout;
  targetFunc->walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    pipelineLayout = subspanOp.getLayout();
    return WalkResult::interrupt();
  });
  if (!pipelineLayout) {
    targetFunc.emitOpError("could not find pipeline layout for scratch binding");
    return signalPassFailure();
  }

  // Scratch binding is the last binding in the pipeline layout.
  // This is by construction: the host-side FormDispatchScratchAllocations pass
  // appends the scratch binding as the final entry.
  int64_t scratchBindingIdx = pipelineLayout.getBindings().size() - 1;

  // Step 3: Create scratch buffer binding at function entry.
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(
      &targetFunc.getFunctionBody().front());

  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  MemRefType scratchBufferType =
      MemRefType::get({totalBytes}, builder.getI8Type());

  Value scratchBuffer = IREE::HAL::InterfaceBindingSubspanOp::create(
      builder, loc, scratchBufferType, pipelineLayout,
      builder.getIndexAttr(scratchBindingIdx), c0,
      /*dynamic_dims=*/ValueRange{},
      builder.getIndexAttr(64),
      /*flags=*/IREE::HAL::DescriptorFlagsAttr{});

  // Step 4: Replace each alloc_scratch with memref.view into scratch buffer.
  for (auto [i, allocOp] : llvm::enumerate(allocOps)) {
    builder.setInsertionPoint(allocOp);
    Value offset = arith::ConstantIndexOp::create(
        builder, allocOp.getLoc(), byteOffsets[i]);
    Value view = memref::ViewOp::create(
        builder, allocOp.getLoc(), allocOp.getResultType(),
        scratchBuffer, offset, ValueRange{});
    allocOp.replaceAllUsesWith(view);
    allocOp.erase();
  }

  // Step 5: Update the export's scratch_size region with the total byte count.
  std::optional<IREE::HAL::ExecutableExportOp> exportOp =
      getEntryPoint(targetFunc);
  if (!exportOp) {
    targetFunc.emitOpError(
        "could not find hal.executable.export for scratch_size update");
    return signalPassFailure();
  }

  Block *scratchSizeBody = exportOp->getScratchSizeBody();
  if (!scratchSizeBody) {
    exportOp->emitOpError("expected scratch_size region to exist");
    return signalPassFailure();
  }

  // Clear placeholder ops (keeps block arguments intact).
  scratchSizeBody->clear();

  builder.setInsertionPointToStart(scratchSizeBody);
  Value totalBytesVal = arith::ConstantIndexOp::create(
      builder, exportOp->getLoc(), totalBytes);
  IREE::HAL::ReturnOp::create(builder, exportOp->getLoc(), totalBytesVal);
}

} // namespace

} // namespace mlir::iree_compiler
