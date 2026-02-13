// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LLVMGPUPromoteAccumulatorToWorkgroup.cpp -------------------------===//
//
// In Stream-K matmul, the k-loop wraps the thread-distributed scf.forall
// (opposite of normal matmul where the forall wraps the k-loop). This causes
// the WMMA accumulator tensor.empty() to not be a direct user of the thread
// forall, so GPUInferMemorySpace marks it as private memory. Each thread then
// only sees its own WMMA fragment, producing all-zeros output.
//
// This pass finds tensor.empty() ops that flow (through vector.transfer_write
// and scf.for iter_args) into thread-mapped scf.forall shared_outs and
// replaces them with bufferization.alloc_tensor{memory_space = workgroup}.
// This ensures the bufferizer allocates the accumulator in workgroup memory
// where all threads can write their WMMA fragments via parallel_insert_slice.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-llvmgpu-promote-accumulator-to-workgroup"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUPROMOTEACCUMULATORTOWORKGROUPPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Trace backward from a value through scf.for iter_args and
/// vector.transfer_write to find the root tensor.empty() op.
/// Returns nullptr if the chain doesn't lead to a tensor.empty().
static tensor::EmptyOp traceToTensorEmpty(Value val) {
  // Limit the trace depth to avoid infinite loops in pathological IR.
  constexpr int kMaxDepth = 16;
  Value current = val;
  for (int depth = 0; depth < kMaxDepth; ++depth) {
    if (!current) {
      return nullptr;
    }
    // Found the root tensor.empty.
    if (auto emptyOp = current.getDefiningOp<tensor::EmptyOp>()) {
      return emptyOp;
    }
    // Trace through vector.transfer_write: result ← base tensor.
    if (auto writeOp =
            current.getDefiningOp<vector::TransferWriteOp>()) {
      current = writeOp.getBase();
      continue;
    }
    // Trace through scf.for: result(i) ← init_arg(i).
    if (auto forResult = dyn_cast<OpResult>(current)) {
      if (auto forOp = dyn_cast<scf::ForOp>(forResult.getOwner())) {
        unsigned idx = forResult.getResultNumber();
        current = forOp.getInitArgs()[idx];
        continue;
      }
    }
    // Trace through scf.for body block arguments (iter_args).
    // Block arguments [iv, iter_arg_0, iter_arg_1, ...] correspond to
    // for op init_args [init_0, init_1, ...].
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      Block *block = blockArg.getOwner();
      if (auto forOp = dyn_cast<scf::ForOp>(block->getParentOp())) {
        // Skip the induction variable (argument 0).
        unsigned argIdx = blockArg.getArgNumber();
        if (argIdx >= forOp.getNumInductionVars()) {
          unsigned initIdx = argIdx - forOp.getNumInductionVars();
          current = forOp.getInitArgs()[initIdx];
          continue;
        }
      }
    }
    // Unknown op in the chain; give up.
    return nullptr;
  }
  return nullptr;
}

struct LLVMGPUPromoteAccumulatorToWorkgroupPass final
    : impl::LLVMGPUPromoteAccumulatorToWorkgroupPassBase<
          LLVMGPUPromoteAccumulatorToWorkgroupPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect,
                    gpu::GPUDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    Attribute workgroupSpace = gpu::AddressSpaceAttr::get(
        ctx, gpu::GPUDialect::getWorkgroupAddressSpace());

    // Collect tensor.empty ops that should be promoted.
    SmallPtrSet<Operation *, 4> toPromote;

    funcOp.walk([&](scf::ForallOp forallOp) {
      // Only consider thread/warp-distributed foralls.
      if (!forallOpHasMappingType<gpu::GPUThreadMappingAttr,
                                  gpu::GPUWarpMappingAttr>(forallOp)) {
        return;
      }

      // Check each shared_outs init value.
      for (Value initVal : forallOp.getOutputs()) {
        tensor::EmptyOp emptyOp = traceToTensorEmpty(initVal);
        if (emptyOp) {
          toPromote.insert(emptyOp.getOperation());
        }
      }
    });

    if (toPromote.empty()) {
      return;
    }

    // For each tensor.empty, find the zero-init transfer_write and promote
    // the result to workgroup memory via alloc_tensor with copy semantics.
    //
    // Before:
    //   %empty = tensor.empty()                          // private
    //   %zeroed = vector.transfer_write %cst, %empty     // zeros in private
    //   scf.for iter_args(%acc = %zeroed)                 // loop uses private
    //
    // After:
    //   %empty = tensor.empty()                           // private (unchanged)
    //   %zeroed = vector.transfer_write %cst, %empty      // zeros in private
    //   %wg = alloc_tensor(%zeroed) {workgroup}           // copy to workgroup
    //   scf.for iter_args(%acc = %wg)                     // loop uses workgroup
    //
    // Bufferization creates memref.alloc{workgroup} + memref.copy(priv→wg),
    // and GPUDistributeCopyUsingForall distributes the memref.copy.
    for (Operation *op : toPromote) {
      auto emptyOp = cast<tensor::EmptyOp>(op);
      RankedTensorType tensorType = emptyOp.getType();

      // Find the vector.transfer_write that zero-initializes this tensor.
      vector::TransferWriteOp writeOp = nullptr;
      for (OpOperand &use : emptyOp.getResult().getUses()) {
        writeOp = dyn_cast<vector::TransferWriteOp>(use.getOwner());
        if (writeOp) {
          break;
        }
      }

      if (writeOp) {
        // The tensor.empty stays in private memory (default).
        // The transfer_write zeros it in private memory (OK).
        // The alloc_tensor with copy causes bufferization to create
        // memref.alloc{workgroup} + memref.copy(private → workgroup).
        // GPUDistributeCopyUsingForall distributes the memref.copy.
        rewriter.setInsertionPointAfter(writeOp);
        auto allocOp = bufferization::AllocTensorOp::create(
            rewriter, writeOp.getLoc(), tensorType,
            /*dynamicSizes=*/ValueRange{},
            /*copy=*/writeOp.getResult());
        allocOp.setMemorySpaceAttr(workgroupSpace);

        rewriter.replaceAllUsesExcept(writeOp.getResult(),
                                      allocOp.getResult(), allocOp);
      } else {
        // No transfer_write found; promote tensor.empty directly.
        rewriter.setInsertionPoint(emptyOp);
        SmallVector<Value> dynamicDims;
        for (int64_t i = 0, e = tensorType.getRank(); i < e; ++i) {
          if (tensorType.isDynamicDim(i)) {
            dynamicDims.push_back(emptyOp.getDynamicSizes()[i]);
          }
        }
        auto allocOp = bufferization::AllocTensorOp::create(
            rewriter, emptyOp.getLoc(), tensorType, dynamicDims);
        allocOp.setMemorySpaceAttr(workgroupSpace);
        rewriter.replaceOp(emptyOp, allocOp.getResult());
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
