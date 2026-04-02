// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPROMOTESHAREDMEMTOPCFALLOCPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

using namespace IREE::PCF;
using namespace IREE::VectorExt;

namespace {

/// Returns true if the alloc_tensor has workgroup memory space.
static bool hasWorkgroupMemorySpace(bufferization::AllocTensorOp allocOp) {
  std::optional<Attribute> memSpace = allocOp.getMemorySpace();
  if (!memSpace) {
    return false;
  }
  auto addrSpaceAttr = dyn_cast<gpu::AddressSpaceAttr>(*memSpace);
  if (!addrSpaceAttr) {
    return false;
  }
  return addrSpaceAttr.getValue() ==
         gpu::GPUDialect::getWorkgroupAddressSpace();
}

/// Process a single bufferization.alloc_tensor with workgroup memory space
/// inside a shared_executor. Converts the alloc_tensor + transfer_write +
/// value_barrier + transfer_read chain to pcf.alloc + sref ops.
static LogicalResult processAllocTensor(bufferization::AllocTensorOp allocOp,
                                        SharedExecutorOp sharedExec,
                                        IRRewriter &rewriter) {
  ScopeAttrInterface scope = sharedExec.getScope();
  Location loc = allocOp.getLoc();

  RankedTensorType tensorType = allocOp.getType();
  ShapedRefType srefType =
      ShapedRefType::get(rewriter.getContext(), tensorType.getShape(),
                         tensorType.getElementType(), scope);

  // Create pcf.alloc in the initializer region.
  Region &initializer = sharedExec.getInitializer();
  Block *initBlock = nullptr;
  if (initializer.empty()) {
    // Create the initializer region with a yield terminator.
    initBlock = &initializer.emplaceBlock();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(initBlock);
    IREE::PCF::YieldOp::create(rewriter, loc);
  } else {
    initBlock = &initializer.front();
  }

  // Insert the pcf.alloc before the yield in the initializer.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Operation *yieldOp = initBlock->getTerminator();
    rewriter.setInsertionPoint(yieldOp);
    Value allocSref = AllocOp::create(rewriter, loc, srefType);

    // Add the sref to the yield operands.
    auto yield = cast<IREE::PCF::YieldOp>(yieldOp);
    SmallVector<Value> yieldOperands(yield.getOperands());
    yieldOperands.push_back(allocSref);
    rewriter.setInsertionPoint(yield);
    IREE::PCF::YieldOp::create(rewriter, loc, yieldOperands);
    rewriter.eraseOp(yield);
  }

  // Add a leading block argument to the execute region for the sref.
  Block &execBlock = sharedExec.getRegion().front();
  int64_t numLeading = sharedExec.getNumLeadingArgs();
  BlockArgument srefArg = execBlock.insertArgument(numLeading, srefType, loc);
  sharedExec.setNumLeadingArgs(numLeading + 1);

  // Now replace the usage chain:
  // alloc_tensor -> transfer_write -> value_barrier -> transfer_read
  //
  // Find all transfer_write ops writing to the alloc_tensor result.
  SmallVector<vector::TransferWriteOp> writesToConvert;
  for (OpOperand &use : allocOp.getResult().getUses()) {
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(use.getOwner())) {
      writesToConvert.push_back(writeOp);
    }
  }

  for (vector::TransferWriteOp writeOp : writesToConvert) {
    rewriter.setInsertionPoint(writeOp);

    // Create iree_vector_ext.transfer_write to the sref.
    SmallVector<bool> inBounds;
    for (Attribute attr : writeOp.getInBounds()) {
      inBounds.push_back(cast<BoolAttr>(attr).getValue());
    }
    AffineMap permMap = writeOp.getPermutationMap();
    TransferWriteOp::create(rewriter, writeOp.getLoc(), writeOp.getVector(),
                            srefArg, writeOp.getIndices(), inBounds, permMap,
                            writeOp.getMask());

    // Find value_barrier ops consuming this transfer_write's result.
    Value writtenTensor = writeOp.getResult();
    SmallVector<IREE::GPU::ValueBarrierOp> barriersToConvert;
    for (OpOperand &use : writtenTensor.getUses()) {
      if (auto barrier = dyn_cast<IREE::GPU::ValueBarrierOp>(use.getOwner())) {
        barriersToConvert.push_back(barrier);
      }
    }

    for (IREE::GPU::ValueBarrierOp barrier : barriersToConvert) {
      rewriter.setInsertionPoint(barrier);

      // Insert gpu.barrier to replace the value_barrier.
      gpu::BarrierOp::create(rewriter, barrier.getLoc());

      // Find transfer_read ops consuming the barrier result that corresponds
      // to our written tensor.
      for (auto [barrierInput, barrierResult] :
           llvm::zip_equal(barrier.getInputs(), barrier.getResults())) {
        if (barrierInput != writtenTensor) {
          continue;
        }

        SmallVector<vector::TransferReadOp> readsToConvert;
        for (OpOperand &use : barrierResult.getUses()) {
          if (auto readOp = dyn_cast<vector::TransferReadOp>(use.getOwner())) {
            readsToConvert.push_back(readOp);
          }
        }

        for (vector::TransferReadOp readOp : readsToConvert) {
          rewriter.setInsertionPoint(readOp);
          VectorType vecType = readOp.getVectorType();

          SmallVector<bool> readInBounds;
          for (Attribute attr : readOp.getInBounds()) {
            readInBounds.push_back(cast<BoolAttr>(attr).getValue());
          }
          AffineMap readPermMap = readOp.getPermutationMap();
          Value newRead = IREE::VectorExt::TransferReadOp::create(
              rewriter, readOp.getLoc(), vecType, srefArg, readOp.getIndices(),
              readOp.getPadding(), readInBounds, readPermMap, readOp.getMask());
          rewriter.replaceOp(readOp, newRead);
        }
      }

      // Erase the value_barrier if all its results are unused.
      if (barrier->use_empty()) {
        rewriter.eraseOp(barrier);
      }
    }

    // Erase the transfer_write (it has been replaced by the sref write).
    if (writeOp->use_empty()) {
      rewriter.eraseOp(writeOp);
    }
  }

  // Erase the alloc_tensor.
  if (allocOp->use_empty()) {
    rewriter.eraseOp(allocOp);
  }

  return success();
}

struct GPUPromoteSharedMemToPCFAllocPass final
    : impl::GPUPromoteSharedMemToPCFAllocPassBase<
          GPUPromoteSharedMemToPCFAllocPass> {
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    getOperation()->walk([&](SharedExecutorOp sharedExec) {
      // Collect alloc_tensor ops with workgroup memory space.
      SmallVector<bufferization::AllocTensorOp> allocOps;
      sharedExec.getRegion().walk([&](bufferization::AllocTensorOp allocOp) {
        if (hasWorkgroupMemorySpace(allocOp)) {
          allocOps.push_back(allocOp);
        }
      });

      for (bufferization::AllocTensorOp allocOp : allocOps) {
        if (failed(processAllocTensor(allocOp, sharedExec, rewriter))) {
          signalPassFailure();
          return;
        }
      }
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
