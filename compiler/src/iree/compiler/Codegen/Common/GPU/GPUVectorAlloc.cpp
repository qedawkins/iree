// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-gpu-vector-alloc"

namespace mlir::iree_compiler {

namespace {
/// Merge insert_slice operation with store/transferWriteOp operation.
class InsertSliceOfTransferWriteOpFolder final
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override;
};

template <typename XferOp, typename ExtractOrInsertOp>
static LogicalResult preconditionsFoldExtractOrInsertWithTransferOp(
    RewriterBase &rewriter, XferOp xferOp,
    ExtractOrInsertOp extractOrInsertSliceOp) {
  if (xferOp.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(xferOp, "out of bounds transfer dim");
  if (xferOp.getMask())
    return rewriter.notifyMatchFailure(xferOp, "masked transfer");
  if (!extractOrInsertSliceOp.hasUnitStride()) {
    return rewriter.notifyMatchFailure(
        xferOp, "non-1 stride insert/extract, requires keeping track of "
                "strides, this may result in needing to insert "
                "vector.insert_strided_slice/extract_strided_slice ops");
  }
  return success();
}

LogicalResult InsertSliceOfTransferWriteOpFolder::matchAndRewrite(
    tensor::InsertSliceOp insertSliceOp, PatternRewriter &rewriter) const {
  auto writeOp = insertSliceOp.getSource()
                     .template getDefiningOp<vector::TransferWriteOp>();
  if (!writeOp)
    return rewriter.notifyMatchFailure(insertSliceOp, "not a transfer_write");

  LogicalResult preconditionResult =
      preconditionsFoldExtractOrInsertWithTransferOp(rewriter, writeOp,
                                                     insertSliceOp);
  if (failed(preconditionResult))
    return preconditionResult;

  SmallVector<Value> indices(writeOp.getIndices().begin(),
                             writeOp.getIndices().end());
  SmallVector<Value> sourceIndices;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, writeOp.getLoc(), insertSliceOp.getMixedOffsets(),
      insertSliceOp.getMixedStrides(), insertSliceOp.getDroppedDims(), indices,
      sourceIndices);

  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
      insertSliceOp, writeOp.getValue(), insertSliceOp.getDest(), sourceIndices,
      AffineMapAttr::get(expandDimsToRank(writeOp.getPermutationMap(),
                                          insertSliceOp.getDestType().getRank(),
                                          insertSliceOp.getDroppedDims())),
      writeOp.getInBoundsAttr());

  return success();
}
} // namespace

// For optimal performance we always want to copy 128 bits.
static constexpr int copyVectorNumBits = 128;

/// Filter to decide which contraction ops need allocations.
static bool contractOpFilter(Operation *op) {
  auto contractOp = dyn_cast<vector::ContractionOp>(op);
  if (!contractOp) {
    return false;
  }
  SmallVector<unsigned> dims;
  for (auto [idx, type] : llvm::enumerate(contractOp.getIteratorTypesArray())) {
    if (type == vector::IteratorType::parallel) {
      dims.push_back(idx);
    }
  }
  SmallVector<int64_t> shapes;
  contractOp.getIterationBounds(shapes);
  // Don't promote vector*matrix kind of case.
  int numNonUnitParallelLoop = 0;
  for (unsigned parallelDim : dims) {
    if (shapes[parallelDim] != 1) {
      numNonUnitParallelLoop++;
    }
  }
  // TODO: Relax this constraint.
  return numNonUnitParallelLoop > 1 && dims.size() >= 2 && dims.size() <= 3;
}

// Allocates a tensor to copy the vector into a la bufferization.alloc_tensor.
// This allocation is always static as vectors are currently always static
// where this is used.
static FailureOr<Value> allocateTensorForVector(OpBuilder &b, Location loc,
                                                Value vector) {
  VectorType vectorType = llvm::cast<VectorType>(vector.getType());
  if (vectorType.isScalable()) {
    return failure();
  }

  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  RankedTensorType tensorType =
      RankedTensorType::get(vectorType.getShape(), vectorType.getElementType(),
                            sharedMemoryAddrSpace);
  // Vectors are always statically shaped.
  auto allocTensorOp = b.create<bufferization::AllocTensorOp>(
      loc, tensorType, ValueRange{}, Value());
  allocTensorOp.setMemorySpaceAttr(sharedMemoryAddrSpace);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value copied = b.create<vector::TransferWriteOp>(loc, vector, allocTensorOp,
                                                   indices, inBounds)
                     .getResult();
  // Create a marker for bufferization to keep this tensor in place. This
  // prevents read/write forwarding of the transfers used to do the copy.
  return b
      .create<bufferization::MaterializeInDestinationOp>(copied.getLoc(),
                                                         copied, copied)
      ->getResult(0);
}

static Value readVectorFromTensor(OpBuilder &b, VectorType vectorType,
                                  Value tensor) {
  Value c0 = b.create<arith::ConstantIndexOp>(tensor.getLoc(), 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  return b
      .create<vector::TransferReadOp>(tensor.getLoc(), vectorType, tensor,
                                      indices, inBounds)
      .getResult();
}

namespace {

struct GPUVectorAllocPass : public GPUVectorAllocBase<GPUVectorAllocPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<gpu::GPUDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    SmallVector<vector::ContractionOp> opsToPromote;
    funcOp.walk([&](vector::ContractionOp op) {
      // Today we only do promotion for certain contractions.
      if (contractOpFilter(op))
        opsToPromote.push_back(op);
    });
    for (vector::ContractionOp contractOp : opsToPromote) {
      OpBuilder builder(contractOp);

      // HACK: Until proper barrier placement is handled later we have to
      // synchronize explicitly in this pass.

      // Synchronize before the write to shared memory to avoid stepping over
      // reads in the previous iteration of a loop.
      builder.create<gpu::BarrierOp>(contractOp->getLoc());

      // Promote both of the input operands, excluding the accumulator.
      OpOperand &lhs = contractOp.getLhsMutable();
      FailureOr<Value> lhsRet =
          allocateTensorForVector(builder, contractOp->getLoc(), lhs.get());
      if (failed(lhsRet)) {
        return signalPassFailure();
      }

      OpOperand &rhs = contractOp.getRhsMutable();
      FailureOr<Value> rhsRet =
          allocateTensorForVector(builder, contractOp->getLoc(), rhs.get());
      if (failed(rhsRet)) {
        return signalPassFailure();
      }

      // Synchronize after the write to shared memory before we read from it.
      builder.create<gpu::BarrierOp>(contractOp->getLoc());

      Value lhsVec =
          readVectorFromTensor(builder, contractOp.getLhsType(), *lhsRet);
      Value rhsVec =
          readVectorFromTensor(builder, contractOp.getRhsType(), *rhsRet);
      lhs.set(lhsVec);
      rhs.set(rhsVec);
    }

    // RewritePatternSet patterns(&getContext());
    // patterns.insert<InsertSliceOfTransferWriteOpFolder>(&getContext());
    // if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    //   return signalPassFailure();
    // }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUVectorAlloc() {
  return std::make_unique<GPUVectorAllocPass>();
}

} // namespace mlir::iree_compiler
