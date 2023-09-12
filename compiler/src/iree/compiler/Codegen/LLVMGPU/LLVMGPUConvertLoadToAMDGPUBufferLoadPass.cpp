// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace {

// Transform vector.transfer_reads to vector.load + vector.insert
struct VectorTransferReadToLoad final
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value vector = op.getVector();
    VectorType vectorType = dyn_cast<VectorType>(vector.getType());
    ArrayRef<int64_t> vectorShape = vectorType.getShape();
    SmallVector<int64_t> shape1D{vectorShape[vectorShape.size() - 1]};
    Value constant;
    SmallVector<int64_t> insertionIndices;
    bool insertionRequired = vectorShape.size() > 1;
    if (insertionRequired) {
      for (int i = 0; i < vectorShape.size() - 1; i++) {
        assert(vectorShape[i] == 1);
        insertionIndices.push_back(0);
      }
      constant = rewriter.create<arith::ConstantOp>(
          loc, vectorType, rewriter.getZeroAttr(vectorType));
    }
    Value source = op.getSource();
    SmallVector<Value> indices(op.getIndices().begin(), op.getIndices().end());
    // Bitcast to integer
    for (int i = 0; i < indices.size(); i++)
      indices[i] = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), indices[i]);
    // TODO: Handle additional transfer read attributes like permutation maps,
    // masks etc.
    Value sgprOffset =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    auto loadOp = rewriter.create<amdgpu::RawBufferLoadOp>(
        loc, VectorType::get(shape1D, vectorType.getElementType()), source,
        indices, rewriter.getBoolAttr(false), rewriter.getI32IntegerAttr(0),
        sgprOffset);
    if (insertionRequired) {
      rewriter.replaceOpWithNewOp<vector::InsertOp>(op, loadOp.getResult(),
                                                    constant, insertionIndices);
    } else {
      rewriter.replaceOp(op, loadOp);
    }
    return success();
  }
};

// Transform vector.transfer_reads to vector.load + vector.insert
struct VectorLoadToRawBufferLoad final
    : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    Attribute spaceAttr = op.getMemRefType().getMemorySpace();
    Attribute globalSpace =
        gpu::AddressSpaceAttr::get(op.getContext(), gpu::AddressSpace::Global);
    /// Address space must either be global or unspecified.
    if (spaceAttr && spaceAttr != globalSpace) {
      return failure();
    }

    auto loadType = op.getVectorType();
    /// Requires 1D vectors.
    if (loadType.getRank() != 1) {
      return failure();
    }

    Type elementType = loadType.getElementType();
    int64_t bitWidth = elementType.getIntOrFloatBitWidth();
    /// Unsupported load type.
    if (bitWidth != 8 && bitWidth != 16 && bitWidth != 32) {
      return failure();
    }

    int64_t numElements = loadType.getNumElements();
    int64_t loadNumBits = bitWidth * numElements;
    /// RawBufferLoads can load at most 128 bits.
    if (!llvm::isPowerOf2_64(loadNumBits) || loadNumBits < 8 ||
        loadNumBits > 128) {
      return failure();
    }

    /// Single element vectors are unsupported by buffer loads, so we instead
    /// load a scalar and broadcast.
    Type targetLoadType = loadType;
    if (loadType.getNumElements() <= 1) {
      targetLoadType = elementType;
    }

    SmallVector<Value> indices(op.getIndices().begin(), op.getIndices().end());
    // Bitcast to i32.
    for (int i = 0; i < indices.size(); i++)
      indices[i] = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), indices[i]);

    Value sgprOffset =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    Value bufferLoadOp = rewriter.create<amdgpu::RawBufferLoadOp>(
        loc, targetLoadType, op.getBase(), indices, rewriter.getBoolAttr(false),
        rewriter.getI32IntegerAttr(0), sgprOffset);

    /// Broadcast to the original type if we have a single element vector.
    if (targetLoadType != loadType) {
      bufferLoadOp =
          rewriter.create<vector::BroadcastOp>(loc, loadType, bufferLoadOp);
    }
    rewriter.replaceOp(op, bufferLoadOp);
    return failure();
  }
};

// Transform vector.transfer_reads to vector.load + vector.insert
struct VectorStoreToRawBufferStore final
    : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    Attribute spaceAttr = op.getMemRefType().getMemorySpace();
    Attribute globalSpace =
        gpu::AddressSpaceAttr::get(op.getContext(), gpu::AddressSpace::Global);
    /// Address space must either be global or unspecified.
    if (spaceAttr && spaceAttr != globalSpace) {
      return failure();
    }

    auto storeType = op.getVectorType();
    /// Requires 1D vectors.
    if (storeType.getRank() != 1) {
      return failure();
    }

    Type elementType = storeType.getElementType();
    int64_t bitWidth = elementType.getIntOrFloatBitWidth();
    /// Unsupported store type.
    if (bitWidth != 8 && bitWidth != 16 && bitWidth != 32) {
      return failure();
    }

    int64_t numElements = storeType.getNumElements();
    int64_t loadNumBits = bitWidth * numElements;
    /// RawBufferLoads can load at most 128 bits.
    if (!llvm::isPowerOf2_64(loadNumBits) || loadNumBits < 8 ||
        loadNumBits > 128) {
      return failure();
    }

    /// Single element vectors are unsupported by buffer loads, so we instead
    /// load a scalar and broadcast.
    Type targetStoreType = storeType;
    Value source = op.getValueToStore();
    if (storeType.getNumElements() <= 1) {
      targetStoreType = elementType;
      SmallVector<Attribute> indices;
      if (storeType.getRank() == 1) {
        indices.push_back(rewriter.getI64IntegerAttr(0));
      }
      source = rewriter.create<vector::ExtractOp>(loc, source,
                                                  storeType.getRank() == 1
                                                      ? ArrayRef<int64_t>{0}
                                                      : ArrayRef<int64_t>{});
    }

    SmallVector<Value> indices(op.getIndices().begin(), op.getIndices().end());
    // Bitcast to i32.
    for (int i = 0; i < indices.size(); i++)
      indices[i] = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), indices[i]);

    Value sgprOffset =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<amdgpu::RawBufferStoreOp>(
        op, TypeRange{}, source, op.getBase(), indices,
        rewriter.getBoolAttr(false), rewriter.getI32IntegerAttr(0), sgprOffset);
    return failure();
  }
};

void populateConvertLoadToAMDGPUPatterns(RewritePatternSet &patterns) {
  patterns.insert<VectorTransferReadToLoad, VectorLoadToRawBufferLoad,
                  VectorStoreToRawBufferStore>(patterns.getContext());
}

struct LLVMGPUConvertLoadToAMDGPUBufferLoadPass final
    : public LLVMGPUConvertLoadToAMDGPUBufferLoadBase<
          LLVMGPUConvertLoadToAMDGPUBufferLoadPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<amdgpu::AMDGPUDialect>();
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(&getContext());
    populateConvertLoadToAMDGPUPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUConvertLoadToAMDGPUBufferLoadPass() {
  return std::make_unique<LLVMGPUConvertLoadToAMDGPUBufferLoadPass>();
}

} // namespace iree_compiler
} // namespace mlir
