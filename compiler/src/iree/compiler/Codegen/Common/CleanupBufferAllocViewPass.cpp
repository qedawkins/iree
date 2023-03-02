// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- BufferAllocViewCleanUpPass.cpp -------------------------------------===//
//
// This pass performs canonicalizations/cleanups related to HAL interface/buffer
// allocations and views. We need a dedicated pass because patterns here involve
// multiple dialects.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

void populateReshapeToInterfaceTensorPatterns(RewritePatternSet &patterns);

namespace {

// TODO(antigainst): enable dynamic shape support once they are needed.
template <typename TensorReshapeOp>
static Optional<Value> getStaticReshapeOpSrc(TensorReshapeOp reshapeOp) {
  auto reshapeSrcType =
      reshapeOp.getSrc().getType().template cast<ShapedType>();
  auto reshapeDstType = reshapeOp.getType().template cast<ShapedType>();
  if (!reshapeSrcType.hasStaticShape() || !reshapeDstType.hasStaticShape())
    return std::nullopt;
  return reshapeOp.getSrc();
}

/// Folds tensor.expand/collapse_shape into the source
/// hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<readonly:tensor<3x3x1x96xf32>>
///   %tensor = flow.dispatch.tensor.load %subspan :
///       !flow.dispatch.tensor<readonly:tensor<3x3x1x96xf32>> ->
///       tensor<3x3x1x96xf32>
///   %0 = linalg.tensor_reshape %tensor [
///         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///       ] : tensor<3x3x1x96xf32> into tensor<864xf32>
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<readonly:tensor<864xf32>>
///   %0 = flow.dispatch.tensor.load %subspan :
///       !flow.dispatch.tensor<readonly:tensor<864xf32>> -> tensor<864xf32>
template <typename TensorReshapeOp>
struct FoldReshapeIntoInterfaceTensorLoad : OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    Optional<Value> reshapeSrc =
        getStaticReshapeOpSrc<TensorReshapeOp>(reshapeOp);
    if (!reshapeSrc) return failure();

    auto loadOp =
        reshapeSrc->template getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!loadOp) return failure();

    // Make sure we are loading the full incoming subspan. Otherwise we cannot
    // simply adjust the subspan's resultant type later.
    if (!loadOp.offsets().empty() || !loadOp.sizes().empty() ||
        !loadOp.strides().empty())
      return failure();

    auto subspanOp =
        loadOp.getSource()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp) return failure();
    assert(subspanOp.getDynamicDims().empty());

    auto tensorAccess = subspanOp.getType()
                            .template cast<IREE::Flow::DispatchTensorType>()
                            .getAccess();
    auto newSubspanType = IREE::Flow::DispatchTensorType::get(
        tensorAccess, reshapeOp.getResultType());

    Value newSubspanOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp.getLoc(), newSubspanType, subspanOp.getSet(),
        subspanOp.getBinding(), subspanOp.getDescriptorType(),
        subspanOp.getByteOffset(), subspanOp.getDynamicDims(),
        subspanOp.getAlignmentAttr(), subspanOp.getDescriptorFlagsAttr());

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        reshapeOp, reshapeOp.getResultType(), newSubspanOp,
        loadOp.getSourceDims());

    return success();
  }
};

/// Folds tensor.expand/collapse_shape into the source
/// hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<writeonly:tensor<3x3x1x96xf32>>
///   %0 = linalg.tensor_reshape %tensor [
///         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///       ] : tensor<864xf32> into tensor<3x3x1x96xf32>
///   %tensor = flow.dispatch.tensor.store %0, %subspan :
///       !flow.dispatch.tensor<writeonly:tensor<3x3x1x96xf32>> ->
///       tensor<3x3x1x96xf32>
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<writeonly:tensor<864xf32>>
///   %0 = flow.dispatch.tensor.store %tensor, %subspan :
///       !flow.dispatch.tensor<writeonly:tensor<864xf32>> -> tensor<864xf32>
struct FoldReshapeIntoInterfaceTensorStore
    : OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpRewritePattern<IREE::Flow::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Make sure we are storing the full incoming subspan. Otherwise we cannot
    // simply adjust the subspan's resultant type later.
    if (!storeOp.offsets().empty() || !storeOp.sizes().empty() ||
        !storeOp.strides().empty())
      return failure();

    auto reshapeOp = storeOp.getValue().getDefiningOp();
    if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(reshapeOp))
      return failure();

    // Dynamic shapes are currently unsupported.
    Optional<Value> reshapeSrc =
        isa<tensor::CollapseShapeOp>(reshapeOp)
            ? getStaticReshapeOpSrc<tensor::CollapseShapeOp>(
                  cast<tensor::CollapseShapeOp>(reshapeOp))
            : getStaticReshapeOpSrc<tensor::ExpandShapeOp>(
                  cast<tensor::ExpandShapeOp>(reshapeOp));
    if (!reshapeSrc) return failure();

    auto subspanOp =
        storeOp.getTarget()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp) return failure();
    assert(subspanOp.getDynamicDims().empty());

    auto tensorAccess = subspanOp.getType()
                            .template cast<IREE::Flow::DispatchTensorType>()
                            .getAccess();
    auto newSubspanType = IREE::Flow::DispatchTensorType::get(
        tensorAccess, reshapeSrc->getType());

    Value newSubspanOp;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(subspanOp);
      newSubspanOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
          subspanOp.getLoc(), newSubspanType, subspanOp.getSet(),
          subspanOp.getBinding(), subspanOp.getDescriptorType(),
          subspanOp.getByteOffset(), subspanOp.getDynamicDims(),
          subspanOp.getAlignmentAttr(), subspanOp.getDescriptorFlagsAttr());
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, *reshapeSrc, newSubspanOp, storeOp.getTargetDims());

    return success();
  }
};

// Removes operations with Allocate MemoryEffects but no uses.
struct RemoveDeadMemAllocs : RewritePattern {
  RemoveDeadMemAllocs(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memEffect || !memEffect.hasEffect<MemoryEffects::Allocate>()) {
      return failure();
    }
    SmallVector<Operation *> deadUsers;
    for (OpOperand &use : op->getUses()) {
      if (auto user = dyn_cast<memref::AssumeAlignmentOp>(use.getOwner())) {
        deadUsers.push_back(user);
        continue;
      }
      // For any other use, return failure;
      return failure();
    }
    for (auto user : deadUsers) {
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Runs canonicalization patterns on interface load/store ops.
struct CleanupBufferAllocViewPass
    : public CleanupBufferAllocViewBase<CleanupBufferAllocViewPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateReshapeToInterfaceTensorPatterns(patterns);
    patterns.insert<RemoveDeadMemAllocs>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateReshapeToInterfaceTensorPatterns(RewritePatternSet &patterns) {
  patterns.insert<FoldReshapeIntoInterfaceTensorLoad<tensor::CollapseShapeOp>,
                  FoldReshapeIntoInterfaceTensorLoad<tensor::ExpandShapeOp>>(
      patterns.getContext());
  patterns.insert<FoldReshapeIntoInterfaceTensorStore>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
createCleanupBufferAllocViewPass() {
  return std::make_unique<CleanupBufferAllocViewPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
