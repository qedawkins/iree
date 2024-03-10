// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-fuse-tensor-slices"

namespace mlir::iree_compiler::IREE::Flow {

static LogicalResult
fuseInsertSliceWithProducerDispatch(RewriterBase &rewriter,
                                    tensor::InsertSliceOp sliceOp) {
  auto dispatchOp = sliceOp.getSource().getDefiningOp<DispatchRegionOp>();
  if (!dispatchOp || !sliceOp.getSource().hasOneUse() ||
      dispatchOp->getNumResults() != 1) {
    return failure();
  }

  // Restrict to cases where the slice and dispatch are in the same block. If
  // there is control flow separating the two
  if (sliceOp->getBlock() != dispatchOp->getBlock()) {
    return failure();
  }

  // Set the insertion point s.t. SSA dominance of the slice w.r.t. its
  // destination is maintained.
  if (auto destOp = sliceOp.getDest().getDefiningOp()) {
    if (!destOp->isBeforeInBlock(dispatchOp)) {
      rewriter.setInsertionPointAfter(destOp);
    } else {
      rewriter.setInsertionPointAfter(dispatchOp);
    }
  }

  Block &body = dispatchOp.getBody().front();
  auto returnOp = cast<Flow::ReturnOp>(body.getTerminator());

  // 1. Get the new return types and dynamic dims for the fused dispatch op.
  SmallVector<Type> newReturnTypes;
  SmallVector<Value> newDynamicDims;
  SmallVector<Value> newYieldVals;
  OpResult sliceSourceResult = cast<OpResult>(sliceOp.getSource());
  ValueRange dynamicDimsList = dispatchOp.getResultDims();
  int64_t dynamicDimIndex = 0;
  Location loc = dispatchOp.getLoc();
  Value sliceSourceReplacement = nullptr;
  for (OpOperand &yieldedValue : returnOp->getOpOperands()) {
    if (yieldedValue.getOperandNumber() !=
        sliceSourceResult.getResultNumber()) {
      // 1a. Keep the same yield value if the producer is not a
      // `tensor.expand_shape` op.
      Type yieldedType = yieldedValue.get().getType();
      newReturnTypes.push_back(yieldedType);
      newYieldVals.push_back(yieldedValue.get());
      if (auto tensorType = dyn_cast<RankedTensorType>(yieldedType)) {
        for (auto size : tensorType.getShape()) {
          if (ShapedType::isDynamic(size)) {
            newDynamicDims.push_back(dynamicDimsList[dynamicDimIndex]);
            dynamicDimIndex++;
          }
        }
      }
      continue;
    }

    // 1b. The return type is same as the type of the result of the slice op.
    RankedTensorType destType = sliceOp.getResultType();
    newReturnTypes.push_back(destType);
    newYieldVals.push_back(sliceOp.getResult());
    sliceSourceReplacement = yieldedValue.get();

    // 1c. Dynamic dims of the result shape are obtained from the destination of
    // the `insert_slice` op. The dynamic dims that used to be associated with
    // the source must be skipped.
    for (auto size : sliceOp.getSourceType().getShape()) {
      if (ShapedType::isDynamic(size)) {
        dynamicDimIndex++;
      }
    }

    for (auto [index, size] : llvm::enumerate(destType.getShape())) {
      if (ShapedType::isDynamic(size)) {
        newDynamicDims.push_back(rewriter.create<tensor::DimOp>(
            sliceOp.getLoc(), sliceOp.getDest(), index));
      }
    }
  }

  // 2. Create the new dispatch op.
  auto newDispatchOp = rewriter.create<DispatchRegionOp>(
      loc, newReturnTypes, newDynamicDims, dispatchOp.getWorkload());

  // 2a. Move the body over, but replace the `flow.return` to use the new yield
  // values.
  Region &newBody = newDispatchOp.getBody();
  rewriter.inlineRegionBefore(dispatchOp.getBody(), newBody, newBody.begin());
  {
    Operation *terminator = newBody.front().getTerminator();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(terminator);
    assert(sliceSourceReplacement && "empty source replacement");
    Value newSlice = rewriter.create<tensor::InsertSliceOp>(
        sliceOp.getLoc(), sliceSourceReplacement, sliceOp.getDest(),
        sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
        sliceOp.getMixedStrides());
    newYieldVals[sliceSourceResult.getResultNumber()] = newSlice;
    rewriter.replaceOpWithNewOp<Flow::ReturnOp>(terminator, newYieldVals);
  }

  // 2b. Move the workgroup count region over.
  Region &workgroupCountRegion = dispatchOp.getWorkgroupCount();
  if (!workgroupCountRegion.empty()) {
    Region &newWorkgroupCountRegion = newDispatchOp.getWorkgroupCount();
    rewriter.inlineRegionBefore(workgroupCountRegion, newWorkgroupCountRegion,
                                newWorkgroupCountRegion.begin());
  }

  // 3. Map the result of the dispatch back to the original users of the
  //    slice operation.
  for (auto [index, returnValue] :
       llvm::enumerate(newDispatchOp.getResults())) {
    Value origResult = newDispatchOp->getResult(index);
    if (index != sliceSourceResult.getResultNumber()) {
      rewriter.replaceAllUsesWith(origResult, returnValue);
      continue;
    }
    rewriter.replaceAllUsesWith(sliceOp, returnValue);
  }

  rewriter.eraseOp(sliceOp);
  rewriter.eraseOp(dispatchOp);
  return success();
}

namespace {
struct ConvertInsertSliceOpToFlow
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    return convertInsertSliceOpToFlowUpdateOp(rewriter, insertSliceOp);
  }
};

/// Pass declaration.
struct FuseConsumerTensorSlicesPass
    : public FuseConsumerTensorSlicesBase<FuseConsumerTensorSlicesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, IREE::Flow::FlowDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void FuseConsumerTensorSlicesPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();

  // Step 1: Rewrite InsertSliceOps to FlowUpdateOps.
  {
    RewritePatternSet patterns(context);
    patterns.add<ConvertInsertSliceOpToFlow>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp->emitOpError("failed when converting tensor.*slice ops to flow");
      return signalPassFailure();
    }
  }

  // Step 2. Attempt to fuse all remaining insert slice ops with producer
  // dispatches of the source slice. First collect a set of potential
  // slices.
  SmallVector<tensor::InsertSliceOp> candidateSlices;
  funcOp->walk([&](tensor::InsertSliceOp insertOp) {
    // TODO: Fusing with multi-use dispatches requires tracking dominance info
    // to determine whether the dispatch is used to construct the destination of
    // the insert_slice, and also how to reorder operations to maintain SSA
    // dominance.
    //
    // This might be easier to do after forming dispatch workgroups, or
    // potentially in Stream.
    if (!insertOp.getSource().hasOneUse()) {
      return;
    }
    // TODO: Fusing with multi-result dispatches similarly requires tracking a
    // backward slice of the program to determine whether one of the other
    // results is a producer of the destination.
    auto dispatchOp = insertOp.getSource().getDefiningOp<DispatchRegionOp>();
    if (!dispatchOp || dispatchOp.getNumResults() != 1) {
      return;
    }
    candidateSlices.push_back(insertOp);
  });

  mlir::TensorDimTrackingRewriter rewriter(funcOp);
  for (tensor::InsertSliceOp sliceOp : candidateSlices) {
    IRRewriter::InsertionGuard g(rewriter);
    if (failed(fuseInsertSliceWithProducerDispatch(rewriter, sliceOp))) {
      funcOp->emitOpError("failed to fuse candidate slice");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFuseConsumerTensorSlicesPass() {
  return std::make_unique<FuseConsumerTensorSlicesPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
