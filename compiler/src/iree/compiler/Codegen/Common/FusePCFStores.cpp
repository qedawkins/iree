// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FUSETENSORTOBUFFERCONVERTERSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Helper function to get all PCF::WriteSliceOps from a PCF::GenericOp or
/// PCF::LoopOp that write to a specific result.
template <typename PCFOpTy>
static FailureOr<SmallVector<IREE::PCF::WriteSliceOp>>
getProducerSlices(PCFOpTy pcfOp, OpResult result) {
  static_assert(std::is_same_v<PCFOpTy, IREE::PCF::GenericOp> ||
                    std::is_same_v<PCFOpTy, IREE::PCF::LoopOp>,
                "PCFOpTy must be PCF::GenericOp or PCF::LoopOp");
  BlockArgument tiedArg = pcfOp.getRegionRefArgs()[result.getResultNumber()];

  // The fusion is only valid if the sref type is return only sync scope.
  auto srefType = dyn_cast<IREE::PCF::ShapedRefType>(tiedArg.getType());
  if (!srefType || !srefType.isReturnOnlySync()) {
    return failure();
  }

  // Collect all WriteSliceOps that use this argument. Skip ReadSliceOps but
  // fail if there are any other users.
  SmallVector<IREE::PCF::WriteSliceOp> writeSlices;
  for (Operation *user : tiedArg.getUsers()) {
    if (isa<IREE::PCF::ReadSliceOp>(user)) {
      continue;
    }
    auto writeSlice = dyn_cast<IREE::PCF::WriteSliceOp>(user);
    if (!writeSlice) {
      return failure();
    }
    writeSlices.push_back(writeSlice);
  }

  return writeSlices;
}

struct FuseStoreToBuffer
    : OpRewritePattern<IREE::Codegen::StoreToBufferOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Codegen::StoreToBufferOp storeOp,
                                PatternRewriter &rewriter) const override {
    Value tensor = storeOp.getTensor();

    Operation *definingOp = tensor.getDefiningOp();
    auto producerLoop = dyn_cast_if_present<IREE::PCF::LoopOp>(definingOp);
    auto producerGeneric =
        dyn_cast_if_present<IREE::PCF::GenericOp>(definingOp);
    if (!producerLoop && !producerGeneric) {
      return failure();
    }

    // Make sure that the buffer operand of the store dominates the producer
    // loop.
    DominanceInfo domInfo(definingOp);
    Value buffer = storeOp.getBuffer();
    if (!domInfo.dominates(buffer, definingOp)) {
      return failure();
    }

    // Get the write_slice ops that produce the result written by the store.
    OpResult result = cast<OpResult>(tensor);
    FailureOr<SmallVector<IREE::PCF::WriteSliceOp>> maybeSlices = failure();
    if (producerLoop) {
      maybeSlices = getProducerSlices(producerLoop, result);
    } else {
      assert(producerGeneric && "unexpected undefined generic");
      maybeSlices = getProducerSlices(producerGeneric, result);
    }
    if (failed(maybeSlices)) {
      return failure();
    }

    SmallVector<IREE::PCF::WriteSliceOp> writeSlices = *maybeSlices;

    // For each WriteSliceOp, write its source to the store op's buffer.
    for (IREE::PCF::WriteSliceOp writeSlice : writeSlices) {
      rewriter.setInsertionPoint(writeSlice);

      // Create subview for the destination.
      Value destSlice = memref::SubViewOp::create(
          rewriter, writeSlice.getLoc(), buffer, writeSlice.getMixedOffsets(),
          writeSlice.getMixedSizes(), writeSlice.getMixedStrides());

      // Handle different source types. Create but don't replace the write_slice
      // ops. We rely on unused result cleanup patterns to drop them when
      // possible.
      Type sourceType = writeSlice.getSourceType();
      if (isa<RankedTensorType>(sourceType)) {
        IREE::Codegen::StoreToBufferOp::create(
            rewriter, storeOp.getLoc(), writeSlice.getSource(), destSlice);
      } else if (isa<MemRefType>(sourceType)) {
        memref::CopyOp::create(rewriter, storeOp.getLoc(),
                               writeSlice.getSource(), destSlice);
      } else if (auto vectorType = dyn_cast<VectorType>(sourceType)) {
        SmallVector<bool> inBounds(vectorType.getRank(), true);
        for (auto [inBound, vecSize, storeSize] :
             llvm::zip_equal(inBounds, vectorType.getShape(),
                             writeSlice.getStaticSizes())) {
          inBound = vecSize == storeSize;
        }
        SmallVector<Value> offsets(
            vectorType.getRank(),
            arith::ConstantIndexOp::create(rewriter, writeSlice.getLoc(), 0));
        vector::TransferWriteOp::create(rewriter, storeOp.getLoc(),
                                        writeSlice.getSource(), destSlice,
                                        offsets, inBounds);
      } else {
        llvm_unreachable("Invalid write_slice operand type");
      }
    }

    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct FuseDispatchTensorStore
    : OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Unimplemented: non-unit stride.
    if (!storeOp.hasUnitStride()) {
      return failure();
    }
    Value value = storeOp.getValue();
    Value target = storeOp.getTarget();

    Operation *definingOp = value.getDefiningOp();
    auto producerLoop = dyn_cast_if_present<IREE::PCF::LoopOp>(definingOp);
    auto producerGeneric =
        dyn_cast_if_present<IREE::PCF::GenericOp>(definingOp);
    if (!producerLoop && !producerGeneric) {
      return failure();
    }

    DominanceInfo domInfo(definingOp);
    if (!domInfo.dominates(target, definingOp)) {
      return failure();
    }

    // Get the write_slice ops that produce the result written by the store.
    OpResult result = cast<OpResult>(value);
    FailureOr<SmallVector<IREE::PCF::WriteSliceOp>> maybeSlices = failure();
    if (producerLoop) {
      maybeSlices = getProducerSlices(producerLoop, result);
    } else {
      assert(producerGeneric && "unexpected undefined generic");
      maybeSlices = getProducerSlices(producerGeneric, result);
    }
    if (failed(maybeSlices)) {
      return failure();
    }

    SmallVector<IREE::PCF::WriteSliceOp> writeSlices = *maybeSlices;

    // Check that all source operands are tensors as that's the only type
    // that can be written to the special dispatch type. Also non-unit stride
    // is currently unsupported.
    if (!llvm::all_of(writeSlices, [](IREE::PCF::WriteSliceOp writeSlice) {
          return isa<RankedTensorType>(writeSlice.getSourceType()) &&
                 writeSlice.hasUnitStride();
        })) {
      return failure();
    }

    // For each WriteSliceOp, create a new DispatchTensorStoreOp of just the
    // written slice.
    AffineExpr d0, d1;
    bindSymbols(rewriter.getContext(), d0, d1);
    AffineExpr add = d0 + d1;
    for (IREE::PCF::WriteSliceOp writeSlice : writeSlices) {
      rewriter.setInsertionPoint(writeSlice);

      // Add the offsets of the WriteSliceOp to the offsets of the store.
      SmallVector<OpFoldResult> newOffsets;
      SmallVector<OpFoldResult> writeOffsets = writeSlice.getMixedOffsets();
      SmallVector<OpFoldResult> storeOffsets = storeOp.getMixedOffsets();

      for (auto [writeOffset, storeOffset] :
           llvm::zip_equal(writeOffsets, storeOffsets)) {
        newOffsets.push_back(affine::makeComposedFoldedAffineApply(
            rewriter, writeSlice.getLoc(), add, {writeOffset, storeOffset}));
      }

      // Get the source to store. If the write_slice source is a rank-reducing
      // insert_slice into tensor.empty, use the insert_slice source directly.
      Value sourceToStore = writeSlice.getSource();
      SmallVector<OpFoldResult> sizesToStore = writeSlice.getMixedSizes();
      if (auto insertOp =
              sourceToStore.getDefiningOp<tensor::InsertSliceOp>()) {
        if (insertOp.getDest().getDefiningOp<tensor::EmptyOp>()) {
          RankedTensorType insertSourceType = insertOp.getSourceType();
          RankedTensorType insertResultType = insertOp.getResultType();
          if (isRankReducedType(insertResultType, insertSourceType) ==
              SliceVerificationResult::Success) {
            sourceToStore = insertOp.getSource();
            sizesToStore = insertOp.getMixedSizes();
          }
        }
      }

      // Use the sizes of the source tensor.
      IREE::TensorExt::DispatchTensorStoreOp::create(
          rewriter, storeOp.getLoc(), sourceToStore, target,
          storeOp.getTargetDims(), newOffsets, sizesToStore,
          storeOp.getMixedStrides());
    }

    rewriter.eraseOp(storeOp);
    return success();
  }
};

/// Clone a map_scatter operation with a new input (tile) and output (buffer),
/// adjusting the transformation body to add the given offsets to the block
/// arguments. This transforms:
///   map_scatter %full_tensor into %buffer { ^bb(%i, %j): ... yield %out_i, %out_j, %mask }
/// Into:
///   map_scatter %tile into %buffer { ^bb(%i, %j): %adj_i = %i + off_i, %adj_j = %j + off_j, ... }
static IREE::LinalgExt::MapScatterOp
cloneMapScatterWithOffsets(IREE::LinalgExt::MapScatterOp origOp,
                           Value newInput, Value newOutput,
                           ArrayRef<OpFoldResult> offsets,
                           PatternRewriter &rewriter) {
  Location loc = origOp.getLoc();

  // Create the new map_scatter with no results (buffer semantics).
  auto newMapScatter = IREE::LinalgExt::MapScatterOp::create(
      rewriter, loc, /*resultTypes=*/TypeRange{}, newInput, newOutput);

  // Clone the transformation region from the original.
  IRMapping mapping;
  origOp.getTransformationRegion().cloneInto(&newMapScatter.getTransformationRegion(),
                                              mapping);

  // Insert offset adjustments at the start of the transformation body.
  // The new block args represent indices in the tile, we need to add offsets
  // to get indices in the full tensor (which the original body expects).
  int64_t numIndices = origOp.getInputRank();
  newMapScatter.insertTransformationAtStart(
      rewriter,
      [&](ArrayRef<BlockArgument> tileIndices) -> SmallVector<Value> {
        // insertTransformationAtStart sets the insertion point to the start of
        // the transformation body before calling this lambda.
        SmallVector<Value> adjustedIndices;
        for (auto [tileIdx, offset] :
             llvm::zip_equal(tileIndices, offsets)) {
          Value offsetVal = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                            offset);
          Value adjusted =
              arith::AddIOp::create(rewriter, loc, tileIdx, offsetVal);
          adjustedIndices.push_back(adjusted);
        }
        return adjustedIndices;
      },
      numIndices);

  return newMapScatter;
}

struct FuseMapScatterIntoPCF
    : OpRewritePattern<IREE::LinalgExt::MapScatterOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::LinalgExt::MapScatterOp mapScatterOp,
                                PatternRewriter &rewriter) const override {
    // Check map_scatter has mixed semantics: tensor input, memref output.
    if (!isa<RankedTensorType>(mapScatterOp.getInputType()) ||
        !isa<MemRefType>(mapScatterOp.getOutputType())) {
      return failure();
    }

    // Get producer PCF op.
    Value tensorInput = mapScatterOp.getInput();
    Operation *definingOp = tensorInput.getDefiningOp();
    auto producerLoop = dyn_cast_if_present<IREE::PCF::LoopOp>(definingOp);
    auto producerGeneric =
        dyn_cast_if_present<IREE::PCF::GenericOp>(definingOp);
    if (!producerLoop && !producerGeneric) {
      return failure();
    }

    // Check that the buffer operand dominates the producer.
    DominanceInfo domInfo(definingOp);
    Value buffer = mapScatterOp.getOutput();
    if (!domInfo.dominates(buffer, definingOp)) {
      return failure();
    }

    // Get the write_slice ops that produce the result consumed by map_scatter.
    OpResult result = cast<OpResult>(tensorInput);
    FailureOr<SmallVector<IREE::PCF::WriteSliceOp>> maybeSlices = failure();
    if (producerLoop) {
      maybeSlices = getProducerSlices(producerLoop, result);
    } else {
      assert(producerGeneric && "unexpected undefined generic");
      maybeSlices = getProducerSlices(producerGeneric, result);
    }
    if (failed(maybeSlices)) {
      return failure();
    }

    SmallVector<IREE::PCF::WriteSliceOp> writeSlices = *maybeSlices;
    if (writeSlices.empty()) {
      return failure();
    }

    // For each write_slice, clone map_scatter with offset adjustment.
    for (IREE::PCF::WriteSliceOp writeSlice : writeSlices) {
      rewriter.setInsertionPoint(writeSlice);

      // Get the tile being written and its offsets.
      Value tile = writeSlice.getSource();
      SmallVector<OpFoldResult> offsets = writeSlice.getMixedOffsets();

      // Clone the map_scatter with the tile as input and buffer as output,
      // adjusting the transformation body to account for tile offsets.
      cloneMapScatterWithOffsets(mapScatterOp, tile, buffer, offsets, rewriter);
    }

    // Erase the original map_scatter.
    rewriter.eraseOp(mapScatterOp);
    return success();
  }
};

struct FuseTensorToBufferConvertersPass final
    : impl::FuseTensorToBufferConvertersPassBase<
          FuseTensorToBufferConvertersPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<FuseStoreToBuffer, FuseDispatchTensorStore, FuseMapScatterIntoPCF>(context);
    IREE::PCF::populatePCFDropUnusedResultPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
