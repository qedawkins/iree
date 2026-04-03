// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTilingInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-pcf-distributed-fuse-consumers"

namespace mlir::iree_compiler::IREE::PCF {

namespace {

/// Core implementation of distributed consumer fusion. Fuses |target| into
/// |producerOp| using PCFTilingOpInterface::getDistributedImplementation.
/// The |params| struct provides the matched operands, results, and slices
/// from matchTilableConsumer.
template <typename OpTy>
static void fuseDistributedConsumerImpl(RewriterBase &rewriter, OpTy producerOp,
                                        PCFTilingOpInterface target,
                                        ConsumerFusionParams &params) {
  assert(!params.results.empty() && "unexpected empty number of results");

  Location loc = target.getLoc();
  Operation *targetOp = target.getOperation();
  // Step 1: Collect tile info from the most dominant write slice.
  // The matcher guarantees slices[0] is the most dominant insertion point.
  WriteSliceOp dominantSlice = params.slices.front();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(dominantSlice);

  // Step 2: Clone the consumer and set up unrealized conversion casts for
  // fused operands (same pattern as fuseIntoWriteSlices).
  auto clonedOp = cast<TilingInterface>(rewriter.clone(*targetOp));
  auto clonedPCF = cast<PCFTilingOpInterface>(clonedOp.getOperation());
  SmallVector<UnrealizedConversionCastOp> unrealizedConversions;
  for (auto [operand, slice] :
       llvm::zip_equal(params.operands, params.slices)) {
    OpOperand &currOperand = clonedOp->getOpOperand(operand);
    Type undistributedType = currOperand.get().getType();
    UnrealizedConversionCastOp conversion = UnrealizedConversionCastOp::create(
        rewriter, loc, undistributedType, slice.getSource());
    currOperand.assign(conversion.getResult(0));
    unrealizedConversions.push_back(conversion);
  }

  // Step 3: Compute iteration domain tile from operand tiles.
  SmallVector<SmallVector<OpFoldResult>> allOffsets = llvm::map_to_vector(
      params.slices, [](WriteSliceOp op) { return op.getMixedOffsets(); });
  SmallVector<SmallVector<OpFoldResult>> allSizes = llvm::map_to_vector(
      params.slices, [](WriteSliceOp op) { return op.getMixedSizes(); });
  SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;
  [[maybe_unused]] LogicalResult res =
      clonedOp.getIterationDomainTileFromOperandTiles(
          rewriter, params.operands, allOffsets, allSizes, iterDomainOffsets,
          iterDomainSizes);
  assert(succeeded(res) && "unexpected iteration domain fetch failed");

  // Step 4: Classify consumer operands and determine which need readonly
  // sref args.
  SmallVector<unsigned> tileableIndices = clonedPCF.getTileableOperandIndices();
  DenseSet<unsigned> tileableSet(tileableIndices.begin(),
                                 tileableIndices.end());
  DenseSet<unsigned> fusedOperandSet(params.operands.begin(),
                                     params.operands.end());

  // Collect readonly inits for non-fused tileable INPUT operands.
  // DPS inits are handled separately as readwrite args (they produce results).
  SmallVector<Value> newReadonlyInits;
  // Map from consumer operand index to index in newReadonlyInits (-1 if not).
  SmallVector<int64_t> operandToNewReadonlyIdx(targetOp->getNumOperands(), -1);

  // Build a set of DPS init operand indices for quick lookup.
  DenseSet<unsigned> dpsInitIndices;
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(*targetOp)) {
    for (OpOperand &init : dpsOp.getDpsInitsMutable()) {
      dpsInitIndices.insert(init.getOperandNumber());
    }
  }

  for (int64_t i = 0, e = targetOp->getNumOperands(); i < e; ++i) {
    unsigned idx = static_cast<unsigned>(i);
    if (fusedOperandSet.contains(idx)) {
      // Fused along — will use write_slice source.
      continue;
    }
    if (dpsInitIndices.contains(idx)) {
      // DPS init — will be handled as readwrite arg (not readonly).
      continue;
    }
    if (!tileableSet.contains(idx)) {
      // Not tileable — will pass through original value.
      continue;
    }
    // Only shaped (tensor) operands become sref args. Scalar operands
    // (e.g. the fill value in linalg.fill) are passed through directly.
    if (!isa<ShapedType>(targetOp->getOperand(i).getType())) {
      continue;
    }
    operandToNewReadonlyIdx[i] = static_cast<int64_t>(newReadonlyInits.size());
    newReadonlyInits.push_back(targetOp->getOperand(i));
  }

  // Step 5: Compute result types, tied args, and dynamic sizes for the
  // consumer's results. Same logic as fuseTilableConsumerImpl.
  SmallVector<bool> newIsTied;
  SmallVector<Value> newTiedArgs;
  SmallVector<Value> newDynamicSizes;
  SmallVector<Type> newResultTypes(targetOp->getResultTypes());

  auto getInitOrCreateEmpty = [&](int64_t resultNumber) -> Value {
    if (OpOperand *tiedInit = producerOp.getTiedInit(resultNumber)) {
      return tiedInit->get();
    }
    return tensor::EmptyOp::create(rewriter, loc,
                                   producerOp.getResultType(resultNumber),
                                   producerOp.getResultDims(resultNumber));
  };

  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(*targetOp)) {
    for (Value init : dpsOp.getDpsInits()) {
      if (init.getDefiningOp() == producerOp) {
        auto result = cast<OpResult>(init);
        if (OpOperand *operand =
                producerOp.getTiedInit(result.getResultNumber())) {
          newIsTied.push_back(true);
          newTiedArgs.push_back(operand->get());
        } else {
          newIsTied.push_back(false);
          ValueRange resultDims =
              producerOp.getResultDims(result.getResultNumber());
          llvm::append_range(newDynamicSizes, resultDims);
        }
      } else {
        newIsTied.push_back(true);
        newTiedArgs.push_back(init);
      }
    }
  } else {
    // ReifyRankedShapedTypeOpInterface path.
    SmallVector<Value> originalOperands;
    rewriter.setInsertionPoint(producerOp);
    for (unsigned operandIndex : params.operands) {
      Value operand = targetOp->getOperand(operandIndex);
      originalOperands.push_back(operand);
      targetOp->getOpOperand(operandIndex)
          .assign(
              getInitOrCreateEmpty(cast<OpResult>(operand).getResultNumber()));
    }

    SmallVector<SmallVector<OpFoldResult>> outputShapes;
    Operation *nextNode = targetOp->getNextNode();
    Block *currBlock = targetOp->getBlock();
    rewriter.moveOpBefore(targetOp, producerOp);

    auto reifyOp = cast<ReifyRankedShapedTypeOpInterface>(*targetOp);
    LogicalResult reifyResult =
        reifyOp.reifyResultShapes(rewriter, outputShapes);
    assert(succeeded(reifyResult) && "unexpected reify result shapes failed");
    (void)reifyResult;
    if (nextNode) {
      rewriter.moveOpBefore(targetOp, nextNode);
    } else {
      rewriter.moveOpAfter(targetOp, &currBlock->back());
    }

    for (ArrayRef<OpFoldResult> outputShape : outputShapes) {
      llvm::append_range(
          newDynamicSizes,
          llvm::map_to_vector(outputShape, [&](OpFoldResult ofr) {
            return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
          }));
    }
    newIsTied.append(outputShapes.size(), false);
  }

  // Step 6: Add readonly and readwrite args to the producer op.
  SmallVector<BlockArgument> newReadonlyBlockArgs;
  SmallVector<BlockArgument> newReadwriteBlockArgs;
  OpTy newProducerOp = addReadonlyAndReadwriteArgs(
      rewriter, producerOp, newReadonlyInits,
      /*newReadwriteInits=*/newTiedArgs.empty() ? ValueRange()
                                                : ValueRange(newTiedArgs),
      newIsTied, newDynamicSizes, newResultTypes, newReadonlyBlockArgs,
      newReadwriteBlockArgs);

  // Restore insertion point into the body, at the dominant slice.
  rewriter.setInsertionPoint(dominantSlice);

  // Step 7: Build DistributedOperandInfo per consumer operand.
  SmallVector<DistributedOperandInfo> operandInfo;
  operandInfo.reserve(targetOp->getNumOperands());
  for (int64_t i = 0, e = targetOp->getNumOperands(); i < e; ++i) {
    unsigned idx = static_cast<unsigned>(i);

    if (fusedOperandSet.contains(idx)) {
      // Fused-along operand: use the write_slice source (the tile the
      // producer already computed).
      // Find the corresponding slice.
      // For single-result fusion, all operands share the dominant slice.
      // For multi-result, each operand has its own slice.
      int64_t sliceIdx = 0;
      for (int64_t j = 0, je = params.operands.size(); j < je; ++j) {
        if (params.operands[j] == idx) {
          sliceIdx = j;
          break;
        }
      }
      operandInfo.push_back(
          {params.slices[sliceIdx].getSource(), /*isTile=*/true});
      continue;
    }

    if (operandToNewReadonlyIdx[i] >= 0) {
      // Non-fused tileable input operand: pass the readonly sref directly.
      // The getDistributedImplementation method will read the correct tile
      // based on the indexing map.
      BlockArgument sref = newReadonlyBlockArgs[operandToNewReadonlyIdx[i]];
      operandInfo.push_back({sref, /*isTile=*/false});
      continue;
    }

    // DPS init operand: pass the corresponding readwrite sref.
    if (dpsInitIndices.contains(idx)) {
      // Find which result this DPS init corresponds to.
      auto dpsOp = cast<DestinationStyleOpInterface>(*targetOp);
      int64_t readwriteIdx = 0;
      for (OpOperand &init : dpsOp.getDpsInitsMutable()) {
        if (init.getOperandNumber() == idx) {
          break;
        }
        ++readwriteIdx;
      }
      BlockArgument sref = newReadwriteBlockArgs[readwriteIdx];
      operandInfo.push_back({sref, /*isTile=*/false});
      continue;
    }

    // Non-tileable operand: pass through original value.
    operandInfo.push_back({targetOp->getOperand(i), /*isTile=*/false});
  }

  // Step 8: Build DistributedResultInfo per consumer result.
  SmallVector<DistributedResultInfo> resultInfo;
  int64_t numConsumerResults = targetOp->getNumResults();
  resultInfo.reserve(numConsumerResults);
  for (int64_t i = 0; i < numConsumerResults; ++i) {
    resultInfo.push_back({newReadwriteBlockArgs[i]});
  }

  // Step 9: Check feasibility, then call getDistributedImplementation.
  assert(succeeded(clonedPCF.canDistribute(iterDomainOffsets, iterDomainSizes,
                                           operandInfo, resultInfo)) &&
         "canDistribute check failed unexpectedly during consumer fusion");
  FailureOr<TilingResult> tilingResult = clonedPCF.getDistributedImplementation(
      rewriter, iterDomainOffsets, iterDomainSizes, operandInfo, resultInfo);
  assert(succeeded(tilingResult) &&
         "unexpected distributed implementation failure");

  // Step 10: Handle returned tiled values. The getDistributedImplementation
  // should have written results to the dest srefs when destSref was set.
  // Only write remaining non-null tiled values (shouldn't happen in the
  // common case where destSref was set for all results).
  for (int64_t i = 0; i < numConsumerResults; ++i) {
    Value tiledValue = tilingResult->tiledValues[i];
    if (!tiledValue) {
      // The implementation already wrote to the sref directly.
      continue;
    }
    // Compute result tile position and write.
    SmallVector<OpFoldResult> resOffsets, resSizes;
    [[maybe_unused]] LogicalResult posRes = clonedOp.getResultTilePosition(
        rewriter, i, iterDomainOffsets, iterDomainSizes, resOffsets, resSizes);
    assert(succeeded(posRes) &&
           "unexpected failure to get result tile position");
    SmallVector<OpFoldResult> strides(resOffsets.size(),
                                      rewriter.getIndexAttr(1));
    WriteSliceOp::create(rewriter, loc, tiledValue, resultInfo[i].destSref,
                         resOffsets, resSizes, strides);
  }

  // Step 11: Clean up unrealized conversion casts.
  for (UnrealizedConversionCastOp unrealizedCast : unrealizedConversions) {
    SmallVector<Operation *> users(unrealizedCast->getUsers());
    for (Operation *user : users) {
      if (auto extract = dyn_cast<tensor::ExtractSliceOp>(user)) {
        if (extract.getResultType() ==
            unrealizedCast->getOperandTypes().front()) {
          rewriter.replaceOp(extract, unrealizedCast.getOperand(0));
        }
      }
    }
    if (unrealizedCast->use_empty()) {
      rewriter.eraseOp(unrealizedCast);
    }
  }

  // Erase the cloned op (it was only used for interface queries).
  rewriter.eraseOp(clonedOp);

  // Step 12: Replace the original consumer with the new producer results.
  ValueRange replacements =
      newProducerOp.getResults().take_back(newResultTypes.size());
  rewriter.replaceOp(targetOp, replacements);
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API: fuseDistributedConsumer
//===----------------------------------------------------------------------===//

void fuseDistributedConsumer(RewriterBase &rewriter, GenericOp genericOp,
                             PCFTilingOpInterface target,
                             ConsumerFusionParams &params) {
  fuseDistributedConsumerImpl(rewriter, genericOp, target, params);
}

void fuseDistributedConsumer(RewriterBase &rewriter, LoopOp loopOp,
                             PCFTilingOpInterface target,
                             ConsumerFusionParams &params) {
  fuseDistributedConsumerImpl(rewriter, loopOp, target, params);
}

} // namespace mlir::iree_compiler::IREE::PCF
