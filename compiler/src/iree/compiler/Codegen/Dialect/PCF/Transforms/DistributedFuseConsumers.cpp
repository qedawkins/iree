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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-pcf-distributed-fuse-consumers"

namespace mlir::iree_compiler::IREE::PCF {

namespace {

/// Creates a new scoped op (LoopOp or GenericOp) with additional readonly and
/// readwrite sref args. Moves the body from the old op to the new one, inserts
/// new block args at the correct positions, and replaces the old op's results
/// with the new op's original results.
///
/// Returns the new op and populates |newReadonlyRefs| and |newReadwriteRefs|
/// with the newly inserted block arguments.
template <typename OpTy>
static OpTy addReadonlyAndReadwriteArgs(
    RewriterBase &rewriter, OpTy op, ValueRange newReadonlyInits,
    ValueRange newReadwriteInits, ArrayRef<bool> newIsTied,
    ArrayRef<Value> newTiedArgs, ArrayRef<Value> newDynamicSizes,
    TypeRange newResultTypes, SmallVectorImpl<BlockArgument> &newReadonlyRefs,
    SmallVectorImpl<BlockArgument> &newReadwriteRefs);

template <>
LoopOp addReadonlyAndReadwriteArgs<LoopOp>(
    RewriterBase &rewriter, LoopOp loopOp, ValueRange newReadonlyInits,
    ValueRange newReadwriteInits, ArrayRef<bool> newIsTied,
    ArrayRef<Value> newTiedArgs, ArrayRef<Value> newDynamicSizes,
    TypeRange newResultTypes, SmallVectorImpl<BlockArgument> &newReadonlyRefs,
    SmallVectorImpl<BlockArgument> &newReadwriteRefs) {
  Location loc = loopOp.getLoc();
  MLIRContext *context = rewriter.getContext();

  // Build combined readonly inits: old + new.
  SmallVector<Value> combinedReadonlyInits(loopOp.getReadonlyInits());
  llvm::append_range(combinedReadonlyInits, newReadonlyInits);

  // Build combined readwrite inits: old + new.
  SmallVector<Value> combinedInits(loopOp.getInits());
  llvm::append_range(combinedInits, newReadwriteInits);

  // Build combined result types: old + new.
  SmallVector<Type> combinedResultTypes(loopOp->getResultTypes());
  llvm::append_range(combinedResultTypes, newResultTypes);

  // Build combined is_tied: old + new.
  SmallVector<bool> combinedIsTied(loopOp.getIsTied());
  llvm::append_range(combinedIsTied, newIsTied);

  // Build combined dynamic sizes: old + new.
  SmallVector<Value> combinedDynamicSizes(loopOp.getDynamicSizes());
  llvm::append_range(combinedDynamicSizes, newDynamicSizes);

  // Build combined tied args (inits for readwrite): old + new.
  // Note: For LoopOp, inits = readonlyInits + readwrite inits.
  // The builder with all params doesn't take readonlyInits separately.
  // We need to use the (resultTypes, scope, count, inits, dynamicSizes,
  // isTied) builder but that doesn't support readonlyInits.
  // Instead, we build manually using the readonly+readwrite builder,
  // but that one infers result types from inits. Since we may have
  // untied results, we need the full builder.

  int64_t numOriginalResults = loopOp->getNumResults();
  int64_t numOriginalReadonlyRefs = loopOp.getNumReadonlyRefs();

  // Use the full builder (resultTypes, scope, count, inits, dynamicSizes,
  // isTied, syncOnReturn) but it doesn't support readonly inits.
  // We need to build it manually.
  LoopOp newLoopOp = LoopOp::create(
      rewriter, loc, combinedResultTypes, loopOp.getScope(), loopOp.getCount(),
      combinedInits, combinedDynamicSizes, combinedIsTied,
      loopOp.getSyncOnReturn());

  // The full builder doesn't add readonly args. We need to add them manually.
  // First move the body over, then fix up block args.
  newLoopOp.getRegion().takeBody(loopOp.getRegion());

  // The old body has block args laid out as:
  //   [old_readonly_refs... | old_readwrite_refs... | id_args...]
  // The new body needs:
  //   [old_readonly_refs... new_readonly_refs... | old_readwrite_refs...
  //    new_readwrite_refs... | id_args...]
  //
  // The full builder already added readwrite sref args for combinedInits
  // and id args at the end. But we took the old body, so we need to insert
  // the new args manually.

  Block *body = newLoopOp.getBody();

  // Insert new readonly sref args after old readonly refs.
  int64_t readonlyInsertIdx = numOriginalReadonlyRefs;
  for (Value init : newReadonlyInits) {
    ShapedType shapedType = cast<ShapedType>(init.getType());
    ShapedRefType srefType = ShapedRefType::get(
        context, shapedType.getShape(), shapedType.getElementType(),
        loopOp.getScope());
    BlockArgument arg =
        body->insertArgument(readonlyInsertIdx, srefType, loc);
    newReadonlyRefs.push_back(arg);
    ++readonlyInsertIdx;
  }

  // Insert new readwrite sref args after old readwrite refs (and after the
  // newly inserted readonly refs).
  // Old layout after readonly insertion:
  //   [old_ro... new_ro... | old_rw... | id_args...]
  // New readwrite args go before id_args.
  int64_t numIdArgs = loopOp.getCount().size();
  int64_t readwriteInsertIdx =
      body->getNumArguments() - numIdArgs;
  Attribute syncScope = SyncOnReturnAttr::get(context);
  for (Type resultType : newResultTypes) {
    ShapedType shapedType = cast<ShapedType>(resultType);
    ShapedRefType srefType =
        ShapedRefType::get(context, shapedType.getShape(),
                           shapedType.getElementType(), loopOp.getScope(),
                           syncScope);
    BlockArgument arg =
        body->insertArgument(readwriteInsertIdx, srefType, loc);
    newReadwriteRefs.push_back(arg);
    ++readwriteInsertIdx;
  }

  // Update the num_readonly_refs property to include the new readonly args.
  newLoopOp.setNumReadonlyRefs(numOriginalReadonlyRefs +
                               newReadonlyInits.size());

  // Update the operand segment sizes to include readonly inits.
  // The full builder set readonlyInits segment to 0; fix it.
  auto &props = newLoopOp.getProperties();
  props.setOperandSegmentSizes(
      {static_cast<int32_t>(loopOp.getCount().size()),
       static_cast<int32_t>(combinedReadonlyInits.size()),
       static_cast<int32_t>(combinedInits.size()),
       static_cast<int32_t>(combinedDynamicSizes.size())});

  // Add the readonly init operands. The full builder didn't include them.
  // We need to rebuild the operand list. Since we can't easily insert operands,
  // we'll set them via the mutable accessor.
  newLoopOp.getReadonlyInitsMutable().assign(combinedReadonlyInits);

  // Replace old loop's results with corresponding new loop results.
  rewriter.replaceOp(loopOp,
                     newLoopOp->getResults().take_front(numOriginalResults));
  return newLoopOp;
}

template <>
GenericOp addReadonlyAndReadwriteArgs<GenericOp>(
    RewriterBase &rewriter, GenericOp genericOp, ValueRange newReadonlyInits,
    ValueRange newReadwriteInits, ArrayRef<bool> newIsTied,
    ArrayRef<Value> newTiedArgs, ArrayRef<Value> newDynamicSizes,
    TypeRange newResultTypes, SmallVectorImpl<BlockArgument> &newReadonlyRefs,
    SmallVectorImpl<BlockArgument> &newReadwriteRefs) {
  Location loc = genericOp.getLoc();
  MLIRContext *context = rewriter.getContext();

  // Build combined readonly inits: old + new.
  SmallVector<Value> combinedReadonlyInits(genericOp.getReadonlyInits());
  llvm::append_range(combinedReadonlyInits, newReadonlyInits);

  // Build combined readwrite inits: old + new.
  SmallVector<Value> combinedInits(genericOp.getInits());
  llvm::append_range(combinedInits, newReadwriteInits);

  // Build combined result types, is_tied, dynamic sizes.
  SmallVector<Type> combinedResultTypes(genericOp->getResultTypes());
  llvm::append_range(combinedResultTypes, newResultTypes);
  SmallVector<bool> combinedIsTied(genericOp.getIsTied());
  llvm::append_range(combinedIsTied, newIsTied);
  SmallVector<Value> combinedDynamicSizes(genericOp.getDynamicSizes());
  llvm::append_range(combinedDynamicSizes, newDynamicSizes);

  int64_t numOriginalResults = genericOp->getNumResults();
  int64_t numOriginalReadonlyRefs = genericOp.getNumReadonlyRefs();

  // Create new GenericOp using the full builder.
  GenericOp newGenericOp = GenericOp::create(
      rewriter, loc, combinedResultTypes, genericOp.getScope(), combinedInits,
      combinedDynamicSizes, combinedIsTied, genericOp.getNumIterators(),
      genericOp.getSyncOnReturn());
  newGenericOp.getRegion().takeBody(genericOp.getRegion());
  newGenericOp.getInitializer().takeBody(genericOp.getInitializer());
  newGenericOp.setNumLeadingArgs(genericOp.getNumLeadingArgs());

  Block *body = &newGenericOp.getRegion().front();

  // Insert new readonly sref args after leading args + old readonly refs.
  int64_t readonlyInsertIdx =
      genericOp.getNumLeadingArgs() + numOriginalReadonlyRefs;
  for (Value init : newReadonlyInits) {
    ShapedType shapedType = cast<ShapedType>(init.getType());
    ShapedRefType srefType = ShapedRefType::get(
        context, shapedType.getShape(), shapedType.getElementType(),
        genericOp.getScope());
    BlockArgument arg =
        body->insertArgument(readonlyInsertIdx, srefType, loc);
    newReadonlyRefs.push_back(arg);
    ++readonlyInsertIdx;
  }

  // Insert new readwrite sref args before index args.
  int64_t numIndexArgs = 2 * genericOp.getNumIterators();
  int64_t readwriteInsertIdx = body->getNumArguments() - numIndexArgs;
  Attribute syncScope = SyncOnReturnAttr::get(context);
  for (Type resultType : newResultTypes) {
    ShapedType shapedType = cast<ShapedType>(resultType);
    ShapedRefType srefType =
        ShapedRefType::get(context, shapedType.getShape(),
                           shapedType.getElementType(), genericOp.getScope(),
                           syncScope);
    BlockArgument arg =
        body->insertArgument(readwriteInsertIdx, srefType, loc);
    newReadwriteRefs.push_back(arg);
    ++readwriteInsertIdx;
  }

  // Update readonly refs count.
  newGenericOp.setNumReadonlyRefs(numOriginalReadonlyRefs +
                                  newReadonlyInits.size());

  // Update operand segment sizes for readonly inits.
  auto &props = newGenericOp.getProperties();
  props.setOperandSegmentSizes(
      {static_cast<int32_t>(combinedReadonlyInits.size()),
       static_cast<int32_t>(combinedInits.size()),
       static_cast<int32_t>(combinedDynamicSizes.size())});
  newGenericOp.getReadonlyInitsMutable().assign(combinedReadonlyInits);

  rewriter.replaceOp(genericOp,
                     newGenericOp->getResults().take_front(numOriginalResults));
  return newGenericOp;
}

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
    UnrealizedConversionCastOp conversion =
        UnrealizedConversionCastOp::create(rewriter, loc, undistributedType,
                                           slice.getSource());
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

  // Collect readonly inits for non-fused tileable operands.
  SmallVector<Value> newReadonlyInits;
  // Map from consumer operand index to index in newReadonlyInits (-1 if not).
  SmallVector<int64_t> operandToNewReadonlyIdx(targetOp->getNumOperands(), -1);
  for (int64_t i = 0, e = targetOp->getNumOperands(); i < e; ++i) {
    unsigned idx = static_cast<unsigned>(i);
    if (fusedOperandSet.contains(idx)) {
      // Fused along — will use write_slice source.
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
    operandToNewReadonlyIdx[i] =
        static_cast<int64_t>(newReadonlyInits.size());
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
          .assign(getInitOrCreateEmpty(
              cast<OpResult>(operand).getResultNumber()));
    }

    SmallVector<SmallVector<OpFoldResult>> outputShapes;
    Operation *nextNode = targetOp->getNextNode();
    Block *currBlock = targetOp->getBlock();
    rewriter.moveOpBefore(targetOp, producerOp);

    [[maybe_unused]] auto reifyOp =
        cast<ReifyRankedShapedTypeOpInterface>(*targetOp);
    assert(succeeded(reifyOp.reifyResultShapes(rewriter, outputShapes)) &&
           "unexpected reify result shapes failed");
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
      /*newReadwriteInits=*/newTiedArgs.empty()
          ? ValueRange()
          : ValueRange(newTiedArgs),
      newIsTied, newTiedArgs, newDynamicSizes, newResultTypes,
      newReadonlyBlockArgs, newReadwriteBlockArgs);

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
      // Non-fused tileable operand: create pcf.read_slice from the new
      // readonly sref arg.
      BlockArgument sref = newReadonlyBlockArgs[operandToNewReadonlyIdx[i]];
      auto srefType = cast<ShapedRefType>(sref.getType());

      // Get result tile position for this operand's tile.
      // We use the cloned op to compute operand tile from iteration domain.
      SmallVector<OpFoldResult> operandOffsets, operandSizes;
      [[maybe_unused]] LogicalResult tileRes =
          clonedOp.getResultTilePosition(rewriter, idx, iterDomainOffsets,
                                         iterDomainSizes, operandOffsets,
                                         operandSizes);
      // If getResultTilePosition fails, fall back to iteration domain
      // offsets/sizes directly.
      if (failed(tileRes)) {
        operandOffsets = SmallVector<OpFoldResult>(iterDomainOffsets);
        operandSizes = SmallVector<OpFoldResult>(iterDomainSizes);
      }

      // Build the read slice type.
      int64_t srefRank = srefType.getRank();
      SmallVector<int64_t> staticSizes;
      staticSizes.reserve(srefRank);
      for (int64_t dim = 0, de = std::min(srefRank,
               static_cast<int64_t>(operandSizes.size())); dim < de; ++dim) {
        if (auto attr = dyn_cast<Attribute>(operandSizes[dim])) {
          staticSizes.push_back(cast<IntegerAttr>(attr).getInt());
        } else {
          staticSizes.push_back(ShapedType::kDynamic);
        }
      }
      // Pad remaining dimensions with original sref sizes.
      for (int64_t dim = staticSizes.size(); dim < srefRank; ++dim) {
        staticSizes.push_back(srefType.getDimSize(dim));
      }
      RankedTensorType readType =
          RankedTensorType::get(staticSizes, srefType.getElementType());
      SmallVector<OpFoldResult> strides(srefRank, rewriter.getIndexAttr(1));

      // Pad offsets/sizes to match sref rank.
      while (static_cast<int64_t>(operandOffsets.size()) < srefRank) {
        operandOffsets.push_back(rewriter.getIndexAttr(0));
      }
      while (static_cast<int64_t>(operandSizes.size()) < srefRank) {
        operandSizes.push_back(
            rewriter.getIndexAttr(srefType.getDimSize(operandSizes.size())));
      }

      Value tile = ReadSliceOp::create(rewriter, loc, readType, sref,
                                       operandOffsets, operandSizes, strides);
      operandInfo.push_back({tile, /*isTile=*/true});
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

  // Step 9: Call getDistributedImplementation.
  FailureOr<TilingResult> tilingResult =
      clonedPCF.getDistributedImplementation(rewriter, iterDomainOffsets,
                                             iterDomainSizes, operandInfo,
                                             resultInfo);
  assert(succeeded(tilingResult) &&
         "unexpected distributed implementation failure");

  // Step 10: Handle returned tiled values. Write non-null results via
  // pcf.write_slice.
  unsigned numResults = clonedOp->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsets(numResults);
  SmallVector<SmallVector<OpFoldResult>> resultSizes(numResults);
  for (auto [idx, v] : llvm::enumerate(clonedOp->getResults())) {
    [[maybe_unused]] LogicalResult posRes = clonedOp.getResultTilePosition(
        rewriter, idx, iterDomainOffsets, iterDomainSizes, resultOffsets[idx],
        resultSizes[idx]);
    assert(succeeded(posRes) &&
           "unexpected failure to get result tile position");
  }

  OpFoldResult one = rewriter.getIndexAttr(1);
  for (int64_t i = 0; i < numConsumerResults; ++i) {
    Value tiledValue = tilingResult->tiledValues[i];
    if (!tiledValue) {
      // The implementation already wrote to the sref directly.
      continue;
    }
    SmallVector<OpFoldResult> strides(resultOffsets[i].size(), one);
    WriteSliceOp::create(rewriter, loc, tiledValue, resultInfo[i].destSref,
                         resultOffsets[i], resultSizes[i], strides);
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
