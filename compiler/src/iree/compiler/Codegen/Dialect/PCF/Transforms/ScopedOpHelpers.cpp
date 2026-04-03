// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// addReadonlyAndReadwriteArgs<LoopOp>
//===----------------------------------------------------------------------===//

template <>
LoopOp addReadonlyAndReadwriteArgs<LoopOp>(
    RewriterBase &rewriter, LoopOp loopOp, ValueRange newReadonlyInits,
    ValueRange newReadwriteInits, ArrayRef<bool> newIsTied,
    ArrayRef<Value> newDynamicSizes, TypeRange newResultTypes,
    SmallVectorImpl<BlockArgument> &newReadonlyRefs,
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

  int64_t numOriginalResults = loopOp->getNumResults();
  int64_t numOriginalReadonlyRefs = loopOp.getNumReadonlyRefs();

  // Create the new loop BEFORE the old loop using the full builder with
  // readonly inits.
  OpBuilder::InsertionGuard createGuard(rewriter);
  rewriter.setInsertionPoint(loopOp);
  LoopOp newLoopOp = LoopOp::create(
      rewriter, loc, combinedResultTypes, loopOp.getScope(), loopOp.getCount(),
      combinedReadonlyInits, combinedInits, combinedDynamicSizes,
      combinedIsTied, loopOp.getSyncOnReturn());

  // Move the old body to the new loop.
  newLoopOp.getRegion().takeBody(loopOp.getRegion());

  // After takeBody, the block has the old layout. Insert new args at the
  // correct positions.
  Block *body = newLoopOp.getBody();

  // Insert new readonly sref args after old readonly refs.
  int64_t readonlyInsertIdx = numOriginalReadonlyRefs;
  for (Value init : newReadonlyInits) {
    ShapedType shapedType = cast<ShapedType>(init.getType());
    ShapedRefType srefType =
        ShapedRefType::get(context, shapedType.getShape(),
                           shapedType.getElementType(), loopOp.getScope());
    BlockArgument arg = body->insertArgument(readonlyInsertIdx, srefType, loc);
    newReadonlyRefs.push_back(arg);
    ++readonlyInsertIdx;
  }

  // Insert new readwrite sref args before id args.
  int64_t numIdArgs = loopOp.getCount().size();
  int64_t readwriteInsertIdx = body->getNumArguments() - numIdArgs;
  Attribute syncOnReturn = SyncOnReturnAttr::get(context);
  for (Type resultType : newResultTypes) {
    ShapedType shapedType = cast<ShapedType>(resultType);
    ShapedRefType srefType = ShapedRefType::get(context, shapedType.getShape(),
                                                shapedType.getElementType(),
                                                loopOp.getScope(), syncOnReturn);
    BlockArgument arg = body->insertArgument(readwriteInsertIdx, srefType, loc);
    newReadwriteRefs.push_back(arg);
    ++readwriteInsertIdx;
  }

  // Replace old loop's results with corresponding new loop results.
  rewriter.replaceOp(loopOp,
                     newLoopOp->getResults().take_front(numOriginalResults));
  return newLoopOp;
}

//===----------------------------------------------------------------------===//
// addReadonlyAndReadwriteArgs<GenericOp>
//===----------------------------------------------------------------------===//

template <>
GenericOp addReadonlyAndReadwriteArgs<GenericOp>(
    RewriterBase &rewriter, GenericOp genericOp, ValueRange newReadonlyInits,
    ValueRange newReadwriteInits, ArrayRef<bool> newIsTied,
    ArrayRef<Value> newDynamicSizes, TypeRange newResultTypes,
    SmallVectorImpl<BlockArgument> &newReadonlyRefs,
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

  // Create the new GenericOp BEFORE the old one. The builder used here
  // (resultTypes, scope, inits, dynamicSizes, isTied) sets readonlyInits=0
  // in the segment sizes and does not add readonly operands. We must insert
  // them separately before updating segment sizes.
  OpBuilder::InsertionGuard createGuard(rewriter);
  rewriter.setInsertionPoint(genericOp);
  GenericOp newGenericOp = GenericOp::create(
      rewriter, loc, combinedResultTypes, genericOp.getScope(), combinedInits,
      combinedDynamicSizes, combinedIsTied, genericOp.getNumIterators(),
      genericOp.getSyncOnReturn());
  newGenericOp.getRegion().takeBody(genericOp.getRegion());
  newGenericOp.getInitializer().takeBody(genericOp.getInitializer());
  newGenericOp.setNumLeadingArgs(genericOp.getNumLeadingArgs());

  // Insert readonly init operands at position 0 (before the inits segment).
  // This must happen BEFORE updating segment sizes so the total operand
  // count is correct when the segment sizes are set.
  if (!combinedReadonlyInits.empty()) {
    newGenericOp->insertOperands(0, combinedReadonlyInits);
  }

  // Now update segment sizes and readonly refs count to include the
  // inserted readonly init operands.
  auto &props = newGenericOp.getProperties();
  props.setOperandSegmentSizes(
      {static_cast<int32_t>(combinedReadonlyInits.size()),
       static_cast<int32_t>(combinedInits.size()),
       static_cast<int32_t>(combinedDynamicSizes.size())});
  newGenericOp.setNumReadonlyRefs(numOriginalReadonlyRefs +
                                  newReadonlyInits.size());

  Block *body = &newGenericOp.getRegion().front();

  // Insert new readonly sref args after leading args + old readonly refs.
  int64_t readonlyInsertIdx =
      genericOp.getNumLeadingArgs() + numOriginalReadonlyRefs;
  for (Value init : newReadonlyInits) {
    ShapedType shapedType = cast<ShapedType>(init.getType());
    ShapedRefType srefType =
        ShapedRefType::get(context, shapedType.getShape(),
                           shapedType.getElementType(), genericOp.getScope());
    BlockArgument arg = body->insertArgument(readonlyInsertIdx, srefType, loc);
    newReadonlyRefs.push_back(arg);
    ++readonlyInsertIdx;
  }

  // Insert new readwrite sref args before index args.
  int64_t numIndexArgs = 2 * genericOp.getNumIterators();
  int64_t readwriteInsertIdx = body->getNumArguments() - numIndexArgs;
  Attribute syncOnReturn = SyncOnReturnAttr::get(context);
  for (Type resultType : newResultTypes) {
    ShapedType shapedType = cast<ShapedType>(resultType);
    ShapedRefType srefType = ShapedRefType::get(
        context, shapedType.getShape(), shapedType.getElementType(),
        genericOp.getScope(), syncOnReturn);
    BlockArgument arg = body->insertArgument(readwriteInsertIdx, srefType, loc);
    newReadwriteRefs.push_back(arg);
    ++readwriteInsertIdx;
  }

  rewriter.replaceOp(genericOp,
                     newGenericOp->getResults().take_front(numOriginalResults));
  return newGenericOp;
}

//===----------------------------------------------------------------------===//
// addReadonlyArgs convenience wrappers
//===----------------------------------------------------------------------===//

template <>
LoopOp
addReadonlyArgs<LoopOp>(RewriterBase &rewriter, LoopOp loopOp,
                        ValueRange newReadonlyInits,
                        SmallVectorImpl<BlockArgument> &newReadonlyRefs) {
  SmallVector<BlockArgument> unusedReadwriteRefs;
  return addReadonlyAndReadwriteArgs(rewriter, loopOp, newReadonlyInits,
                                     /*newReadwriteInits=*/ValueRange(),
                                     /*newIsTied=*/ArrayRef<bool>(),
                                     /*newDynamicSizes=*/ArrayRef<Value>(),
                                     /*newResultTypes=*/TypeRange(),
                                     newReadonlyRefs, unusedReadwriteRefs);
}

template <>
GenericOp
addReadonlyArgs<GenericOp>(RewriterBase &rewriter, GenericOp genericOp,
                           ValueRange newReadonlyInits,
                           SmallVectorImpl<BlockArgument> &newReadonlyRefs) {
  SmallVector<BlockArgument> unusedReadwriteRefs;
  return addReadonlyAndReadwriteArgs(rewriter, genericOp, newReadonlyInits,
                                     /*newReadwriteInits=*/ValueRange(),
                                     /*newIsTied=*/ArrayRef<bool>(),
                                     /*newDynamicSizes=*/ArrayRef<Value>(),
                                     /*newResultTypes=*/TypeRange(),
                                     newReadonlyRefs, unusedReadwriteRefs);
}

} // namespace mlir::iree_compiler::IREE::PCF
