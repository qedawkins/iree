// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- BufferizationExternalModels.cpp -----------------------------------===//
//
// This file implements bufferization interfaces for PCF ops.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/PCF/ExternalInterfaces/BufferizationExternalModels.h"

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

namespace mlir::iree_compiler::IREE::PCF {

namespace {

struct GenericOpInterface
    : bufferization::BufferizableOpInterface::ExternalModel<GenericOpInterface,
                                                            PCF::GenericOp> {
  /// Returns true if the given operand is a readonly init.
  static bool isReadonlyInitOperand(PCF::GenericOp genericOp,
                                    OpOperand &opOperand) {
    OperandRange readonlyInits = genericOp.getReadonlyInits();
    if (readonlyInits.empty()) {
      return false;
    }
    int64_t roBegin = readonlyInits.getBeginOperandIndex();
    int64_t opNum = opOperand.getOperandNumber();
    return opNum >= roBegin &&
           opNum < roBegin + static_cast<int64_t>(readonlyInits.size());
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    auto genericOp = cast<PCF::GenericOp>(op);
    // Readonly inits are read.
    if (isReadonlyInitOperand(genericOp, opOperand)) {
      return true;
    }
    // Readwrite inits in parallel ops can be treated as though they never
    // read.
    return false;
  }

  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    auto genericOp = cast<PCF::GenericOp>(op);
    // Readonly inits are never written.
    if (isReadonlyInitOperand(genericOp, opOperand)) {
      return false;
    }
    // Readwrite inits must always be assumed to write.
    return true;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    auto genericOp = cast<PCF::GenericOp>(op);
    // Readonly inits have no tied result.
    OpResult tiedResult = genericOp.getTiedResult(opOperand);
    if (!tiedResult) {
      return {};
    }

    return {{tiedResult, bufferization::BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto genericOp = cast<PCF::GenericOp>(op);
    Location loc = genericOp.getLoc();

    // Bufferize readonly inits.
    SmallVector<Value> newReadonlyInits;
    newReadonlyInits.reserve(genericOp.getReadonlyInits().size());
    for (Value init : genericOp.getReadonlyInits()) {
      if (isa<RankedTensorType>(init.getType())) {
        FailureOr<Value> newInit =
            bufferization::getBuffer(rewriter, init, options, state);
        if (failed(newInit)) {
          return op->emitOpError("failed to get readonly init buffer");
        }
        newReadonlyInits.push_back(*newInit);
      } else {
        newReadonlyInits.push_back(init);
      }
    }

    // Bufferize readwrite inits.
    SmallVector<Value> newInits;
    newInits.reserve(genericOp.getInits().size());
    for (Value init : genericOp.getInits()) {
      if (isa<RankedTensorType>(init.getType())) {
        FailureOr<Value> newInit =
            bufferization::getBuffer(rewriter, init, options, state);
        if (failed(newInit)) {
          return op->emitOpError("failed to get init buffer");
        }
        newInits.push_back(*newInit);
      } else {
        newInits.push_back(init);
      }
    }

    SmallVector<Type> newResultTypes;
    for (Value result : genericOp.getResults()) {
      if (isa<TensorType>(result.getType())) {
        FailureOr<bufferization::BufferLikeType> resultType =
            bufferization::getBufferType(result, options, state);
        if (failed(resultType)) {
          return failure();
        }
        newResultTypes.push_back(*resultType);
      } else {
        newResultTypes.push_back(result.getType());
      }
    }

    auto newGenericOp = PCF::GenericOp::create(
        rewriter, loc, newResultTypes, genericOp.getScope(), newReadonlyInits,
        newInits, genericOp.getDynamicSizes(), genericOp.getIsTied(),
        genericOp.getNumIterators(), genericOp.getSyncOnReturn());
    // The builder doesn't set num_leading_args (it defaults to 0), but we
    // need to preserve it from the original op since the execute region's
    // block arguments include leading args from the initialize region.
    newGenericOp.setNumLeadingArgs(genericOp.getNumLeadingArgs());

    newGenericOp.getRegion().takeBody(genericOp.getRegion());
    newGenericOp.getInitializer().takeBody(genericOp.getInitializer());

    // For results with tied inits, use the init buffer directly so
    // bufferization knows the result aliases the init (avoids extraneous
    // copies). Self-allocated results use the new op's result.
    SmallVector<Value> replacements;
    for (int64_t i = 0, e = genericOp->getNumResults(); i < e; ++i) {
      OpOperand *tiedInit = genericOp.getTiedInit(i);
      if (tiedInit) {
        replacements.push_back(
            newGenericOp->getOperand(tiedInit->getOperandNumber()));
      } else {
        replacements.push_back(newGenericOp->getResult(i));
      }
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }

  FailureOr<bufferization::BufferLikeType>
  getBufferType(Operation *op, Value value,
                const bufferization::BufferizationOptions &options,
                const bufferization::BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto genericOp = cast<PCF::GenericOp>(op);

    // Block arguments are `pcf.sref`, so this must always be an opresult.
    auto result = cast<OpResult>(value);
    assert(result.getOwner() == op && "invalid value");

    // If the result has a tied init, use that as the buffer type.
    OpOperand *tiedInit = genericOp.getTiedInit(result.getResultNumber());
    if (tiedInit) {
      return bufferization::detail::asMemRefType(bufferization::getBufferType(
          tiedInit->get(), options, state, invocationStack));
    }

    auto resultType = cast<RankedTensorType>(result.getType());

    // Else query the scope for the memory space to allocate for.
    FailureOr<Attribute> memSpace =
        genericOp.getScope().getAllocMemSpace(op->getContext());
    if (failed(memSpace)) {
      return failure();
    }
    return cast<bufferization::BufferLikeType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(resultType,
                                                             *memSpace));
  }
};

struct LoopOpInterface
    : bufferization::BufferizableOpInterface::ExternalModel<LoopOpInterface,
                                                            PCF::LoopOp> {
  /// Returns true if the given operand is a readonly init.
  static bool isReadonlyInitOperand(PCF::LoopOp loopOp, OpOperand &opOperand) {
    OperandRange readonlyInits = loopOp.getReadonlyInits();
    if (readonlyInits.empty()) {
      return false;
    }
    int64_t roBegin = readonlyInits.getBeginOperandIndex();
    int64_t opNum = opOperand.getOperandNumber();
    return opNum >= roBegin &&
           opNum < roBegin + static_cast<int64_t>(readonlyInits.size());
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    auto loopOp = cast<PCF::LoopOp>(op);
    // Readonly inits are read.
    if (isReadonlyInitOperand(loopOp, opOperand)) {
      return true;
    }
    // Readwrite inits in parallel ops can be treated as though they never
    // read.
    return false;
  }

  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    auto loopOp = cast<PCF::LoopOp>(op);
    // Readonly inits are never written.
    if (isReadonlyInitOperand(loopOp, opOperand)) {
      return false;
    }
    // Readwrite inits must always be assumed to write.
    return true;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    auto loopOp = cast<PCF::LoopOp>(op);
    // Readonly inits have no tied result.
    OpResult tiedResult = loopOp.getTiedResult(opOperand);
    if (!tiedResult) {
      return {};
    }

    return {{tiedResult, bufferization::BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto loopOp = cast<PCF::LoopOp>(op);
    Location loc = loopOp.getLoc();

    // Bufferize readonly inits.
    SmallVector<Value> newReadonlyInits;
    newReadonlyInits.reserve(loopOp.getReadonlyInits().size());
    for (Value init : loopOp.getReadonlyInits()) {
      if (isa<RankedTensorType>(init.getType())) {
        FailureOr<Value> newInit =
            bufferization::getBuffer(rewriter, init, options, state);
        if (failed(newInit)) {
          return op->emitOpError("failed to get readonly init buffer");
        }
        newReadonlyInits.push_back(*newInit);
      } else {
        newReadonlyInits.push_back(init);
      }
    }

    // Bufferize readwrite inits.
    SmallVector<Value> newInits;
    newInits.reserve(loopOp.getInits().size());
    for (Value init : loopOp.getInits()) {
      if (isa<RankedTensorType>(init.getType())) {
        FailureOr<Value> newInit =
            bufferization::getBuffer(rewriter, init, options, state);
        if (failed(newInit)) {
          return op->emitOpError("failed to get init buffer");
        }
        newInits.push_back(*newInit);
      } else {
        newInits.push_back(init);
      }
    }

    SmallVector<Type> newResultTypes;
    for (Value result : loopOp.getResults()) {
      if (isa<TensorType>(result.getType())) {
        FailureOr<bufferization::BufferLikeType> resultType =
            bufferization::getBufferType(result, options, state);
        if (failed(resultType)) {
          return failure();
        }
        newResultTypes.push_back(*resultType);
      } else {
        newResultTypes.push_back(result.getType());
      }
    }

    auto newLoopOp = PCF::LoopOp::create(
        rewriter, loc, newResultTypes, loopOp.getScope(), loopOp.getCount(),
        newReadonlyInits, newInits, loopOp.getDynamicSizes(),
        loopOp.getIsTied(), loopOp.getSyncOnReturn());

    newLoopOp.getRegion().takeBody(loopOp.getRegion());

    // For results with tied inits, use the init buffer directly so
    // bufferization knows the result aliases the init (avoids extraneous
    // copies). Self-allocated results use the new op's result.
    SmallVector<Value> replacements;
    for (int64_t i = 0, e = loopOp->getNumResults(); i < e; ++i) {
      OpOperand *tiedInit = loopOp.getTiedInit(i);
      if (tiedInit) {
        replacements.push_back(
            newLoopOp->getOperand(tiedInit->getOperandNumber()));
      } else {
        replacements.push_back(newLoopOp->getResult(i));
      }
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }

  FailureOr<bufferization::BufferLikeType>
  getBufferType(Operation *op, Value value,
                const bufferization::BufferizationOptions &options,
                const bufferization::BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto loopOp = cast<PCF::LoopOp>(op);

    // Block arguments are `pcf.sref`, so this must always be an opresult.
    auto result = cast<OpResult>(value);
    assert(result.getOwner() == op && "invalid value");

    // If the result has a tied init, use that as the buffer type.
    OpOperand *tiedInit = loopOp.getTiedInit(result.getResultNumber());
    if (tiedInit) {
      return bufferization::detail::asMemRefType(bufferization::getBufferType(
          tiedInit->get(), options, state, invocationStack));
    }

    auto resultType = cast<RankedTensorType>(result.getType());

    // Else query the scope for the memory space to allocate for.
    FailureOr<Attribute> memSpace =
        loopOp.getScope().getAllocMemSpace(op->getContext());
    if (failed(memSpace)) {
      return failure();
    }
    return cast<bufferization::BufferLikeType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(resultType,
                                                             *memSpace));
  }
};

struct WriteSliceOpInterface
    : bufferization::BufferizableOpInterface::ExternalModel<
          WriteSliceOpInterface, PCF::WriteSliceOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    // The only valid tensor operand is the source which is always read.
    return true;
  }

  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    // The only valid tensor operand is the source which is only read.
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto writeOp = cast<PCF::WriteSliceOp>(op);

    if (isa<RankedTensorType>(writeOp.getSourceType())) {
      FailureOr<Value> newSrc = bufferization::getBuffer(
          rewriter, writeOp.getSource(), options, state);
      if (failed(newSrc)) {
        return failure();
      }
      writeOp.getSourceMutable().assign(*newSrc);
    }
    return success();
  }
};

struct ToSrefOpInterface
    : bufferization::BufferizableOpInterface::ExternalModel<ToSrefOpInterface,
                                                            PCF::ToSrefOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    // Returns true because the tensor input is consumed to create the sref
    // view. While to_sref is logically a view (not a read), returning false
    // here causes MLIR's alias analysis to traverse the Equivalent aliasing
    // chain and call isValueRead on the sref result, which is not a
    // TensorType and triggers an assertion failure.
    return true;
  }

  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    // to_sref is a readonly binding — it never writes.
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    // The sref result aliases the input buffer.
    return {{op->getResult(0), bufferization::BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto toSrefOp = cast<PCF::ToSrefOp>(op);

    // Get the buffer for the tensor input.
    if (isa<RankedTensorType>(toSrefOp.getInput().getType())) {
      FailureOr<Value> newInput = bufferization::getBuffer(
          rewriter, toSrefOp.getInput(), options, state);
      if (failed(newInput)) {
        return failure();
      }
      toSrefOp.getInputMutable().assign(*newInput);
    }
    return success();
  }
};

struct ReadSliceOpInterface
    : bufferization::BufferizableOpInterface::ExternalModel<
          ReadSliceOpInterface, PCF::ReadSliceOp> {
  /// Build a memref type for the slice result using the read_slice's sizes.
  /// The result shape comes from the slice sizes, not the full source shape.
  /// Layout and memory space are maximally dynamic (unknown until resolving
  /// sref types, after which both are propagated to this operation's users).
  static MemRefType getSliceBufferType(MLIRContext *context,
                                       PCF::ReadSliceOp readOp) {
    auto resultTensorType = cast<RankedTensorType>(readOp.getResultType());
    int64_t rank = resultTensorType.getRank();
    SmallVector<int64_t> strides(rank, ShapedType::kDynamic);
    auto layout =
        StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);
    return MemRefType::get(resultTensorType.getShape(),
                           resultTensorType.getElementType(), layout,
                           /*memorySpace=*/nullptr);
  }
  FailureOr<bufferization::BufferLikeType>
  getBufferType(Operation *op, Value value,
                const bufferization::BufferizationOptions &options,
                const bufferization::BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto readOp = cast<PCF::ReadSliceOp>(op);
    return cast<bufferization::BufferLikeType>(
        getSliceBufferType(op->getContext(), readOp));
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto readOp = cast<PCF::ReadSliceOp>(op);

    // Skip vector results.
    if (!isa<RankedTensorType>(readOp.getResultType())) {
      return success();
    }

    // Create result type with maximally dynamic layout and no memory space.
    // Uses the slice sizes, not the full source shape.
    MemRefType resultType = getSliceBufferType(op->getContext(), readOp);

    // GetMemrefOp lets us get a memref out of a read_slice. Accesses to srefs
    // are allowed to ignore accesses to this memref.
    auto getMemrefOp = PCF::GetMemrefOp::create(
        rewriter, readOp.getLoc(), resultType, readOp.getSource(),
        readOp.getMixedOffsets(), readOp.getMixedSizes(),
        readOp.getMixedStrides());
    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 getMemrefOp.getResult());
    return success();
  }
};

} // namespace

void registerBufferizationExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, PCF::PCFDialect *dialect) {
    GenericOp::attachInterface<GenericOpInterface>(*ctx);
    LoopOp::attachInterface<LoopOpInterface>(*ctx);
    ReadSliceOp::attachInterface<ReadSliceOpInterface>(*ctx);
    ToSrefOp::attachInterface<ToSrefOpInterface>(*ctx);
    WriteSliceOp::attachInterface<WriteSliceOpInterface>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::PCF
