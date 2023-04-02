// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/UKernelOps.h"

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/Utils/EncodingInfo.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/UKernelOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Codegen {

/// Returns true if the dimensions of ShapedType are compatible.
static bool isShapedTypeDimCompatible(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic ||
         lhs == rhs;
}

/// Returns true if the dimensions of ShapedType are compatible.
static bool areShapesCompatible(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return llvm::all_of(llvm::zip(lhs, rhs), [](std::tuple<int64_t, int64_t> it) {
    return isShapedTypeDimCompatible(std::get<0>(it), std::get<1>(it));
  });
}

/// Helper method to generate a function declaration at a module scope,
/// and a call to that function
static FailureOr<func::CallOp> createFunctionCall(RewriterBase &rewriter,
                                                  Operation *op,
                                                  StringRef fnName,
                                                  TypeRange callArgumentTypes,
                                                  TypeRange callReturnTypes,
                                                  ValueRange callOperands) {
  FunctionType functionType =
      rewriter.getFunctionType(callArgumentTypes, callReturnTypes);

  // Create a declaration for the function type.
  Location loc = op->getLoc();
  auto moduleOp = SymbolTable::getNearestSymbolTable(op);
  // Check for duplicates.
  auto fnDecl = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(moduleOp, fnName));
  if (!fnDecl) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&moduleOp->getRegion(0).front());
    fnDecl = rewriter.create<func::FuncOp>(loc, fnName, functionType);
    SymbolTable::setSymbolVisibility(fnDecl, SymbolTable::Visibility::Private);
  } else if (fnDecl.getFunctionType() != functionType) {
    return rewriter.notifyMatchFailure(
        op, llvm::formatv("mismatch in function type computed during lowering "
                          "({0}) and already declared function ({1})",
                          functionType, fnDecl.getFunctionType()));
  }

  // Insert the function call.
  return rewriter.create<func::CallOp>(loc, fnDecl, callOperands);
}

//===---------------------------------------------------------------------===//
// UKernelGenericOp
//===---------------------------------------------------------------------===//

std::pair<int64_t, int64_t> UKernelGenericOp::getDpsInitsPositionRange() {
  auto [pos, size] = getODSOperandIndexAndLength(1);
  return {static_cast<int64_t>(pos), static_cast<int64_t>(pos + size)};
}

/// Map type of operand of a `iree_codegen.ukernel.generic` operation to
/// the type(s) of the function call arguments(s) it lowers to.
static LogicalResult getCallOpType(MLIRContext *context,
                                   Type microKernelOpOperandType,
                                   SmallVector<Type> &callOperandTypes) {
  return TypeSwitch<Type, LogicalResult>(microKernelOpOperandType)
      .Case<FloatType, IndexType, IntegerType>([&](auto scalarType) {
        callOperandTypes.push_back(scalarType);
        return success();
      })
      .Case<MemRefType>([&](MemRefType memrefType) {
        // 0D memref lowers to a 0D memref operand type. Other nD memrefs
        // lower to 0D memref, offset, and n-1 strides.

        // Base ptr.
        callOperandTypes.push_back(MemRefType::get(
            ArrayRef<int64_t>{}, memrefType.getElementType(),
            MemRefLayoutAttrInterface{}, memrefType.getMemorySpace()));
        if (memrefType.getRank() == 0) return success();

        auto indexType = IndexType::get(context);
        // Offset
        callOperandTypes.push_back(indexType);

        // Strides.
        callOperandTypes.resize(
            callOperandTypes.size() + memrefType.getRank() - 1, indexType);
        return success();
      })
      .Default([&](Type t) { return failure(); });
}

/// Map `operand` of a `ukernel.generic` operation to the operand(s) of
/// the function call it lowers to.
static LogicalResult lowerToCallOperands(Location loc, RewriterBase &rewriter,
                                         Value operand,
                                         SmallVector<Value> &callOperands) {
  return TypeSwitch<Type, LogicalResult>(operand.getType())
      .Case<FloatType, IndexType, IntegerType>([&](auto scalarType) {
        callOperands.push_back(operand);
        return success();
      })
      .Case<MemRefType>([&](MemRefType memrefType) {
        auto extractStridedMetadataOp =
            rewriter.create<memref::ExtractStridedMetadataOp>(loc, operand);
        // Base ptr.
        callOperands.push_back(extractStridedMetadataOp.getBaseBuffer());
        if (memrefType.getRank() == 0) {
          return success();
        }
        // Offset.
        callOperands.push_back(extractStridedMetadataOp.getOffset());
        // Strides.
        for (auto stride : extractStridedMetadataOp.getStrides().drop_back()) {
          callOperands.push_back(stride);
        }
        return success();
      })
      .Default([](Type) { return failure(); });
}

FailureOr<func::CallOp> UKernelGenericOp::lowerToFunctionCall(
    RewriterBase &rewriter) {
  // Create the function type based on the operands and results.
  SmallVector<Type> callArgumentTypes;
  for (auto microKernelOpOperandType : getOperation()->getOperandTypes()) {
    if (failed(getCallOpType(rewriter.getContext(), microKernelOpOperandType,
                             callArgumentTypes))) {
      return rewriter.notifyMatchFailure(
          getOperation(), llvm::formatv("failed to lower operand type {0}",
                                        microKernelOpOperandType));
    }
  }
  SmallVector<Type> callResultTypes;
  for (auto resultType : getResultTypes()) {
    if (resultType.isa<ShapedType>()) {
      return rewriter.notifyMatchFailure(
          getOperation(),
          "cannot lower a `ShapedType` return value to function call");
    }
    if (failed(getCallOpType(rewriter.getContext(), resultType,
                             callResultTypes))) {
      return rewriter.notifyMatchFailure(
          getOperation(),
          llvm::formatv("failed to lower result type {0}", resultType));
    }
  }

  // Get the operands for the function call.
  SmallVector<Value> callOperands;
  Location loc = getLoc();
  for (auto operand : getOperands()) {
    if (failed(lowerToCallOperands(loc, rewriter, operand, callOperands))) {
      return rewriter.notifyMatchFailure(
          getOperation(), "failed to lower operands to function call operands");
    }
  }
  return createFunctionCall(rewriter, getOperation(), getUKernelFnName(),
                            callArgumentTypes, callResultTypes, callOperands);
}

//===---------------------------------------------------------------------===//
// UKernelMmt4DOp
//===---------------------------------------------------------------------===//

std::pair<int64_t, int64_t> UKernelMmt4DOp::getDpsInitsPositionRange() {
  auto [pos, size] = getODSOperandIndexAndLength(2);
  return {static_cast<int64_t>(pos), static_cast<int64_t>(pos + size)};
}

static FailureOr<SmallVector<Type>> getFunctionArgTypesForUKernelMmt4D(
    MLIRContext *context, UKernelMmt4DOp mmt4dUKernelOp) {
  SmallVector<Type> callArgumentTypes;
  auto indexType = IndexType::get(context);

  auto processMemrefTypeOperand = [&](Value memRefValue) -> LogicalResult {
    auto memRefType = memRefValue.getType().dyn_cast<MemRefType>();
    if (!memRefType) {
      return mmt4dUKernelOp->emitOpError(
          llvm::formatv("unable to lower {0} to function call argument types",
                        memRefValue.getType()));
    }
    // base-ptr
    callArgumentTypes.push_back(MemRefType::get(
        /*shape=*/{}, memRefType.getElementType(), MemRefLayoutAttrInterface{},
        memRefType.getMemorySpace()));
    // offset
    callArgumentTypes.push_back(indexType);
    // stride[0]
    callArgumentTypes.push_back(indexType);
    return success();
  };
  /// LHS, RHS, Out
  if (failed(processMemrefTypeOperand(mmt4dUKernelOp.getLhs())) ||
      failed(processMemrefTypeOperand(mmt4dUKernelOp.getRhs())) ||
      failed(processMemrefTypeOperand(mmt4dUKernelOp.getOutput()))) {
    return failure();
  }
  // m, n, k
  callArgumentTypes.resize(callArgumentTypes.size() + 3, indexType);
  // m0, n0, k0
  auto i32Type = IntegerType::get(context, 32);
  callArgumentTypes.resize(callArgumentTypes.size() + 3, i32Type);
  // flags
  callArgumentTypes.push_back(i32Type);
  return callArgumentTypes;
}

static FailureOr<SmallVector<Value>> getFunctionArgValuesForUKernelMmt4D(
    RewriterBase &rewriter, Location loc, UKernelMmt4DOp mmt4dUKernelOp) {
  SmallVector<Value> callOperands;
  auto processMemrefTypeOperand = [&](Value memRefValue) {
    auto extractStridedMetadataOp =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, memRefValue);
    // Base ptr.
    callOperands.push_back(extractStridedMetadataOp.getBaseBuffer());
    // offset.
    callOperands.push_back(extractStridedMetadataOp.getOffset());
    // strides.
    callOperands.push_back(extractStridedMetadataOp.getStrides().front());
  };
  auto getDimAsI32 = [&](Value value, int dim) {
    auto dimValue = rewriter.create<memref::DimOp>(loc, value, dim);
    auto asI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                     dimValue);
    callOperands.push_back(asI32);
  };
  // LHS
  processMemrefTypeOperand(mmt4dUKernelOp.getLhs());
  // RHS
  processMemrefTypeOperand(mmt4dUKernelOp.getRhs());
  // Out
  processMemrefTypeOperand(mmt4dUKernelOp.getOutput());
  // M
  callOperands.push_back(
      rewriter.create<memref::DimOp>(loc, mmt4dUKernelOp.getLhs(), 0));
  // N
  callOperands.push_back(
      rewriter.create<memref::DimOp>(loc, mmt4dUKernelOp.getRhs(), 0));
  // K
  callOperands.push_back(
      rewriter.create<memref::DimOp>(loc, mmt4dUKernelOp.getLhs(), 1));
  // M0
  getDimAsI32(mmt4dUKernelOp.getLhs(), 2);
  // N0
  getDimAsI32(mmt4dUKernelOp.getRhs(), 2);
  // K0
  getDimAsI32(mmt4dUKernelOp.getLhs(), 3);
  // Flags;
  int flags = 0;
  if (mmt4dUKernelOp.getAccumulate()) {
    flags |= IREE_UK_FLAG_ACCUMULATE;
  }
  callOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags)));
  return callOperands;
}

FailureOr<func::CallOp> UKernelMmt4DOp::lowerToFunctionCall(
    RewriterBase &rewriter) {
  // TODO: handle op with return values if they are scalar.
  if (getNumResults() != 0) {
    return rewriter.notifyMatchFailure(
        getOperation(), "cannot lower to function call operation with results");
  }

  std::optional<MatmulType> matmulType = getMatmulType(
      getLhsElementType(), getRhsElementType(), getOutputElementType());
  if (!matmulType) {
    return emitOpError(
        "unhandled element types of operands for lowering to micro kernel "
        "function call");
  }

  // Function name.
  std::string fnName = "vmvx.mmt4d.";
  switch (matmulType.value()) {
    case MatmulType::I8I8I32:
      fnName.append("i8.i8.i32");
      break;
    case MatmulType::F32F32F32:
      fnName.append("f32.f32.f32");
      break;
  }

  // Create the function type.
  FailureOr<SmallVector<Type>> fnArgTypes =
      getFunctionArgTypesForUKernelMmt4D(rewriter.getContext(), *this);
  if (failed(fnArgTypes)) {
    return emitOpError(
        "unable to get function type to lower micro kernel op to");
  }
  // Create the function call operands.
  FailureOr<SmallVector<Value>> fnCallOperands =
      getFunctionArgValuesForUKernelMmt4D(rewriter, getLoc(), *this);
  if (failed(fnCallOperands)) {
    return emitOpError(
        "unable to get the function call operands to lower micro kernel op");
  }
  return createFunctionCall(
      rewriter, getOperation(), fnName, fnArgTypes.value(),
      /*callReturnTypes=*/TypeRange{}, fnCallOperands.value());
}

}  // namespace Codegen
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
