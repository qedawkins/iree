// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPROMOTEMATMULOPERANDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Creates a `linalg.copy` on the given tensor value and sets the lowering
/// configuration for the copy to `#iree_gpu.derived_thread_config`.
Value promoteValue(OpBuilder &builder, Location loc, Value v) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  SmallVector<OpFoldResult> mixedSizes = tensor::getMixedSizes(builder, loc, v);
  Value empty = builder.create<tensor::EmptyOp>(loc, mixedSizes,
                                                tensorType.getElementType());
  auto copy = builder.create<linalg::CopyOp>(loc, v, empty);
  setLoweringConfig(
      copy, IREE::GPU::DerivedThreadConfigAttr::get(builder.getContext()));
  return copy.getResult(0);
}

/// Inserts a `linalg.copy` directly before the given operation on the
/// specified operand, for example with operand index = 1:
///
///   linalg.matmul ins(%0, %1)
///
/// becomes
///
///   %empty = tensor.empty()
///   %copy = linalg.copy %1 to %empty {
///     lowering_config = #iree_gpu.derived_thread_config}
///   linalg.matmul ins(%0, %copy)
///
/// If the producer is already a tilable op, the producer is just annotated with
/// #iree_gpu.derived_thread_config to indicate that it should be distributed
/// to threads independently of the matmul.
void promoteOperand(OpBuilder &builder, Operation *op, unsigned index) {
  Value operand = op->getOperand(index);
  if (auto producer = operand.getDefiningOp<TilingInterface>()) {
    setLoweringConfig(producer, IREE::GPU::DerivedThreadConfigAttr::get(
                                    builder.getContext()));
    return;
  }

  auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorType) {
    return;
  }

  Value replacement = promoteValue(builder, op->getLoc(), operand);
  op->setOperand(index, replacement);
}

/// Promotes the |index|th result of |op| by inserting a `linalg.copy` with
/// `iree_gpu.derived_thread_config` for the lowering config.
/// TODO: Currently this always creates a copy because derived_thread_config
/// does not properly handle multiple results currently. Try to omit the
/// copy given certain fusions once multiple results is properly supported.
void promoteResult(OpBuilder &builder, Operation *op, unsigned index) {
  Value result = op->getResult(index);

  auto tensorType = cast<RankedTensorType>(result.getType());
  SmallVector<Value> dynamicSizes;
  for (auto [idx, size] : llvm::enumerate(tensorType.getShape())) {
    if (ShapedType::isDynamic(size)) {
      dynamicSizes.push_back(
          builder.create<tensor::DimOp>(op->getLoc(), result, idx));
    }
  }
  Attribute addressSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto alloc = builder.create<bufferization::AllocTensorOp>(
      op->getLoc(), tensorType, dynamicSizes);
  alloc.setMemorySpaceAttr(addressSpace);
  auto copy =
      builder.create<linalg::CopyOp>(op->getLoc(), result, alloc.getResult());

  Value replacement = promoteValue(builder, op->getLoc(), copy.getResult(0));
  result.replaceAllUsesExcept(replacement, copy);
}

bool isNonMatvecContraction(linalg::LinalgOp linalgOp) {
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return false;
  }

  if (contractionDims->k.size() < 1 || contractionDims->m.size() < 1 ||
      contractionDims->n.size() < 1) {
    return false;
  }

  auto getElementCount = [&](ArrayRef<unsigned> dims) {
    int64_t acc = 1;
    for (auto mDim : dims) {
      int64_t size = bounds[mDim];
      if (ShapedType::isDynamic(size)) {
        return size;
      }
      acc *= size;
    }
    return acc;
  };
  return getElementCount(contractionDims->m) != 1 &&
         getElementCount(contractionDims->n) != 1;
}

struct GPUPromoteMatmulOperandsPass final
    : impl::GPUPromoteMatmulOperandsPassBase<GPUPromoteMatmulOperandsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    OpBuilder builder(funcOp);
    SmallVector<linalg::LinalgOp> promotionTargets;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (isNonMatvecContraction(linalgOp)) {
        promotionTargets.push_back(linalgOp);
      }
    });

    for (auto linalgOp : promotionTargets) {
      builder.setInsertionPoint(linalgOp);
      promoteOperand(builder, linalgOp, 0);
      promoteOperand(builder, linalgOp, 1);

      auto loweringConfig =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
      if (loweringConfig &&
          loweringConfig.getAttributes().contains("promote_c")) {
        builder.setInsertionPointAfter(linalgOp);
        promoteResult(builder, linalgOp, 0);
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
