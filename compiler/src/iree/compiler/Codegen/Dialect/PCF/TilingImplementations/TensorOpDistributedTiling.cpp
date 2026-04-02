// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/TilingImplementations/RegisterAll.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::PCF;

namespace {

//===----------------------------------------------------------------------===//
// PadOp distributed tiling.
//===----------------------------------------------------------------------===//

/// PadOp has one shaped operand (the source tensor) and produces one result.
/// The iteration domain is the result shape.
///
/// If the source operand is an sref, we read the needed source tile from it.
/// If destSref is set, we write the padded tile to it.
struct PadOpDistributedTilingModel
    : public PCFTilingOpInterface::ExternalModel<PadOpDistributedTilingModel,
                                                  tensor::PadOp> {

  SmallVector<unsigned> getTileableOperandIndices(Operation *op) const {
    // PadOp has one shaped operand (the source tensor) at index 0.
    return {0};
  }

  FailureOr<TilingResult> getDistributedImplementation(
      Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      ArrayRef<DistributedOperandInfo> operandInfo,
      ArrayRef<DistributedResultInfo> resultInfo) const {
    Location loc = op->getLoc();

    // If the source operand is an sref, we need to read the appropriate
    // source tile from it. The pad op's source tile position is derived
    // from the result tile position minus the padding.
    Value sourceValue = operandInfo[0].value;
    if (isa<ShapedRefType>(sourceValue.getType())) {
      // For pad ops, the source tile is the result tile minus padding.
      // Delegate to the existing TilingInterface to compute the tile,
      // but we need to replace the source first.
      // TODO: Implement proper sref source handling for PadOp.
      return failure();
    }

    // If the source is already a tensor (tile or full), delegate to the
    // existing TilingInterface implementation.
    auto tilingIface = cast<TilingInterface>(op);
    FailureOr<TilingResult> tiledResult =
        tilingIface.getTiledImplementation(b, offsets, sizes);
    if (failed(tiledResult)) {
      return failure();
    }

    // Handle sref destination writing if requested.
    if (!tiledResult->tiledValues.empty() && resultInfo[0].destSref) {
      Value result = tiledResult->tiledValues[0];
      int64_t rank = cast<ShapedType>(result.getType()).getRank();
      SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
      WriteSliceOp::create(b, loc, result, resultInfo[0].destSref,
                           SmallVector<OpFoldResult>(offsets),
                           SmallVector<OpFoldResult>(sizes), strides);
      tiledResult->tiledValues[0] = Value();
    }

    return tiledResult;
  }

  // Reduction methods not applicable to pad ops.
  SmallVector<Type> getReductionIterArgTypes(
      Operation *op, OpBuilder &b,
      const MultiLevelTilingParams &params) const {
    return {};
  }
  SmallVector<Value> emitReductionInit(
      Operation *op, OpBuilder &b, ValueRange resultSrefs,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      const MultiLevelTilingParams &params) const {
    return {};
  }
  void emitReductionWriteback(
      Operation *op, OpBuilder &b, ValueRange reductionResults,
      ValueRange resultSrefs, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      const MultiLevelTilingParams &params) const {}
};

} // namespace

namespace mlir::iree_compiler::IREE::PCF {

void attachTensorDistributedTilingModels(MLIRContext *ctx) {
  tensor::PadOp::attachInterface<PadOpDistributedTilingModel>(*ctx);
}

void registerTensorDistributedTilingModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
        attachTensorDistributedTilingModels(ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::PCF
