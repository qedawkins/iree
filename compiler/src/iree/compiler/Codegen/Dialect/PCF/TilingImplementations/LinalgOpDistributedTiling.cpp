// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/TilingImplementations/RegisterAll.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::PCF;
using namespace mlir::linalg;

namespace {

/// Computes the offsets and sizes for a single operand tile given the
/// iteration domain offsets/sizes and the operand's indexing map.
/// Returns std::nullopt if the operand is not tiled (all mapped dims have
/// zero tile size).
static std::optional<std::pair<SmallVector<OpFoldResult>,
                               SmallVector<OpFoldResult>>>
computeOperandTilePosition(OpBuilder &b, Location loc, LinalgOp linalgOp,
                           OpOperand &opOperand,
                           ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes) {
  AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&opOperand);
  int64_t rank = indexingMap.getNumResults();
  SmallVector<OpFoldResult> operandOffsets;
  SmallVector<OpFoldResult> operandSizes;
  operandOffsets.reserve(rank);
  operandSizes.reserve(rank);

  for (AffineExpr expr : indexingMap.getResults()) {
    // Only handle simple dim expressions for now.
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      return std::nullopt;
    }
    unsigned pos = dimExpr.getPosition();
    operandOffsets.push_back(offsets[pos]);
    operandSizes.push_back(sizes[pos]);
  }

  return std::make_pair(operandOffsets, operandSizes);
}

/// Reads a tile from an sref using pcf.read_slice.
static Value readTileFromSref(OpBuilder &b, Location loc, Value sref,
                              ArrayRef<OpFoldResult> tileOffsets,
                              ArrayRef<OpFoldResult> tileSizes) {
  ShapedRefType srefType = cast<ShapedRefType>(sref.getType());
  int64_t rank = srefType.getRank();

  // Build the result tensor type with static sizes where possible.
  SmallVector<int64_t> staticSizes;
  staticSizes.reserve(rank);
  for (OpFoldResult size : tileSizes) {
    if (auto attr = dyn_cast<Attribute>(size)) {
      staticSizes.push_back(cast<IntegerAttr>(attr).getInt());
    } else {
      staticSizes.push_back(ShapedType::kDynamic);
    }
  }
  RankedTensorType resultType =
      RankedTensorType::get(staticSizes, srefType.getElementType());

  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  return ReadSliceOp::create(b, loc, resultType, sref,
                             tileOffsets, tileSizes, strides);
}

/// External model implementing PCFTilingOpInterface for all LinalgOps.
///
/// The implementation:
/// 1. For each operand: if it's an sref, compute tile position from the
///    indexing map and read via pcf.read_slice. If it's a tensor tile
///    already, use it directly.
/// 2. Clone the linalg op with the tensor tile operands.
/// 3. For each result: if destSref is set, write via pcf.write_slice.
template <typename LinalgOpTy>
struct LinalgOpDistributedTilingModel
    : public PCFTilingOpInterface::ExternalModel<
          LinalgOpDistributedTilingModel<LinalgOpTy>, LinalgOpTy> {

  SmallVector<unsigned> getTileableOperandIndices(Operation *op) const {
    // All shaped operands of a LinalgOp are tileable. Scalar operands
    // (e.g., the fill value in linalg.fill) are not.
    SmallVector<unsigned> indices;
    for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
      if (isa<ShapedType>(op->getOperand(i).getType())) {
        indices.push_back(i);
      }
    }
    return indices;
  }

  FailureOr<TilingResult> getDistributedImplementation(
      Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      ArrayRef<DistributedOperandInfo> operandInfo,
      ArrayRef<DistributedResultInfo> resultInfo) const {
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);

    // Build the tiled operand list. For each operand:
    // - If operandInfo says it's already a tile (isTile=true) and the value
    //   is a tensor, use it directly.
    // - If the value is an sref, compute the tile position from the indexing
    //   map and read the tile via pcf.read_slice.
    // - If the value is a non-shaped type (scalar), pass through.
    SmallVector<Value> tiledOperands;
    tiledOperands.reserve(operandInfo.size());
    for (auto [i, info] : llvm::enumerate(operandInfo)) {
      Value operandValue = info.value;
      Type operandType = operandValue.getType();

      // Scalar operand — pass through.
      if (!isa<ShapedType, ShapedRefType>(operandType)) {
        tiledOperands.push_back(operandValue);
        continue;
      }

      // If already a tensor tile, use directly.
      if (info.isTile && isa<RankedTensorType>(operandType)) {
        tiledOperands.push_back(operandValue);
        continue;
      }

      // Need to extract a tile. Compute tile position from indexing map.
      OpOperand &opOperand = linalgOp->getOpOperand(i);
      std::optional<std::pair<SmallVector<OpFoldResult>,
                              SmallVector<OpFoldResult>>>
          tilePos = computeOperandTilePosition(b, loc, linalgOp, opOperand,
                                               offsets, sizes);
      if (!tilePos) {
        return failure();
      }
      auto &[tileOffsets, tileSizes] = *tilePos;

      // Use a plain OpBuilder for creating slice ops to avoid issues
      // with the greedy rewriter's notification/folding system.
      OpBuilder sliceBuilder(b.getContext());
      sliceBuilder.setInsertionPoint(b.getInsertionBlock(),
                                     b.getInsertionPoint());
      if (isa<ShapedRefType>(operandType)) {
        // Read from sref.
        Value tile = readTileFromSref(sliceBuilder, loc, operandValue,
                                      tileOffsets, tileSizes);
        tiledOperands.push_back(tile);
      } else {
        // Tensor — extract slice.
        RankedTensorType tensorType = cast<RankedTensorType>(operandType);
        int64_t rank = tensorType.getRank();
        SmallVector<OpFoldResult> strides(rank, sliceBuilder.getIndexAttr(1));
        Value tile = tensor::ExtractSliceOp::create(
            sliceBuilder, loc, operandValue, tileOffsets, tileSizes, strides);
        tiledOperands.push_back(tile);
      }
    }

    // Compute result types from the tiled DPS init operands.
    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(linalgOp, tiledOperands);

    // Clone the linalg op with tiled operands.
    // Use a plain OpBuilder for cloning to avoid greedy rewriter
    // notification issues during region cloning.
    OpBuilder plainBuilder(b.getContext());
    plainBuilder.setInsertionPoint(b.getInsertionBlock(),
                                   b.getInsertionPoint());
    Operation *tiledOp =
        clone(plainBuilder, linalgOp, resultTensorTypes, tiledOperands);
    offsetIndices(b, cast<LinalgOp>(tiledOp), offsets);

    // Handle results. For each result, either write to dest sref or return
    // the tile value.
    SmallVector<Value> tiledValues;
    SmallVector<Operation *> tiledOps = {tiledOp};
    for (auto [i, result] : llvm::enumerate(tiledOp->getResults())) {
      if (resultInfo[i].destSref) {
        // Compute result tile position from the output indexing map.
        OpOperand *initOperand = linalgOp.getDpsInitOperand(i);
        AffineMap indexingMap = linalgOp.getMatchingIndexingMap(initOperand);

        SmallVector<OpFoldResult> resultOffsets;
        SmallVector<OpFoldResult> resultSizes;
        for (AffineExpr expr : indexingMap.getResults()) {
          auto dimExpr = cast<AffineDimExpr>(expr);
          unsigned pos = dimExpr.getPosition();
          resultOffsets.push_back(offsets[pos]);
          resultSizes.push_back(sizes[pos]);
        }

        int64_t resultRank = cast<ShapedType>(result.getType()).getRank();
        SmallVector<OpFoldResult> strides(resultRank, b.getIndexAttr(1));
        // Use a plain builder to avoid greedy rewriter notification issues.
        OpBuilder writeBuilder(b.getContext());
        writeBuilder.setInsertionPointAfter(tiledOp);
        WriteSliceOp::create(writeBuilder, loc, result,
                             resultInfo[i].destSref, resultOffsets,
                             resultSizes, strides);
        tiledValues.push_back(Value());
      } else {
        tiledValues.push_back(result);
      }
    }

    return TilingResult{tiledOps, tiledValues, /*generatedSlices=*/{}};
  }
};

} // namespace

/// Register for a single linalg op.
template <typename OpType>
static void registerOneLinalgOp(MLIRContext *ctx) {
  OpType::template attachInterface<LinalgOpDistributedTilingModel<OpType>>(
      *ctx);
}

/// Variadic helper to register for all linalg ops.
template <typename... OpTypes>
static void registerAllLinalgOps(MLIRContext *ctx) {
  (registerOneLinalgOp<OpTypes>(ctx), ...);
}

#define GET_OP_LIST

namespace mlir::iree_compiler::IREE::PCF {

void attachLinalgDistributedTilingModels(MLIRContext *ctx) {
  // Register for linalg.generic.
  registerOneLinalgOp<linalg::GenericOp>(ctx);
  // Register for all named structured linalg ops.
  registerAllLinalgOps<
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >(ctx);
}

void registerLinalgDistributedTilingModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
        attachLinalgDistributedTilingModels(ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::PCF
