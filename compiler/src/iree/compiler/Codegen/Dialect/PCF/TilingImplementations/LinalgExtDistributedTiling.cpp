// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/TilingImplementations/RegisterAll.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler::IREE::PCF {
namespace {

/// Helper to read a tile from an sref or extract from a tensor.
static Value readTile(OpBuilder &b, Location loc, Value value,
                      ArrayRef<OpFoldResult> tileOffsets,
                      ArrayRef<OpFoldResult> tileSizes) {
  if (ShapedRefType srefType = dyn_cast<ShapedRefType>(value.getType())) {
    int64_t rank = srefType.getRank();
    SmallVector<int64_t> staticSizes;
    staticSizes.reserve(rank);
    for (OpFoldResult size : tileSizes) {
      if (Attribute attr = dyn_cast<Attribute>(size)) {
        staticSizes.push_back(cast<IntegerAttr>(attr).getInt());
      } else {
        staticSizes.push_back(ShapedType::kDynamic);
      }
    }
    RankedTensorType resultType =
        RankedTensorType::get(staticSizes, srefType.getElementType());
    SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
    return ReadSliceOp::create(b, loc, resultType, value, tileOffsets,
                               tileSizes, strides);
  }
  int64_t rank = cast<RankedTensorType>(value.getType()).getRank();
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  return tensor::ExtractSliceOp::create(b, loc, value, tileOffsets, tileSizes,
                                        strides);
}

//===----------------------------------------------------------------------===//
// ScatterOp distributed tiling.
//===----------------------------------------------------------------------===//

/// ScatterOp scatters updates into a destination at positions determined by
/// indices. When the destination is an sref, the scatter writes directly to
/// it — the op accepts sref as its `original` operand and produces no
/// tensor results.
struct ScatterOpDistributedTilingModel
    : public PCFTilingOpInterface::ExternalModel<
          ScatterOpDistributedTilingModel, IREE::LinalgExt::ScatterOp> {

  SmallVector<unsigned> getTileableOperandIndices(Operation *op) const {
    // updates (0) and original (2) are tileable. indices (1) are tiled along
    // batch dims internally.
    return {0, 2};
  }

  FailureOr<TilingResult> getDistributedImplementation(
      Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      ArrayRef<DistributedOperandInfo> operandInfo,
      ArrayRef<DistributedResultInfo> resultInfo) const {
    auto scatterOp = cast<IREE::LinalgExt::ScatterOp>(op);
    Location loc = op->getLoc();

    // Tile updates.
    Value updates = operandInfo[0].value;
    if (!operandInfo[0].isTile) {
      int64_t updateRank = scatterOp.getUpdateType().getRank();
      SmallVector<OpFoldResult> updateOffsets(offsets.begin(), offsets.end());
      SmallVector<OpFoldResult> updateSizes(sizes.begin(), sizes.end());
      while (static_cast<int64_t>(updateOffsets.size()) < updateRank) {
        int64_t dim = updateOffsets.size();
        updateOffsets.push_back(b.getIndexAttr(0));
        updateSizes.push_back(
            IREE::LinalgExt::getDim(b, loc, updates, dim));
      }
      updates = readTile(b, loc, updates, updateOffsets, updateSizes);
    }

    // Tile indices along batch dims.
    Value indices = operandInfo[1].value;
    if (!operandInfo[1].isTile) {
      int64_t indicesRank = scatterOp.getIndicesType().getRank();
      int64_t batchRank = scatterOp.getBatchRank();
      SmallVector<OpFoldResult> indicesOffsets(offsets.begin(),
                                               offsets.begin() + batchRank);
      SmallVector<OpFoldResult> indicesSizes(sizes.begin(),
                                             sizes.begin() + batchRank);
      if (batchRank != indicesRank) {
        indicesOffsets.push_back(b.getIndexAttr(0));
        indicesSizes.push_back(b.getIndexAttr(scatterOp.getIndexDepth()));
      }
      indices = readTile(b, loc, indices, indicesOffsets, indicesSizes);
    }

    // Handle destination. If it's an sref, pass it directly — the scatter
    // op now accepts sref as its `original` operand and writes to it
    // without producing a tensor result.
    Value original = operandInfo[2].value;
    bool destIsSref = isa<ShapedRefType>(original.getType());

    SmallVector<Type> resultTypes;
    if (!destIsSref) {
      resultTypes.push_back(original.getType());
    }

    Operation *tiledOp =
        mlir::clone(b, op, resultTypes, {updates, indices, original});

    SmallVector<Value> tiledValues;
    if (destIsSref) {
      // No results — the scatter wrote directly to the sref.
    } else if (!tiledOp->getResults().empty()) {
      tiledValues.push_back(tiledOp->getResult(0));
    }

    return TilingResult{{tiledOp}, tiledValues, /*generatedSlices=*/{}};
  }

  // Reduction methods not applicable to scatter ops.
  SmallVector<Type>
  getReductionIterArgTypes(Operation *op, OpBuilder &b,
                           const MultiLevelTilingParams &params) const {
    return {};
  }
  SmallVector<Value>
  emitReductionInit(Operation *op, OpBuilder &b, ValueRange resultSrefs,
                    ArrayRef<OpFoldResult> offsets,
                    ArrayRef<OpFoldResult> sizes,
                    const MultiLevelTilingParams &params) const {
    return {};
  }
  void emitReductionWriteback(Operation *op, OpBuilder &b,
                              ValueRange reductionResults,
                              ValueRange resultSrefs,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes,
                              const MultiLevelTilingParams &params) const {}
};

//===----------------------------------------------------------------------===//
// MapStoreOp distributed tiling.
//===----------------------------------------------------------------------===//

/// MapStoreOp stores input elements to output at mapped positions.
/// When the output is an sref, the op writes directly to it.
struct MapStoreOpDistributedTilingModel
    : public PCFTilingOpInterface::ExternalModel<
          MapStoreOpDistributedTilingModel, IREE::LinalgExt::MapStoreOp> {

  SmallVector<unsigned> getTileableOperandIndices(Operation *op) const {
    // input (0) is tileable. output (1) is the destination.
    return {0};
  }

  FailureOr<TilingResult> getDistributedImplementation(
      Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      ArrayRef<DistributedOperandInfo> operandInfo,
      ArrayRef<DistributedResultInfo> resultInfo) const {
    Location loc = op->getLoc();

    // Tile input.
    Value tiledInput = operandInfo[0].value;
    if (!operandInfo[0].isTile) {
      tiledInput = readTile(b, loc, tiledInput, offsets, sizes);
    }

    // Handle output destination. If it's an sref, pass directly — the
    // map_store op accepts sref as its output and writes to it.
    Value output = operandInfo[1].value;
    bool destIsSref = isa<ShapedRefType>(output.getType());

    SmallVector<Type> resultTypes;
    if (!destIsSref) {
      resultTypes.push_back(output.getType());
    }

    Operation *tiledOp = mlir::clone(b, op, resultTypes, {tiledInput, output});
    auto tiledMapStoreOp = cast<IREE::LinalgExt::MapStoreOp>(tiledOp);

    // Compose the tiling offsets into the transformation body.
    auto indexTransformBuilder =
        [&](ArrayRef<BlockArgument> srcIndices) -> SmallVector<Value> {
      SmallVector<OpFoldResult> offsetIndices;
      AffineMap addMap =
          AffineMap::get(2, 0, {b.getAffineDimExpr(0) + b.getAffineDimExpr(1)});
      for (auto [srcIdx, offset] : llvm::zip_equal(srcIndices, offsets)) {
        offsetIndices.push_back(affine::makeComposedFoldedAffineApply(
            b, loc, addMap, {OpFoldResult(srcIdx), offset}));
      }
      return getValueOrCreateConstantIndexOp(b, loc, offsetIndices);
    };
    tiledMapStoreOp.insertTransformationAtStart(b, indexTransformBuilder,
                                                offsets.size());

    SmallVector<Value> tiledValues;
    if (destIsSref) {
      // No results — the map_store wrote directly to the sref.
    } else if (!tiledOp->getResults().empty()) {
      tiledValues.push_back(tiledOp->getResult(0));
    }

    return TilingResult{{tiledOp}, tiledValues, /*generatedSlices=*/{}};
  }

  // Reduction methods not applicable to map_store ops.
  SmallVector<Type>
  getReductionIterArgTypes(Operation *op, OpBuilder &b,
                           const MultiLevelTilingParams &params) const {
    return {};
  }
  SmallVector<Value>
  emitReductionInit(Operation *op, OpBuilder &b, ValueRange resultSrefs,
                    ArrayRef<OpFoldResult> offsets,
                    ArrayRef<OpFoldResult> sizes,
                    const MultiLevelTilingParams &params) const {
    return {};
  }
  void emitReductionWriteback(Operation *op, OpBuilder &b,
                              ValueRange reductionResults,
                              ValueRange resultSrefs,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes,
                              const MultiLevelTilingParams &params) const {}
};

} // namespace

void attachLinalgExtDistributedTilingModels(MLIRContext *ctx) {
  IREE::LinalgExt::ScatterOp::attachInterface<ScatterOpDistributedTilingModel>(
      *ctx);
  IREE::LinalgExt::MapStoreOp::attachInterface<
      MapStoreOpDistributedTilingModel>(*ctx);
}

void registerLinalgExtDistributedTilingModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::LinalgExt::IREELinalgExtDialect *dialect) {
        attachLinalgExtDistributedTilingModels(ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::PCF
