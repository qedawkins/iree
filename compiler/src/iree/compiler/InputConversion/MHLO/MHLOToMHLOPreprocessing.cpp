// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include <random>

#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

namespace {

static bool isIota(ArrayRef<int64_t> array) {
  for (auto it : llvm::enumerate(array)) {
    if (it.index() != it.value()) {
      return false;
    }
  }
  return true;
}

static DenseIntElementsAttr make1DElementsAttr(OpBuilder &b,
                                               ArrayRef<int64_t> integers) {
  auto type = RankedTensorType::get({static_cast<int64_t>(integers.size())},
                                    b.getIntegerType(64));
  return DenseIntElementsAttr::get(type, integers);
}

static DenseIntElementsAttr make1DElementsAttr(OpBuilder &b, int64_t start,
                                               int64_t num) {
  return make1DElementsAttr(
      b, llvm::to_vector<4>(llvm::seq<int64_t>(start, start + num)));
}

static Value getF32Const(ImplicitLocOpBuilder b, ArrayRef<int64_t> shapes,
                         ArrayRef<float> values) {
  RankedTensorType ty = RankedTensorType::get(shapes, b.getF32Type());
  return b.create<mhlo::ConstantOp>(DenseFPElementsAttr::get(ty, values))
      .getResult();
}

// Guarantee that the input dimensions are ordered batch, spatial_dims, feature
// dim.
class ReorderConvOpInputDimensions
    : public OpRewritePattern<mhlo::ConvolutionOp> {
 public:
  using OpRewritePattern<mhlo::ConvolutionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = op.getLhs().getType().cast<ShapedType>();
    auto lhsShape = lhsType.getShape();
    if (!lhsType.hasRank()) {
      return failure();
    }

    auto dimensionNumbers = op.getDimensionNumbers();
    auto spatialDims = dimensionNumbers.getInputSpatialDimensions();

    // Compute the permutation required to create a standard order.
    llvm::SmallVector<int64_t, 4> permutations;
    permutations.push_back(dimensionNumbers.getInputBatchDimension());
    permutations.append(spatialDims.begin(), spatialDims.end());
    permutations.push_back(dimensionNumbers.getInputFeatureDimension());

    // If the permutation is iota then no reordering is required.
    if (isIota(permutations)) {
      return failure();
    }

    llvm::SmallVector<int64_t, 4> transposeShape;
    for (auto p : permutations) {
      transposeShape.push_back(lhsShape[p]);
    }

    auto transposed = rewriter.create<mhlo::TransposeOp>(
        op.getLoc(),
        RankedTensorType::get(transposeShape, lhsType.getElementType()),
        op.getLhs(), rewriter.getI64TensorAttr(permutations));

    llvm::SmallVector<int64_t, 4> newSpatialDimensions(spatialDims.size());
    std::iota(newSpatialDimensions.begin(), newSpatialDimensions.end(), 1);

    auto newDimensionNumbers = mhlo::ConvDimensionNumbersAttr::get(
        op.getContext(),
        /*input_batch_dimension=*/0,
        /*input_feature_dimension=*/newSpatialDimensions.size() + 1,
        /*input_spatial_dimensions=*/newSpatialDimensions,
        dimensionNumbers.getKernelInputFeatureDimension(),
        dimensionNumbers.getKernelOutputFeatureDimension(),
        dimensionNumbers.getKernelSpatialDimensions(),
        dimensionNumbers.getOutputBatchDimension(),
        dimensionNumbers.getOutputFeatureDimension(),
        dimensionNumbers.getOutputSpatialDimensions());

    SmallVector<Value, 2> operands = {transposed, op.getRhs()};
    auto newConv = rewriter.create<mhlo::ConvolutionOp>(
        op.getLoc(), op.getType(), operands, op->getAttrs());
    newConv.setDimensionNumbersAttr(newDimensionNumbers);
    rewriter.replaceOp(op, newConv.getResult());

    return success();
  }
};

struct ReorderConvOpKernelDimensions
    : public OpRewritePattern<mhlo::ConvolutionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    auto kernel = op.getRhs();
    auto kernelType = kernel.getType().cast<ShapedType>();
    if (!kernelType.hasRank()) return failure();
    auto kernelShape = kernelType.getShape();

    auto dimensionNumbers = op.getDimensionNumbers();

    auto spatialDims = dimensionNumbers.getKernelSpatialDimensions();

    auto inputFeatureDimension =
        dimensionNumbers.getKernelInputFeatureDimension();
    auto outputFeatureDimension =
        dimensionNumbers.getKernelOutputFeatureDimension();

    // Compute the permutation for the transpose.
    llvm::SmallVector<int64_t, 4> permutation(spatialDims.begin(),
                                              spatialDims.end());
    permutation.push_back(inputFeatureDimension);
    permutation.push_back(outputFeatureDimension);

    // If the permutation is iota, then no transpose is required.
    if (isIota(permutation)) return failure();

    llvm::SmallVector<int64_t, 4> transposeShape;
    for (auto perm : permutation) {
      transposeShape.push_back(kernelShape[perm]);
    }

    llvm::SmallVector<int64_t, 4> newSpatialDimensions(spatialDims.size());
    std::iota(newSpatialDimensions.begin(), newSpatialDimensions.end(), 0);

    auto transposeKernel = rewriter.create<mhlo::TransposeOp>(
        op.getLoc(),
        RankedTensorType::get(transposeShape, kernelType.getElementType()),
        kernel, rewriter.getI64TensorAttr(permutation));

    auto newDimensionNumbers = mhlo::ConvDimensionNumbersAttr::get(
        op.getContext(), dimensionNumbers.getInputBatchDimension(),
        dimensionNumbers.getInputFeatureDimension(),
        dimensionNumbers.getInputSpatialDimensions(),
        /*kernel_input_feature_dimension=*/
        newSpatialDimensions.size(),
        /*kernel_output_feature_dimension=*/
        newSpatialDimensions.size() + 1, newSpatialDimensions,
        dimensionNumbers.getOutputBatchDimension(),
        dimensionNumbers.getOutputFeatureDimension(),
        dimensionNumbers.getOutputSpatialDimensions());

    SmallVector<Value, 2> operands = {op.getLhs(), transposeKernel};
    mhlo::ConvolutionOp newConv = rewriter.create<mhlo::ConvolutionOp>(
        op.getLoc(), op.getType(), operands, op->getAttrs());
    newConv.setDimensionNumbersAttr(newDimensionNumbers);

    rewriter.replaceOp(op, {newConv.getResult()});
    return success();
  }
};

// Guarantee that the output dimensions are ordered batch, spatial_dims, feature
// dim.
class ReorderConvOpOutputDimensions
    : public OpRewritePattern<mhlo::ConvolutionOp> {
 public:
  using OpRewritePattern<mhlo::ConvolutionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType().cast<ShapedType>();
    auto resultShape = resultType.getShape();
    if (!resultType.hasRank()) {
      return failure();
    }

    auto dimensionNumbers = op.getDimensionNumbers();
    auto spatialDims = dimensionNumbers.getOutputSpatialDimensions();

    // Compute the permutation to transpose to an ordered output.
    llvm::SmallVector<int64_t, 4> permutation;
    permutation.push_back(dimensionNumbers.getOutputBatchDimension());
    permutation.append(spatialDims.begin(), spatialDims.end());
    permutation.push_back(dimensionNumbers.getOutputFeatureDimension());

    // If the permutation is iota then no reordering is required.
    if (isIota(permutation)) {
      return failure();
    }

    // Compute what the new conv shape should be.
    llvm::SmallVector<int64_t, 4> convShape;
    for (auto p : permutation) {
      convShape.push_back(resultShape[p]);
    }

    // Compute the inverse transpose to unordered and ordered output.
    llvm::SmallVector<int64_t, 4> invertPermutation(permutation.size());
    for (auto it : llvm::enumerate(permutation)) {
      invertPermutation[it.value()] = it.index();
    }

    llvm::SmallVector<int64_t, 4> newSpatialDimensions(spatialDims.size());
    std::iota(newSpatialDimensions.begin(), newSpatialDimensions.end(), 1);

    auto newDimensionNumbers = mhlo::ConvDimensionNumbersAttr::get(
        op.getContext(), dimensionNumbers.getInputBatchDimension(),
        dimensionNumbers.getInputFeatureDimension(),
        dimensionNumbers.getInputSpatialDimensions(),
        dimensionNumbers.getKernelInputFeatureDimension(),
        dimensionNumbers.getKernelOutputFeatureDimension(),
        dimensionNumbers.getKernelSpatialDimensions(),
        /*output_batch_dimension=*/0,
        /*output_feature_dimension=*/newSpatialDimensions.size() + 1,
        /*output_spatial_dimensions=*/newSpatialDimensions);

    SmallVector<Value, 2> operands = {op.getLhs(), op.getRhs()};
    auto newConv = rewriter.create<mhlo::ConvolutionOp>(
        op.getLoc(),
        RankedTensorType::get(convShape, resultType.getElementType()), operands,
        op->getAttrs());
    newConv.setDimensionNumbersAttr(newDimensionNumbers);

    auto transposed = rewriter.create<mhlo::TransposeOp>(
        op.getLoc(), resultType, newConv,
        rewriter.getI64TensorAttr(invertPermutation));

    rewriter.replaceOp(op, transposed.getResult());
    return success();
  }
};

bool isConsecutive(ArrayRef<int64_t> array) {
  for (int i = 1; i < array.size(); ++i) {
    if (array[i] - array[i - 1] != 1) return false;
  }
  return true;
}

// Rewrites mhlo.dot_general so lhs contraction dimensions are innermost and rhs
// contraction dimensions are dims right after batch dimension. The pattern
// inserts transposes so the dot_general always has the form:
// {batch_dims, parallel_dims, contraction_dims}.
//   {batch_dims, contraction_dims, parallel_dims}
// After that, batch_dims, contraction_dims, parallel_dims are
// in consecutive order and not spliting the domain. This pattern inserts
// reshapes to collapse consecutive reduction and parallel dims to always
// generate a rank-3 dot_general op.
class TransposeReshapeGenericDotGeneral
    : public OpRewritePattern<mhlo::DotGeneralOp> {
 public:
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;

  Value TransposeIfNonConsecutive(OpBuilder &b, Location loc, Value src,
                                  ArrayRef<int64_t> targetOrder) const {
    if (isConsecutive(targetOrder)) return src;
    auto type = src.getType().cast<RankedTensorType>();
    SmallVector<int64_t, 4> transposeShape;
    for (auto i : targetOrder) {
      transposeShape.push_back(type.getDimSize(i));
    }
    return b.create<mhlo::TransposeOp>(
        loc, RankedTensorType::get(transposeShape, type.getElementType()), src,
        b.getI64TensorAttr(targetOrder));
  }

  Value ReshapeIfMorethan3D(OpBuilder &b, Location loc, Value src,
                            size_t dimsBorder0, size_t dimsBorder1) const {
    auto type = src.getType().cast<RankedTensorType>();
    if (type.getRank() <= 3) return src;
    auto shape = type.getShape();
    SmallVector<int64_t, 4> result_shape = {
        std::accumulate(shape.begin(), shape.begin() + dimsBorder0, 1,
                        std::multiplies<int64_t>()),
        std::accumulate(shape.begin() + dimsBorder0,
                        shape.begin() + dimsBorder1, 1,
                        std::multiplies<int64_t>()),
        std::accumulate(shape.begin() + dimsBorder1, shape.end(), 1,
                        std::multiplies<int64_t>())};
    return b.create<mhlo::ReshapeOp>(
        loc, RankedTensorType::get(result_shape, type.getElementType()), src);
  }

  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsShapeType = op.getLhs().getType().dyn_cast<RankedTensorType>();
    auto rhsShapeType = op.getRhs().getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!lhsShapeType || !rhsShapeType || !resultType) return failure();

    // TODO(jpienaar): This pattern is not safe for dynamic shapes and seems to
    // be (now) redundant with later pass that does handle them. To decouple
    // fixing and verifying redundant, this just limits to static shapes and
    // then will remove this in follow up.
    if (!lhsShapeType.hasStaticShape() || !rhsShapeType.hasStaticShape())
      return failure();

    SmallVector<int64_t> lhsTargetOrder, rhsTargetOrder;
    mhlo::DotDimensionNumbersAttr dimNumbers = op.getDotDimensionNumbers();
    auto lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();

    // No contraction dims means this can be represented as a mul.
    if (lhsContractingDims.size() == 0 || rhsContractingDims.size() == 0)
      return rewriter.notifyMatchFailure(op, "can be represented as mhlo.mul");

    // No batching dimensions means this can be represented a dot.
    if (lhsBatchingDims.size() == 0 || rhsBatchingDims.size() == 0)
      return rewriter.notifyMatchFailure(op, "can be represented as mhlo.dot");

    SmallVector<bool> isLhsParallel(lhsShapeType.getRank(), true);
    for (auto i : lhsBatchingDims) {
      lhsTargetOrder.push_back(i);
      isLhsParallel[i] = false;
    }
    for (auto i : lhsContractingDims) {
      isLhsParallel[i] = false;
    }
    for (int64_t i = 0, e = lhsShapeType.getRank(); i < e; ++i) {
      if (isLhsParallel[i]) {
        lhsTargetOrder.push_back(i);
      }
    }
    for (auto i : lhsContractingDims) {
      lhsTargetOrder.push_back(i);
    }

    SmallVector<bool> isRhsParallel(rhsShapeType.getRank(), true);

    for (auto i : rhsBatchingDims) {
      rhsTargetOrder.push_back(i);
      isRhsParallel[i] = false;
    }
    for (auto i : rhsContractingDims) {
      rhsTargetOrder.push_back(i);
      isRhsParallel[i] = false;
    }
    for (int64_t i = 0, e = rhsShapeType.getRank(); i < e; ++i) {
      if (isRhsParallel[i]) {
        rhsTargetOrder.push_back(i);
      }
    }

    Value lhs = TransposeIfNonConsecutive(rewriter, op.getLoc(), op.getLhs(),
                                          lhsTargetOrder);
    Value rhs = TransposeIfNonConsecutive(rewriter, op.getLoc(), op.getRhs(),
                                          rhsTargetOrder);

    // The dimensions of this will always be transposed into {batch_dims,
    // parallel_dims, contraction_dims}, and the
    // following logic is based on this assumption.
    // TODO(#7443): If we consider transpose performance, the above assumptions
    // may not be true.
    int64_t numLhsContractionDims = lhsContractingDims.size();
    int64_t lhsContractionBase = lhsShapeType.getRank() - numLhsContractionDims;
    int64_t rhsContractionBase = rhsBatchingDims.size();
    int64_t numRhsContractionDims =
        rhsContractionBase + rhsContractingDims.size();

    lhs = ReshapeIfMorethan3D(rewriter, op.getLoc(), lhs,
                              lhsBatchingDims.size(), lhsContractionBase);
    rhs = ReshapeIfMorethan3D(rewriter, op.getLoc(), rhs,
                              rhsBatchingDims.size(), numRhsContractionDims);

    if (lhs == op.getLhs() && rhs == op.getRhs())
      return rewriter.notifyMatchFailure(op, "already in canonical form");

    auto dimensionNumbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatchingDimensions=*/0,
        /*rhsBatchingDimensions=*/0,
        /*lhsContractingDimensions=*/
        lhs.getType().cast<ShapedType>().getRank() - 1,
        /*rhsContractingDimensions=*/1);
    auto lhsNewType = lhs.getType().cast<RankedTensorType>();
    auto rhsNewType = rhs.getType().cast<RankedTensorType>();

    // if lhs's shape or rhs's shape has collapsed, we need reshape the result
    bool needReshapeResult = lhsNewType.getRank() < lhsShapeType.getRank() ||
                             rhsNewType.getRank() < rhsShapeType.getRank();
    // batching、lhs parallel、rhs parallel this order is a convension
    SmallVector<int64_t, 4> newShape = {lhsNewType.getShape()[0],
                                        lhsNewType.getShape()[1]};
    if (rhsNewType.getRank() > 2) newShape.push_back(rhsNewType.getDimSize(2));

    auto newResultType =
        needReshapeResult
            ? RankedTensorType::get(newShape, resultType.getElementType())
            : op.getType();

    auto newOp = rewriter.create<mhlo::DotGeneralOp>(
        op.getLoc(), newResultType, lhs, rhs, dimensionNumbers,
        op.getPrecisionConfigAttr());

    // Copy over unknown attributes as we currently rely on it to let user tune
    // lowering parameters.
    ArrayRef<StringRef> odsAttrs = op.getAttributeNames();
    for (NamedAttribute kv : op->getAttrs()) {
      if (!llvm::is_contained(odsAttrs, kv.getName().getValue())) {
        newOp->setAttr(kv.getName(), kv.getValue());
      }
    }

    Value result = newOp.getResult();
    if (needReshapeResult) {
      result =
          rewriter.create<mhlo::ReshapeOp>(op.getLoc(), resultType, result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ScatterInt64Indices : public OpRewritePattern<mhlo::ScatterOp> {
  using OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                PatternRewriter &rewriter) const final {
    auto indices = op.getScatterIndices();
    auto indicesTy = indices.getType();
    auto indicesETy = indicesTy.getElementType();
    if (indicesETy.isInteger(32))
      return rewriter.notifyMatchFailure(op, "already has i32 index type");

    if (!indicesTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "cannot validate legal size");

    uint64_t maxSize = std::numeric_limits<int32_t>::max();
    if (indicesETy.getIntOrFloatBitWidth() > 32) {
      for (int i = 0, s = indicesTy.getRank(); i < s; ++i) {
        if (indicesTy.getDimSize(i) > maxSize) {
          return rewriter.notifyMatchFailure(op, "index may exceed i32 max");
        }
      }
    }

    indices = rewriter.create<mhlo::ConvertOp>(
        op.getLoc(), indicesTy.clone(rewriter.getI32Type()), indices);

    auto newScatter = rewriter.create<mhlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), indices,
        op.getUpdates(), op.getScatterDimensionNumbers(),
        op.getIndicesAreSorted(), op.getUniqueIndices());

    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());

    return success();
  }
};

// If the indices tensor has an implicit index vector dim we expand and make it
// an explicit dim.
struct ScatterImplicitIndex : public OpRewritePattern<mhlo::ScatterOp> {
  using OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                PatternRewriter &rewriter) const final {
    auto dimNumbers = op.getScatterDimensionNumbers();
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    auto indices = op.getScatterIndices();
    auto indicesTy = indices.getType().cast<ShapedType>();

    // Check indices vector has an implicit dim.
    if (indexVectorDim != indicesTy.getRank()) {
      return rewriter.notifyMatchFailure(op, "no implicit index dim");
    }

    // Materialize the implicit indices dim.
    SmallVector<ReassociationExprs, 4> reassociationMap;
    reassociationMap.resize(indicesTy.getRank());
    SmallVector<int64_t> newShape;
    for (int i = 0, s = indicesTy.getRank(); i < s; i++) {
      reassociationMap[i].push_back(rewriter.getAffineDimExpr(i));
      newShape.push_back(indicesTy.getDimSize(i));
    }
    if (!reassociationMap.empty()) {
      reassociationMap.back().push_back(
          rewriter.getAffineDimExpr(indicesTy.getRank()));
    }
    newShape.push_back(1);
    indicesTy = RankedTensorType::get(newShape, indicesTy.getElementType());
    indices = rewriter.create<tensor::ExpandShapeOp>(op.getLoc(), indicesTy,
                                                     indices, reassociationMap);

    auto newScatter = rewriter.create<mhlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), indices,
        op.getUpdates(), dimNumbers, op.getIndicesAreSorted(),
        op.getUniqueIndices());
    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
  }
};

struct ScatterImplicitBatch : public OpRewritePattern<mhlo::ScatterOp> {
  using OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

  static Value addUnitBatchDim(Location loc, Value value,
                               PatternRewriter &rewriter) {
    ShapedType valueTy = value.getType().cast<ShapedType>();
    if (!valueTy.hasRank()) return nullptr;

    // Materialize the implicit indices dim.
    SmallVector<ReassociationExprs, 4> reassociationMap(valueTy.getRank());
    if (!reassociationMap.empty()) {
      reassociationMap.front().push_back(rewriter.getAffineDimExpr(0));
    }

    SmallVector<int64_t> newShape = {1};
    for (int i = 0, s = valueTy.getRank(); i < s; i++) {
      reassociationMap[i].push_back(rewriter.getAffineDimExpr(i + 1));
      newShape.push_back(valueTy.getDimSize(i));
    }

    valueTy = RankedTensorType::get(newShape, valueTy.getElementType());
    return rewriter.create<tensor::ExpandShapeOp>(loc, valueTy, value,
                                                  reassociationMap);
  }

  LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                PatternRewriter &rewriter) const final {
    auto dimNumbers = op.getScatterDimensionNumbers();
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    auto indices = op.getScatterIndices().cast<Value>();
    auto indicesTy = indices.getType().dyn_cast<RankedTensorType>();

    // Check whether indices has no batch dimension.
    if (!indicesTy) return failure();
    if (indicesTy.getRank() != 1 || indexVectorDim != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "no implicit batch dimension to add.");
    }

    indices = addUnitBatchDim(op.getLoc(), indices, rewriter);
    if (!indices) {
      return rewriter.notifyMatchFailure(
          op, "Unable to add implicit batch dim to indice.");
    }

    llvm::SmallVector<int64_t> newUpdateWindowDims;
    for (auto dim : dimNumbers.getUpdateWindowDims()) {
      // Batch dimension is inserted at the start so window dimensions are shift
      // forwards.
      newUpdateWindowDims.push_back(dim + 1);
    }

    llvm::SmallVector<Value> updates;
    for (Value update : op.getUpdates()) {
      update = addUnitBatchDim(op.getLoc(), update, rewriter);
      if (!update) {
        return rewriter.notifyMatchFailure(
            op, "Unable to add implicit batch dim to update.");
      }
      updates.push_back(update);
    }

    auto newDimNumbers = mhlo::ScatterDimensionNumbersAttr::get(
        op.getContext(), newUpdateWindowDims,
        dimNumbers.getInsertedWindowDims(),
        dimNumbers.getScatterDimsToOperandDims(),
        dimNumbers.getIndexVectorDim() + 1);

    auto newScatter = rewriter.create<mhlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), indices, updates,
        newDimNumbers, op.getIndicesAreSorted(), op.getUniqueIndices());
    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
  }
};

struct ScatterCollapseBatch : public OpRewritePattern<mhlo::ScatterOp> {
  using OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

  static Value collapseBatchDims(Location loc, Value value, int64_t batchCount,
                                 PatternRewriter &rewriter) {
    auto valueTy = value.getType().dyn_cast<ShapedType>();
    if (!valueTy) return nullptr;

    SmallVector<ReassociationExprs, 4> reassociationMap(1);
    reassociationMap.reserve(valueTy.getRank() - batchCount + 1);
    int64_t batchSize = 1;
    for (int i = 0, s = batchCount; i < s; i++) {
      reassociationMap.front().push_back(rewriter.getAffineDimExpr(i));
      bool isDynamic =
          valueTy.isDynamicDim(i) || batchSize == ShapedType::kDynamic;
      batchSize =
          isDynamic ? ShapedType::kDynamic : valueTy.getDimSize(i) * batchSize;
    }

    SmallVector<int64_t> newShape = {batchSize};
    for (int i = batchCount, s = valueTy.getRank(); i < s; i++) {
      reassociationMap.push_back({rewriter.getAffineDimExpr(i)});
      newShape.push_back(valueTy.getDimSize(i));
    }

    valueTy = RankedTensorType::get(newShape, valueTy.getElementType());
    return rewriter.create<tensor::CollapseShapeOp>(loc, valueTy, value,
                                                    reassociationMap);
  }

  LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                PatternRewriter &rewriter) const final {
    auto dimNumbers = op.getScatterDimensionNumbers();
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    auto indices = op.getScatterIndices().cast<Value>();
    auto indicesTy = indices.getType().cast<ShapedType>();
    auto updatedWindowDims = dimNumbers.getUpdateWindowDims();

    if (!indicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "indices has unknown rank");
    }

    // Check for an explicit indice dimension.
    if (indexVectorDim != indicesTy.getRank() - 1) {
      return rewriter.notifyMatchFailure(op, "no explicit indices dimension");
    }

    // Check that there are multiple batch dimensions.
    if (indicesTy.getRank() < 3) {
      return rewriter.notifyMatchFailure(op, "no multiple batch dimensions");
    }

    const int64_t batchCount = indicesTy.getRank() - 1;
    for (auto it : llvm::enumerate(updatedWindowDims)) {
      if (it.index() != it.value() - batchCount) {
        return rewriter.notifyMatchFailure(
            op, "update windows should be at the end.");
      }
    }

    indices = collapseBatchDims(op.getLoc(), indices, batchCount, rewriter);
    if (!indices) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot collapse indices batch dims");
    }

    llvm::SmallVector<Value> updates;
    for (Value update : op.getUpdates()) {
      update = collapseBatchDims(op.getLoc(), update, batchCount, rewriter);
      if (!update) {
        return rewriter.notifyMatchFailure(op,
                                           "cannot collapse update batch dims");
      }
      updates.push_back(update);
    }

    llvm::SmallVector<int64_t> newUpdatedWindowDims;
    for (auto dim : updatedWindowDims) {
      newUpdatedWindowDims.push_back(dim - batchCount + 1);
    }

    auto newDimNumbers = mhlo::ScatterDimensionNumbersAttr::get(
        op.getContext(), newUpdatedWindowDims,
        dimNumbers.getInsertedWindowDims(),
        dimNumbers.getScatterDimsToOperandDims(),
        /*indexVectorDim=*/1);

    auto newScatter = rewriter.create<mhlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), indices, updates,
        newDimNumbers, op.getIndicesAreSorted(), op.getUniqueIndices());
    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
  }
};

// Ensure the batch dimensions of both the indices and updates are the first
// dimensions. If they are not, transpose them to the start.
struct ScatterBatchFirst : public OpRewritePattern<mhlo::ScatterOp> {
  using OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                PatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto dimNumbers = op.getScatterDimensionNumbers();

    // If the index vector dim is not implicitly or explicitly at the end
    // we need to transpose the batch dimensions to the start.
    Value indices = op.getScatterIndices();
    auto indicesTy = indices.getType().cast<ShapedType>();
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    if (indexVectorDim < indicesTy.getRank() - 1) {
      llvm::SmallVector<int64_t> perm;
      perm.reserve(indicesTy.getRank());
      for (int i = 0, s = indicesTy.getRank(); i < s; ++i)
        if (i != indexVectorDim) perm.push_back(i);

      if (perm.size() < indicesTy.getRank()) perm.push_back(indexVectorDim);

      llvm::SmallVector<int64_t> newShape;
      for (int i = 0, s = perm.size(); i < s; ++i)
        newShape.push_back(indicesTy.getDimSize(perm[i]));

      indices = builder.create<mhlo::TransposeOp>(
          indicesTy.clone(newShape), indices, builder.getI64TensorAttr(perm));
      indicesTy = indices.getType().cast<RankedTensorType>();
      indexVectorDim = indicesTy.getRank() - 1;
    }

    // Compute the permutation require to transpose the batch dimensions to
    // the beginning.
    auto updates = op.getUpdates();
    auto updates0 = updates.front();
    auto updates0Ty = updates0.getType().cast<ShapedType>();
    auto updatedWindowDims = dimNumbers.getUpdateWindowDims();

    // Determine which dimensions are batch dimensions.
    llvm::SmallVector<bool> isBatch(updates0Ty.getRank(), true);
    for (int i = 0, s = updatedWindowDims.size(); i < s; ++i)
      isBatch[updatedWindowDims[i]] = false;

    // Permute batch dimensions to the start of the update tensor.
    llvm::SmallVector<int64_t> updatePerm;
    updatePerm.reserve(updates0Ty.getRank());
    for (int i = 0, s = isBatch.size(); i < s; ++i)
      if (isBatch[i]) updatePerm.push_back(i);
    updatePerm.append(updatedWindowDims.begin(), updatedWindowDims.end());

    llvm::SmallVector<int64_t> newUpdatedWindowDims;
    int64_t batchCount = updates0Ty.getRank() - updatedWindowDims.size();
    for (int i = batchCount, s = updates0Ty.getRank(); i < s; i++)
      newUpdatedWindowDims.push_back(i);

    bool indicesChanged = indices != op.getScatterIndices();
    bool updatesChanged =
        llvm::any_of(llvm::enumerate(updatePerm),
                     [](auto it) { return it.index() != it.value(); });
    llvm::SmallVector<Value> newUpdates(updates.begin(), updates.end());
    if (updatesChanged) {
      for (Value &update : newUpdates) {
        auto updateTy = update.getType().cast<ShapedType>();
        llvm::SmallVector<int64_t> newShape;
        newShape.reserve(updateTy.getRank());
        for (int i = 0, s = updatePerm.size(); i < s; i++)
          newShape.push_back(updateTy.getDimSize(updatePerm[i]));
        update = builder.create<mhlo::TransposeOp>(
            updateTy.clone(newShape), update,
            builder.getI64TensorAttr(updatePerm));
      }
    }

    if (!indicesChanged && !updatesChanged)
      return rewriter.notifyMatchFailure(
          op, "batch dimensions are already leading");

    auto newDimNumbers = mhlo::ScatterDimensionNumbersAttr::get(
        op.getContext(), newUpdatedWindowDims,
        dimNumbers.getInsertedWindowDims(),
        dimNumbers.getScatterDimsToOperandDims(),
        /*indexVectorDim=*/indexVectorDim);

    auto newScatter = rewriter.create<mhlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), indices, newUpdates,
        newDimNumbers, op.getIndicesAreSorted(), op.getUniqueIndices());
    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
  }
};

// mhlo.scatter can materialize a unit dimension at both indexed dimensions or
// at unary dimensions in the destination matrix. linalg_ext.scatter only
// allows unit dimensions at indexed dimensions. This pattern inserts all
// unary dimensions that are not index dimensions to be compatible with
// linalg_ext.scatter.
//
// If converts an mhlo.scatter as below:
//  %result = "mhlo.scatter"(...) ({
//    indices_are_sorted = true,
//    scatter_dimension_numbers = #mhlo.scatter<
//            update_window_dims = [1],
//            inserted_window_dims = [0, 2],
//            scatter_dims_to_operand_dims = [0],
//            index_vector_dim = 1>,
//    unique_indices = true} :
//        (tensor<5x4x1xi32>, tensor<1x1xi32>, tensor<1x4xi32>)
//
// To:
//  %result = "mhlo.scatter"(...) ({
//    indices_are_sorted = true,
//    scatter_dimension_numbers = #mhlo.scatter<
//            update_window_dims = [1, 2],
//            inserted_window_dims = [0],
//            scatter_dims_to_operand_dims = [0],
//            index_vector_dim = 1>,
//     unique_indices = true} :
//        (tensor<5x4x1xi32>, tensor<1x1xi32>, tensor<1x4x1xi32>)
//  return %0 : tensor<5x4x1xi32>
struct ScatterMaterializeInsertedDim
    : public OpRewritePattern<mhlo::ScatterOp> {
  using OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                PatternRewriter &rewriter) const final {
    auto indices = op.getScatterIndices();
    auto operand = op.getInputs().front();
    auto indicesTy = indices.getType().cast<ShapedType>();
    auto operandTy = operand.getType().cast<ShapedType>();

    if (!operandTy.hasRank() || !indicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "operand/indices have no rank");
    }

    auto dimNumbers = op.getScatterDimensionNumbers();
    auto updateDims = dimNumbers.getUpdateWindowDims();

    if (indicesTy.getRank() != 2 || dimNumbers.getIndexVectorDim() != 1) {
      return rewriter.notifyMatchFailure(
          op, "indices is not of shape [batch, indices]");
    }

    if (!updateDims.empty() && updateDims.front() == 0) {
      return rewriter.notifyMatchFailure(
          op, "updates is not of shape [batch, ...]");
    }

    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    llvm::SmallVector<bool> isIndexDim(operandTy.getRank(), false);
    for (auto val : scatterDimsToOperandDims) {
      isIndexDim[val] = true;
    }

    int64_t firstNonIndex = 0;
    for (int64_t s = scatterDimsToOperandDims.size(); firstNonIndex < s;
         ++firstNonIndex) {
      if (!isIndexDim[firstNonIndex]) break;
    }

    llvm::SmallVector<bool> isInsertDims(operandTy.getRank(), false);
    for (auto val : dimNumbers.getInsertedWindowDims()) {
      isInsertDims[val] = true;
    }

    int64_t frontInsertedDims = 0;
    for (; frontInsertedDims < firstNonIndex; ++frontInsertedDims) {
      if (!isInsertDims[frontInsertedDims]) {
        break;
      }
    }

    llvm::ArrayRef<bool> toInsertDims =
        llvm::ArrayRef<bool>(isInsertDims).drop_front(frontInsertedDims);
    if (!llvm::any_of(toInsertDims, [](auto d) { return d; })) {
      return rewriter.notifyMatchFailure(op, "no dimensions to insert");
    }

    // Create a reassociation map that starts with the batch dims.
    SmallVector<ReassociationExprs, 4> reassociationMap;
    reassociationMap.push_back({rewriter.getAffineDimExpr(0)});

    for (auto it : llvm::enumerate(llvm::ArrayRef<bool>(toInsertDims))) {
      if (!it.value()) reassociationMap.push_back({});
      reassociationMap.back().push_back(
          rewriter.getAffineDimExpr(it.index() + 1));
    }

    llvm::SmallVector<Value> expandedUpdates;
    for (auto update : op.getUpdates()) {
      auto updatesTy = update.getType().cast<ShapedType>();

      llvm::SmallVector<int64_t> newShape;
      for (int i = 0, s = reassociationMap.size(); i < s; ++i) {
        newShape.push_back(updatesTy.getDimSize(i));
        for (int j = 1, s = reassociationMap[i].size(); j < s; ++j) {
          newShape.push_back(1);
        }
      }

      Value expandUpdate = rewriter.create<tensor::ExpandShapeOp>(
          op.getLoc(),
          RankedTensorType::get(newShape, updatesTy.getElementType()), update,
          reassociationMap);
      expandedUpdates.push_back(expandUpdate);
    }

    llvm::SmallVector<int64_t> newUpdatedWindowDims(toInsertDims.size());
    llvm::SmallVector<int64_t> newInsertedWindowDims(frontInsertedDims);
    std::iota(newUpdatedWindowDims.begin(), newUpdatedWindowDims.end(), 1);
    std::iota(newInsertedWindowDims.begin(), newInsertedWindowDims.end(), 0);

    auto newDimNumbers = mhlo::ScatterDimensionNumbersAttr::get(
        op.getContext(), newUpdatedWindowDims, newInsertedWindowDims,
        dimNumbers.getScatterDimsToOperandDims(),
        /*indexVectorDim=*/1);

    auto newScatter = rewriter.create<mhlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(),
        op.getScatterIndices(), expandedUpdates, newDimNumbers,
        op.getIndicesAreSorted(), op.getUniqueIndices());
    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
  }
};

// Traverse upward past common operations to see if the value came from a
// boolean tensor.
bool isFromBool(Value val) {
  while (true) {
    Operation *op = val.getDefiningOp();
    if (!op) return false;

    if (auto convertOp = dyn_cast<mhlo::ConvertOp>(op)) {
      auto inTy = convertOp.getOperand().getType().cast<ShapedType>();
      if (inTy.getElementType().isInteger(1)) {
        return true;
      }
      val = convertOp.getOperand();
      continue;
    }

    if (isa<mhlo::DynamicBroadcastInDimOp>(op) ||
        isa<mhlo::BroadcastInDimOp>(op) || isa<mhlo::BroadcastOp>(op)) {
      val = op->getOperand(0);
      continue;
    }

    return false;
  }
}

// Mhlo of non-finite values (e.g. NaN, inf) and 0.0 produce 0.0 for XLA. For
// linalg we need to conver these to select operations.
class MulCastOfBool : public OpRewritePattern<mhlo::MulOp> {
 public:
  using OpRewritePattern<mhlo::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::MulOp op,
                                PatternRewriter &rewriter) const override {
    auto resultTy = op.getType().cast<ShapedType>();
    if (!resultTy.getElementType().isa<FloatType>()) return failure();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    bool lhsIsBool = isFromBool(lhs);
    bool rhsIsBool = isFromBool(rhs);

    if (lhsIsBool == rhsIsBool) return failure();
    if (rhsIsBool) std::swap(lhs, rhs);

    Type eType = resultTy.getElementType();
    auto lhsTy = lhs.getType().cast<ShapedType>();
    Value lhsBool = rewriter.create<mhlo::ConvertOp>(
        op.getLoc(), lhsTy.clone(rewriter.getIntegerType(1)), lhs);
    Value zero = rewriter.create<mhlo::ConstantOp>(
        op.getLoc(), DenseElementsAttr::get(RankedTensorType::get({}, eType),
                                            rewriter.getZeroAttr(eType)));

    auto lhsShape = rewriter.create<shape::ShapeOfOp>(
        op.getLoc(),
        RankedTensorType::get({lhsTy.getRank()}, rewriter.getIndexType()), lhs);

    int64_t resultRank = resultTy.getRank();
    auto broadcast = [&](Value value) -> Value {
      auto valueTy = value.getType().cast<ShapedType>();
      auto newTy =
          RankedTensorType::get(resultTy.getShape(), valueTy.getElementType());
      if (valueTy == newTy) return value;
      auto dimensions = llvm::to_vector<4>(
          llvm::seq<int64_t>(resultRank - valueTy.getRank(), resultRank));
      return rewriter.create<mhlo::DynamicBroadcastInDimOp>(
          op.getLoc(), newTy, value, lhsShape,
          rewriter.getI64TensorAttr(dimensions));
    };

    zero = broadcast(zero);

    rewriter.replaceOpWithNewOp<mhlo::SelectOp>(op, resultTy, lhsBool, rhs,
                                                zero);
    return success();
  }
};

// Generates Gaussian noise with uniform random generator based on Box-Muller
// transform.
class ExpandRngNormal : public OpRewritePattern<mhlo::RngOp> {
 public:
  using OpRewritePattern<mhlo::RngOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::RngOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRngDistribution() != mhlo::RngDistribution::NORMAL)
      return failure();

    auto resTy = op.getType().dyn_cast<RankedTensorType>();
    // We can support static shapes, but it's easier to implement Box-Muller
    // transform if we know the number of elements.
    if (!resTy || !resTy.hasStaticShape()) return failure();

    // The algorithm requires even numbers and will generate pairs.
    auto numElems = resTy.getNumElements();
    if (numElems & 1) numElems++;
    auto halfNumElems = numElems / 2;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Explicitly set the seed to 0, so we have stateless generator. This is not
    // a hard limit. Random generator is still a new topic, and we start with
    // stateless random generator.
    std::mt19937 rng{0};
    std::uniform_real_distribution<> runif(0.0, 1.0);
    SmallVector<float> sqrtValues(halfNumElems), cosValues(halfNumElems),
        sinValues(halfNumElems);
    for (auto i : llvm::seq<unsigned>(0, numElems / 2)) {
      constexpr float kEpsilon = std::numeric_limits<float>::epsilon();
      constexpr float kTwoPi = static_cast<float>(2.0 * M_PI);
      float u1, u2;
      do {
        u1 = runif(rng);
        u2 = runif(rng);
      } while (u1 <= kEpsilon);
      sqrtValues[i] = -2.0 * log(u1);
      cosValues[i] = cos(kTwoPi * u2);
      sinValues[i] = sin(kTwoPi * u2);
    }

    // mag = sigma * sqrt(-2.0 * log(u1));
    Value mag = getF32Const(b, /*shapes=*/{halfNumElems}, sqrtValues);
    Value sigma = b.create<mhlo::BroadcastOp>(
        mag.getType(), op.getB(), make1DElementsAttr(b, halfNumElems));
    mag = b.create<mhlo::MulOp>(sigma, b.create<mhlo::SqrtOp>(mag));

    // z0 = mag * cos(two_pi * u2) + mu;
    // z1 = mag * sin(two_pi * u2) + mu;
    Value mu = b.create<mhlo::BroadcastOp>(mag.getType(), op.getA(),
                                           make1DElementsAttr(b, halfNumElems));
    Value z0 = getF32Const(b, /*shapes=*/{halfNumElems}, cosValues);
    z0 = b.create<mhlo::MulOp>(mag, z0);
    z0 = b.create<mhlo::AddOp>(z0, mu);
    Value z1 = getF32Const(b, /*shapes=*/{halfNumElems}, sinValues);
    z1 = b.create<mhlo::MulOp>(mag, z1);
    z1 = b.create<mhlo::AddOp>(z1, mu);

    Value res = b.create<mhlo::ConcatenateOp>(ValueRange{z0, z1},
                                              b.getI64IntegerAttr(0));
    if (numElems != resTy.getNumElements()) {
      OpFoldResult zero = b.getIndexAttr(0);
      OpFoldResult one = b.getIndexAttr(1);
      OpFoldResult size = b.getIndexAttr(resTy.getNumElements());
      res = b.create<tensor::ExtractSliceOp>(res, zero, size, one);
    }
    if (resTy.getRank() != 1) {
      res = b.create<mhlo::ReshapeOp>(resTy, res);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

// clang-format off
//
// Reorder BroadcastInDimOp and N-ary elementwise op.
//
// Rewrites the following pattern (take binary elementwise op as example)
//
// %bcastx = "mhlo.broadcast_in_dim"(%x) {broadcast_dimensions = %[[BCAST_DIMS]]} : (%[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
// %bcasty = "mhlo.broadcast_in_dim"(%y) {broadcast_dimensions = %[[BCAST_DIMS]]} : (%[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
// %result = "BinaryElementwiseOpT"(%bcastx, %bcasty) : (%[[SHAPE_AFTER_BCAST]], %[[SHAPE_AFTER_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
//
// into
//
// %z = "BinaryElementwiseOpT"(%x, %y) : (%[[SHAPE_BEFORE_BCAST]], %[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_BEFORE_BCAST]]
// %result = "mhlo.broadcast_in_dim"(%z) {broadcast_dimensions = %[[BCAST_DIMS]]} : (%[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
//
// clang-format on
template <typename ElementwiseOpT>
class ReorderBroadcastInDimOpAndElementwiseOp
    : public OpRewritePattern<ElementwiseOpT> {
 public:
  using OpRewritePattern<ElementwiseOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseOpT op,
                                PatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    assert(operation->getNumOperands() >= 1 && operation->getNumResults() == 1);

    // Verify if all operands are from BroadcastInDimOp and its
    // broadcast_dimensions is the same.
    llvm::SmallVector<mhlo::BroadcastInDimOp, 2> bcastOps;
    for (auto operand : operation->getOperands()) {
      if (auto bcastOp = operand.getDefiningOp<mhlo::BroadcastInDimOp>()) {
        bcastOps.push_back(bcastOp);
      } else {
        return failure();
      }
    }

    if (llvm::any_of(bcastOps, [&bcastOps](mhlo::BroadcastInDimOp bcastOp) {
          return bcastOp.getBroadcastDimensions() !=
                 bcastOps[0].getBroadcastDimensions();
        })) {
      return failure();
    }

    // Verify if all operands of BroadcastInDimOp are of same type and have
    // static shape.
    auto bcastOperandType =
        bcastOps[0].getOperand().getType().template dyn_cast<ShapedType>();
    llvm::SmallVector<Value, 2> bcastOperands;
    for (auto bcastOp : bcastOps) {
      auto bcastOperand = bcastOp.getOperand();
      auto type = bcastOperand.getType().template dyn_cast<ShapedType>();
      if (!type || !type.hasStaticShape() || type != bcastOperandType) {
        return failure();
      }
      bcastOperands.push_back(bcastOperand);
    }

    // Some elementwise ops, mhlo::RealOp for example, do not have
    // SameOperandsAndResultType trait, so resultType might be different
    // from bcastOperandType.
    auto elementType = getElementTypeOrSelf(op.getResult());
    auto resultShape = bcastOperandType.getShape();
    auto resultType = RankedTensorType::get(resultShape, elementType);

    Value result =
        rewriter.create<ElementwiseOpT>(op.getLoc(), resultType, bcastOperands);
    rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
        op, op.getType(), result, bcastOps[0].getBroadcastDimensions());

    for (auto bcastOp : bcastOps) {
      if (bcastOp.getOperation()->use_empty()) {
        rewriter.eraseOp(bcastOp);
      }
    }

    return success();
  }
};

// Identifies cases where a dense operation has inputs that come from widening
// operations. For instance, a dot product widening from FP16 to FP32 is better
// to have the casting operation fused into the dot operation. This decreases
// the loading required during a dense computation.
template <class Op>
struct FuseWidenOperands : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      auto convertOp =
          dyn_cast_or_null<mhlo::ConvertOp>(operand.getDefiningOp());
      if (convertOp) {
        auto inputType = getElementTypeOrSelf(convertOp.getOperand().getType());
        auto castedType = getElementTypeOrSelf(convertOp.getResult().getType());
        bool isSameCast =
            (inputType.isa<IntegerType>() && castedType.isa<IntegerType>()) ||
            (inputType.isa<FloatType>() && castedType.isa<FloatType>());
        if (isSameCast && inputType.getIntOrFloatBitWidth() <
                              castedType.getIntOrFloatBitWidth()) {
          operands.push_back(convertOp.getOperand());
          continue;
        }
      }
      operands.push_back(operand);
    }

    if (llvm::all_of(
            llvm::zip_equal(operands, op->getOperands()),
            [](auto pair) { return std::get<0>(pair) == std::get<1>(pair); }))
      return failure();

    rewriter.replaceOpWithNewOp<Op>(op, op->getResultTypes(), operands,
                                    op->getAttrs());
    return success();
  }
};

struct DotToMul : public OpRewritePattern<mhlo::DotOp> {
  using OpRewritePattern<mhlo::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DotOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
    auto resultTy = op.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy) {
      return rewriter.notifyMatchFailure(op, "lhs and rhs must be ranked");
    }

    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "lhs and rhs must be rank-2");
    }

    if (lhsTy.getDimSize(1) != 1) return failure();

    // Dynamically compute the shape of the result of the DotOp by querying
    // the 0-th dimensions, of the left, and the 1st dimension of the right.
    // Concatenating them togething to make the final shape.
    Value batchSize = rewriter.create<mhlo::GetDimensionSizeOp>(
        op.getLoc(), lhs, rewriter.getI64IntegerAttr(0));
    Value batchSize1 = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), RankedTensorType::get({1}, rewriter.getI32Type()),
        batchSize);

    Value featureSize = rewriter.create<mhlo::GetDimensionSizeOp>(
        op.getLoc(), rhs, rewriter.getI64IntegerAttr(1));
    Value featureSize1 = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), RankedTensorType::get({1}, rewriter.getI32Type()),
        featureSize);

    Value outSize = rewriter.create<mhlo::ConcatenateOp>(
        op.getLoc(), RankedTensorType::get({2}, rewriter.getI32Type()),
        ValueRange{batchSize1, featureSize1}, rewriter.getI64IntegerAttr(0));

    lhs = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        op.getLoc(), resultTy.clone(lhsTy.getElementType()), lhs, outSize,
        rewriter.getI64TensorAttr({0, 1}));

    rhs = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        op.getLoc(), resultTy.clone(rhsTy.getElementType()), rhs, outSize,
        rewriter.getI64TensorAttr({0, 1}));

    auto computeETy = lhsTy.getElementType();
    if (computeETy.getIntOrFloatBitWidth() < rhsTy.getElementTypeBitWidth())
      computeETy = rhsTy.getElementType();
    if (computeETy.getIntOrFloatBitWidth() < resultTy.getElementTypeBitWidth())
      computeETy = resultTy.getElementType();

    auto computeTy = resultTy.clone(computeETy);

    rhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), computeTy, rhs);
    lhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), computeTy, lhs);

    auto result = rewriter.create<mhlo::MulOp>(
        op.getLoc(), resultTy.clone(computeETy), lhs, rhs);
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, resultTy, result);
    return success();
  }
};

// Similar to DotIsMul, this finds the case where a dot general
// can be represented using a mul operation. This includes possibly making
// an implicit cast explicit prior the mul.
struct DotGeneralIsMul : public OpRewritePattern<mhlo::DotGeneralOp> {
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs().cast<Value>();
    auto rhs = op.getRhs().cast<Value>();
    auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
    auto resultTy = op.getType().dyn_cast<RankedTensorType>();
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    if (!lhsTy || !rhsTy || !resultTy) return failure();

    auto dNums = op.getDotDimensionNumbers();
    auto batchDimsL = dNums.getLhsBatchingDimensions();
    auto batchDimsR = dNums.getRhsBatchingDimensions();
    auto contractDimsL = dNums.getLhsContractingDimensions();
    auto contractDimsR = dNums.getRhsContractingDimensions();

    llvm::SmallVector<bool> isLhsParallelDim(lhsTy.getRank(), true);
    llvm::SmallVector<bool> isRhsParallelDim(rhsTy.getRank(), true);

    for (auto dim : batchDimsL) isLhsParallelDim[dim] = false;
    for (auto dim : batchDimsR) isRhsParallelDim[dim] = false;
    for (auto dim : contractDimsL) isLhsParallelDim[dim] = false;
    for (auto dim : contractDimsR) isRhsParallelDim[dim] = false;

    for (auto dim : contractDimsL) {
      if (lhsTy.getDimSize(dim) != 1) {
        return rewriter.notifyMatchFailure(op, "Non unit contract dimensions");
      }
    }

    // Generate the permutation matrix to order BatchDims, ParallelDims,
    // ContractDims.
    llvm::SmallVector<int64_t> permLhs;
    llvm::SmallVector<int64_t> permRhs;
    permLhs.append(batchDimsL.begin(), batchDimsL.end());
    permRhs.append(batchDimsR.begin(), batchDimsR.end());

    for (auto it : llvm::enumerate(isLhsParallelDim)) {
      if (it.value()) permLhs.push_back(it.index());
    }

    for (auto it : llvm::enumerate(isRhsParallelDim)) {
      if (it.value()) permRhs.push_back(it.index());
    }

    permLhs.append(contractDimsL.begin(), contractDimsL.end());
    permRhs.append(contractDimsR.begin(), contractDimsR.end());

    // Determine the transpose shape based on the generate permutations.
    llvm::SmallVector<int64_t> lhsTransposeShape;
    llvm::SmallVector<int64_t> rhsTransposeShape;
    for (auto dim : permLhs) lhsTransposeShape.push_back(lhsTy.getDimSize(dim));
    for (auto dim : permRhs) rhsTransposeShape.push_back(rhsTy.getDimSize(dim));

    // Transpose the left hand side and the right hand side.
    lhs = builder.create<mhlo::TransposeOp>(
        RankedTensorType::get(lhsTransposeShape, lhsTy.getElementType()), lhs,
        builder.getI64TensorAttr(permLhs));
    lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = builder.create<mhlo::TransposeOp>(
        RankedTensorType::get(rhsTransposeShape, rhsTy.getElementType()), rhs,
        builder.getI64TensorAttr(permRhs));
    rhsTy = rhs.getType().cast<RankedTensorType>();

    auto dimI32Ty = RankedTensorType::get({1}, builder.getI32Type());

    // Drop all of the non-concat dimensions from the lhs.
    llvm::SmallVector<Value> lhsReshapeDims;
    for (int i = 0, s = lhsTy.getRank() - contractDimsL.size(); i < s; i++) {
      Value dim = builder.create<mhlo::GetDimensionSizeOp>(lhs, i);
      lhsReshapeDims.push_back(builder.create<mhlo::ReshapeOp>(dimI32Ty, dim));
    }
    Value lhsDynShape = builder.create<mhlo::ConcatenateOp>(
        RankedTensorType::get({static_cast<int64_t>(lhsReshapeDims.size())},
                              builder.getI32Type()),
        lhsReshapeDims, 0);
    lhsTy =
        RankedTensorType::get(lhsTy.getShape().drop_back(contractDimsL.size()),
                              lhsTy.getElementType());
    lhs = builder.create<mhlo::DynamicReshapeOp>(lhsTy, lhs, lhsDynShape);

    // Drop all of the non concat dimensions from the rhs.
    llvm::SmallVector<Value> rhsReshapeDims;
    for (int i = 0, s = rhsTy.getRank() - contractDimsR.size(); i < s; i++) {
      Value dim = builder.create<mhlo::GetDimensionSizeOp>(rhs, i);
      rhsReshapeDims.push_back(builder.create<mhlo::ReshapeOp>(dimI32Ty, dim));
    }
    Value rhsDynShape = builder.create<mhlo::ConcatenateOp>(
        RankedTensorType::get({static_cast<int64_t>(rhsReshapeDims.size())},
                              builder.getI32Type()),
        rhsReshapeDims, 0);
    rhsTy =
        RankedTensorType::get(rhsTy.getShape().drop_back(contractDimsR.size()),
                              rhsTy.getElementType());
    rhs = builder.create<mhlo::DynamicReshapeOp>(rhsTy, rhs, rhsDynShape);

    // Compute the size of the output shape with dynamic shape support using the
    // lhs and rhs dimensions.
    llvm::SmallVector<Value> outputDims;
    outputDims.append(lhsReshapeDims);
    outputDims.append(rhsReshapeDims.begin() + batchDimsR.size(),
                      rhsReshapeDims.end());
    Value outputShape = builder.create<mhlo::ConcatenateOp>(
        RankedTensorType::get({resultTy.getRank()}, builder.getI32Type()),
        outputDims, 0);

    // Broadcast the left hand side to match the expect output shape.
    llvm::SmallVector<int64_t> lhsDimMapping(lhsTy.getRank());
    std::iota(lhsDimMapping.begin(), lhsDimMapping.end(), 0);
    auto lhsBroadcastTy =
        RankedTensorType::get(resultTy.getShape(), lhsTy.getElementType());
    lhs = builder.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        lhsBroadcastTy, lhs, outputShape,
        rewriter.getI64TensorAttr(lhsDimMapping));

    // Broadcast the right hand side to match the expected output shape.
    llvm::SmallVector<int64_t> rhsDimMapping(rhsTy.getRank());
    std::iota(rhsDimMapping.begin(), rhsDimMapping.begin() + batchDimsR.size(),
              0);
    std::iota(rhsDimMapping.begin() + batchDimsR.size(), rhsDimMapping.end(),
              lhsTy.getRank());
    auto rhsBroadcastTy =
        RankedTensorType::get(resultTy.getShape(), rhsTy.getElementType());
    rhs = builder.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        rhsBroadcastTy, rhs, outputShape,
        rewriter.getI64TensorAttr(rhsDimMapping));

    lhs = builder.createOrFold<mhlo::ConvertOp>(resultTy, lhs);
    rhs = builder.createOrFold<mhlo::ConvertOp>(resultTy, rhs);
    rewriter.replaceOpWithNewOp<mhlo::MulOp>(op, resultTy, lhs, rhs);
    return success();
  }
};

struct MHLOToMHLOPreprocessingPass
    : public MHLOToMHLOPreprocessingBase<MHLOToMHLOPreprocessingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, mhlo::MhloDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget conversionTarget(*context);
    RewritePatternSet conversionPatterns(&getContext());
    // Note that various input modalities may do their own legalization of
    // CHLO. Converting here allows IREE to accept CHLO dialect regardless of
    // whether it was legalized away at a higher level.
    // chlo::PopulateLegalizeChloToHloPatterns(context, &conversionPatterns);
    conversionTarget.addLegalDialect<
        shape::ShapeDialect, chlo::ChloDialect, mhlo::MhloDialect,
        math::MathDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect,
        mlir::tensor::TensorDialect>();
    // conversionTarget.addIllegalDialect<chlo::ChloDialect>();
    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    // TODO: Remove once we have a general contraction to matmul pass.
    mhlo::populateEinsumToDotGeneralPatterns(context, &patterns);
    mhlo::populateUnfuseBatchNormPatterns(context, &patterns);
    mhlo::populateComplexLoweringPatterns(context, &patterns);
    mhlo::populateGatherToTorchIndexSelectPatterns(context, &patterns);
    patterns.insert<ExpandRngNormal, MulCastOfBool>(context);

    // scatter canonicalization patterns
    patterns.insert<ScatterInt64Indices, ScatterImplicitIndex,
                    ScatterImplicitBatch, ScatterMaterializeInsertedDim,
                    ScatterCollapseBatch, ScatterBatchFirst>(context);

    // dot_general canoncalization patterns.
    mhlo::populateGeneralDotOpLoweringPatterns(&patterns, context);
    // TODO(jpienaar): This may be redundant with lower_general_dot. Remove if
    // so.
    patterns.insert<TransposeReshapeGenericDotGeneral>(context,
                                                       /*benefit=*/200);
    patterns.insert<DotGeneralIsMul>(context, /*benefit=*/300);

    // Fusion operations.
    patterns.insert<FuseWidenOperands<mhlo::DotOp>,
                    FuseWidenOperands<mhlo::DotGeneralOp>,
                    FuseWidenOperands<mhlo::ConvolutionOp>>(context,
                                                            /*benefit=*/400);

    // Additional canonicalizers that simplify to computationally
    // less-complex operations.
    patterns.insert<DotToMul>(context);

    // Unary elementwise op.
    patterns.insert<
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::AbsOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::CeilOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ConvertOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ClzOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::CosineOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ExpOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::Expm1Op>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::FloorOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ImagOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::IsFiniteOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::LogOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::Log1pOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::LogisticOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::NotOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::NegOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::PopulationCountOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::RealOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::RoundOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::RsqrtOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::SignOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::SineOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::SqrtOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::TanhOp>>(context);
    // Binary elementwise op.
    patterns.insert<
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::AddOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::Atan2Op>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ComplexOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::DivOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::MaxOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::MinOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::MulOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::PowOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::RemOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ShiftLeftOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ShiftRightArithmeticOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::ShiftRightLogicalOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::SubtractOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::AndOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::OrOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mhlo::XorOp>>(context);
    if (orderConvFeatures) {
      patterns.insert<ReorderConvOpInputDimensions>(context);
      patterns.insert<ReorderConvOpKernelDimensions>(context);
      patterns.insert<ReorderConvOpOutputDimensions>(context);
    }
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createMHLOToMHLOPreprocessingPass() {
  return std::make_unique<MHLOToMHLOPreprocessingPass>();
}

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir
