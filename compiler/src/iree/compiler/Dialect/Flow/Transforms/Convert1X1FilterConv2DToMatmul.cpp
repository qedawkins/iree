// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

static void swapMatmulBlockArgs(linalg::GenericOp genericOp) {
  Value leftOperand = genericOp.getRegion().front().getArgument(0);
  Value rightOperand = genericOp.getRegion().front().getArgument(1);
  auto leftUses = leftOperand.getUses();
  auto rightUses = rightOperand.getUses();
  leftOperand.replaceUsesWithIf(rightOperand, [&](OpOperand &use) {
    return llvm::any_of(leftUses, [&](OpOperand &newUse) {
              return newUse.getOwner() == use.getOwner() &&
                    newUse.getOperandNumber() == use.getOperandNumber();
            });
  });
  rightOperand.replaceUsesWithIf(leftOperand, [&](OpOperand &use) {
    return llvm::any_of(rightUses, [&](OpOperand &newUse) {
              return newUse.getOwner() == use.getOwner() &&
                    newUse.getOperandNumber() == use.getOperandNumber();
            });
  });
  llvm::SmallDenseSet<Operation *, 2> leftOps;
  llvm::SmallDenseSet<Operation *, 2> rightOps;
  for (Operation *user : leftOperand.getUsers())
    leftOps.insert(user);
  for (Operation *user : rightOperand.getUsers())
    rightOps.insert(user);

  for (Operation &op : genericOp.getRegion().front()) {
    if (leftOps.contains(&op) || rightOps.contains(&op))
      continue;

    int leftIdx = -1;
    int rightIdx = -1;
    for (auto &operand : op.getOpOperands()) {
      auto definingOp = operand.get().getDefiningOp();
      if (!definingOp) continue;
      if (leftOps.contains(definingOp)) {
        assert(leftIdx < 0 && "Found non-unary or binary op");
        leftIdx = operand.getOperandNumber();
      }

      if (rightOps.contains(definingOp)) {
        assert(rightIdx < 0 && "Found non-unary or binary op");
        rightIdx = operand.getOperandNumber();
      }
    }
    if (leftIdx >= 0 && rightIdx >= 0) {
      auto leftVal = op.getOpOperand(leftIdx).get();
      auto rightVal = op.getOpOperand(rightIdx).get();
      op.setOperand(leftIdx, rightVal);
      op.setOperand(rightIdx, leftVal);
      continue;
    }
    if (leftIdx >= 0)
      leftOps.insert(&op);
    if (rightIdx >= 0)
      rightOps.insert(&op);
  }
}

class Convert1x1FilterConvGenericToMatmul : public OpRewritePattern<linalg::GenericOp> {
 public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    mlir::linalg::detail::ConvolutionDimensions dimensions;
    if (!linalg::detail::getMatchConvolutionMessage(
                linalg::detail::isConvolutionInterfaceImpl(genericOp, &dimensions)).empty())
      return failure();

    if (dimensions.outputImage.size() < 2 || dimensions.filterLoop.size() < 2)
      return failure();

    auto inputType = genericOp.getInputs()[0].getType().cast<ShapedType>();
    auto filterType = genericOp.getInputs()[1].getType().cast<ShapedType>();
    auto outputType = genericOp.getOutputs()[0].getType().cast<ShapedType>();

    if (!filterType.hasStaticShape())
      return failure();

    if (!inputType.hasStaticShape())
      return failure();

    if (!llvm::all_of(
        dimensions.dilations, [](unsigned element) { return element == 1; }))
      return failure();

    if (!llvm::all_of(
        dimensions.strides, [](unsigned element) { return element == 1; }))
      return failure();

    Value input = genericOp.getInputs()[0];
    Value filter = genericOp.getInputs()[1];
    Value output = genericOp.getOutputs()[0];

    bool isNchw = dimensions.outputChannel[0] < dimensions.outputImage[0];
    bool isBatched = dimensions.batch.size() > 0;

    auto dimSizes = genericOp.getStaticLoopRanges();

    int n = 0;
    if (isBatched) n = dimSizes[dimensions.batch[0]];
    int oc = dimSizes[dimensions.outputChannel[0]];
    int oh = dimSizes[dimensions.outputImage[0]];
    int ow = dimSizes[dimensions.outputImage[1]];
    int ic = dimSizes[dimensions.inputChannel[0]];
    int fh = dimSizes[dimensions.filterLoop[0]];
    int fw = dimSizes[dimensions.filterLoop[1]];

    if (fh != 1 || fw != 1)
      return failure();

    auto loc = genericOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    SmallVector<ReassociationIndices> filterReassocIndices =
        isNchw ? SmallVector<ReassociationIndices>{{0}, {1, 2, 3}}
               : SmallVector<ReassociationIndices>{{0, 1, 2}, {3}};
    SmallVector<int64_t> filterShape =
        isNchw ? SmallVector<int64_t>{oc, ic * fh * fw}
               : SmallVector<int64_t>{ic * fh * fw, oc};
    auto reshapedFilterType =
        RankedTensorType::get({oc, ic * fh * fw}, inputType.getElementType());
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, filterReassocIndices);

    SmallVector<ReassociationIndices> outputReassocIndices;
    SmallVector<int64_t> outputShape;
    SmallVector<int64_t> inputShape;
    if (isNchw) {
      outputReassocIndices =
          isBatched ? SmallVector<ReassociationIndices>{{0}, {1}, {2, 3}}
                    : SmallVector<ReassociationIndices>{{0}, {1, 2}};
      outputShape =
          isBatched ? SmallVector<int64_t>{n, oc, oh * ow}
                    : SmallVector<int64_t>{oc, oh * ow};
      inputShape =
          isBatched ? SmallVector<int64_t>{n, ic, oh * ow}
                    : SmallVector<int64_t>{ic, oh * ow};
    } else {
      outputReassocIndices =
          isBatched ? SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}}
                    : SmallVector<ReassociationIndices>{{0, 1}, {2}};
      outputShape =
          isBatched ? SmallVector<int64_t>{n, oh * ow, oc}
                    : SmallVector<int64_t>{oh * ow, oc};
      inputShape =
          isBatched ? SmallVector<int64_t>{n, oh * ow, ic}
                    : SmallVector<int64_t>{oh * ow, ic};
    }
    auto reshapedOutputType =
        RankedTensorType::get(outputShape, outputType.getElementType());
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, outputReassocIndices);
    auto reshapedInputType =
        RankedTensorType::get(inputShape, inputType.getElementType());
    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, outputReassocIndices);

    // Because the filter does not share the same batch dimension,
    // the batch dimension is only used in indexing the input and output. Thus
    // we cannot use existing linalg named ops like linalg.batch_matmul.
    // i.e. M x K * (B x) K x N = (B x) M x N
    AffineExpr bDim, mDim, nDim, kDim;
    if (isBatched)
      bindDims(context, bDim, mDim, nDim, kDim);
    else
      bindDims(context, mDim, nDim, kDim);

    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    int numMatmulLoops = isBatched ? 4 : 3;
    auto lhsMap = AffineMap::get(numMatmulLoops, 0,
            isBatched && !isNchw ? SmallVector<AffineExpr>{bDim, mDim, kDim}
                                 : SmallVector<AffineExpr>{mDim, kDim}, context);
    auto rhsMap = AffineMap::get(numMatmulLoops, 0,
            isBatched && isNchw ? SmallVector<AffineExpr>{bDim, kDim, nDim}
                                : SmallVector<AffineExpr>{kDim, nDim}, context);
    auto resultMap = AffineMap::get(numMatmulLoops, 0,
            isBatched ? SmallVector<AffineExpr>{bDim, mDim, nDim}
                      : SmallVector<AffineExpr>{mDim, nDim}, context);
    SmallVector<utils::IteratorType> genericIterators;
    if (isBatched)
      genericIterators = {parallel, parallel, parallel, reduction};
    else
      genericIterators = {parallel, parallel, reduction};
    auto newGenericOp = rewriter.create<linalg::GenericOp>(
        loc, reshapedOutputType,
        /*inputs=*/isNchw ? ValueRange{reshapedFilter, reshapedInput}
                          : ValueRange{reshapedInput, reshapedFilter},
        /*outputs=*/ValueRange{reshapedOutput},
        ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators);
    IRMapping mapper;
    genericOp.getRegion().cloneInto(&newGenericOp.getRegion(), mapper);

    if (isNchw) swapMatmulBlockArgs(newGenericOp);

    Value result = newGenericOp.getResults().front();

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputType, result, outputReassocIndices);

    rewriter.replaceOp(genericOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
template <typename Conv2DOpType>
class Convert1x1FilterConvToMatmul : public OpRewritePattern<Conv2DOpType> {
 public:
  using OpRewritePattern<Conv2DOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOpType convOp,
                                PatternRewriter &rewriter) const override {
    auto inputShapeType = convOp.getDpsInputOperand(0)
                              ->get()
                              .getType()
                              .template dyn_cast<RankedTensorType>();
    auto filterShapeType = convOp.getDpsInputOperand(1)
                               ->get()
                               .getType()
                               .template dyn_cast<RankedTensorType>();
    auto outputShapeType = convOp.getDpsInitOperand(0)
                               ->get()
                               .getType()
                               .template dyn_cast<RankedTensorType>();

    const bool isNCHW = isa<linalg::Conv2DNchwFchwOp>(convOp);
    const bool isNHWC = isa<linalg::Conv2DNhwcHwcfOp>(convOp);
    if (!isNCHW & !isNHWC) return failure();

    if (!inputShapeType || !filterShapeType || !outputShapeType)
      return failure();

    auto inputShape = inputShapeType.getShape();
    auto filterShape = filterShapeType.getShape();
    auto outputShape = outputShapeType.getShape();

    // Adjusting dimension indices based on Conv2DOpType.
    const int nIndex = 0;
    const int kcIndex = isNHWC ? 2 : 1;
    const int kfIndex = isNHWC ? 3 : 0;
    const int khIndex = isNHWC ? 0 : 2;
    const int kwIndex = isNHWC ? 1 : 3;
    const int ohIndex = isNHWC ? 1 : 2;
    const int owIndex = isNHWC ? 2 : 3;
    const int ocIndex = isNHWC ? 3 : 1;

    bool isInputHWDynamic = inputShape[ohIndex] == ShapedType::kDynamic &&
                            inputShape[owIndex] == ShapedType::kDynamic;

    // We cannot merge the width and height if they are both dynamic as we
    // cannot expand them back to their dynamic values.
    if (isInputHWDynamic) return failure();

    if (filterShape[khIndex] != 1 || filterShape[kwIndex] != 1)
      return failure();

    // TODO(ataei): Support conversion to linalg.batch_matmul.
    if (inputShape[0] != 1) return failure();

    if (!llvm::all_of(convOp.getStrides(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();
    if (!llvm::all_of(convOp.getDilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    auto combineDims = [](int64_t a, int64_t b) {
      if (a == ShapedType::kDynamic || b == ShapedType::kDynamic)
        return ShapedType::kDynamic;
      return a * b;
    };

    SmallVector<ReassociationIndices, 4> reassociationInputOutputIndices;
    SmallVector<ReassociationIndices, 4> reassociationFilterIndices;
    SmallVector<int64_t> reshapedInputShape(2, 0);
    SmallVector<int64_t> reshapedFilterShape(2, 0);
    SmallVector<int64_t> reshapedOutputShape(2, 0);
    if (isNHWC) {
      // Generate reassociation indices.
      reassociationInputOutputIndices = {{nIndex, ohIndex, owIndex}, {ocIndex}};
      reassociationFilterIndices = {{khIndex, kwIndex, kcIndex}, {kfIndex}};

      // Generate matmul shapes from 1x1 conv.
      reshapedInputShape = {
          combineDims(inputShape[ohIndex], inputShape[owIndex]),
          inputShape[ocIndex]};
      reshapedFilterShape = {filterShape[kcIndex], filterShape[kfIndex]};
      reshapedOutputShape = {
          combineDims(outputShape[ohIndex], outputShape[owIndex]),
          outputShape[ocIndex]};
    } else if (isNCHW) {
      // Generate reassociation indices.
      reassociationInputOutputIndices = {{nIndex, ocIndex}, {ohIndex, owIndex}};
      reassociationFilterIndices = {{kfIndex}, {kcIndex, khIndex, kwIndex}};

      // Generate matmul shapes from 1x1 conv.
      reshapedInputShape = {
          inputShape[ocIndex],
          combineDims(inputShape[ohIndex], inputShape[owIndex])};
      reshapedFilterShape = {filterShape[kfIndex], filterShape[kcIndex]};
      reshapedOutputShape = {
          outputShape[ocIndex],
          combineDims(outputShape[ohIndex], outputShape[owIndex])};
    }

    auto reshapedInputType = RankedTensorType::get(
        reshapedInputShape, inputShapeType.getElementType());

    auto reshapedFilterType = RankedTensorType::get(
        reshapedFilterShape, filterShapeType.getElementType());

    auto reshapedOutputType = RankedTensorType::get(
        reshapedOutputShape, outputShapeType.getElementType());

    Value input = convOp.getDpsInputOperand(0)->get();
    Value filter = convOp.getDpsInputOperand(1)->get();
    Value output = convOp.getDpsInitOperand(0)->get();
    auto loc = convOp.getLoc();

    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, reassociationInputOutputIndices);
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, reassociationFilterIndices);
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, reassociationInputOutputIndices);

    SmallVector<Value, 2> matmulInput;
    if (isNHWC) {
      matmulInput = {reshapedInput, reshapedFilter};
    } else if (isNCHW) {
      matmulInput = {reshapedFilter, reshapedInput};
    }
    auto matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType, matmulInput, ArrayRef<Value>{reshapedOutput});

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputShapeType, matmulResult.getResults()[0],
        reassociationInputOutputIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

struct Convert1X1FilterConv2DToMatmulPass
    : public Convert1X1FilterConv2DToMatmulBase<
          Convert1X1FilterConv2DToMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<Convert1x1FilterConvToMatmul<linalg::Conv2DNhwcHwcfOp>,
                    Convert1x1FilterConvToMatmul<linalg::Conv2DNchwFchwOp>>(
        context);
    patterns.insert<Convert1x1FilterConvGenericToMatmul>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createConvert1X1FilterConv2DToMatmulPass() {
  return std::make_unique<Convert1X1FilterConv2DToMatmulPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
