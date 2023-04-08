// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-preprocessing-convert-conv-to-channels-last"

namespace mlir {
namespace iree_compiler {
namespace IREE {

using TransposeIndices = SmallVector<int64_t, 4>;
using ConvBuilderFn = std::function<Value(OpBuilder &b, Location loc,
        linalg::LinalgOp srcConv, Value input, Value filter, Value output,
        AffineMap inputMap, AffineMap filterMap, AffineMap outputMap,
        SmallVector<unsigned> newDimOrder)>;
using linalg::detail::MatchConvolutionResult;

static Value defaultConvBuilderFn(OpBuilder &b, Location loc,
        linalg::LinalgOp srcConv, Value input, Value filter, Value output,
        AffineMap inputMap, AffineMap filterMap, AffineMap outputMap,
        SmallVector<unsigned> newDimOrder) {
    DenseMap<AffineExpr, AffineExpr> dimMap;
    for (auto [newDim, oldDim] : llvm::enumerate(newDimOrder))
      dimMap[b.getAffineDimExpr(oldDim)] = b.getAffineDimExpr(newDim);
    auto newInputMap = inputMap.replace(dimMap,
            /*numResultDims=*/newDimOrder.size(), /*numResultSymbols=*/0);
    auto newFilterMap = filterMap.replace(dimMap,
            /*numResultDims=*/newDimOrder.size(), /*numResultSymbols=*/0);
    auto newOutputMap = outputMap.replace(dimMap,
            /*numResultDims=*/newDimOrder.size(), /*numResultSymbols=*/0);
    auto genericConv =
        b.create<linalg::GenericOp>(loc, output.getType(),
                                    ValueRange{input, filter}, output,
                                    ArrayRef<AffineMap>{newInputMap, newFilterMap, newOutputMap},
                                    srcConv.getIteratorTypesArray());
    IRMapping mapper;
    srcConv->getRegion(0).cloneInto(&genericConv.getRegion(), mapper);
    return genericConv.getResult(0);
}

template <typename sourceNamedConvTy, typename targetNamedConvTy>
static Value namedConvBuilderFn(OpBuilder &b, Location loc,
        linalg::LinalgOp srcConv, Value input, Value filter, Value output,
        AffineMap inputMap, AffineMap filterMap, AffineMap outputMap, SmallVector<unsigned> newDimOrder) {
    sourceNamedConvTy namedConv = cast<sourceNamedConvTy>(srcConv);
    return b.create<targetNamedConvTy>(loc, output.getType(),
                                    ValueRange{input, filter},
                                    output, namedConv.getStrides(),
                                    namedConv.getDilations()).getResult(0);
}

static TransposeIndices getNormalizedIndices(TransposeIndices targetIndices) {
  int startDim = *std::min_element(targetIndices.begin(), targetIndices.end());
  TransposeIndices normalized(targetIndices.size());
  for (auto i : llvm::enumerate(targetIndices))
    normalized[i.index()] = i.value() - startDim;
  return normalized;
}

static TransposeIndices invertIndices(TransposeIndices targetIndices) {
  int startDim = *std::min_element(targetIndices.begin(), targetIndices.end());
  TransposeIndices inverted(targetIndices.size());
  for (auto i : llvm::enumerate(targetIndices)) {
    inverted[i.value() - startDim] = i.index() + startDim;
  }
  return inverted;
}

static bool isInnerIdentityIndices(TransposeIndices indices, int64_t rank) {
  return indices.empty() ||
         (llvm::all_of(llvm::enumerate(indices), [indices](auto e) {
            if (e.index() == 0) return true;
            return indices[e.index()-1] < e.value();
          }) && indices.back() == rank - 1);
}

// Helper to shuffle vectors according to the transpose indices.
template <typename T>
static SmallVector<T> shuffleFromIndices(SmallVector<T> unshuffled,
                                         TransposeIndices targetIndices) {
  int startDim = *std::min_element(targetIndices.begin(), targetIndices.end());
  SmallVector<T> shuffled(unshuffled);
  for (auto i : llvm::enumerate(targetIndices)) {
    shuffled[i.index() + startDim] = unshuffled[i.value()];
  }
  return shuffled;
}

template <typename T>
static SmallVector<T> getPackedVector(SmallVector<T> vec,
                                         TransposeIndices targetIndices) {
  SmallVector<T> packedShape;
  for (auto [i, val] : llvm::enumerate(vec))
    if (!llvm::is_contained(targetIndices, i))
      packedShape.push_back(val);
  for (auto i : targetIndices)
    packedShape.push_back(vec[i]);
  return packedShape;
}

static SmallVector<ReassociationIndices, 4> getUntiledPackReassociationMap(
        TransposeIndices targetIndices, int64_t rank) {
  int startDim = *std::min_element(targetIndices.begin(), targetIndices.end());
  int dimCount = targetIndices.size();
  SmallVector<ReassociationIndices, 4> reassociationMap;
  for (int i = 0; i <= startDim; i++)
    reassociationMap.push_back({i});
  for (int i = startDim + 1; i < dimCount + startDim + 1; i++)
    reassociationMap[startDim].push_back(i);
  for (int i = dimCount + startDim + 1; i < dimCount + rank; i++)
    reassociationMap.push_back({i});
  return reassociationMap;
}

// Transpose the given tensor based on the given transpose indices. Marks the
// created transpose based on the propagation direction.
static std::tuple<Value, std::optional<tensor::PackOp>, AffineMap>
createTransposeAsTensorPack(PatternRewriter &rewriter, Location loc,
                             Value input, AffineMap inputMap, TransposeIndices targetIndices) {
  if (isInnerIdentityIndices(targetIndices, inputMap.getNumResults()))
    return std::make_tuple(input, std::nullopt, inputMap);

  RankedTensorType inType = input.getType().cast<RankedTensorType>();
  auto elementType = inType.getElementType();
  auto inputShape(inType.getShape());

  SmallVector<OpFoldResult> transposedTileSizes;
  for (auto i : targetIndices) {
    if (ShapedType::isDynamic(inputShape[i]))
      transposedTileSizes.push_back(rewriter.create<tensor::DimOp>(loc, input, i).getResult());
    else
      transposedTileSizes.push_back(rewriter.getIndexAttr(inputShape[i]));
  }

  // Pack the input tensor.
  auto empty = tensor::PackOp::createDestinationTensor(
      rewriter, loc, input, transposedTileSizes, targetIndices, SmallVector<int64_t>{});
  auto packedInput = rewriter.create<tensor::PackOp>(
      loc, input, empty, targetIndices, transposedTileSizes,
      /*padding=*/std::nullopt, SmallVector<int64_t>{});

  // Collapse the unit dims created by tensor.pack.
  auto reassociationMap = getUntiledPackReassociationMap(targetIndices, inType.getRank());
  auto transposedInputShape =
      getPackedVector<int64_t>(llvm::to_vector(inputShape), targetIndices);

  auto collapsed = rewriter.create<tensor::CollapseShapeOp>(
    loc, RankedTensorType::get(transposedInputShape, elementType),
    packedInput, reassociationMap);

  SmallVector<AffineExpr> mapResults(inputMap.getResults());
  AffineMap transposedMap = AffineMap::get(inputMap.getNumDims(), inputMap.getNumSymbols(),
          getPackedVector<AffineExpr>(mapResults, targetIndices),
          input.getContext());
  return std::make_tuple(collapsed.getResult(), packedInput, transposedMap);
}

// Transpose the given tensor based on the given transpose indices. Marks the
// created transpose based on the propagation direction.
static Value
createTransposeAsTensorUnPack(PatternRewriter &rewriter, Location loc,
                             Value output, TransposeIndices targetIndices) {
  RankedTensorType outType = output.getType().cast<RankedTensorType>();
  int64_t rank = outType.getRank();
  if (isInnerIdentityIndices(targetIndices, rank))
    return output;

  auto elementType = outType.getElementType();
  auto outputShape(outType.getShape());

  int startDim = *std::min_element(targetIndices.begin(), targetIndices.end());
  SmallVector<int64_t> expandedOutputShape;
  for (int i = 0, e = startDim; i < e; i++)
    expandedOutputShape.push_back(outputShape[i]);
  for (int i = 0, e = targetIndices.size(); i < e; i++)
    expandedOutputShape.push_back(1);
  for (int i = startDim, e = rank; i < e; i++)
    expandedOutputShape.push_back(outputShape[i]);

  auto reassociationMap = getUntiledPackReassociationMap(targetIndices, rank);
  auto expandedOutput = rewriter.create<tensor::ExpandShapeOp>(
    loc, RankedTensorType::get(expandedOutputShape, elementType),
    output, reassociationMap);

  SmallVector<OpFoldResult> tileSizes;
  for (auto i : getNormalizedIndices(targetIndices)) {
    int64_t dim = i + rank - targetIndices.size();
    if (ShapedType::isDynamic(outputShape[dim]))
      tileSizes.push_back(rewriter.create<tensor::DimOp>(loc, output, dim).getResult());
    else
      tileSizes.push_back(rewriter.getIndexAttr(outputShape[dim]));
  }

  SmallVector<OpFoldResult> transposedOutputShape;
  int64_t nChannels = 0;
  for (int64_t i = 0, end = rank; i < end; i++) {
    auto *where = llvm::find(targetIndices, i);
    int64_t dim;
    if (where == targetIndices.end()) {
      dim = i - nChannels;
    } else {
      dim = rank - targetIndices.size() + (where - targetIndices.begin());
      nChannels++;
    }
    if (ShapedType::isDynamic(outputShape[dim]))
      transposedOutputShape.push_back(rewriter.create<tensor::DimOp>(loc, output, dim).getResult());
    else
      transposedOutputShape.push_back(rewriter.getIndexAttr(outputShape[dim]));
  }

  Value empty = rewriter.create<tensor::EmptyOp>(
    loc, transposedOutputShape, elementType);

  auto unpackedOutput = rewriter.create<tensor::UnPackOp>(
    loc, expandedOutput, empty, targetIndices,
    tileSizes, SmallVector<int64_t>{});
  unpackedOutput->setAttr("__unpack__", rewriter.getUnitAttr());
  return unpackedOutput.getResult();
}

static TransposeIndices collectChannelTransposeIndices(AffineMap map,
        SmallVector<SmallVector<unsigned, 2>> transposeDimTargets) {
  SmallVector<TransposeIndices> channelIndices(transposeDimTargets.size());
  for (auto [index, result] : llvm::enumerate(map.getResults())) {
    if (result.isa<AffineDimExpr>()) {
      for (auto [channelVec, dimCategory] : llvm::zip_equal(channelIndices, transposeDimTargets)) {
        if (llvm::is_contained(dimCategory, result.cast<AffineDimExpr>().getPosition())) {
          channelVec.push_back(index);
          break;
        }
      }
    }
  }

  TransposeIndices indices;
  for (auto channelVec : channelIndices)
    indices.append(channelVec);
  return indices;
}

static LogicalResult transposeConvLikeLinalgOp(PatternRewriter &rewriter,
                                               linalg::LinalgOp convOp,
                                               ConvBuilderFn convBuilder = defaultConvBuilderFn) {
  Location loc = convOp.getLoc();

  linalg::detail::ConvolutionDimensions convDims;
  auto errString = getMatchConvolutionMessage(
          linalg::detail::isConvolutionInterfaceImpl(convOp, &convDims));
  if (!errString.empty())
    return failure();

  ////if (convOp.getIteratorTypesArray().size() == 6) {
  //{
  //  llvm::errs() << "Found conv-like op\n";
  //  convOp.dump();
  //  llvm::errs() << "\n";
  //  llvm::errs() << "\n";
  //  llvm::interleaveComma(convDims.batch, llvm::errs() << "Batch: ");
  //  llvm::errs() << "\n";
  //  llvm::interleaveComma(convDims.outputImage, llvm::errs() << "OutputImage: ");
  //  llvm::errs() << "\n";
  //  llvm::interleaveComma(convDims.outputChannel, llvm::errs() << "OutputChannel: ");
  //  llvm::errs() << "\n";
  //  llvm::interleaveComma(convDims.filterLoop, llvm::errs() << "FilterLoop: ");
  //  llvm::errs() << "\n";
  //  llvm::interleaveComma(convDims.inputChannel, llvm::errs() << "InputChannel: ");
  //  llvm::errs() << "\n";
  //  llvm::interleaveComma(convDims.depth, llvm::errs() << "Depth: ");
  //  llvm::errs() << "\n";
  //  llvm::interleaveComma(convDims.strides, llvm::errs() << "Strides: ");
  //  llvm::errs() << "\n";
  //  llvm::interleaveComma(convDims.dilations, llvm::errs() << "Dilations: ");
  //  llvm::errs() << "\n";
  //  llvm::errs() << "\n";
  //}

  // TODO: Support depthwise convolutions
  if (!convDims.depth.empty())
    return failure();

  Value input = convOp->getOperand(0);
  Value filter = convOp->getOperand(1);
  Value output = convOp->getOperand(2);

  auto inputMap = convOp.getIndexingMapsArray()[0];
  auto filterMap = convOp.getIndexingMapsArray()[1];
  auto outputMap = convOp.getIndexingMapsArray()[2];

  auto inputIndices = collectChannelTransposeIndices(inputMap, {convDims.inputChannel});
  auto filterIndices = collectChannelTransposeIndices(filterMap,
          {convDims.inputChannel, convDims.outputChannel});
  auto outputIndices = collectChannelTransposeIndices(outputMap, {convDims.outputChannel});

  // Don't transpose if there's no change to the op.
  if (isInnerIdentityIndices(inputIndices, inputMap.getNumResults()) &&
          isInnerIdentityIndices(filterIndices, filterMap.getNumResults()) &&
          isInnerIdentityIndices(outputIndices, outputMap.getNumResults()))
    return failure();

  auto [transposedInput, inputPack, transposedInputMap] =
      createTransposeAsTensorPack(rewriter, loc, input, inputMap, inputIndices);
  auto [transposedFilter, filterPack, transposedFilterMap] =
      createTransposeAsTensorPack(rewriter, loc, filter, filterMap, filterIndices);
  auto [transposedOutput, outputPack, transposedOutputMap] =
      createTransposeAsTensorPack(rewriter, loc, output, outputMap, outputIndices);

  // Don't transpose if there's no change to the op.
  if (transposedInputMap == inputMap &&
          transposedFilterMap == filterMap &&
          transposedOutputMap == outputMap)
    return failure();

  Value convDest = transposedOutput;
  if (auto fillOp = output.getDefiningOp<linalg::FillOp>()) {
    if (outputPack) {
      auto outputDest = outputPack->getDest().getDefiningOp<tensor::EmptyOp>();
      auto elementType = outputDest.getType().getElementType();

      auto dimToTileMapping = outputPack->getDimAndTileMapping();
      SmallVector<OpFoldResult> mixedSizes = outputDest.getMixedSizes();
      SmallVector<OpFoldResult> collapsedSizes;
      for (auto [index, size] : llvm::enumerate(mixedSizes))
        if (!dimToTileMapping.count(index))
          collapsedSizes.push_back(size);

      auto emptyOp =
        rewriter.create<tensor::EmptyOp>(loc, collapsedSizes, elementType);

      convDest =
        rewriter.create<linalg::FillOp>(loc, fillOp.getInputs(), emptyOp.getResult()).result();
    }
  }

  SmallVector<unsigned> newDimOrder;
  newDimOrder.append(convDims.batch);
  newDimOrder.append(convDims.outputImage);
  newDimOrder.append(convDims.outputChannel);
  newDimOrder.append(convDims.filterLoop);
  newDimOrder.append(convDims.inputChannel);

  Value transposedConvResult = convBuilder(rewriter, loc, convOp,
          transposedInput, transposedFilter, convDest,
          transposedInputMap, transposedFilterMap, transposedOutputMap, newDimOrder);

  auto returnToNCHW =
      createTransposeAsTensorUnPack(rewriter, loc,
              transposedConvResult, outputIndices);

  rewriter.replaceOp(convOp, returnToNCHW);
  return success();
}

namespace {

/*
 *  Convolution conversion patterns
 */

struct ConvertLinalgConvNchwFchw : OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  using OpRewritePattern::OpRewritePattern;
  ConvertLinalgConvNchwFchw(MLIRContext *context, PatternBenefit benefit = 2) :
      OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    return transposeConvLikeLinalgOp(rewriter, convOp,
            namedConvBuilderFn<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>);
  }
};

struct ConvertLinalgPoolingNchwMax
    : OpRewritePattern<linalg::PoolingNchwMaxOp> {
  using OpRewritePattern::OpRewritePattern;
  ConvertLinalgPoolingNchwMax(MLIRContext *context, PatternBenefit benefit = 2) :
      OpRewritePattern<linalg::PoolingNchwMaxOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::PoolingNchwMaxOp poolOp,
                                PatternRewriter &rewriter) const override {
    return transposeConvLikeLinalgOp(rewriter, poolOp,
            namedConvBuilderFn<linalg::PoolingNchwMaxOp, linalg::PoolingNhwcMaxOp>);
  }
};

struct ConvertLinalgPoolingNchwSum
    : OpRewritePattern<linalg::PoolingNchwSumOp> {
  using OpRewritePattern::OpRewritePattern;
  ConvertLinalgPoolingNchwSum(MLIRContext *context, PatternBenefit benefit = 2) :
      OpRewritePattern<linalg::PoolingNchwSumOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::PoolingNchwSumOp poolOp,
                                PatternRewriter &rewriter) const override {
    return transposeConvLikeLinalgOp(rewriter, poolOp,
            namedConvBuilderFn<linalg::PoolingNchwMaxOp, linalg::PoolingNhwcSumOp>);
  }
};

struct ConvertLinalgConvOp
    : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  ConvertLinalgConvOp(MLIRContext *context, PatternBenefit benefit = 1) :
      OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return transposeConvLikeLinalgOp(rewriter, op);
  }
};

template <typename PackOrUnPackOpTy>
class GeneralizeUntiledPackOrUnPackOp final
    : public OpRewritePattern<PackOrUnPackOpTy> {
 public:
  using OpRewritePattern<PackOrUnPackOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOrUnPackOpTy op,
                                PatternRewriter &rewriter) const override {
    if (!op.getMixedTiles().empty())
      return failure();
    TransposeIndices perm(op.getOuterDimsPerm());
    if (std::is_same<PackOrUnPackOpTy, tensor::UnPackOp>::value)
      perm = invertIndices(perm);
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(op,
            op.getSource(), op.getDest(), perm);
    return success();
  }
};

class GeneralizeLinalgTransposeOp final
    : public OpRewritePattern<linalg::TransposeOp> {
 public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = cast<linalg::LinalgOp>(*op);
    auto transpose = rewriter.create<linalg::GenericOp>(
      op.getLoc(), op.getResult().getType(), op.getInput(), op.getInit(),
      linalgOp.getIndexingMapsArray(), linalgOp.getIteratorTypesArray(),
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args[0]);
      }).getResult(0);
    rewriter.replaceOp(op, transpose);
    return success();
  }
};

class FoldCancellingUnPackPackOps final
    : public OpRewritePattern<tensor::UnPackOp> {
 public:
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    return tensor::UnPackOp::canonicalize(unpackOp, rewriter);
  }
};

class FoldCancellingPackUnPackOps final
    : public OpRewritePattern<tensor::PackOp> {
 public:
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    return tensor::PackOp::canonicalize(packOp, rewriter);
  }
};

struct ConvertConvToChannelsLastPass
    : public ConvertConvToChannelsLastBase<ConvertConvToChannelsLastPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = &getContext();

    {
      RewritePatternSet patterns(context);
      patterns.insert<ConvertLinalgConvNchwFchw>(context);
      patterns.insert<ConvertLinalgPoolingNchwMax>(context);
      patterns.insert<ConvertLinalgPoolingNchwSum>(context);
      patterns.insert<ConvertLinalgConvOp>(context);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(context);
      linalg::populateDataLayoutPropagationPatterns(
              patterns, [](Operation *op) { return true; });
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(context);
      patterns.insert<FoldCancellingPackUnPackOps>(context);
      patterns.insert<FoldCancellingUnPackPackOps>(context);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(context);
      patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern,
                   linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(context);
      patterns.insert<GeneralizeLinalgTransposeOp>(context);
      patterns.insert<GeneralizeUntiledPackOrUnPackOp<tensor::PackOp>>(context);
      patterns.insert<GeneralizeUntiledPackOrUnPackOp<tensor::UnPackOp>>(context);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
createConvertConvToChannelsLastPass() {
  return std::make_unique<ConvertConvToChannelsLastPass>();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
