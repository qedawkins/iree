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

static TransposeIndices invertIndices(TransposeIndices targetIndices) {
  int startDim = *std::min_element(targetIndices.begin(), targetIndices.end());
  TransposeIndices inverted(targetIndices.size());
  for (auto i : llvm::enumerate(targetIndices)) {
    inverted[i.value() - startDim] = i.index() + startDim;
  }
  return inverted;
}

static bool isMinorIdentityIndices(TransposeIndices indices) {
  return llvm::all_of(llvm::enumerate(indices), [indices](auto e) {
            if (e.index() == 0) return true;
            return indices[e.index()-1] < e.value();
          });
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

// Transpose the given tensor based on the given transpose indices. Marks the
// created transpose based on the propagation direction.
static std::tuple<Value, AffineMap>
createTransposeAsTensorPack(PatternRewriter &rewriter, Location loc,
                             Value input, AffineMap inputMap, TransposeIndices targetIndices) {
  if (isMinorIdentityIndices(targetIndices))
    return std::make_tuple(input, inputMap);

  // Pack the input tensor.
  auto empty = tensor::PackOp::createDestinationTensor(
      rewriter, loc, input, SmallVector<OpFoldResult>{}, SmallVector<int64_t>{}, targetIndices);
  auto packedInput = rewriter.create<tensor::PackOp>(
    //loc, input, empty, invertIndices(targetIndices),
    //shuffleFromIndices<OpFoldResult>(tileSizes, invertIndices(targetIndices)),
    loc, input, empty, SmallVector<int64_t>{},
    SmallVector<OpFoldResult>{},
    /*padding=*/std::nullopt, targetIndices);

  SmallVector<AffineExpr> mapResults(inputMap.getResults());
  AffineMap transposedMap = AffineMap::get(inputMap.getNumDims(), inputMap.getNumSymbols(),
          shuffleFromIndices<AffineExpr>(mapResults, targetIndices),
          input.getContext());
  return std::make_tuple(packedInput, transposedMap);
}

// Transpose the given tensor based on the given transpose indices. Marks the
// created transpose based on the propagation direction.
static Value
createTransposeAsTensorUnPack(PatternRewriter &rewriter, Location loc,
                             Value output, TransposeIndices targetIndices) {
  if (isMinorIdentityIndices(targetIndices))
    return output;

  RankedTensorType outType = output.getType().cast<RankedTensorType>();
  auto elementType = outType.getElementType();
  auto outputShape(outType.getShape());

  SmallVector<OpFoldResult> transposedOutputShape;
  for (int64_t i = 0, end = outType.getRank(); i < end; i++) {
    if (ShapedType::isDynamic(outputShape[i]))
      transposedOutputShape.push_back(rewriter.create<tensor::DimOp>(loc, output, i).getResult());
    else
      transposedOutputShape.push_back(rewriter.getIndexAttr(outputShape[i]));
  }
  transposedOutputShape = shuffleFromIndices<OpFoldResult>(transposedOutputShape, targetIndices);

  Value empty = rewriter.create<tensor::EmptyOp>(
    loc, transposedOutputShape, elementType);

  auto unpackedOutput = rewriter.create<tensor::UnPackOp>(
    loc, output, empty, SmallVector<int64_t>{},
    SmallVector<OpFoldResult>{}, invertIndices(targetIndices));
  return unpackedOutput.getResult();
}

static TransposeIndices collectChannelTransposeIndices(AffineMap map,
        SmallVector<SmallVector<unsigned, 2>> transposeDimTargets) {
  TransposeIndices indices;
  SmallVector<TransposeIndices> channelIndices(transposeDimTargets.size());
  for (auto [index, result] : llvm::enumerate(map.getResults())) {
    // Separate the input channel indices from others while maintaining the order of indices.
    if (result.isa<AffineDimExpr>()) {
      bool foundDim = false;
      for (auto [channelVec, dimCategory] : llvm::zip_equal(channelIndices, transposeDimTargets)) {
        if (llvm::is_contained(dimCategory, result.cast<AffineDimExpr>().getPosition())) {
          foundDim = true;
          channelVec.push_back(index);
          break;
        }
      }
      if (foundDim)
        continue;
    }
    indices.push_back(index);
  }

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
  if (isMinorIdentityIndices(inputIndices) &&
          isMinorIdentityIndices(filterIndices) &&
          isMinorIdentityIndices(outputIndices))
    return failure();

  auto [transposedInput, transposedInputMap] =
      createTransposeAsTensorPack(rewriter, loc, input, inputMap, inputIndices);
  auto [transposedFilter, transposedFilterMap] =
      createTransposeAsTensorPack(rewriter, loc, filter, filterMap, filterIndices);
  auto [transposedOutput, transposedOutputMap] =
      createTransposeAsTensorPack(rewriter, loc, output, outputMap, outputIndices);

  // Don't transpose if there's no change to the op.
  if (transposedInputMap == inputMap &&
          transposedFilterMap == filterMap &&
          transposedOutputMap == outputMap)
    return failure();

  SmallVector<unsigned> newDimOrder;
  newDimOrder.append(convDims.batch);
  newDimOrder.append(convDims.outputImage);
  newDimOrder.append(convDims.outputChannel);
  newDimOrder.append(convDims.filterLoop);
  newDimOrder.append(convDims.inputChannel);

  Value transposedConvResult = convBuilder(rewriter, loc, convOp,
          transposedInput, transposedFilter, transposedOutput,
          transposedInputMap, transposedFilterMap, transposedOutputMap, newDimOrder);

  auto returnToNCHW =
      createTransposeAsTensorUnPack(rewriter, loc,
              transposedConvResult, invertIndices(outputIndices));

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
      //patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern,
      //             linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(context);
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
