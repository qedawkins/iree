// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-convert-conv-nchw-to-nhwc"

#define TRANSPOSE_ATTR_NAME "_ConvNchwToNhwcTranspose"
#define GENERIC_ATTR_NAME "_NormalGeneric"
#define CLAST "CLast"
#define CFIRST "CFirst"
#define FLAST "FLast"
#define FFIRST "FFirst"
#define TRANSPOSE_INIT "TransposeInit"

namespace mlir {
namespace iree_compiler {

// Utils ----------------------------------------------

static LogicalResult propagateTagThroughOp(Operation *op) {
  // We don't want to overwrite existing tags
  if (op->hasAttr(TRANSPOSE_ATTR_NAME)) return success();

  if (op->getNumResults() != 1) return success();

  auto result = op->getResults()[0];
  if (result.use_empty()) return success();

  RankedTensorType outputType = dyn_cast<RankedTensorType>(result.getType());
  if (!outputType || outputType.getRank() != 4) return success();

  MLIRContext *context = op->getContext();

  auto owner = result.use_begin()->getOwner();
  if (!owner->hasAttr(TRANSPOSE_ATTR_NAME)) return success();

  Attribute tag = owner->getAttr(TRANSPOSE_ATTR_NAME);

  if (llvm::all_of(op->getResults()[0].getUses(), [&tag](const OpOperand &use) {
        auto owner = use.getOwner();
        return owner->getAttr(TRANSPOSE_ATTR_NAME) == tag;
      })) {
    op->setAttr(TRANSPOSE_ATTR_NAME, tag);
    if (dyn_cast<linalg::GenericOp>(op)) {
      op->setAttr(GENERIC_ATTR_NAME, UnitAttr::get(context));
    }
  }
  return success();
}

static SmallVector<uint64_t> getShuffleIndicesFromTag(MLIRContext *context,
                                                      Attribute tag) {
  SmallVector<uint64_t> targetIndices;
  if (tag == StringAttr::get(context, CLAST)) {
    targetIndices.append({0, 2, 3, 1});
  } else if (tag == StringAttr::get(context, CFIRST)) {
    targetIndices.append({0, 3, 1, 2});
  } else if (tag == StringAttr::get(context, FLAST)) {
    targetIndices.append({2, 3, 1, 0});
  } else {
    targetIndices.append({3, 2, 0, 1});
  }
  return targetIndices;
}

template <typename T>
static SmallVector<T> shuffle4DFromTag(MLIRContext *context,
                                       SmallVector<T> unshuffled,
                                       Attribute tag) {
  SmallVector<uint64_t> targetIndices = getShuffleIndicesFromTag(context, tag);
  SmallVector<T> shuffled(
      {unshuffled[targetIndices[0]], unshuffled[targetIndices[1]],
       unshuffled[targetIndices[2]], unshuffled[targetIndices[3]]});
  return shuffled;
}

static Value create4DTransposeWithAttr(PatternRewriter &rewriter, Location loc,
                                       Value input,
                                       SmallVector<uint64_t> targetIndices,
                                       Attribute tag) {
  RankedTensorType inType = input.getType().cast<RankedTensorType>();
  auto inputRank = inType.getRank();
  assert(inputRank == 4);
  auto elementType = inType.getElementType();
  SmallVector<int64_t> inputShape(inType.getShape());

  MLIRContext *context = rewriter.getContext();

  SmallVector<AffineExpr> idExprs;
  for (auto i = 0; i < inputRank; i++)
    idExprs.push_back(getAffineDimExpr(i, context));

  SmallVector<int64_t> outputShape =
      shuffle4DFromTag<int64_t>(context, inputShape, tag);
  SmallVector<AffineExpr> swapExprs =
      shuffle4DFromTag<AffineExpr>(context, idExprs, tag);

  Value output =
      rewriter.create<linalg::InitTensorOp>(loc, outputShape, elementType);

  output.getDefiningOp()->setAttr(TRANSPOSE_ATTR_NAME,
                                  StringAttr::get(context, TRANSPOSE_INIT));

  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(inputRank, 0, idExprs, context),
      AffineMap::get(inputRank, 0, swapExprs, context)};
  SmallVector<StringRef> iteratorTypes(inputRank, "parallel");
  auto transpose = rewriter.create<linalg::GenericOp>(
      loc, output.getType(), input, output, indexingMaps, iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args[0]);
      });
  transpose->setAttr(TRANSPOSE_ATTR_NAME, tag);
  return transpose.getResult(0);
}

static Value createNchwTransposeWithAttr(PatternRewriter &rewriter,
                                         Location loc, Value input,
                                         bool inputIsNchw) {
  StringAttr tag;
  MLIRContext *context = rewriter.getContext();
  if (inputIsNchw) {
    tag = StringAttr::get(context, CLAST);
  } else {
    tag = StringAttr::get(context, CFIRST);
  }
  SmallVector<uint64_t> targetIndices = getShuffleIndicesFromTag(context, tag);
  return create4DTransposeWithAttr(rewriter, loc, input, targetIndices, tag);
}

static Value createFchwTransposeWithAttr(PatternRewriter &rewriter,
                                         Location loc, Value input,
                                         bool inputIsFchw) {
  StringAttr tag;
  MLIRContext *context = rewriter.getContext();
  if (inputIsFchw) {
    tag = StringAttr::get(rewriter.getContext(), FLAST);
  } else {
    tag = StringAttr::get(rewriter.getContext(), FFIRST);
  }
  SmallVector<uint64_t> targetIndices = getShuffleIndicesFromTag(context, tag);
  return create4DTransposeWithAttr(rewriter, loc, input, targetIndices, tag);
}

static Value createTransposeWithAttrFromTag(PatternRewriter &rewriter,
                                            Location loc, Value input,
                                            Attribute tag, bool inputIsFirst) {
  MLIRContext *context = rewriter.getContext();
  if (!inputIsFirst) {
    if (tag == StringAttr::get(context, CLAST)) {
      tag = StringAttr::get(context, CFIRST);
    } else if (tag == StringAttr::get(context, FLAST)) {
      tag = StringAttr::get(context, FFIRST);
    }
  }
  SmallVector<uint64_t> targetIndices = getShuffleIndicesFromTag(context, tag);
  return create4DTransposeWithAttr(rewriter, loc, input, targetIndices, tag);
}

template <typename ConvOpTy, typename ConvTargetOpTy>
static LogicalResult convertConvLikeNchwToNhwc(PatternRewriter &rewriter,
                                               ConvOpTy convOp,
                                               bool transposeFilter) {
  LLVM_DEBUG(llvm::dbgs() << "inspecting " << convOp << "\n");

  Location loc = convOp.getLoc();

  // This pattern does not handle convolutions with dilation?
  if (auto dilations = convOp.getDilations()) {
    auto values = dilations.template getValues<APInt>();
    if (llvm::any_of(values, [](const APInt &value) {
          return value.getSExtValue() != 1;
        })) {
      return failure();
    }
  }

  Value input = convOp.image();
  Value filter = convOp.filter();
  Value output = convOp.getOutputs()[0];

  auto inputType = input.getType().cast<RankedTensorType>();
  auto filterType = filter.getType().cast<RankedTensorType>();
  auto outputType = output.getType().cast<RankedTensorType>();

  // The filter/input/output view should have static sizes to convert.
  if (!inputType.hasStaticShape() || !filterType.hasStaticShape() ||
      !outputType.hasStaticShape()) {
    return failure();
  }

  // Require rank 4 (might already be in the verifier)
  if (inputType.getRank() != 4 ||
      (transposeFilter && filterType.getRank() != 4)) {
    return failure();
  }

  // Require uniform stride for now
  auto strides = convOp.getStrides().template getValues<int64_t>();
  int64_t commonStride = strides[0];
  if (llvm::any_of(strides, [&commonStride](const int64_t &stride) {
        return stride != commonStride;
      })) {
    return failure();
  }

  auto transposedInput =
      createNchwTransposeWithAttr(rewriter, loc, input, true);
  auto transposedFilter = filter;
  if (transposeFilter)
    transposedFilter = createFchwTransposeWithAttr(rewriter, loc, filter, true);
  auto transposedOutput =
      createNchwTransposeWithAttr(rewriter, loc, output, true);

  auto conv =
      rewriter
          .create<ConvTargetOpTy>(loc, transposedOutput.getType(),
                                  ValueRange{transposedInput, transposedFilter},
                                  transposedOutput, convOp.getStrides(),
                                  convOp.getDilations())
          .getResult(0);

  auto returnToNCHW = createNchwTransposeWithAttr(rewriter, loc, conv, false);

  rewriter.replaceOp(convOp, returnToNCHW);
  return success();
}

// Conversion Patterns ------------------------------------

namespace {

/*
 *  Convolution conversion
 */

struct ConvertLinalgConvNchwFchw : OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    return convertConvLikeNchwToNhwc<linalg::Conv2DNchwFchwOp,
                                     linalg::Conv2DNhwcHwcfOp>(rewriter, convOp,
                                                               true);
  }
};

struct ConvertLinalgPoolingNchwMax
    : OpRewritePattern<linalg::PoolingNchwMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNchwMaxOp poolOp,
                                PatternRewriter &rewriter) const override {
    return convertConvLikeNchwToNhwc<linalg::PoolingNchwMaxOp,
                                     linalg::PoolingNhwcMaxOp>(rewriter, poolOp,
                                                               false);
  }
};

struct ConvertLinalgPoolingNchwSum
    : OpRewritePattern<linalg::PoolingNchwSumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNchwSumOp poolOp,
                                PatternRewriter &rewriter) const override {
    return convertConvLikeNchwToNhwc<linalg::PoolingNchwSumOp,
                                     linalg::PoolingNhwcSumOp>(rewriter, poolOp,
                                                               false);
  }
};

/*
 *  Transpose propagation
 */

struct PropagateThroughTensorPad : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (!padOp->hasAttr(TRANSPOSE_ATTR_NAME)) return failure();
    LLVM_DEBUG(llvm::dbgs() << "propagating " << padOp << "\n");
    Attribute tag = padOp->getAttr(TRANSPOSE_ATTR_NAME);

    Location loc = padOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto input = padOp.getSource();
    SmallVector<OpFoldResult> mixedLow =
        shuffle4DFromTag<OpFoldResult>(context, padOp.getMixedLowPad(), tag);
    SmallVector<OpFoldResult> mixedHigh =
        shuffle4DFromTag<OpFoldResult>(context, padOp.getMixedHighPad(), tag);

    auto transposedInput =
        createTransposeWithAttrFromTag(rewriter, loc, input, tag, true);

    SmallVector<int64_t> outputShape(padOp.getResultType().getShape());
    SmallVector<int64_t> transposedOutputShape =
        shuffle4DFromTag<int64_t>(context, outputShape, tag);
    RankedTensorType transposedOutputType = RankedTensorType::get(
        transposedOutputShape, padOp.getResultType().getElementType());

    auto newPad = rewriter.create<tensor::PadOp>(loc, transposedOutputType,
                                                 transposedInput, mixedLow,
                                                 mixedHigh, padOp.getNofold());
    BlockAndValueMapping mapper;
    padOp.getRegion().cloneInto(&newPad.getRegion(), mapper);
    newPad->removeAttr(TRANSPOSE_ATTR_NAME);

    auto returnToNCHW = createTransposeWithAttrFromTag(
        rewriter, loc, newPad.getResult(), tag, false);

    rewriter.replaceOp(padOp, returnToNCHW);
    return success();
  }
};

struct PropagateThroughLinalgFill : OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (!fillOp->hasAttr(TRANSPOSE_ATTR_NAME)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "propagating " << fillOp << "\n");
    Attribute tag = fillOp->getAttr(TRANSPOSE_ATTR_NAME);

    Location loc = fillOp.getLoc();

    auto transposedOutput = createTransposeWithAttrFromTag(
        rewriter, loc, fillOp.output(), tag, true);

    auto newTensor =
        rewriter.create<linalg::FillOp>(loc, fillOp.value(), transposedOutput)
            .getResult(0);

    auto returnToNCHW =
        createTransposeWithAttrFromTag(rewriter, loc, newTensor, tag, false);

    rewriter.replaceOp(fillOp, returnToNCHW);
    return success();
  }
};

struct PropagateThroughLinalgGeneric : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp->hasAttr(GENERIC_ATTR_NAME)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "propagating " << genericOp << "\n");
    Attribute tag = genericOp->getAttr(TRANSPOSE_ATTR_NAME);

    Location loc = genericOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    // For now we are restricting to single outputs.
    auto output = genericOp.getOutputs()[0];
    auto transposedOutput =
        createTransposeWithAttrFromTag(rewriter, loc, output, tag, true);

    auto indexingMaps = genericOp.getIndexingMapsArray();
    AffineMap outMap = indexingMaps.back();
    SmallVector<AffineExpr> outExprs(outMap.getResults());
    SmallVector<AffineExpr> exprs =
        shuffle4DFromTag<AffineExpr>(context, outExprs, tag);
    indexingMaps[indexingMaps.size() - 1] =
        AffineMap::get(outMap.getNumDims(), outMap.getNumSymbols(), exprs,
                       genericOp->getContext());

    SmallVector<Value> newInputs;
    for (auto input : llvm::enumerate(genericOp.getInputs())) {
      if (input.value().getDefiningOp()->hasAttr(TRANSPOSE_ATTR_NAME)) {
        auto transposedInput = createTransposeWithAttrFromTag(
            rewriter, loc, input.value(), tag, true);
        AffineMap inMap = indexingMaps[input.index()];
        SmallVector<AffineExpr> inputExprs(inMap.getResults());
        SmallVector<AffineExpr> shuffledInputExprs =
            shuffle4DFromTag<AffineExpr>(context, inputExprs, tag);
        indexingMaps[input.index()] =
            AffineMap::get(inMap.getNumDims(), inMap.getNumSymbols(),
                           shuffledInputExprs, genericOp->getContext());
        newInputs.push_back(transposedInput);
      } else {
        newInputs.push_back(input.value());
      }
    }

    SmallVector<StringRef> iteratorTypes = llvm::to_vector(llvm::map_range(
        genericOp.getIteratorTypes(),
        [](Attribute attr) { return attr.cast<StringAttr>().getValue(); }));

    auto newGeneric = rewriter.create<linalg::GenericOp>(
        loc, transposedOutput.getType().cast<RankedTensorType>(), newInputs,
        transposedOutput, indexingMaps, iteratorTypes);
    BlockAndValueMapping mapper;
    genericOp.getRegion().cloneInto(&newGeneric.getRegion(), mapper);
    newGeneric->removeAttr(TRANSPOSE_ATTR_NAME);
    newGeneric->removeAttr(GENERIC_ATTR_NAME);

    auto returnToNCHW = createTransposeWithAttrFromTag(
        rewriter, loc, newGeneric.getResult(0), tag, false);

    rewriter.replaceOp(genericOp, returnToNCHW);
    return success();
  }
};

struct PropagateThroughLinalgInitTensor
    : OpRewritePattern<linalg::InitTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::InitTensorOp initTensorOp,
                                PatternRewriter &rewriter) const override {
    if (!initTensorOp->hasAttr(TRANSPOSE_ATTR_NAME) ||
        initTensorOp->getAttr(TRANSPOSE_ATTR_NAME) ==
            StringAttr::get(initTensorOp.getContext(), TRANSPOSE_INIT)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "propagating " << initTensorOp << "\n");
    Attribute tag = initTensorOp->getAttr(TRANSPOSE_ATTR_NAME);

    Location loc = initTensorOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    SmallVector<OpFoldResult> mixedSizes = shuffle4DFromTag<OpFoldResult>(
        context, initTensorOp.getMixedSizes(), tag);

    auto newTensor = rewriter.create<linalg::InitTensorOp>(
        loc, mixedSizes, initTensorOp.getType().getElementType());
    auto returnToNCHW = createTransposeWithAttrFromTag(
        rewriter, loc, newTensor.getResult(), tag, false);

    rewriter.replaceOp(initTensorOp, returnToNCHW);
    return success();
  }
};

struct PropagateThroughArithConstant : OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                PatternRewriter &rewriter) const override {
    if (!constantOp->hasAttr(TRANSPOSE_ATTR_NAME)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "propagating " << constantOp << "\n");
    Attribute tag = constantOp->getAttr(TRANSPOSE_ATTR_NAME);

    Location loc = constantOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    RankedTensorType outputType =
        dyn_cast<RankedTensorType>(constantOp.getType());

    SmallVector<int64_t> outputShape(outputType.getShape());
    SmallVector<int64_t> transposedOutputShape =
        shuffle4DFromTag<int64_t>(context, outputShape, tag);
    RankedTensorType transposedOutputType = RankedTensorType::get(
        transposedOutputShape, outputType.getElementType());

    DenseElementsAttr elements;
    if (!(elements = constantOp.getValue().dyn_cast<DenseElementsAttr>())) {
      return failure();
    }
    DenseElementsAttr newElements = elements.reshape(transposedOutputType);

    auto newTensor = rewriter.create<arith::ConstantOp>(
        loc, transposedOutputType, newElements);
    auto returnToNCHW = createTransposeWithAttrFromTag(
        rewriter, loc, newTensor.getResult(), tag, false);

    rewriter.replaceOp(constantOp, returnToNCHW);
    return success();
  }
};

/*
 *  Folding away cancelling transposes
 */

struct CancelNCHWToNHWCTranspose : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp transposeOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "trying to fold " << transposeOp << "\n");

    if (!transposeOp->hasAttr(TRANSPOSE_ATTR_NAME)) return failure();

    MLIRContext *context = transposeOp->getContext();

    Attribute transposeType = transposeOp->getAttr(TRANSPOSE_ATTR_NAME);
    StringAttr cLastAttr = StringAttr::get(context, CLAST);
    StringAttr cFirstAttr = StringAttr::get(context, CFIRST);
    StringAttr fLastAttr = StringAttr::get(context, FLAST);
    StringAttr fFirstAttr = StringAttr::get(context, FFIRST);

    if (transposeType == cLastAttr) {
      auto parentOp =
          transposeOp->getOperand(0).getDefiningOp<linalg::GenericOp>();
      if (parentOp && parentOp->getAttr(TRANSPOSE_ATTR_NAME) == cFirstAttr) {
        rewriter.replaceOp(transposeOp, parentOp->getOperand(0));
        return success();
      }
    } else if (transposeType == fLastAttr) {
      auto parentOp =
          transposeOp->getOperand(0).getDefiningOp<linalg::GenericOp>();
      if (parentOp && parentOp->getAttr(TRANSPOSE_ATTR_NAME) == fFirstAttr) {
        rewriter.replaceOp(transposeOp, parentOp->getOperand(0));
        return success();
      }
    }

    return failure();
  }
};

struct LinalgConvNCHWToNHWCPass
    : public LinalgConvNCHWToNHWCBase<LinalgConvNCHWToNHWCPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    {
      RewritePatternSet patterns(context);
      patterns.insert<ConvertLinalgConvNchwFchw>(context);
      patterns.insert<ConvertLinalgPoolingNchwMax>(context);
      patterns.insert<ConvertLinalgPoolingNchwSum>(context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    auto transposePropagationFn = [&](Operation *op) -> WalkResult {
      return TypeSwitch<Operation *, LogicalResult>(op)
          .Case<tensor::PadOp, linalg::FillOp, linalg::InitTensorOp,
                // linalg::GenericOp, arith::ConstantOp>([&](auto taggableOp) {
                linalg::GenericOp>([&](auto taggableOp) {
            return propagateTagThroughOp(taggableOp);
          })
          .Default([&](Operation *op) -> LogicalResult { return success(); });
    };

    for (Block &block : llvm::reverse(funcOp.getBody().getBlocks())) {
      for (Operation &op : llvm::reverse(block.getOperations())) {
        transposePropagationFn(&op);
      }
    }

    {
      RewritePatternSet patterns(context);
      patterns.insert<PropagateThroughTensorPad>(context);
      patterns.insert<PropagateThroughLinalgInitTensor>(context);
      patterns.insert<PropagateThroughLinalgFill>(context);
      patterns.insert<PropagateThroughLinalgGeneric>(context);
      // patterns.insert<PropagateThroughArithConstant>(context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(context);
      patterns.insert<CancelNCHWToNHWCTranspose>(context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgConvNCHWToNHWCPass() {
  return std::make_unique<LinalgConvNCHWToNHWCPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
