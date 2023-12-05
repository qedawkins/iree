// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- PropagateLinalgTranspose.cpp - Pass to propagate transposes ---------==//
//
// The pass is to propagate linalg.transpose operations through a restricted
// set of operations based on whether the propagation is locally lucrative.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-opt-propagate-linalg-transpose"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

static bool isIdentityPermutation(ArrayRef<int64_t> perm) {
  for (auto [index, dim] : llvm::enumerate(perm)) {
    if (index != dim) {
      return false;
    }
  }
  return true;
}

static Value createTranspose(OpBuilder &builder, Value source,
                             ArrayRef<int64_t> perm) {
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, source.getLoc(), source);
  applyPermutationToVector(mixedSizes, perm);
  Type elemType = cast<RankedTensorType>(source.getType()).getElementType();
  Value empty =
      builder.create<tensor::EmptyOp>(source.getLoc(), mixedSizes, elemType)
          .getResult();
  return builder
      .create<linalg::TransposeOp>(source.getLoc(), source, empty, perm)
      ->getResult(0);
}

static RankedTensorType getPermutedTensorType(RankedTensorType type,
                                              SmallVector<int64_t> perm) {
  SmallVector<int64_t> permutedShape = applyPermutation(type.getShape(), perm);
  return RankedTensorType::get(permutedShape, type.getElementType());
}

//===----------------------------------------------------------------------===//
// Transpose specialization
//===----------------------------------------------------------------------===//

static bool isaTransposeOpInterface(linalg::LinalgOp linalgOp) {
  // Structural.
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
    return false;

  // Operands and maps.
  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1)
    return false;
  auto mapRange = linalgOp.getIndexingMapsArray();
  if (mapRange.size() != 2 || !mapRange.front().isPermutation() ||
      !mapRange.back().isPermutation() || mapRange.front() == mapRange.back()) {
    return false;
  }
  // Region.
  return llvm::hasSingleElement(linalgOp.getBlock()->getOperations());
}

static void specializeGenericTransposeOp(RewriterBase &rewriter,
                                         linalg::GenericOp genericOp) {
  if (isaTransposeOpInterface(genericOp)) {
    auto mapRange = genericOp.getIndexingMapsArray();
    AffineMap transposeMap = mapRange.front().compose(mapRange.back());
    SmallVector<int64_t> perm;
    for (AffineExpr expr : transposeMap.getResults()) {
      perm.push_back(cast<AffineDimExpr>(expr).getPosition());
    }
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0],
        perm);
  }
}

//===----------------------------------------------------------------------===//
// Transpose propagation
//===----------------------------------------------------------------------===//

namespace {

// Combines two transposes into one. This shouldn't be strictly necessary as
// fusion should cancel inverse transposes, but doing this here can open up
// new propagation opportunities and eases the analysis in fusion.
class ComposeTransposes : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp consumer,
                                PatternRewriter &rewriter) const override {
    Value input = consumer.getInput();
    auto producer = input.getDefiningOp<linalg::TransposeOp>();
    if (!producer) {
      return failure();
    }

    ArrayRef<int64_t> producerPerm = producer.getPermutation();
    ArrayRef<int64_t> consumerPerm = consumer.getPermutation();
    SmallVector<int64_t> composedPerm =
        applyPermutation(producerPerm, consumerPerm);

    Value transposedSource = producer.getInput();
    if (!isIdentityPermutation(composedPerm)) {
      transposedSource =
          createTranspose(rewriter, transposedSource, composedPerm);
    }
    rewriter.replaceOp(consumer, transposedSource);
    return success();
  }
};

// Sinks a transpose through a tensor.extract_slice iff the transpose turns
// the extracted slice into a contiguous slice.
class SinkTransposeThroughExtractSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    Value source = extractOp.getSource();
    auto transposeOp = source.getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp) {
      return failure();
    }

    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    auto invPerm = invertPermutationVector(perm);

    SmallVector<OpFoldResult> offsets = extractOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = extractOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = extractOp.getMixedStrides();
    ArrayRef<int64_t> srcShape = extractOp.getSourceType().getShape();

    applyPermutationToVector(offsets, invPerm);
    applyPermutationToVector(sizes, invPerm);
    applyPermutationToVector(strides, invPerm);
    SmallVector<int64_t> baseShape = applyPermutation(srcShape, invPerm);

    if (!IREE::Flow::isOffsetSizeAndStrideMappableToFlow(offsets, sizes,
                                                         strides, baseShape)) {
      return failure();
    }

    ArrayRef<int64_t> staticSizes = extractOp.getStaticSizes();
    ArrayRef<int64_t> sliceShape = extractOp.getResultType().getShape();
    llvm::SmallDenseSet<unsigned> rankReducingMask =
        *mlir::computeRankReductionMask(staticSizes, sliceShape);

    int64_t dim = 0;
    llvm::SmallDenseMap<int64_t, int64_t> rankReducedMap;
    for (int64_t i = 0, e = perm.size(); i < e; ++i) {
      if (!rankReducingMask.contains(i)) {
        rankReducedMap[i] = dim++;
      }
    }

    SmallVector<int64_t> rankReducedPerm;
    for (int64_t i : perm) {
      if (!rankReducingMask.contains(i)) {
        rankReducedPerm.push_back(rankReducedMap[i]);
      }
    }

    auto rankReducedInvPerm = invertPermutationVector(rankReducedPerm);

    RankedTensorType sliceType = getPermutedTensorType(
        cast<RankedTensorType>(extractOp.getType()), rankReducedInvPerm);
    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        extractOp.getLoc(), sliceType, transposeOp.getInput(), offsets, sizes,
        strides);
    if (!isIdentityPermutation(rankReducedPerm)) {
      slice = createTranspose(rewriter, slice, rankReducedPerm);
    }
    rewriter.replaceOp(extractOp, slice);
    return success();
  }
};

// Sinks a transpose through a tensor.expand_shape.
class SinkTransposeThroughExpandShape
    : public OpRewritePattern<tensor::ExpandShapeOp> {
public:
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    Value source = expandOp.getSrc();
    auto transposeOp = source.getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp) {
      return failure();
    }

    ArrayRef<int64_t> perm = transposeOp.getPermutation();

    auto invPerm = invertPermutationVector(perm);
    SmallVector<ReassociationIndices> reassociations =
        expandOp.getReassociationIndices();
    applyPermutationToVector(reassociations, invPerm);

    SmallVector<int64_t> newInvPerm;
    SmallVector<ReassociationIndices> newReassociations;
    int64_t expandedDim = 0;
    for (auto reassoc : reassociations) {
      ReassociationIndices newReassoc;
      for (auto dim : reassoc) {
        newInvPerm.push_back(dim);
        newReassoc.push_back(expandedDim++);
      }
      newReassociations.push_back(newReassoc);
    }

    auto newPerm = invertPermutationVector(newInvPerm);

    RankedTensorType expandedType = getPermutedTensorType(
        cast<RankedTensorType>(expandOp.getType()), newInvPerm);
    Value transposedReshape = rewriter.create<tensor::ExpandShapeOp>(
        expandOp.getLoc(), expandedType, transposeOp.getInput(),
        newReassociations);
    Value originalReshape =
        createTranspose(rewriter, transposedReshape, newPerm);
    rewriter.replaceOp(expandOp, originalReshape);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {
struct PropagateLinalgTransposePass
    : public PropagateLinalgTransposeBase<PropagateLinalgTransposePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void PropagateLinalgTransposePass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  // First, specialize all transposes to `linalg.transpose`. This dramatically
  // simplifies all subsequent propagation patterns, both in matching and
  // rewriting.
  {
    SmallVector<linalg::GenericOp> genericCandidates;
    funcOp.walk([&](linalg::GenericOp genericOp) {
      if (IREE::Flow::isNonNullAndOutsideDispatch(genericOp)) {
        genericCandidates.push_back(genericOp);
      }
    });
    IRRewriter rewriter(&getContext());
    for (auto genericOp : genericCandidates) {
      rewriter.setInsertionPoint(genericOp);
      specializeGenericTransposeOp(rewriter, genericOp);
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After specializing transpose ops ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  {
    RewritePatternSet bubblingPatterns(context);
    bubblingPatterns.insert<ComposeTransposes>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(bubblingPatterns)))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "\n--- After bubbling transpose ops up ---\n";
      funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    RewritePatternSet sinkingPatterns(context);
    bubblingPatterns.insert<ComposeTransposes>(context);
    sinkingPatterns.insert<SinkTransposeThroughExtractSlice>(context);
    sinkingPatterns.insert<SinkTransposeThroughExpandShape>(context);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(sinkingPatterns)))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "\n--- After sinking transpose ops down ---\n";
      funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  // Re-generalize any remaining transposes. Later pipelines expect it.
  {
    SmallVector<linalg::LinalgOp> transposeCandidates;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (!IREE::Flow::isNonNullAndOutsideDispatch(linalgOp)) {
        return;
      }
      if (isa_and_nonnull<linalg::TransposeOp>(linalgOp.getOperation())) {
        transposeCandidates.push_back(linalgOp);
      }
    });
    IRRewriter rewriter(&getContext());
    for (auto linalgOp : transposeCandidates) {
      rewriter.setInsertionPoint(linalgOp);
      FailureOr<linalg::GenericOp> generalizedOp =
          linalg::generalizeNamedOp(rewriter, linalgOp);
      if (failed(generalizedOp)) {
        linalgOp->emitOpError("failed to generalize operation");
        return signalPassFailure();
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After propagating transpose ops ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createPropagateLinalgTransposePass() {
  return std::make_unique<PropagateLinalgTransposePass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
