// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-tiling-level"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYTILINGLEVELPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUApplyTilingLevelPass final
    : impl::GPUApplyTilingLevelPassBase<GPUApplyTilingLevelPass> {
  using GPUApplyTilingLevelPassBase::GPUApplyTilingLevelPassBase;
  void runOnOperation() override;
};
} // namespace

LogicalResult swapExpandShapeWithSlice(RewriterBase &rewriter,
                                       tensor::ExpandShapeOp expandShapeOp,
                                       tensor::ExtractSliceOp sliceOp) {

  SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();

  // Helper variables and function for accumulating the new offset and length
  // values.
  Location loc = expandShapeOp->getLoc();
  AffineExpr d0, d1, d2;
  bindDims(rewriter.getContext(), d0, d1, d2);
  // Multiply two integers.
  auto mul = [&](OpFoldResult v1, OpFoldResult v2) {
    auto mulMap = AffineMap::get(2, 0, {d0 * d1});
    return affine::makeComposedFoldedAffineApply(rewriter, loc, mulMap,
                                                 {v1, v2});
  };
  auto mulAdd = [&](OpFoldResult v1, OpFoldResult v2, OpFoldResult v3) {
    auto mulMap = AffineMap::get(3, 0, {d0 * d1 + d2});
    return affine::makeComposedFoldedAffineApply(rewriter, loc, mulMap,
                                                 {v1, v2, v3});
  };

  SmallVector<OpFoldResult> outputShape =
      getMixedValues(expandShapeOp.getStaticOutputShape(),
                     expandShapeOp.getOutputShape(), rewriter);

  // Compute new offsets, lengths, and strides.
  SmallVector<OpFoldResult> newOffsets, newLengths, newStrides;

  auto isZeroOffsetAndFullSize = [](OpFoldResult offset, OpFoldResult sliceSize,
                                    OpFoldResult size) {
    if (!isConstantIntValue(offset, 0))
      return false;
    FailureOr<bool> maybeEqual =
        ValueBoundsConstraintSet::areEqual(sliceSize, size);
    return llvm::succeeded(maybeEqual) && maybeEqual.value();
  };

  for (const ReassociationIndices &indices :
       expandShapeOp.getReassociationIndices()) {
    OpFoldResult newOffset = rewriter.getIndexAttr(0);
    OpFoldResult newSize = rewriter.getIndexAttr(1);

    int64_t i = 0;
    int64_t e = indices.size();
    for (; i < e; ++i) {
      int64_t expandedDim = indices[i];
      if (!isConstantIntValue(sizes[expandedDim], 1))
        break;

      newOffset =
          mulAdd(newOffset, outputShape[expandedDim], offsets[expandedDim]);
    }

    if (i != e) {
      int64_t expandedDim = indices[i];
      newOffset =
          mulAdd(newOffset, outputShape[expandedDim], offsets[expandedDim]);
      newSize = sizes[expandedDim];
      i++;
    }

    for (; i < e; ++i) {
      int64_t expandedDim = indices[i];
      OpFoldResult offset = offsets[expandedDim];
      OpFoldResult fullSize = outputShape[expandedDim];
      if (!isZeroOffsetAndFullSize(offset, sizes[expandedDim], fullSize)) {
        return failure();
      }

      newOffset = mul(newOffset, fullSize);
      newSize = mul(newSize, fullSize);
    }

    newOffsets.push_back(newOffset);
    newLengths.push_back(newSize);

    // Only unit stride supported.
    newStrides.push_back(rewriter.getIndexAttr(1));
  }

  // The shape of the result can be obtained from the sizes passed in.
  SmallVector<Value> dynDims;
  SmallVector<int64_t> shape;
  dispatchIndexOpFoldResults(sizes, dynDims, shape);
  RankedTensorType resultType = RankedTensorType::get(
      shape, expandShapeOp.getResultType().getElementType());

  // Create a new ExtractSliceOp and ExpandShapeOp.
  Value newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, expandShapeOp.getSrc(), newOffsets, newLengths, newStrides);
  auto newExpandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
      loc, resultType, newSliceOp, expandShapeOp.getReassociationIndices(),
      sizes);
  rewriter.replaceOp(sliceOp, newExpandShapeOp);
  return success();
}

/// tensor.empty does not define any tensor contents, so an unpadded pack
/// can be folded away.
struct SwapExpandShapeWithSlicePattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = sliceOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
      return failure();
    }

    if (!sliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "unsupported: non-unit stride");
    }

    return swapExpandShapeWithSlice(rewriter, expandOp, sliceOp);
  }
};

/// This collects the set of operations to tile + fuse starting from the given
/// root |op| and walking up to its producers. Stops at operations given by
/// |exclude| which are expected to receive their own independent tiling for the
/// given level.
static llvm::SmallDenseSet<Operation *>
collectTiledAndFusedOps(Operation *op,
                        llvm::SmallDenseSet<TilingInterface> exclude) {
  SmallVector<Operation *> worklist;
  llvm::SmallDenseSet<Operation *> producers;
  worklist.push_back(op);
  producers.insert(op);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      auto producer = operand.get().getDefiningOp<TilingInterface>();
      if (!producer || producers.contains(producer) ||
          exclude.contains(producer))
        continue;
      worklist.push_back(producer);
      producers.insert(producer);
    }
  }
  return producers;
}

/// Apply a tile and fuse transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
static LogicalResult
applyTileAndFuseToEachRoot(RewriterBase &rewriter,
                           llvm::SmallDenseSet<TilingInterface> &payloadOps,
                           IREE::GPU::TilingLevel tilingLevel) {
  MLIRContext *context = rewriter.getContext();
  for (TilingInterface tilingInterfaceOp : payloadOps) {
    mlir::DominanceInfo dominanceInfo(tilingInterfaceOp);

    llvm::SmallDenseSet<Operation *> tiledAndFusedOps =
        collectTiledAndFusedOps(tilingInterfaceOp, payloadOps);
    llvm::DenseSet<Operation *> yieldReplacementsFor;
    for (auto op : tiledAndFusedOps) {
      if (llvm::any_of(op->getUsers(), [&](Operation *user) {
            return dominanceInfo.properlyDominates(tilingInterfaceOp, user);
          })) {
        yieldReplacementsFor.insert(op);
      }
    }

    rewriter.setInsertionPoint(tilingInterfaceOp);
    SmallVector<OpFoldResult> tileSizes =
        getLoweringConfig(tilingInterfaceOp)
            .getTilingLevelSizes(rewriter, llvm::to_underlying(tilingLevel),
                                 tilingInterfaceOp);

    // Pad the tile sizes with zero.
    auto zero = rewriter.getIndexAttr(0);
    int64_t numLoops = tilingInterfaceOp.getLoopIteratorTypes().size();
    if (tileSizes.size() > numLoops) {
      return failure();
    }
    while (tileSizes.size() < numLoops) {
      tileSizes.push_back(zero);
    }

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    if (tilingLevel == IREE::GPU::TilingLevel::Thread ||
        tilingLevel == IREE::GPU::TilingLevel::Subgroup) {
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

      // TODO: Add some helpers to construct this based on the enum type rather
      // than doing it here.
      SmallVector<Attribute> mapping;
      int idx = 0;
      for (auto size : tileSizes) {
        if (!isConstantIntValue(size, 0)) {
          unsigned mappingId =
              static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx++;
          if (tilingLevel == IREE::GPU::TilingLevel::Thread) {
            mapping.push_back(gpu::GPUThreadMappingAttr::get(
                context, static_cast<gpu::MappingId>(mappingId)));
          } else {
            // Else it must be subgroup tiling.
            mapping.push_back(gpu::GPUWarpMappingAttr::get(
                context, static_cast<gpu::MappingId>(mappingId)));
          }
        }
      }
      tilingOptions.setMapping(llvm::to_vector(llvm::reverse(mapping)));
    }

    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);

    scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand)
        -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      Operation *owner = originalProducer.getOwner();
      bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
      bool shouldFuse = false;
      if (auto tilingOwner = dyn_cast<TilingInterface>(owner)) {
        shouldFuse = !payloadOps.contains(tilingOwner);
      }
      // Do not fuse destination operands.
      shouldFuse &= !isDestinationOperand;
      if (shouldFuse) {
        return scf::SCFTileAndFuseOptions::ControlFnResult{
            yieldProducerReplacement};
      }
      return std::nullopt;
    };
    tileAndFuseOptions.setFusionControlFn(controlFn);

    // Avoid cleanup for subgroup level tiling because cleanup/fusion must
    // happen later during lane tiling because failure to fuse at the lane
    // tiling level is irrecoverable if fusion happens now.
    if (tilingLevel != IREE::GPU::TilingLevel::Subgroup) {
      RewritePatternSet patterns(context);
      tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      patterns.add<SwapExpandShapeWithSlicePattern>(context);
      tileAndFuseOptions.cleanupPatterns = std::move(patterns);
    }

    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                  tileAndFuseOptions);
    if (failed(tiledResults)) {
      return failure();
    }

    // Perform the replacement of tiled and fused values.
    SmallVector<Operation *> opsToReplace{tilingInterfaceOp};
    llvm::append_range(opsToReplace, tiledResults->fusedProducers);
    for (Operation *toReplace : opsToReplace) {
      for (OpResult res : toReplace->getResults())
        if (auto replacement = tiledResults->replacements.lookup(res)) {
          Operation *replacementOp = replacement.getDefiningOp();
          rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand &use) {
            Operation *user = use.getOwner();
            return dominanceInfo.properlyDominates(replacementOp, user);
          });
        }

      if (toReplace->use_empty()) {
        rewriter.eraseOp(toReplace);
      }
    }
  }
  return success();
}

static llvm::SmallDenseSet<TilingInterface>
getTiledOps(Operation *funcOp, IREE::GPU::TilingLevel tilingLevel) {
  llvm::SmallDenseSet<TilingInterface> targets;
  unsigned opaqueLevel = llvm::to_underlying(tilingLevel);
  funcOp->walk([&](TilingInterface target) {
    // TODO: This would probably be easier with a lowering config interface
    // method that checks whether a particular level is tiled.
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(target)) {
      if (loweringConfig.hasTilingLevel(opaqueLevel)) {
        targets.insert(target);
      }
    }
  });
  return targets;
}

void GPUApplyTilingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  if (tilingLevel != IREE::GPU::TilingLevel::Reduction &&
      tilingLevel != IREE::GPU::TilingLevel::Thread &&
      tilingLevel != IREE::GPU::TilingLevel::Subgroup) {
    funcOp.emitError() << "unsupported tiling level: "
                       << IREE::GPU::stringifyEnum(tilingLevel) << "\n";
    return signalPassFailure();
  }

  llvm::SmallDenseSet<TilingInterface> targetOps =
      getTiledOps(funcOp, tilingLevel);

  IRRewriter rewriter(funcOp);
  if (failed(applyTileAndFuseToEachRoot(rewriter, targetOps, tilingLevel))) {
    funcOp.emitError() << "tiling of level "
                       << IREE::GPU::stringifyEnum(tilingLevel) << " failed\n";
    return signalPassFailure();
  }

  MLIRContext *context = &getContext();

  // Apply cleanup patterns.
  {
    RewritePatternSet patterns(context);
    // Merge consecutive insert/extract slice ops to simplify later loop
    // hoisting patterns.
    tensor::populateFoldTensorEmptyPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "tiling cleanup failed\n";
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
