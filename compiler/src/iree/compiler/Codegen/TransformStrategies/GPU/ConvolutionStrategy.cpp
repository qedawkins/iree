// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/GPU/ConvolutionStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/MappingInfo.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/MathExtras.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// TODO: significantly better namespacing.
using iree_compiler::buildPad;
using iree_compiler::buildSelectFirstNonEmpty;
using iree_compiler::buildTileFuseDistToForallWithNumThreads;
using iree_compiler::buildTileFuseDistToForallWithTileSizes;
using iree_compiler::TileToForallAndFuseAndDistributeResult;
using iree_compiler::gpu::buildBufferize;
using iree_compiler::gpu::buildConvertToAsyncCopies;
using iree_compiler::gpu::buildConvertToTensorCoreOp;
using iree_compiler::gpu::buildDistributeMatmulCopies;
using iree_compiler::gpu::ConvolutionStrategy;
using iree_compiler::gpu::MappingInfo;
using iree_compiler::IREE::transform_dialect::ApplyBufferOptimizationsOp;
using iree_compiler::IREE::transform_dialect::
    ApplyFoldReshapeIntoTensorHalInterfacePatternsOp;
using iree_compiler::IREE::transform_dialect::EliminateGpuBarriersOp;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform_ext::RegisterMatchCallbacksOp;

static llvm::cl::list<int64_t> clBlockTileSizes(
    "td-convolution-strategy-blk-sizes",
    llvm::cl::desc("block tile size for dims (x,y,z) for the transform "
                   "dialect convolution strategy"),
    llvm::cl::CommaSeparated);
static llvm::cl::list<int64_t> clNumThreads(
    "td-convolution-strategy-num-threads",
    llvm::cl::desc("number of threads for dims (x,y,z) for the transform "
                   "dialect convolution strategy"),
    llvm::cl::CommaSeparated);
static llvm::cl::list<int64_t> clNumWarps(
    "td-convolution-strategy-num-warps",
    llvm::cl::desc("number of warps for dims (x,y,z) for the transform "
                   "dialect convolution strategy"),
    llvm::cl::CommaSeparated);

void ConvolutionStrategy::initDefaultValues(const GPUModel &gpuModel) {
  blockTileSizes =
      SmallVector<int64_t>{clBlockTileSizes.begin(), clBlockTileSizes.end()};
  numThreads = SmallVector<int64_t>{clNumThreads.begin(), clNumThreads.end()};
  numWarps = SmallVector<int64_t>{clNumWarps.begin(), clNumWarps.end()};

  /// Default configuration based on hardware properties and problem bit widths.
  if (clBlockTileSizes.getNumOccurrences()) {
    blockTileSizes =
        SmallVector<int64_t>(clBlockTileSizes.begin(), clBlockTileSizes.end());
  } else {
    blockTileSizes = SmallVector<int64_t>{4, 16, 1};
    while (
        captures
            .convolutionOpSizes[captures.convolutionDims.outputImage.front()] %
        blockTileSizes[0])
      blockTileSizes[0] /= 2;
  }

  if (clNumThreads.getNumOccurrences()) {
    numThreads = SmallVector<int64_t>(clNumThreads.begin(), clNumThreads.end());
  } else {
    // Infer from warp counts if present.
    if (clNumWarps.getNumOccurrences()) {
      numThreads = SmallVector<int64_t>(clNumWarps.begin(), clNumWarps.end());
      numThreads[0] *= subgroupSize;
    } else {
      numThreads = SmallVector<int64_t>{64, 1, 1};
    }
  }
  if (clNumWarps.getNumOccurrences()) {
    numWarps = SmallVector<int64_t>(clNumWarps.begin(), clNumWarps.end());
  } else {
    numWarps = numThreads;
    numWarps[0] = mlir::ceilDiv(numWarps[0], subgroupSize);
  }
}

LLVM_DUMP_METHOD void ConvolutionStrategy::dump() const { print(llvm::errs()); }

void ConvolutionStrategy::print(llvm::raw_ostream &os) const {
  os << "\n--- Convolution strategy ---\n";
  os << "- block tile sizes: {";
  bool isFirst = true;
  for (int64_t blockTileSize : blockTileSizes) {
    if (!isFirst)
      os << ", ";
    os << blockTileSize;
    isFirst = false;
  }
  os << "}\n";
  os << "- number of threads: {";
  isFirst = true;
  for (int64_t numThreadsForDim : numThreads) {
    if (!isFirst)
      os << ", ";
    os << numThreadsForDim;
    isFirst = false;
  }
  os << "}\n";

  os << "- number of warps: {";
  isFirst = true;
  for (int64_t numWarpsForDim : numWarps) {
    if (!isFirst)
      os << ", ";
    os << numWarpsForDim;
    isFirst = false;
  }
  os << "\n-- Derived quantities --\n";
  os << "- block mapping:\n";
  getBlockMapping().print(os << "    -> ");
  os << "- compute mapping:\n";
  computeMapping().print(os << "    -> ");
}

// TODO: implement validator.
LogicalResult ConvolutionStrategy::validate(const GPUModel &gpuModel) const {
  return success();
}

static std::tuple<Value, Value, Value, Value, Value>
buildConvolutionStrategyBlockDistribution(ImplicitLocOpBuilder &b,
                                          Value variantH,
                                          const ConvolutionStrategy &strategy) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [padH, fillH, convH, maybeTrailingH] = unpackRegisteredMatchCallback<4>(
      b, "convolution", transform::FailurePropagationMode::Propagate, variantH);

  // Step 2. Create the block/mapping tiling level and fusee.
  auto [fusionTargetH, fusionGroupH] =
      buildSelectFirstNonEmpty(b, maybeTrailingH, convH);
  MappingInfo blockMapping = strategy.getBlockMapping();
  TileToForallAndFuseAndDistributeResult tileResult =
      buildTileFuseDistToForallWithTileSizes(
          /*builder=*/b,
          /*variantH=*/variantH,
          /*rootH=*/fusionTargetH,
          /*opsToFuseH=*/fusionGroupH,
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(blockMapping.tileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(blockMapping.threadMapping));

  auto [blockConvH, maybeBlockTrailingH] = buildSelectFirstNonEmpty(
      b, tileResult.resultingFusedOpsHandles.front(), tileResult.tiledOpH);

  Value fusedPadH =
      b.create<FuseIntoContainingOp>(padH, tileResult.forallH).getFusedOp();
  Value fusedFillH =
      b.create<FuseIntoContainingOp>(fillH, tileResult.forallH).getFusedOp();

  // Handle the workgroup count region.
  b.create<IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp>(
      tileResult.forallH);

  return std::make_tuple(fusedPadH, fusedFillH, blockConvH, maybeBlockTrailingH,
                         tileResult.forallH);
}

/// Builds the common part of the schedule for matmuls and batched matmuls.
static void buildCommonConvolutionLikeThreadSchedule(
    ImplicitLocOpBuilder &b, Value variantH, Value padH, Value fillH,
    Value convH, Value trailingH, const ConvolutionStrategy &strategy) {
  using mlir::iree_compiler::buildLowerVectorMasksAndCleanup;
  using mlir::iree_compiler::buildTileFuseToScfFor;
  using namespace mlir::iree_compiler::gpu;

  // Tile the outer input channel dimension.
  if (strategy.captures.convolutionDims.inputChannel.size() > 1) {
    SmallVector<int64_t> tileSizes(
        strategy.captures.convolutionDims.outputChannel.size(), 0);
    tileSizes.append(strategy.captures.convolutionDims.outputImage.size(), 0);
    // tileSizes.append(strategy.captures.convolutionDims.filterLoop.size(), 0);
    tileSizes.push_back(1);

    // Avoid canonicalizing before the pad to avoid folding away the
    // extract_slice on the output needed to hoist the output pad.
    auto tileReductionResult = buildTileFuseToScfFor(
        b, variantH, convH, {}, getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)),
        /*canonicalize=*/false);
    convH = tileReductionResult.tiledOpH;
  }

  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);

  // Step 5. Tile the filter loop dimensions.
  SmallVector<int64_t> tileSizes(
      strategy.captures.convolutionDims.outputChannel.size(), 0);
  tileSizes.append(strategy.captures.convolutionDims.outputImage.size(), 0);
  tileSizes.append(strategy.captures.convolutionDims.filterLoop.size(), 1);

  auto tileReductionResult = buildTileFuseToScfFor(
      b, variantH, convH, {}, getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)),
      /*canonicalize=*/true);
  Value filterTiledConvH = tileReductionResult.tiledOpH;

  // Step 6. Distribute to threads: SIMT programming model.
  MappingInfo computeMapping = strategy.computeMapping();
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, filterTiledConvH, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
      b.getArrayAttr(computeMapping.threadMapping));
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, fillH, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
      b.getArrayAttr(computeMapping.threadMapping));
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, trailingH, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
      b.getArrayAttr(computeMapping.threadMapping));

  // Step 7. Apply vectorization + cleanups to what remains.
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<ApplyFoldReshapeIntoTensorHalInterfacePatternsOp>(loc);
    b.create<transform::ApplyFoldUnitExtentDimsViaSlicesPatternsOp>(loc);
    b.create<transform::ApplyCastAwayVectorLeadingOneDimPatternsOp>(loc);
  });
  funcH = iree_compiler::buildVectorize(b, funcH,
                                        /*vectorizeNdExtract=*/false,
                                        /*vectorizePadding=*/false,
                                        /*useIreePadHandling=*/true,
                                        /*applyCleanups=*/true);

  // Step 8. Bufferize and drop HAL descriptor from memref ops.
  variantH = buildBufferize(b, variantH);

  // Step 9. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, needs hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH,
                                    /*blockSize=*/strategy.numThreads,
                                    /*warpDims=*/strategy.numWarps,
                                    /*subgroupSize=*/strategy.subgroupSize);
  // This currently spins forever.
  // funcH = b.create<EliminateGpuBarriersOp>(funcH);

  // Step 10. Cleanup.
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);
  b.create<iree_compiler::IREE::transform_dialect::HoistStaticAllocOp>(funcH);
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyFoldMemrefAliasOpsPatternsOp>(loc);
  });
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyExtractAddressComputationsPatternsOp>(loc);
  });
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);

  // Value forH = b.create<transform::MatchOp>(
  //     transform::OperationType::get(b.getContext(), "scf.for"), funcH,
  //     b.getStrArrayAttr({scf::ForOp::getOperationName()}),
  //     /*matchInterfaceEnum=*/transform::MatchInterfaceEnumAttr(),
  //     /*opAttrs=*/DictionaryAttr(),
  //     /*filterResultType=*/TypeAttr());
  // // TODO: At this time, this synchronization is needed for applying the
  // // HoistRedundantVectorTransfersOp transform correctly. This is because the
  // // transform does not take parallelism into accound.
  // // In the future, HoistRedundantVectorTransfersOp + SynchronizeLoopOp need
  // to
  // // be replaced by a single transform.
  // b.create<SynchronizeLoopOp>(forH);

  // TODO: not a functional style transform and avoid returning funcH.
  // funcH = b.create<transform::HoistRedundantVectorTransfersOp>(
  //    transform::AnyOpType::get(b.getContext()), funcH);
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);
  b.create<ApplyBufferOptimizationsOp>(funcH);

  // // Post-hoc elimiation of barriers.
  // funcH = b.create<EliminateGpuBarriersOp>(funcH);

  // Step 11. Late lowerings and cleanups.
  buildLowerVectorMasksAndCleanup(b, funcH);
}

void iree_compiler::gpu::buildConvolutionStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const ConvolutionStrategy &strategy) {
  LLVM_DEBUG(strategy.print(DBGS()));

  // Step 1. Apply block-level part of the strategy, keeps everything fused.
  auto [padH, fillH, convH, trailingH, forall] =
      buildConvolutionStrategyBlockDistribution(b, variantH, strategy);
  buildCommonConvolutionLikeThreadSchedule(b, variantH, padH, fillH, convH,
                                           trailingH, strategy);
}