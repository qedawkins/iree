// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/MatmulImplicitGemmStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::buildPad;
using iree_compiler::buildSelectFirstNonEmpty;
using iree_compiler::buildTileFuseDistToForallWithNumThreads;
using iree_compiler::buildTileFuseDistToForallWithTileSizes;
using iree_compiler::TileToForallAndFuseAndDistributeResult;
using iree_compiler::gpu::buildBufferize;
using iree_compiler::gpu::buildConvertToAsyncCopies;
using iree_compiler::gpu::buildConvertToTensorCoreOp;
using iree_compiler::gpu::buildDistributeCopies;
using iree_compiler::gpu::buildHoistOutputPaddingOp;
using iree_compiler::gpu::buildMatmulVectorization;
using iree_compiler::gpu::buildMultiBuffering;
using iree_compiler::gpu::buildPipelineSharedMemoryCopies;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::gpu::MatmulImplicitGemmStrategy;
using iree_compiler::gpu::scaleUpByBitWidth;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using transform::ConvertConv2DToImg2ColOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::TileToScfForOp;
using transform_ext::RegisterMatchCallbacksOp;

/// Options to set the default values of the matmul strategy.

void MatmulImplicitGemmStrategy::initDefaultValues(bool optUseMmaSync) {
  assert(captures.convolutionDims.batch.size() == 1 &&
         "requires two output channel dimensions");
  assert(captures.convolutionDims.outputChannel.size() <= 2 &&
         "requires two output channel dimensions");
  assert(captures.convolutionDims.inputChannel.size() <= 2 &&
         "requires two input channel dimensions");
  assert(captures.convolutionDims.outputImage.size() == 0 &&
         "requires no output image dimensions");
  assert(captures.convolutionDims.filterLoop.size() == 0 &&
         "requires no filter loop dimensions");

  derivedM = 1;
  for (auto dim : captures.convolutionDims.batch)
    derivedM *= captures.convolutionOpSizes[dim];
  derivedN = 1;
  for (auto dim : captures.convolutionDims.outputChannel)
    derivedN *= captures.convolutionOpSizes[dim];
  derivedK = 1;
  for (auto dim : captures.convolutionDims.inputChannel)
    derivedK *= captures.convolutionOpSizes[dim];

  // Pull in tile configs from flags.
  AbstractGemmLikeStrategy::initDefaultValues(optUseMmaSync);

  // Set the elemental bit widths.
  lhsElementalBitWidth = captures.inputElementType.getIntOrFloatBitWidth();
  rhsElementalBitWidth = captures.filterElementType.getIntOrFloatBitWidth();
  resElementalBitWidth = captures.outputElementType.getIntOrFloatBitWidth();

  // Set the configuration for padding the gemm.
  paddingValueTypes =
      SmallVector<Type>{captures.inputElementType, captures.filterElementType};
  paddingValueTypes.push_back(captures.outputElementType);
  paddingDimensions = {0, 1, 2};
  // TODO: Re-enable once padding works with the img2col op.
  packingDimensions = SmallVector<int64_t>{1, 1, 1};

  // TODO: Enable async-copies.
  useAsyncCopies = false;
  pipelineDepth = k() / reductionTileSize > 2 ? 1 : 0;
}

void MatmulImplicitGemmStrategy::adjustBlockTileSizesForShape() {
  // while (blockTileSizes[0] > n()) blockTileSizes[0] /= 2;
  // while (blockTileSizes[1] > m()) blockTileSizes[1] /= 2;
  // while (reductionTileSize > k()) reductionTileSize /= 2;

  // This is forcing alignment on 16 (alignment precondition).
  // TODO: Enable unaligned tile sizes.
  while (n() % blockTileSizes[0] != 0) blockTileSizes[0] /= 2;
  while (m() % blockTileSizes[1] != 0) blockTileSizes[1] /= 2;
  while (k() % reductionTileSize != 0) reductionTileSize /= 2;

  while (blockTileSizes[0] < numThreads[0] && numWarps[0] > 1) {
    numThreads[0] /= 2;
    numWarps[0] /= 2;
  }
  while (blockTileSizes[1] < (numThreads[1] * numThreads[0]) &&
         numWarps[1] > 1) {
    numThreads[1] /= 2;
    numWarps[1] /= 2;
  }

  // Force distribution of leftover output channel along the z-axis.
  if (captures.convolutionDims.outputChannel.size() == 2) {
    if (tiledBlockTileN() != 1) {
      numWarps[2] = tiledBlockTileN();
      numThreads[2] = tiledBlockTileN();
    }
  }

  //// Force distribution of leftover output channel along the x-axis.
  // if (captures.convolutionDims.outputChannel.size() == 2) {
  //   if (tiledBlockTileN() != 1) {
  //     numWarps[0] *= tiledBlockTileN();
  //     numThreads[0] *= tiledBlockTileN();
  //     blockTileSizes[0] /= tiledBlockTileN();
  //   }
  // }
}

LLVM_DUMP_METHOD void MatmulImplicitGemmStrategy::dump() const {
  print(llvm::errs());
}

void MatmulImplicitGemmStrategy::print(llvm::raw_ostream &os) const {
  os << "\n--- Matmul Implicit GEMM strategy ---\n";

  AbstractGemmLikeStrategy::print(os);

  os << "- derived problem shape (MNK): " << m() << ", " << n() << ", " << k()
     << '\n';
  os << "- convolution dim types: \n";
  llvm::interleaveComma(captures.convolutionDims.batch, os << "Batch: ");
  os << "\n";
  llvm::interleaveComma(captures.convolutionDims.outputImage,
                        os << "OutputImage: ");
  os << "\n";
  llvm::interleaveComma(captures.convolutionDims.outputChannel,
                        os << "OutputChannel: ");
  os << "\n";
  llvm::interleaveComma(captures.convolutionDims.filterLoop,
                        os << "FilterLoop: ");
  os << "\n";
  llvm::interleaveComma(captures.convolutionDims.inputChannel,
                        os << "InputChannel: ");
  os << "\n";
  llvm::interleaveComma(captures.convolutionDims.depth, os << "Depth: ");
  os << "\n";
}

static std::tuple<Value, Value, Value, Value>
buildConvolutionStrategyBlockDistribution(
    ImplicitLocOpBuilder &b, Value variantH,
    const MatmulImplicitGemmStrategy &strategy) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [fillH, matmulH, maybeTrailingH] = unpackRegisteredMatchCallback<3>(
      b, "convolution", transform::FailurePropagationMode::Propagate, variantH);

  // Step 2. Create the block/mapping tiling level and fuse.
  auto [fusionTargetH, fusionGroupH] =
      buildSelectFirstNonEmpty(b, maybeTrailingH, matmulH);
  MatmulImplicitGemmStrategy::MappingInfo blockMapping =
      strategy.getBlockMapping();
  TileToForallAndFuseAndDistributeResult tileResult =
      buildTileFuseDistToForallWithTileSizes(
          /*builder=*/b,
          /*isolatedParentOpH=*/variantH,
          /*rootH=*/fusionTargetH,
          /*opsToFuseH=*/fusionGroupH,
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(blockMapping.tileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(blockMapping.threadMapping));

  // Handle the workgroup count region.
  b.create<IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp>(
      tileResult.forallH);

  // Rematch the fill because earlier handle is invalidated.
  Value newFillH =
      b.create<MatchOp>(variantH, linalg::FillOp::getOperationName());
  fillH =
      b.create<FuseIntoContainingOp>(newFillH, tileResult.forallH).getResult();

  auto [blockMatmulH, maybeBlockTrailingH] = buildSelectFirstNonEmpty(
      b, tileResult.resultingFusedOpsHandles.front(), tileResult.tiledOpH);

  // TODO: handle trailing op.
  return std::make_tuple(fillH, blockMatmulH, maybeBlockTrailingH,
                         tileResult.forallH);
}

// TODO: Merge with buildTileFuseToScfFor
static mlir::iree_compiler::TileToScfForAndFuseResult
buildTileFuseToSingleScfFor(ImplicitLocOpBuilder &b, Value isolatedParentOpH,
                            Value rootH, ArrayRef<int64_t> tileSizes) {
  iree_compiler::TileToScfForAndFuseResult result;
  Type rootType = rootH.getType();
  auto tiletoScfForOp = b.create<TileToScfForOp>(
      TypeRange{rootType, rootType}, rootH, ValueRange{}, tileSizes);
  result.forLoops = tiletoScfForOp.getLoops();
  result.tiledOpH = tiletoScfForOp.getTiledLinalgOp();

  // Avoid canonicalization for now to avoid prematurely folding away the pad
  // ops. ApplyPatternsOpPatterns configuration; isolatedParentOpH =
  //     mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
  //         b, configuration, isolatedParentOpH);
  return result;
}

void iree_compiler::gpu::buildMatmulImplicitGemmStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const MatmulImplicitGemmStrategy &strategy) {
  assert(strategy.totalNumThreads() ==
             strategy.totalNumWarps() * kCudaWarpSize &&
         "Number of threads specified by warps must match total number of "
         "threads");
  // Step 1. Apply block-level part of the strategy, keeps everything fused.
  auto [fillH, matmulH, maybeTiledTrailingBlockH, forall] =
      buildConvolutionStrategyBlockDistribution(b, variantH, strategy);
  // Tile reduction loop.
  SmallVector<int64_t> tileSizes;
  for (int i = 0, e = strategy.captures.convolutionDims.outputChannel.size() +
                      strategy.captures.convolutionDims.batch.size();
       i < e; i++) {
    tileSizes.push_back(0);
  }
  tileSizes.push_back(strategy.tiledReductionTileSize());
  auto tileReductionResult =
      buildTileFuseToSingleScfFor(b, variantH, matmulH, tileSizes);

  // Step 2. Pad the matmul op.
  auto paddedMatmulOpH =
      buildPad(b, tileReductionResult.tiledOpH,
               strategy.getZeroPadAttrFromElementalTypes(b).getValue(),
               strategy.paddingDimensions, strategy.packingDimensions);

  // Step 3. Hoist the padding of the output operand above the reduction loop.
  // The resulting fillOp will be mapped with the contraction using an SIMD
  // programming model.
  Value fillOpH;
  if (!strategy.alignedRes()) {
    fillOpH = buildHoistOutputPaddingOp(b, variantH, paddedMatmulOpH);
  } else {
    fillOpH = b.create<transform::MatchOp>(variantH,
                                           linalg::FillOp::getOperationName());
    ApplyPatternsOpPatterns config;
    iree_compiler::buildCanonicalizationAndEnablingTransforms(b, config,
                                                              variantH);
  }

  // Step 4. Distribute pad and copies: SIMT programming model.
  auto [lhsCopyOpH, rhsCopyOpH, copyBackOpH] =
      buildDistributeCopies(b, variantH, paddedMatmulOpH, strategy);

  // Step 4.5. Distribute trailing elementwise: SIMT for WMMA.
  AbstractGemmLikeStrategy::MappingInfo resCopyMapping =
      strategy.resCopyMapping();
  iree_compiler::buildTileFuseDistToForallWithNumThreads(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*rootH=*/maybeTiledTrailingBlockH,
      /*opsHToFuse=*/{},
      /*numThreads=*/
      getAsOpFoldResult(b.getI64ArrayAttr(resCopyMapping.numThreads)),
      /*threadDimMapping=*/b.getArrayAttr(resCopyMapping.threadMapping));

  // Currently we distribute the fill with SIMT for WMMA with trailing
  // elementwise to avoid producing a huge number of scalar loads since we're
  // promoting the C matix anyway.
  if (!strategy.useMmaSync) {
    buildTileFuseDistToForallWithNumThreads(
        b, variantH, fillOpH, ValueRange(),
        getAsOpFoldResult(b.getI64ArrayAttr(resCopyMapping.numThreads)),
        b.getArrayAttr(resCopyMapping.threadMapping));
  }

  // Step 5. Distribute to warps: SIMD programming model.
  // TODO: get the number of warps from strategy.
  MatmulImplicitGemmStrategy::MappingInfo computeMapping =
      strategy.computeMapping();
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, paddedMatmulOpH, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
      b.getArrayAttr(computeMapping.threadMapping));
  if (strategy.useMmaSync) {
    buildTileFuseDistToForallWithNumThreads(
        b, variantH, fillOpH, ValueRange(),
        getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
        b.getArrayAttr(computeMapping.threadMapping));
  }

  // Step 6. Rank-reduce and vectorize.
  buildMatmulVectorization(b, variantH, lhsCopyOpH, rhsCopyOpH, copyBackOpH,
                           strategy);

  // Step 7. Bufferize and drop HAL descriptor from memref ops.
  variantH = buildBufferize(b, variantH);

  // Step 8. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, needs hardening.
  // TODO: extract info from strategy.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, strategy.numThreads,
                                    strategy.numWarps);

  // Step 9. Convert to tensor core ops.
  // TODO: avoid consuming handles and returning here.
  funcH = buildConvertToTensorCoreOp(b, funcH, strategy);

  // TODO: Enable async copies/multibuffering/pipelining.
  if (strategy.pipelineDepth > 0) {
    // Step 10. Multi-buffering.
    buildMultiBuffering(b, funcH, strategy);

    ApplyPatternsOpPatterns patterns;
    patterns.foldMemrefAliases = true;
    b.create<ApplyPatternsOp>(funcH, patterns);

    // TODO: Enable async copies.

    // Step 11. Pipeline shared memory copies.
    buildPipelineSharedMemoryCopies(b, funcH, strategy);
  }

  // Step 13. Late lowerings and cleanups.
  // TODO: not a functional style op to avoid invalidating artificially.
  if (!strategy.alignedLhs() || !strategy.alignedRhs() ||
      !strategy.alignedRes()) {
    funcH = b.create<transform::LowerMasksOp>(
        pdl::OperationType::get(b.getContext()), funcH);
    // TODO: not a functional style op to avoid invalidating artificially.
    funcH = b.create<transform::MaterializeMasksOp>(
        pdl::OperationType::get(b.getContext()), funcH);
  }

  if (strategy.pipelineDepth == 0) {
    ApplyPatternsOpPatterns config;
    config.foldMemrefAliases = true;
    iree_compiler::buildCanonicalizationAndEnablingTransforms(b, config, funcH);
  }
}
