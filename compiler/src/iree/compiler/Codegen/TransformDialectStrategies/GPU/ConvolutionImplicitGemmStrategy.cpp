// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/ConvolutionImplicitGemmStrategy.h"

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
using iree_compiler::gpu::ImplicitGemmStrategy;
using iree_compiler::gpu::kCudaWarpSize;
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

void ImplicitGemmStrategy::initDefaultValues(bool optUseMmaSync) {
  assert(captures.convolutionDims.outputChannel.size() >= 1 &&
         "requires at least one output channel dimension");
  assert(captures.convolutionDims.inputChannel.size() >= 1 &&
         "requires at least one input channel dimension");
  assert(captures.convolutionDims.outputImage.size() >= 1 &&
         "requires at least one output image dimension");
  assert(captures.convolutionDims.filterLoop.size() >= 1 &&
         "requires at least one filter loop dimension");

  // It is an NCHW conv if the output channel precedes the output image
  // dimensions.
  // TODO: This should be inferred directly from the shape of the input (i.e.
  // input indexing map) rather than overall iterator classes.
  filterLHS = captures.convolutionDims.outputChannel.back() <
              captures.convolutionDims.outputImage.back();

  int64_t channelSize = 1;
  for (auto dim : captures.convolutionDims.outputChannel)
    channelSize *= captures.convolutionOpSizes[dim];
  int64_t imageSize = 1;
  for (auto dim : captures.convolutionDims.outputImage)
    imageSize *= captures.convolutionOpSizes[dim];

  derivedN = channelSize;
  derivedM = imageSize;
  if (filterLHS) std::swap(derivedM, derivedN);

  derivedK = 1;
  for (auto dim : captures.convolutionDims.filterLoop)
    derivedK *= captures.convolutionOpSizes[dim];
  for (auto dim : captures.convolutionDims.inputChannel)
    derivedK *= captures.convolutionOpSizes[dim];

  // Pull in tile configs from flags.
  AbstractGemmLikeStrategy::initDefaultValues(optUseMmaSync);

  // Set the elemental bit widths.
  int64_t inputWidth = captures.inputElementType.getIntOrFloatBitWidth();
  int64_t filterWidth = captures.filterElementType.getIntOrFloatBitWidth();
  lhsElementalBitWidth = filterLHS ? filterWidth : inputWidth;
  rhsElementalBitWidth = filterLHS ? inputWidth : filterWidth;
  resElementalBitWidth = captures.outputElementType.getIntOrFloatBitWidth();

  // Set the configuration for padding the gemm.
  paddingValueTypes = filterLHS ? SmallVector<Type>{captures.filterElementType,
                                                    captures.inputElementType}
                                : SmallVector<Type>{captures.inputElementType,
                                                    captures.filterElementType};
  paddingValueTypes.push_back(captures.outputElementType);
  int64_t batchCount = captures.convolutionDims.batch.size();
  paddingDimensions = {0 + batchCount, 1 + batchCount, 2 + batchCount};
  // TODO: Re-enable once padding works with the img2col op.
  packingDimensions =
      filterLHS ? SmallVector<int64_t>{1, 0, 1} : SmallVector<int64_t>{0, 1, 1};

  // TODO: Enable async-copies and pipelining
  useAsyncCopies = false;
  pipelineDepth = 0;
}

void ImplicitGemmStrategy::adjustBlockTileSizesForShape() {
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
  while (blockTileSizes[1] < (numThreads[1] * numThreads[0]) && numWarps[1] > 1) {
    numThreads[1] /= 2;
    numWarps[1] /= 2;
  }

  while (blockTileSizes[1] / numWarps[1] < 16 && numWarps[1] > 1) {
    numWarps[1] /= 2;
    numThreads[1] /= 2;
  }

  while (blockTileSizes[0] / numWarps[0] < 16 && numWarps[0] > 1) {
    numWarps[0] /= 2;
    numThreads[0] /= 2;
  }

  // Force distribution of leftover output channel along the y-axis.
  if (captures.convolutionDims.outputChannel.size() == 2) {
    if (tiledBlockTileN() != 1) {
      numWarps[1] = tiledBlockTileN();
      numThreads[1] = tiledBlockTileN();
    } else {
      //numWarps[0] *= numWarps[1];
      //numThreads[0] *= numThreads[1];
      numWarps[1] = 1;
      numThreads[1] = 1;
    }
    while (blockTileM() / numWarps[0] < 16 && numWarps[0] > 1) {
      numWarps[0] /= 2;
      numThreads[0] /= 2;
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

LLVM_DUMP_METHOD void ImplicitGemmStrategy::dump() const {
  print(llvm::errs());
}

void ImplicitGemmStrategy::print(llvm::raw_ostream &os) const {
  os << "\n--- Implicit GEMM strategy ---\n";

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

  if (filterLHS)
    os << "- filter is the LHS\n";
  else
    os << "- filter is the RHS\n";
}

static std::tuple<Value, Value, Value, Value, Value>
buildConvolutionStrategyBlockDistribution(
    ImplicitLocOpBuilder &b, Value variantH,
    const ImplicitGemmStrategy &strategy) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [fillH, convolutionH, maybeTrailingH] = unpackRegisteredMatchCallback<3>(
      b, "convolution", transform::FailurePropagationMode::Propagate, variantH);

  // Step 2. Do Img2Col on the convolution to get the GEMM + img2col op.
  Type convType = convolutionH.getType();
  auto conv2DToImg2Col = b.create<ConvertConv2DToImg2ColOp>(
      TypeRange{convType, convType}, convolutionH);
  Value img2colH = conv2DToImg2Col.getImg2colTensor();
  Value transformedH = conv2DToImg2Col.getTransformed();

  // The matmul is the producer of the transformed handle (expand back to
  // convolution shape).
  Value matmulH = b.create<transform::GetProducerOfOperand>(
      transformedH.getType(), transformedH, 0);

  // Bubble the expand_shape from img2col through the trailing elementwise
  ApplyPatternsOpPatterns configuration;
  configuration.bubbleCollapse = true;
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<ApplyPatternsOp>(funcH, configuration);

  // Step 3. Create the block/mapping tiling level and fuse.
  auto [fusionTargetH, fusionGroupH] =
      buildSelectFirstNonEmpty(b, maybeTrailingH, matmulH);
  ImplicitGemmStrategy::MappingInfo blockMapping = strategy.getBlockMapping();
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

  Value tiledImg2colH =
      b.create<FuseIntoContainingOp>(img2colH, tileResult.forallH).getResult();

  auto [blockMatmulH, maybeBlockTrailingH] = buildSelectFirstNonEmpty(
      b, tileResult.resultingFusedOpsHandles.front(), tileResult.tiledOpH);

  // TODO: handle trailing op.
  return std::make_tuple(fillH, tiledImg2colH, blockMatmulH,
                         maybeBlockTrailingH, tileResult.forallH);
}

// TODO: Merge with buildTileFuseToScfFor
static mlir::iree_compiler::TileToScfForAndFuseResult
buildTileFuseToSingleScfFor(ImplicitLocOpBuilder &b, Value isolatedParentOpH,
                            Value rootH, Value opHToFuse,
                            ArrayRef<int64_t> tileSizes) {
  iree_compiler::TileToScfForAndFuseResult result;
  Type rootType = rootH.getType();
  auto tiletoScfForOp = b.create<TileToScfForOp>(
      TypeRange{rootType, rootType}, rootH, ValueRange{}, tileSizes);
  result.forLoops = tiletoScfForOp.getLoops();
  result.tiledOpH = tiletoScfForOp.getTiledLinalgOp();

  assert(result.forLoops.size() == 1 && "More than one loop");

  // TODO: Allow fusing more than one op.
  b.create<FuseIntoContainingOp>(opHToFuse, result.forLoops[0]);

  // Avoid canonicalization for now to avoid prematurely folding away the pad
  // ops. ApplyPatternsOpPatterns configuration; isolatedParentOpH =
  //     mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
  //         b, configuration, isolatedParentOpH);
  return result;
}

void iree_compiler::gpu::buildConvolutionImplicitGemmStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const ImplicitGemmStrategy &strategy) {
  assert(strategy.totalNumThreads() ==
             strategy.totalNumWarps() * kCudaWarpSize &&
         "Number of threads specified by warps must match total number of "
         "threads");
  // Step 1. Apply block-level part of the strategy, keeps everything fused.
  auto [fillH, img2colH, matmulH, maybeTiledTrailingBlockH, forall] =
      buildConvolutionStrategyBlockDistribution(b, variantH, strategy);
  // Tile reduction loop.
  SmallVector<int64_t> tileSizes(strategy.captures.convolutionDims.batch.size(),
                                 0);
  for (int i = 0,
           e = strategy.captures.convolutionDims.outputChannel.size() - 1;
       i < e; i++)
    tileSizes.push_back(0);
  tileSizes.append({0, 0, strategy.tiledReductionTileSize()});
  auto tileReductionResult =
      buildTileFuseToSingleScfFor(b, variantH, matmulH, img2colH, tileSizes);

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
  ImplicitGemmStrategy::MappingInfo computeMapping = strategy.computeMapping();
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
  {
    ApplyPatternsOpPatterns config;
    config.foldMemrefAliases = true;
    iree_compiler::buildCanonicalizationAndEnablingTransforms(b, config, funcH);
  }
}