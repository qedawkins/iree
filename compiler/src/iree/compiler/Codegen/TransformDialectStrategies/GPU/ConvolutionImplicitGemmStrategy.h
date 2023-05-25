// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/AbstractGemmLikeStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

using iree_compiler::gpu::MMAShape;

class ImplicitGemmStrategy : public AbstractGemmLikeStrategy {
 public:
  ImplicitGemmStrategy(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures,
      bool optUseMmaSync, MMAShape targetWmmaShape)
      : AbstractGemmLikeStrategy(targetWmmaShape),
        ctx(context),
        captures(captures) {
    initDefaultValues(optUseMmaSync);
    adjustBlockTileSizesForShape();
  }

  ImplicitGemmStrategy(const ImplicitGemmStrategy &) = default;
  ImplicitGemmStrategy &operator=(const ImplicitGemmStrategy &) = default;

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedConvolutionCaptures captures;

  void initDefaultValues(bool optUseMmaSync = false);

  void adjustBlockTileSizesForShape();

  int64_t m() const override { return derivedM; }
  int64_t n() const override { return derivedN; }
  int64_t k() const override { return derivedK; }

  int64_t blockTileM() const override {
    assert(blockTileSizes.size() >= 3 && "need at least 3 block tile sizes");
    return blockTileSizes[1];
  }
  int64_t blockTileN() const override {
    assert(blockTileSizes.size() >= 3 && "need at least 3 block tile sizes");
    return blockTileSizes[0];
  }
  int64_t blockTileK() const override { return reductionTileSize; }

  int64_t tiledBlockTileM() const {
    assert(blockTileSizes.size() >= 3 && "need at least 3 block tile sizes");
    return blockTileSizes[1];
  }
  int64_t tiledBlockTileN() const {
    assert(blockTileSizes.size() >= 3 && "need at least 3 block tile sizes");
    if (captures.convolutionDims.outputChannel.size() == 2)
      return blockTileSizes[0] /
             captures.convolutionOpSizes[captures.convolutionDims.outputChannel
                                             .back()];
    return blockTileSizes[0];
  }
  int64_t tiledReductionTileSize() const {
    if (captures.convolutionDims.inputChannel.size() == 2)
      return reductionTileSize /
             captures.convolutionOpSizes[captures.convolutionDims.inputChannel
                                             .back()];
    return reductionTileSize;
  }

  using AbstractGemmLikeStrategy::MappingInfo;

  MappingInfo getBlockMapping() const override {
    assert(batchSize() <= 1 && "More than one batch dimension unsupported");
    SmallVector<int64_t> tileSizes(batchSize(), blockTileSizes[2]);
    if (captures.convolutionDims.outputChannel.size() == 2)
      tileSizes.append({tiledBlockTileN(), tiledBlockTileM()});
    else
      tileSizes.append({tiledBlockTileM(), tiledBlockTileN()});
    SmallVector<Attribute> threadMapping(batchSize(), blockZ(ctx));
    threadMapping.append({blockY(ctx), blockX(ctx)});
    return MappingInfo{/*numThreads=*/{},
                       /*tileSizes=*/tileSizes,
                       /*threadMapping=*/threadMapping};
  }

  // LHS copy or img2col is of size mxk.
  MappingInfo lhsCopyMapping() const override {
    if (captures.convolutionDims.inputChannel.size() == 2)
      return tiledLhsCopyMapping();

    assert(reductionTileSize % lhsCopyVectorSize() == 0 &&
           "vector size must divide reductionTileSize");
    int64_t numThreadsK = reductionTileSize / lhsCopyVectorSize();
    assert(totalNumThreads() % numThreadsK == 0 &&
           "num threads must be divisible by num threads along k");
    int64_t numThreadsM = totalNumThreads() / numThreadsK;
    assert(blockTileM() % numThreadsM == 0 &&
           "blockTileM must be divisible by numThreadsM");
    assert(reductionTileSize % numThreadsK == 0 &&
           "reductionTileSize must be divisible by numThreadsK");

    // Filter does not have the batch dimension so we check where the filter is.
    SmallVector<int64_t> threadCounts((1 - filterLHS) * batchSize(), 0);
    threadCounts.append({numThreadsM, numThreadsK});
    SmallVector<int64_t> tileSizes((1 - filterLHS) * batchSize(), 0);
    tileSizes.append(
        {blockTileM() / numThreadsM, reductionTileSize / numThreadsK});
    return MappingInfo{/*numThreads=*/threadCounts,
                       /*tileSizes=*/tileSizes,
                       /*threadMapping=*/{linearIdX(ctx), linearIdY(ctx)}};
  }
  // RHS copy or img2col is of size kxn.
  MappingInfo rhsCopyMapping() const override {
    if (captures.convolutionDims.outputChannel.size() == 2 ||
        captures.convolutionDims.outputChannel.size() == 2)
      return tiledRhsCopyMapping();

    assert(blockTileN() % rhsCopyVectorSize() == 0 &&
           "vector size must divide blockTileN");
    int64_t numThreadsN = blockTileN() / rhsCopyVectorSize();
    assert(totalNumThreads() % numThreadsN == 0 &&
           "num threads must be divisible by num threads along n");
    int64_t numThreadsK = totalNumThreads() / numThreadsN;
    assert(reductionTileSize % numThreadsK == 0 &&
           "reductionTileSize must be divisible by numThreadsK");
    assert(blockTileN() % numThreadsN == 0 &&
           "blockTileN must be divisible by numThreadsN");

    // Filter does not have the batch dimension so we check where the filter is.
    SmallVector<int64_t> threadCounts(filterLHS * batchSize(), 0);
    threadCounts.append({numThreadsK, numThreadsN});
    SmallVector<int64_t> tileSizes(filterLHS * batchSize(), 0);
    tileSizes.append(
        {reductionTileSize / numThreadsK, blockTileN() / numThreadsN});
    return MappingInfo{/*numThreads=*/threadCounts,
                       /*tileSizes=*/tileSizes,
                       /*threadMapping=*/{linearIdY(ctx), linearIdX(ctx)}};
  }
  // RES copy is of size mxn.
  MappingInfo resCopyMapping() const override {
    if (captures.convolutionDims.outputChannel.size() == 2)
      return tiledResCopyMapping();

    assert(blockTileN() % resCopyVectorSize() == 0 &&
           "vector size must divide n");
    int64_t numThreadsN = blockTileN() / resCopyVectorSize();
    assert(totalNumThreads() % numThreadsN == 0 &&
           "num threads must be divisible by num threads along n");
    int64_t numThreadsM = totalNumThreads() / numThreadsN;
    assert(blockTileM() % numThreadsM == 0 &&
           "blockTileSizes[1] must be divisible by numThreadsM");
    assert(blockTileN() % numThreadsN == 0 &&
           "blockTileSizes[0] must be divisible by numThreadsN");

    SmallVector<int64_t> threadCounts(batchSize(), 0);
    threadCounts.append({numThreadsM, numThreadsN});
    SmallVector<int64_t> tileSizes(batchSize(), 1);
    tileSizes.append({blockTileM() / numThreadsM, blockTileN() / numThreadsN});
    return MappingInfo{/*numThreads=*/threadCounts,
                       /*tileSizes=*/tileSizes,
                       /*threadMapping=*/{linearIdY(ctx), linearIdX(ctx)}};
  }
  // COMPUTE is of size mxn.
  MappingInfo computeMapping() const override {
    SmallVector<int64_t> warpCounts(batchSize(), 0);
    SmallVector<Attribute> threadMapping;
    if (captures.convolutionDims.outputChannel.size() == 2) {
      warpCounts.push_back(numWarps[2]);
      threadMapping.push_back(warpZ(ctx));
    }
    warpCounts.append({numWarps[1], numWarps[0]});
    threadMapping.append({warpY(ctx), warpX(ctx)});
    return MappingInfo{/*numThreads=*/warpCounts,
                       /*tileSizes=*/{},
                       /*threadMapping=*/threadMapping};
  }

  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;

 private:
  int64_t batchSize() const { return captures.convolutionDims.batch.size(); }

  MappingInfo mapThreadsAlongShape(SmallVector<int64_t> tileShape) const {
    int64_t numIds = 0;
    int64_t remainingThreads = totalNumThreads();
    SmallVector<int64_t> tileSizes;
    SmallVector<int64_t> threadCounts;
    for (int64_t size : tileShape) {
      if (size % 2 != 0) {
        tileSizes.push_back(0);
        threadCounts.push_back(0);
        continue;
      }
      int64_t tileSize = size;
      int64_t threadCount = 1;
      while (tileSize % 2 == 0 && remainingThreads > 1) {
        tileSize /= 2;
        remainingThreads /= 2;
        threadCount *= 2;
      }
      tileSizes.push_back(tileSize);
      threadCounts.push_back(threadCount);
      numIds++;
      if (remainingThreads == 1) break;
    }
    // llvm::interleaveComma(tileShape, llvm::errs() << "tile shape: ");
    // llvm::errs() << "\n";
    // llvm::interleaveComma(tileSizes, llvm::errs() << "tile sizes: ");
    // llvm::errs() << "\n";
    // llvm::interleaveComma(threadCounts, llvm::errs() << "thread counts: ");
    // llvm::errs() << "\n";
    assert(numIds <= 3 && "Can only map up to 3 ids");
    assert(remainingThreads == 1 && "Failed to map all threads in copy");
    SmallVector<Attribute> threadMapping;
    switch (numIds) {
      case 1:
        threadMapping = {linearIdX(ctx)};
        break;
      case 2:
        threadMapping = {linearIdY(ctx), linearIdX(ctx)};
        break;
      case 3:
        threadMapping = {linearIdZ(ctx), linearIdY(ctx), linearIdX(ctx)};
        break;
    }
    return MappingInfo{/*numThreads=*/threadCounts,
                       /*tileSizes=*/tileSizes,
                       /*threadMapping=*/threadMapping};
  }

  MappingInfo tiledLhsCopyMapping() const {
    SmallVector<int64_t> lhsTileShape;
    for (int i = 0, e = batchSize(); i < e; i++) lhsTileShape.push_back(1);
    lhsTileShape.push_back(tiledReductionTileSize());
    lhsTileShape.push_back(blockTileM());
    lhsTileShape.push_back(
        captures
            .convolutionOpSizes[captures.convolutionDims.inputChannel.back()]);
    return mapThreadsAlongShape(lhsTileShape);
  }

  MappingInfo tiledRhsCopyMapping() const {
    SmallVector<int64_t> rhsTileShape;
    rhsTileShape.push_back(tiledBlockTileN());
    rhsTileShape.push_back(tiledReductionTileSize());
    if (captures.convolutionDims.inputChannel.size() == 2)
      rhsTileShape.push_back(
          captures.convolutionOpSizes[captures.convolutionDims.inputChannel
                                          .back()]);
    if (captures.convolutionDims.outputChannel.size() == 2)
      rhsTileShape.push_back(
          captures.convolutionOpSizes[captures.convolutionDims.outputChannel
                                          .back()]);
    return mapThreadsAlongShape(rhsTileShape);
  }

  MappingInfo tiledResCopyMapping() const {
    SmallVector<int64_t> resTileShape;
    for (int i = 0, e = batchSize(); i < e; i++) resTileShape.push_back(1);
    resTileShape.push_back(tiledBlockTileN());
    resTileShape.push_back(blockTileM());
    resTileShape.push_back(
        captures
            .convolutionOpSizes[captures.convolutionDims.outputChannel.back()]);
    return mapThreadsAlongShape(resTileShape);
  }

  // For NCHW convolutions, the filter will be the LHS of the GEMM.
  bool filterLHS = false;

  int64_t derivedM = 0;
  int64_t derivedN = 0;
  int64_t derivedK = 0;
};

void buildConvolutionImplicitGemmStrategy(ImplicitLocOpBuilder &b,
                                          Value variantH,
                                          const ImplicitGemmStrategy &strategy);

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_
