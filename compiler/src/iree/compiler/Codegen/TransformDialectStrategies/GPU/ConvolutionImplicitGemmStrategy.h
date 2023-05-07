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

class ImplicitGemmStrategy : public AbstractGemmLikeStrategy {
 public:
  ImplicitGemmStrategy(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures)
      : AbstractGemmLikeStrategy(), ctx(context), captures(captures) {
    initDefaultValues();
  }

  ImplicitGemmStrategy(const ImplicitGemmStrategy &) = default;
  ImplicitGemmStrategy &operator=(const ImplicitGemmStrategy &) = default;

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedConvolutionCaptures captures;

  void initDefaultValues() override;

  int64_t m() const override { return derivedM; }
  int64_t n() const override { return derivedN; }
  int64_t k() const override { return derivedK; }

  using AbstractGemmLikeStrategy::MappingInfo;

  MappingInfo getBlockMapping() const override {
    // 2D named convolutions are always batched.
    return MappingInfo{
        /*numThreads=*/{},
        /*tileSizes=*/{blockTileSizes[2], blockTileSizes[1], blockTileSizes[0]},
        /*threadMapping=*/{blockZ(ctx), blockY(ctx), blockX(ctx)}};
  }

  // LHS copy or img2col is of size mxk.
  MappingInfo lhsCopyMapping() const override {
    assert(reductionTileSize % lhsCopyVectorSize() == 0 &&
           "vector size must divide reductionTileSize");
    int64_t numThreadsK = reductionTileSize / lhsCopyVectorSize();
    assert(totalNumThreads() % numThreadsK == 0 &&
           "num threads must be divisible by num threads along k");
    int64_t numThreadsM = totalNumThreads() / numThreadsK;
    assert(blockTileSizes[1] % numThreadsM == 0 &&
           "blockTileSizes[1] must be divisible by numThreadsM");
    assert(reductionTileSize % numThreadsK == 0 &&
           "reductionTileSize must be divisible by numThreadsK");
    // Filter does not have the batch dimension so we check where the filter is.
    return MappingInfo{
        /*numThreads=*/
        filterLHS ? SmallVector<int64_t>{numThreadsM, numThreadsK}
                  : SmallVector<int64_t>{0, numThreadsM, numThreadsK},
        /*tileSizes=*/
        filterLHS ? SmallVector<int64_t>{blockTileSizes[1] / numThreadsM,
                                         reductionTileSize / numThreadsK}
                  : SmallVector<int64_t>{blockTileSizes[1] / numThreadsM,
                                         reductionTileSize / numThreadsK},
        /*threadMapping=*/{linearIdX(ctx), linearIdY(ctx)}};
  }
  // RHS copy or img2col is of size kxn.
  MappingInfo rhsCopyMapping() const override {
    assert(blockTileSizes[0] % rhsCopyVectorSize() == 0 &&
           "vector size must divide blockTileSizes[0]");
    int64_t numThreadsN = blockTileSizes[0] / rhsCopyVectorSize();
    assert(totalNumThreads() % numThreadsN == 0 &&
           "num threads must be divisible by num threads along n");
    int64_t numThreadsK = totalNumThreads() / numThreadsN;
    assert(reductionTileSize % numThreadsK == 0 &&
           "reductionTileSize must be divisible by numThreadsK");
    assert(blockTileSizes[0] % numThreadsN == 0 &&
           "blockTileSizes[0] must be divisible by numThreadsN");
    return MappingInfo{
        /*numThreads=*/
        filterLHS ? SmallVector<int64_t>{0, numThreadsK, numThreadsN}
                  : SmallVector<int64_t>{numThreadsK, numThreadsN},
        /*tileSizes=*/
        filterLHS ? SmallVector<int64_t>{0, reductionTileSize / numThreadsK,
                                         blockTileSizes[0] / numThreadsN}
                  : SmallVector<int64_t>{reductionTileSize / numThreadsK,
                                         blockTileSizes[0] / numThreadsN},
        /*threadMapping=*/{linearIdY(ctx), linearIdX(ctx)}};
  }
  // RES copy is of size mxn.
  MappingInfo resCopyMapping() const override {
    assert(blockTileSizes[0] % resCopyVectorSize() == 0 &&
           "vector size must divide n");
    int64_t numThreadsN = blockTileSizes[0] / resCopyVectorSize();
    assert(totalNumThreads() % numThreadsN == 0 &&
           "num threads must be divisible by num threads along n");
    int64_t numThreadsM = totalNumThreads() / numThreadsN;
    assert(blockTileSizes[1] % numThreadsM == 0 &&
           "blockTileSizes[1] must be divisible by numThreadsM");
    assert(blockTileSizes[0] % numThreadsN == 0 &&
           "blockTileSizes[0] must be divisible by numThreadsN");
    return MappingInfo{
        /*numThreads=*/{0, numThreadsM, numThreadsN},
        /*tileSizes=*/
        {1, blockTileSizes[1] / numThreadsM, blockTileSizes[0] / numThreadsN},
        /*threadMapping=*/{linearIdY(ctx), linearIdX(ctx)}};
  }
  // COMPUTE is of size mxn.
  MappingInfo computeMapping() const override {
    return MappingInfo{/*numThreads=*/{0, numWarps[1], numWarps[2]},
                       /*tileSizes=*/{},
                       /*threadMapping=*/{warpY(ctx), warpX(ctx)}};
  }

  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;

 private:
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
