// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/AbstractConvolutionStrategy.h"

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

class ConvolutionImplicitGemmStrategy : public AbstractConvolutionStrategy {
 public:
  static ConvolutionImplicitGemmStrategy create(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures,
      const ConvolutionConfig &convolutionConfig);

  ConvolutionImplicitGemmStrategy(const ConvolutionImplicitGemmStrategy &) = default;
  ConvolutionImplicitGemmStrategy &operator=(const ConvolutionImplicitGemmStrategy &) = default;

  std::array<int64_t, 3> getNumThreadsInBlock() const override {
    return {numThreadsXInBlock, 1, 1};
  }

  std::array<int64_t, 3> getNumWarpsInBlock() const override {
    return {numWarpsXInBlock, 1, 1};
  }

  std::array<int64_t, 3> getFullThreadsTileSizes() const {
    if (tileM)
      return {0, numThreadsXInBlock, 0};
    return {0, 0, numThreadsXInBlock};
  }

  std::array<int64_t, 3> getThreadsTileSizes() const override {
    if (tileM)
      return {0, numThreadsXToDistribute, 0};
    return {0, 0, numThreadsXToDistribute};
  }

  std::array<int64_t, 3> getWarpsTileSizes() const override {
    if (tileM)
      return {0, numWarpsXInBlock, 0};
    return {0, 0, numWarpsXInBlock};
  }

  std::array<int64_t, 4> getInnerLoopTileSizes() const override {
    return {0, 0, 0, innerLoopTileSize};
  }

  int64_t getImplicitGemmFilterOperandIndex() const {
    if (captures.convolutionAffineInputDims[0] + 1 == captures.convolutionAffineInputDims[1])
      return 0;
    return 1;
  }

  bool getIsSpirv() const {
    return isSpirv;
  }

 private:
  ConvolutionImplicitGemmStrategy(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures)
      : AbstractConvolutionStrategy(context, captures) {}

  void configure(const ConvolutionConfig &convolutionConfig);

  int64_t numThreadsXInBlock;
  int64_t numThreadsXToDistribute;
  int64_t numWarpsXInBlock;
  int64_t innerLoopTileSize;

  bool tileM = false;
  bool isSpirv = false;
};

void buildConvolutionImplicitGemmStrategy(ImplicitLocOpBuilder &b, Value variantH,
                                  const ConvolutionImplicitGemmStrategy &strategy);

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_
