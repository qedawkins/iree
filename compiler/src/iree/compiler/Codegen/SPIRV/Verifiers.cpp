// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"

#define DEBUG_TYPE "iree-spirv-verifier"

namespace mlir {
namespace iree_compiler {

constexpr unsigned kWorkgroupTileLevel = 0;
constexpr unsigned kThreadTileLevel = 1;
constexpr unsigned kReductionTileLevel = 2;

LogicalResult verifySPIRVMatmulPromoteVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  // Verify that the translation info is using the right pipeline.
  if (translationInfo.getDispatchLoweringPassPipeline() !=
      IREE::Codegen::DispatchLoweringPassPipeline::
          SPIRVMatmulPromoteVectorize) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << stringifyEnum(IREE::Codegen::DispatchLoweringPassPipeline::
                                SPIRVMatmulPromoteVectorize);
  }

  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return success();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "verifying op: " << *op << "\n";
    llvm::dbgs() << "chosen workgroup size: [";
    llvm::interleaveComma(workgroupSize, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  // Get spirv.target_env attributes
  const spirv::TargetEnvAttr targetEnvAttr = getSPIRVTargetEnvAttr(op);
  const spirv::TargetEnv targetEnv(targetEnvAttr);
  const auto limits = targetEnv.getResourceLimits();
  LLVM_DEBUG(llvm::dbgs() << "target environment: " << targetEnvAttr << "\n");

  const int subgroupSize = limits.getSubgroupSize();
  const int maxSharedMemory = limits.getMaxComputeSharedMemorySize();
  const int maxThreads = limits.getMaxComputeWorkgroupInvocations();
  const auto maxWorkGroupSize = llvm::to_vector<3>(llvm::map_range(
      limits.getMaxComputeWorkgroupSize().getAsValueRange<IntegerAttr>(),
      [](const APInt &dim) { return dim.getSExtValue(); }));

  // Verify each dimension of workgroupSize should be power of two.
  if (!llvm::isPowerOf2_64(workgroupSize[0]) ||
      !llvm::isPowerOf2_64(workgroupSize[1]) ||
      !llvm::isPowerOf2_64(workgroupSize[2])) {
    return op->emitOpError(
        "expected each workgroup size dimension to be power of two");
  }

  // Verify each dimension of workgroup size should not exceed the corresponding
  // limit of maxWorkGroupSize.
  if (workgroupSize[0] > maxWorkGroupSize[0] ||
      workgroupSize[1] > maxWorkGroupSize[1] ||
      workgroupSize[2] > maxWorkGroupSize[2]) {
    return op->emitOpError("expected workgroup size dimensions not exceeding ")
           << "[" << maxWorkGroupSize[0] << ", " << maxWorkGroupSize[1] << ", "
           << maxWorkGroupSize[2] << "]";
  }

  // Verify the total workgroup size should not exceed maxThreads.
  int64_t totalWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  if (totalWorkgroupSize > maxThreads) {
    return op->emitOpError(
               "expected total invocation count in workgroup to be <= ")
           << maxThreads << ", got " << totalWorkgroupSize;
  }

  // Verify the total workgroup size should be multiple of subgroupSize.
  if (totalWorkgroupSize % subgroupSize != 0) {
    return op->emitOpError("expected total workgroup size to be multiple of ")
           << subgroupSize;
  }

  ArrayRef<int64_t> lhsShape =
      op->getOperand(0).getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape =
      op->getOperand(1).getType().cast<ShapedType>().getShape();
  Type inputType = op->getOperand(0).getType();

  SmallVector<int64_t> workgroupTileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
  SmallVector<int64_t> threadTileSizes =
      loweringConfig.getTileSizeVals(kThreadTileLevel);
  SmallVector<int64_t> reductionTileSizes =
      loweringConfig.getTileSizeVals(kReductionTileLevel);

  if (loweringConfig.getTileSizes().size() != 3) {
    return op->emitOpError("expected 3 levels of tiling sizes, got ")
           << loweringConfig.getTileSizes().size();
  }

  // For BatchMatmul, the first dimension is the batch dimension.
  // We don't check the batch.
  if (isa<linalg::BatchMatmulOp>(op)) {
    lhsShape = lhsShape.drop_front(1);
    rhsShape = rhsShape.drop_front(1);
    workgroupTileSizes.erase(workgroupTileSizes.begin());
    threadTileSizes.erase(threadTileSizes.begin());
    reductionTileSizes.erase(reductionTileSizes.begin());
  }

  // Verify the tile size divides the matmul inputs A [M x K] & B [K x N].
  const int64_t dimM = lhsShape[0], dimN = rhsShape[1], dimK = lhsShape[1];
  if (dimM % workgroupTileSizes[0] != 0 || dimK % reductionTileSizes[2] != 0) {
    return op->emitOpError("LHS shape is indivisible by first level tile size");
  }
  if (dimK % reductionTileSizes[2] != 0 || dimN % workgroupTileSizes[1] != 0) {
    return op->emitOpError("RHS shape is indivisible by first level tile size");
  }

  // Verify that workgroup_tile_size = thread_tile_size * workgroup_size.
  if (threadTileSizes[0] * workgroupSize[1] != workgroupTileSizes[0] ||
      threadTileSizes[1] * workgroupSize[0] != workgroupTileSizes[1]) {
    return op->emitOpError(
        "expected workgroup tile sizes to be the product of thread tile "
        "sizes and workgroup sizes");
  }

  auto pipelineDepth = translationInfo.getSoftwarePipelineDepth();
  pipelineDepth = pipelineDepth ? pipelineDepth : 1;

  // Verify shared memory usage of operands after tiling <= maxSharedMemory.
  unsigned tilingSharedMemSizeBytes = getTileBytes(
      workgroupTileSizes[0], workgroupTileSizes[1], reductionTileSizes[2],
      inputType.cast<ShapedType>().getElementType().getIntOrFloatBitWidth());
  unsigned totalSharedMemSizeBytes = getMultiBufferMemoryUsage(
      tilingSharedMemSizeBytes, pipelineDepth,
      translationInfo.getSoftwarePipelineStoreStage());

  if (totalSharedMemSizeBytes > maxSharedMemory) {
    return op->emitOpError("expected shared memory usage <= ")
           << maxSharedMemory << ", got " << totalSharedMemSizeBytes;
  }
  return success();
}

LogicalResult verifySPIRVCooperativeMatrixVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  // Verify that the translation info is using the right pipeline.
  if (translationInfo.getDispatchLoweringPassPipeline() !=
      IREE::Codegen::DispatchLoweringPassPipeline::
          SPIRVCooperativeMatrixVectorize) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << stringifyEnum(IREE::Codegen::DispatchLoweringPassPipeline::
                                SPIRVCooperativeMatrixVectorize);
  }

  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return success();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "verifying op: " << *op << "\n";
    llvm::dbgs() << "chosen workgroup size: [";
    llvm::interleaveComma(workgroupSize, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  // Get spirv.target_env attributes
  const spirv::TargetEnvAttr targetEnvAttr = getSPIRVTargetEnvAttr(op);
  const spirv::TargetEnv targetEnv(targetEnvAttr);
  const auto limits = targetEnv.getResourceLimits();
  LLVM_DEBUG(llvm::dbgs() << "target environment: " << targetEnvAttr << "\n");

  const int subgroupSize = limits.getSubgroupSize();
  const int maxSharedMemory = limits.getMaxComputeSharedMemorySize();
  const int maxThreads = limits.getMaxComputeWorkgroupInvocations();
  const auto maxWorkGroupSize = llvm::to_vector<3>(llvm::map_range(
      limits.getMaxComputeWorkgroupSize().getAsValueRange<IntegerAttr>(),
      [](const APInt &dim) { return dim.getSExtValue(); }));

  // Verify each dimension of workgroupSize should be power of two.
  if (!llvm::isPowerOf2_64(workgroupSize[0]) ||
      !llvm::isPowerOf2_64(workgroupSize[1]) ||
      !llvm::isPowerOf2_64(workgroupSize[2])) {
    return op->emitOpError(
        "expected each workgroup size dimension to be power of two");
  }

  // Verify each dimension of workgroup size should not exceed the corresponding
  // limit of maxWorkGroupSize.
  if (workgroupSize[0] > maxWorkGroupSize[0] ||
      workgroupSize[1] > maxWorkGroupSize[1] ||
      workgroupSize[2] > maxWorkGroupSize[2]) {
    return op->emitOpError("expected workgroup size dimensions not exceeding ")
           << "[" << maxWorkGroupSize[0] << ", " << maxWorkGroupSize[1] << ", "
           << maxWorkGroupSize[2] << "]";
  }

  // Verify the total workgroup size should not exceed maxThreads.
  int64_t totalWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  if (totalWorkgroupSize > maxThreads) {
    return op->emitOpError(
               "expected total invocation count in workgroup to be <= ")
           << maxThreads << ", got " << totalWorkgroupSize;
  }

  // Verify the total workgroup size should be multiple of subgroupSize.
  if (totalWorkgroupSize % subgroupSize != 0) {
    return op->emitOpError("expected total workgroup size to be multiple of ")
           << subgroupSize;
  }

  // Verify the total workgroup size should be equal or larger than 2 *
  // subgroupSize.
  if (totalWorkgroupSize / subgroupSize < 2) {
    return op->emitOpError("expected total workgroup size to be >= ")
           << 2 * subgroupSize;
  }

  // Verify that there are four level of tile sizes.
  if (loweringConfig.getTileSizes().size() != 4) {
    return op->emitOpError("expected 4 levels of tiling sizes, got ")
           << loweringConfig.getTileSizes().size();
  }

  ArrayRef<int64_t> lhsShape =
      op->getOperand(0).getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape =
      op->getOperand(1).getType().cast<ShapedType>().getShape();

  SmallVector<int64_t> workgroupTileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
  SmallVector<int64_t> subgroupTileSizes =
      loweringConfig.getTileSizeVals(kThreadTileLevel);
  SmallVector<int64_t> reductionTileSizes =
      loweringConfig.getTileSizeVals(kReductionTileLevel);
  SmallVector<int64_t> nativeVectorSizes = loweringConfig.getTileSizeVals(3);

  // For BatchMatmul, the first dimension is the batch dimension.
  // We don't check the batch.
  if (isa<linalg::BatchMatmulOp>(op)) {
    lhsShape = lhsShape.drop_front(1);
    rhsShape = rhsShape.drop_front(1);
    workgroupTileSizes.erase(workgroupTileSizes.begin());
    subgroupTileSizes.erase(subgroupTileSizes.begin());
    reductionTileSizes.erase(reductionTileSizes.begin());
    nativeVectorSizes.erase(nativeVectorSizes.begin());
  }

  auto getElementType = [](Value v) {
    return v.getType().cast<ShapedType>().getElementType();
  };

  Type lhsType = getElementType(op->getOperand(0));
  Type rhsType = getElementType(op->getOperand(1));
  Type resultType = getElementType(op->getOperand(2));

  auto properties = limits.getCooperativeMatrixPropertiesNv()
                        .getAsRange<spirv::CooperativeMatrixPropertiesNVAttr>();

  // Verify that the fourth level tile sizes match cooperative matrix,
  // and subgroup tile sizes should be multiple of cooperative matrix (M, N, K)
  // sizes.
  bool isNativeVectorSizeAccepted = false;
  for (auto p : properties) {
    if (p.getAType() == lhsType && p.getBType() == rhsType &&
        p.getCType() == resultType &&
        p.getScope().getValue() == spirv::Scope::Subgroup &&
        p.getMSize() == nativeVectorSizes[0] &&
        p.getNSize() == nativeVectorSizes[1] &&
        p.getKSize() == nativeVectorSizes[2]) {
      isNativeVectorSizeAccepted = true;
      if (subgroupTileSizes[0] % p.getMSize() != 0 ||
          subgroupTileSizes[1] % p.getNSize() != 0 ||
          reductionTileSizes[2] % p.getKSize() != 0) {
        return op->emitOpError(
                   "expected subgroup tile sizes to be multiple of ")
               << "[" << p.getMSize() << ", " << p.getNSize() << ", "
               << p.getKSize() << "]";
      }
    }
  }

  if (!isNativeVectorSizeAccepted) {
    return op->emitOpError(
        "expected the fourth level tile sizes to match cooperative matrix "
        "sizes");
  }

  // Verify the tile size divides the matmul inputs A [M x K] & B [K x N].
  const int64_t dimM = lhsShape[0], dimN = rhsShape[1], dimK = lhsShape[1];
  if (dimM % workgroupTileSizes[0] != 0 || dimK % reductionTileSizes[2] != 0) {
    return op->emitOpError("LHS shape is indivisible by first level tile size");
  }
  if (dimK % reductionTileSizes[2] != 0 || dimN % workgroupTileSizes[1] != 0) {
    return op->emitOpError("RHS shape is indivisible by first level tile size");
  }

  // Verify workgroup_size_x = warp_size * wg_tile_n / subgroup_tile_n.
  if (workgroupSize[0] * subgroupTileSizes[1] !=
      subgroupSize * workgroupTileSizes[1]) {
    return op->emitOpError(
        "expected workgroup x component equals to (warp_size * wg_tile_n / "
        "subgroup_tile_n)");
  }

  // Verify workgroup_size_y = wg_tile_m / subgroup_tile_m.
  if (workgroupSize[1] * subgroupTileSizes[0] != workgroupTileSizes[0]) {
    return op->emitOpError(
        "expected workgroup y component equals to (wg_tile_m / "
        "subgroup_tile_m)");
  }

  // Verify shared memory usage of operands after tiling <= maxSharedMemory.
  unsigned tilingSharedMemSizeBytes =
      getTileBytes(workgroupTileSizes[0], workgroupTileSizes[1],
                   reductionTileSizes[2], lhsType.getIntOrFloatBitWidth());
  unsigned totalSharedMemSizeBytes = getMultiBufferMemoryUsage(
      tilingSharedMemSizeBytes, translationInfo.getSoftwarePipelineDepth());

  if (totalSharedMemSizeBytes > maxSharedMemory) {
    return op->emitOpError("expected shared memory usage <= ")
           << maxSharedMemory << ", got " << totalSharedMemSizeBytes;
  }
  return success();
}

LogicalResult verifySPIRVBaseVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  // Verify that the translation info is using the right pipeline.
  if (translationInfo.getDispatchLoweringPassPipeline() !=
      IREE::Codegen::DispatchLoweringPassPipeline::SPIRVBaseVectorize) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << stringifyEnum(IREE::Codegen::DispatchLoweringPassPipeline::
                                SPIRVBaseVectorize);
  }

  if (!isa<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(op)) {
    return success();
  }

  const int numTileSizeLevels = loweringConfig.getTileSizes().size();
  SmallVector<int64_t> workgroupTileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
  SmallVector<int64_t> threadTileSizes =
      loweringConfig.getTileSizeVals(kThreadTileLevel);
  SmallVector<int64_t> reductionTileSizes =
      loweringConfig.getTileSizeVals(kReductionTileLevel);

  if (numTileSizeLevels != 4) {
    return op->emitOpError("expected 4 levels of tiling sizes, got ")
           << numTileSizeLevels;
  }

  ArrayRef<int64_t> outputShape =
      op->getOperand(2).getType().cast<ShapedType>().getShape();
  const int64_t oh = outputShape[1], ow = outputShape[2], oc = outputShape[3];

  // Verify the first level tile size divides the Convolution
  // output size [OH, OW, OC].
  if (oh % workgroupTileSizes[1] != 0 || ow % workgroupTileSizes[2] != 0 ||
      oc % workgroupTileSizes[3] != 0) {
    return op->emitOpError(
        "expected first level tile size divides the output size [OH, OW, "
        "OC]");
  }

  // Verify that workgroup_tile_size = thread_tile_size * workgroup_size.
  if (threadTileSizes[1] * workgroupSize[2] != workgroupTileSizes[1] ||
      threadTileSizes[2] * workgroupSize[1] != workgroupTileSizes[2] ||
      threadTileSizes[3] * workgroupSize[0] != workgroupTileSizes[3]) {
    return op->emitOpError(
        "expected workgroup tile sizes to be the product of thread tile size "
        "and workgroup size");
  }

  // Verify that the tile sizes for KH and KW should be 1.
  if (reductionTileSizes[4] != 1 || reductionTileSizes[5] != 1) {
    return op->emitOpError("expected tile sizes for KH and KW to be 1");
  }

  // Verify the fourth level of tile size.
  SmallVector<int64_t> fourthLevelTileSizes = loweringConfig.getTileSizeVals(3);
  if (fourthLevelTileSizes[0] != 0 || fourthLevelTileSizes[1] != 1 ||
      fourthLevelTileSizes[2] != 0 || fourthLevelTileSizes[3] != 0) {
    return op->emitOpError(
        "expected the fourth level of tile size to be [0, 1, 0, 0]");
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
