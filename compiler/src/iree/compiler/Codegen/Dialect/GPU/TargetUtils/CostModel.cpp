// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/CostModel.h"

#include "llvm/Support/MathExtras.h"

namespace mlir::iree_compiler::IREE::GPU {

//===----------------------------------------------------------------------===//
// MMA Register Cost
//===----------------------------------------------------------------------===//

/// Compute elements held per thread from a MMASingleSubgroupLayout.
/// Elements per thread = product(outer) * product(element).
static int64_t
computeElementsPerThread(const MMASingleSubgroupLayout &layout) {
  int64_t result = 1;
  for (int64_t o : layout.outer)
    result *= o;
  for (int64_t e : layout.element)
    result *= e;
  return result;
}

int64_t computeMMAOperandVGPRs(const MMASingleSubgroupLayout &layout,
                               int64_t elementBits) {
  int64_t elts = computeElementsPerThread(layout);
  return llvm::divideCeil(elts * elementBits, 32);
}

MMARegisterCost computeMMARegisterCost(MMAIntrinsic intrinsic,
                                       int64_t lhsBits, int64_t rhsBits,
                                       int64_t accBits) {
  MMASingleSubgroupLayout lhsLayout =
      getSingleSubgroupLayout(intrinsic, kMMAOperandLhs);
  MMASingleSubgroupLayout rhsLayout =
      getSingleSubgroupLayout(intrinsic, kMMAOperandRhs);
  MMASingleSubgroupLayout accLayout =
      getSingleSubgroupLayout(intrinsic, kMMAOperandAcc);

  return MMARegisterCost{
      computeMMAOperandVGPRs(lhsLayout, lhsBits),
      computeMMAOperandVGPRs(rhsLayout, rhsBits),
      computeMMAOperandVGPRs(accLayout, accBits),
  };
}

//===----------------------------------------------------------------------===//
// VGPR Budget
//===----------------------------------------------------------------------===//

std::optional<VGPRBudget> computeVGPRBudget(TargetAttr target,
                                            int64_t occupancy) {
  TargetWgpAttr wgp = target.getWgp();
  std::optional<int32_t> vgprBits = wgp.getVgprSpaceBits();
  if (!vgprBits || occupancy <= 0)
    return std::nullopt;

  int64_t totalVGPRs = *vgprBits / (occupancy * 32);
  return VGPRBudget{totalVGPRs, occupancy};
}

int64_t computeMaxOccupancyFromVGPRs(TargetAttr target, int64_t vgprsUsed) {
  TargetWgpAttr wgp = target.getWgp();
  std::optional<int32_t> vgprBits = wgp.getVgprSpaceBits();
  if (!vgprBits || vgprsUsed <= 0)
    return 0;

  int64_t totalVGPRsAtOcc1 = *vgprBits / 32;
  return totalVGPRsAtOcc1 / vgprsUsed;
}

//===----------------------------------------------------------------------===//
// Accumulator Budget
//===----------------------------------------------------------------------===//

std::optional<int64_t> computeAccumulatorVGPRs(MMAIntrinsic intrinsic,
                                               int64_t subgroupM,
                                               int64_t subgroupN,
                                               int64_t accBits) {
  int64_t mmaM = getMSize(intrinsic);
  int64_t mmaN = getNSize(intrinsic);

  if (subgroupM % mmaM != 0 || subgroupN % mmaN != 0)
    return std::nullopt;

  int64_t numTilesM = subgroupM / mmaM;
  int64_t numTilesN = subgroupN / mmaN;
  int64_t numTiles = numTilesM * numTilesN;

  MMASingleSubgroupLayout accLayout =
      getSingleSubgroupLayout(intrinsic, kMMAOperandAcc);
  int64_t accVGPRsPerTile = computeMMAOperandVGPRs(accLayout, accBits);
  return numTiles * accVGPRsPerTile;
}

//===----------------------------------------------------------------------===//
// Global Load Staging Budget
//===----------------------------------------------------------------------===//

GlobalLoadVGPRs computeGlobalLoadVGPRs(int64_t lhsTileElements,
                                        int64_t rhsTileElements,
                                        int64_t numThreads,
                                        int64_t elementBits) {
  int64_t lhsEltsPerThread = llvm::divideCeil(lhsTileElements, numThreads);
  int64_t rhsEltsPerThread = llvm::divideCeil(rhsTileElements, numThreads);

  int64_t lhsVGPRs = llvm::divideCeil(lhsEltsPerThread * elementBits, 32);
  int64_t rhsVGPRs = llvm::divideCeil(rhsEltsPerThread * elementBits, 32);

  return GlobalLoadVGPRs{lhsVGPRs, rhsVGPRs};
}

//===----------------------------------------------------------------------===//
// LDS Allocation
//===----------------------------------------------------------------------===//

std::optional<LDSAllocation>
computeLDSAllocation(int64_t lhsTileElements, int64_t rhsTileElements,
                     int64_t elementBytes, int64_t bufferDepth,
                     int64_t maxLDSBytes) {
  int64_t lhsBytes = lhsTileElements * elementBytes;
  int64_t rhsBytes = rhsTileElements * elementBytes;
  int64_t totalBytes = (lhsBytes + rhsBytes) * bufferDepth;

  if (totalBytes > maxLDSBytes)
    return std::nullopt;

  return LDSAllocation{lhsBytes, rhsBytes, totalBytes, bufferDepth};
}

int64_t computeMaxBufferDepth(int64_t lhsTileElements,
                              int64_t rhsTileElements, int64_t elementBytes,
                              int64_t maxLDSBytes) {
  int64_t singleBufferBytes =
      (lhsTileElements + rhsTileElements) * elementBytes;
  if (singleBufferBytes <= 0)
    return 0;
  return maxLDSBytes / singleBufferBytes;
}

} // namespace mlir::iree_compiler::IREE::GPU
