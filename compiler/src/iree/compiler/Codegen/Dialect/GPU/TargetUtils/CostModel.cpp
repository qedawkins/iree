// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cost model functions that depend on IREEGPUDialect at link time (via
// getSingleSubgroupLayout, getMSize, getNSize, getKSize, TargetAttr accessors).
// Pure arithmetic functions live in CostModelArith.cpp.

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/CostModel.h"

#include "llvm/Support/MathExtras.h"

namespace mlir::iree_compiler::IREE::GPU {

//===----------------------------------------------------------------------===//
// MMA Register Cost (intrinsic-based, uses getSingleSubgroupLayout)
//===----------------------------------------------------------------------===//

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
// VGPR Budget (needs TargetAttr)
//===----------------------------------------------------------------------===//

std::optional<VGPRBudget> computeVGPRBudget(TargetAttr target,
                                            int64_t occupancy) {
  if (occupancy <= 0)
    return std::nullopt;
  TargetWgpAttr wgp = target.getWgp();
  if (std::optional<int32_t> vgprBits = wgp.getVgprSpaceBits()) {
    int64_t totalVGPRs = *vgprBits / (occupancy * kVGPRBitWidth);
    return VGPRBudget{totalVGPRs, occupancy};
  }
  return std::nullopt;
}

std::optional<int64_t> computeMaxOccupancyFromVGPRs(TargetAttr target,
                                                     int64_t vgprsUsed) {
  if (vgprsUsed <= 0)
    return std::nullopt;
  TargetWgpAttr wgp = target.getWgp();
  if (std::optional<int32_t> vgprBits = wgp.getVgprSpaceBits()) {
    int64_t totalVGPRsAtOcc1 = *vgprBits / kVGPRBitWidth;
    return totalVGPRsAtOcc1 / vgprsUsed;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Accumulator Budget (uses getSingleSubgroupLayout)
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
// LDS Quarter Read Budget (uses getSingleSubgroupLayout)
//===----------------------------------------------------------------------===//

std::optional<LDSQuarterReadVGPRs>
computeLDSQuarterReadVGPRs(MMAIntrinsic intrinsic, int64_t subgroupM,
                            int64_t subgroupN, int64_t quarterK,
                            int64_t lhsBits, int64_t rhsBits) {
  int64_t mmaM = getMSize(intrinsic);
  int64_t mmaN = getNSize(intrinsic);
  int64_t mmaK = getKSize(intrinsic);

  if (subgroupM % mmaM != 0 || subgroupN % mmaN != 0 || quarterK % mmaK != 0)
    return std::nullopt;

  int64_t tilesM = subgroupM / mmaM;
  int64_t tilesN = subgroupN / mmaN;
  int64_t tilesK = quarterK / mmaK;

  MMASingleSubgroupLayout lhsLayout =
      getSingleSubgroupLayout(intrinsic, kMMAOperandLhs);
  MMASingleSubgroupLayout rhsLayout =
      getSingleSubgroupLayout(intrinsic, kMMAOperandRhs);

  int64_t lhsVGPRsPerTile = computeMMAOperandVGPRs(lhsLayout, lhsBits);
  int64_t rhsVGPRsPerTile = computeMMAOperandVGPRs(rhsLayout, rhsBits);

  int64_t lhsVGPRs = tilesM * tilesK * lhsVGPRsPerTile;
  int64_t rhsVGPRs = tilesK * tilesN * rhsVGPRsPerTile;

  return LDSQuarterReadVGPRs{lhsVGPRs, rhsVGPRs};
}

} // namespace mlir::iree_compiler::IREE::GPU
