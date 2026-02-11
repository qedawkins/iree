// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Schedule configuration derivation. Depends on IREEGPUDialect at link time
// (via getMSize, getNSize, getKSize, getSingleSubgroupLayout, TargetAttr).

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ScheduleConfig.h"

#include "llvm/Support/MathExtras.h"

namespace mlir::iree_compiler::IREE::GPU {

/// Default timing values used when the target doesn't have timing data in
/// its extra dict. Conservative estimates that should work for any target.
static constexpr InstructionTiming kDefaultMMATiming = {2, 32};
static constexpr InstructionTiming kDefaultLDSReadTiming = {2, 25};
static constexpr InstructionTiming kDefaultLDSWriteTiming = {2, 10};
static constexpr InstructionTiming kDefaultGlobalLoadTiming = {2, 300};
static constexpr int64_t kDefaultBarrierCycles = 20;

/// Get timing from target, falling back to defaults.
static InstructionTiming getTimingOrDefault(TargetAttr target, StringRef cat,
                                            InstructionTiming fallback) {
  if (std::optional<InstructionTiming> t = getInstructionTiming(target, cat))
    return *t;
  return fallback;
}

std::optional<ScheduleConfig>
deriveScheduleConfig(int64_t workgroupM, int64_t workgroupN,
                     MMAIntrinsic intrinsic, TargetAttr target,
                     const ScheduleDerivationParams &params) {
  // Get MMA tile dimensions.
  int64_t mmaM = getMSize(intrinsic);
  int64_t mmaN = getNSize(intrinsic);
  int64_t mmaK = getKSize(intrinsic);
  if (mmaM <= 0 || mmaN <= 0 || mmaK <= 0)
    return std::nullopt;

  // Get VGPR budget at target occupancy.
  std::optional<VGPRBudget> budget =
      computeVGPRBudget(target, params.targetOccupancy);
  if (!budget)
    return std::nullopt;
  int64_t availableVGPRs = budget->totalVGPRs;

  // Get subgroup size from target.
  TargetWgpAttr wgp = target.getWgp();
  ArrayRef<int32_t> subgroupSizes = wgp.getSubgroupSizeChoices();
  if (subgroupSizes.empty())
    return std::nullopt;
  int64_t subgroupSize = subgroupSizes.front();

  // Get instruction timing from target (or defaults).
  InstructionTiming mmaTiming =
      getTimingOrDefault(target, "mma", kDefaultMMATiming);
  InstructionTiming ldsReadTiming =
      getTimingOrDefault(target, "lds_read", kDefaultLDSReadTiming);
  InstructionTiming ldsWriteTiming =
      getTimingOrDefault(target, "lds_write", kDefaultLDSWriteTiming);
  InstructionTiming globalLoadTiming =
      getTimingOrDefault(target, "global_load", kDefaultGlobalLoadTiming);

  // Barrier cycles: use average of issue+exec from timing data, or default.
  int64_t barrierCycles = kDefaultBarrierCycles;
  if (std::optional<InstructionTiming> bt =
          getInstructionTiming(target, "barrier")) {
    barrierCycles = bt->issueCycles + bt->execCycles;
  }

  // Element sizes in bytes.
  int64_t inputBytes = params.inputBits / 8;
  int64_t accBits = params.accBits;

  // Track the best configuration found.
  std::optional<ScheduleConfig> bestConfig;
  int64_t bestScore = INT64_MAX; // Lower is better (cycles per iteration).

  // Sweep over all candidate configurations.
  for (auto [sgCountM, sgCountN] : params.subgroupLayoutCandidates) {
    int64_t numSubgroups = sgCountM * sgCountN;
    int64_t numThreads = numSubgroups * subgroupSize;

    // Subgroup tile dimensions.
    if (workgroupM % sgCountM != 0 || workgroupN % sgCountN != 0)
      continue;
    int64_t sgM = workgroupM / sgCountM;
    int64_t sgN = workgroupN / sgCountN;

    // Subgroup tile must be divisible by MMA tile.
    if (sgM % mmaM != 0 || sgN % mmaN != 0)
      continue;

    // Check max workgroup size.
    int64_t maxThreads = wgp.getMaxThreadCountPerWorkgroup();
    if (maxThreads > 0 && numThreads > maxThreads)
      continue;

    // Compute accumulator VGPRs.
    std::optional<int64_t> accVGPRs =
        computeAccumulatorVGPRs(intrinsic, sgM, sgN, accBits);
    if (!accVGPRs)
      continue;

    for (int64_t kTile : params.kTileCandidates) {
      // K tile must be divisible by mmaK.
      if (kTile % mmaK != 0)
        continue;

      // Determine number of quarters.
      int64_t numQuarters = std::min(int64_t(4), kTile / mmaK);
      int64_t quarterK = kTile / numQuarters;
      if (quarterK % mmaK != 0)
        continue;

      // Compute global load staging VGPRs.
      int64_t lhsTileElements = workgroupM * kTile;
      int64_t rhsTileElements = kTile * workgroupN;
      GlobalLoadVGPRs glVGPRs = computeGlobalLoadVGPRs(
          lhsTileElements, rhsTileElements, numThreads, params.inputBits);

      // Compute LDS quarter read VGPRs.
      std::optional<LDSQuarterReadVGPRs> qReadVGPRs =
          computeLDSQuarterReadVGPRs(intrinsic, sgM, sgN, quarterK,
                                     params.inputBits, params.inputBits);
      if (!qReadVGPRs)
        continue;

      // Compute index overhead.
      int64_t indexVGPRs = computeIndexOverheadVGPRs(
          /*numLoadOperands=*/2, numQuarters, /*splitCopy=*/sgCountM > 1);

      // Check LDS allocation.
      std::optional<LDSAllocation> ldsAlloc = computeLDSAllocation(
          lhsTileElements, rhsTileElements, inputBytes,
          /*bufferDepth=*/1, params.maxLDSBytes);
      if (!ldsAlloc)
        continue;

      // Try both schedule variants.
      for (bool earlyWrite : {true, false}) {
        PeakVGPRUsage peak = computePeakVGPRUsage(
            *accVGPRs, glVGPRs, *qReadVGPRs, indexVGPRs, availableVGPRs,
            earlyWrite);

        // Check VGPR budget.
        if (peak.headroom < params.minHeadroom)
          continue;

        // Compute latency estimate.
        int64_t iterCycles = computeIterationCycles(
            workgroupM, workgroupN, sgM, sgN, kTile, mmaM, mmaN, mmaK,
            numThreads, params.inputBits, earlyWrite, mmaTiming, ldsReadTiming,
            ldsWriteTiming, globalLoadTiming, barrierCycles);

        // WMMAs per iteration.
        int64_t wmmaPerIter =
            numQuarters * (sgM / mmaM) * (sgN / mmaN) * (quarterK / mmaK);
        int64_t flopsPerIter = wmmaPerIter * 2 * mmaM * mmaN * mmaK;

        // Score: lower cycles is better. Tie-break: prefer more headroom.
        // Normalize by FLOPs to compare different K tiles fairly.
        // Score = cycles_per_flop (lower = better throughput).
        int64_t score = (iterCycles * 1000) / std::max(flopsPerIter, int64_t(1));

        if (!bestConfig || score < bestScore ||
            (score == bestScore && peak.headroom > bestConfig->headroom)) {
          ScheduleConfig config;
          config.workgroupM = workgroupM;
          config.workgroupN = workgroupN;
          config.kTile = kTile;
          config.subgroupCountM = sgCountM;
          config.subgroupCountN = sgCountN;
          config.subgroupM = sgM;
          config.subgroupN = sgN;
          config.numSubgroups = numSubgroups;
          config.subgroupSize = subgroupSize;
          config.numThreads = numThreads;
          config.mmaIntrinsic = intrinsic;
          config.mmaM = mmaM;
          config.mmaN = mmaN;
          config.mmaK = mmaK;
          config.variant =
              earlyWrite ? ScheduleVariant::EarlyWrite : ScheduleVariant::Original;
          config.numQuarters = numQuarters;
          config.quarterK = quarterK;
          config.peakVGPRs = peak.totalVGPRs;
          config.availableVGPRs = availableVGPRs;
          config.headroom = peak.headroom;
          config.ldsBytes = ldsAlloc->totalBytes;
          config.bufferDepth = ldsAlloc->bufferDepth;
          config.cyclesPerIteration = iterCycles;
          config.wmmaPerIteration = wmmaPerIter;
          config.flopsPerIteration = flopsPerIter;

          bestConfig = config;
          bestScore = score;
        }
      }
    }
  }

  return bestConfig;
}

} // namespace mlir::iree_compiler::IREE::GPU
