// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_SCHEDULECONFIG_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_SCHEDULECONFIG_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/CostModel.h"

namespace mlir::iree_compiler::IREE::GPU {

//===----------------------------------------------------------------------===//
// Schedule Variant
//===----------------------------------------------------------------------===//

/// Schedule variants for the quarter-K pingpong schedule.
enum class ScheduleVariant {
  /// Original schedule: Phase 5 reads q2+q3 simultaneously.
  /// Peak has 2 simultaneous quarter reads + global load staging.
  Original,

  /// Early-write schedule: LDS writes happen before the second quarter read.
  /// Peak has only 1 quarter read live at a time. Saves ~32 VGPRs on f16.
  EarlyWrite,
};

//===----------------------------------------------------------------------===//
// Schedule Configuration
//===----------------------------------------------------------------------===//

/// Derived schedule configuration for a matmul workgroup tile.
///
/// All fields are derived analytically from the problem shape, target hardware,
/// and MMA intrinsic. This struct is the output of deriveScheduleConfig().
struct ScheduleConfig {
  //--- Problem parameters (echoed back for convenience) ---//
  int64_t workgroupM = 0;   /// Workgroup tile M dimension.
  int64_t workgroupN = 0;   /// Workgroup tile N dimension.
  int64_t kTile = 0;        /// K tile size per iteration.

  //--- Subgroup layout ---//
  int64_t subgroupCountM = 0; /// Number of subgroups along M.
  int64_t subgroupCountN = 0; /// Number of subgroups along N.
  int64_t subgroupM = 0;      /// Per-subgroup tile M = workgroupM / subgroupCountM.
  int64_t subgroupN = 0;      /// Per-subgroup tile N = workgroupN / subgroupCountN.

  //--- Threading ---//
  int64_t numSubgroups = 0;   /// Total subgroups = subgroupCountM * subgroupCountN.
  int64_t subgroupSize = 0;   /// Threads per subgroup (wave32=32, wave64=64).
  int64_t numThreads = 0;     /// Total threads = numSubgroups * subgroupSize.

  //--- MMA ---//
  MMAIntrinsic mmaIntrinsic{}; /// Selected MMA intrinsic.
  int64_t mmaM = 0;            /// MMA tile M.
  int64_t mmaN = 0;            /// MMA tile N.
  int64_t mmaK = 0;            /// MMA tile K.

  //--- Schedule ---//
  ScheduleVariant variant = ScheduleVariant::Original; /// Original or early-write.
  int64_t numQuarters = 0;    /// Number of K quarters per iteration.
  int64_t quarterK = 0;       /// K elements per quarter = kTile / numQuarters.

  //--- Budgets ---//
  int64_t peakVGPRs = 0;         /// Predicted peak VGPR usage.
  int64_t availableVGPRs = 0;    /// VGPR budget at target occupancy.
  int64_t headroom = 0;          /// availableVGPRs - peakVGPRs (positive = fits).
  int64_t ldsBytes = 0;          /// Total LDS allocation (all buffers).
  int64_t bufferDepth = 0;       /// Multi-buffering depth.

  //--- Latency estimate ---//
  int64_t cyclesPerIteration = 0; /// Estimated cycles per K iteration.
  int64_t wmmaPerIteration = 0;   /// Total WMMA instructions per K iteration.
  int64_t flopsPerIteration = 0;  /// FLOPs per K iteration.
};

//===----------------------------------------------------------------------===//
// Schedule Derivation Parameters
//===----------------------------------------------------------------------===//

/// Parameters controlling the schedule derivation search.
struct ScheduleDerivationParams {
  /// Target occupancy (waves per SIMD). Default 1 for maximum VGPRs.
  int64_t targetOccupancy = 1;

  /// Minimum VGPR headroom required. Configs with less headroom are rejected.
  int64_t minHeadroom = 0;

  /// Maximum LDS bytes available per workgroup.
  int64_t maxLDSBytes = 65536;

  /// Candidate K tile sizes to evaluate.
  SmallVector<int64_t> kTileCandidates = {32, 64};

  /// Candidate subgroup counts to evaluate. Each entry is (countM, countN).
  SmallVector<std::pair<int64_t, int64_t>> subgroupLayoutCandidates = {
      {2, 2}, {2, 4}, {4, 2}, {4, 4}, {4, 8}, {8, 4}};

  /// Input element bitwidth (e.g., 16 for f16).
  int64_t inputBits = 16;

  /// Accumulator element bitwidth (e.g., 32 for f32).
  int64_t accBits = 32;
};

//===----------------------------------------------------------------------===//
// Schedule Derivation
//===----------------------------------------------------------------------===//

/// Derive the optimal schedule configuration for a matmul workgroup tile.
///
/// Sweeps over subgroup layouts, K tile sizes, and schedule variants.
/// For each candidate, computes VGPR budget, LDS allocation, and estimated
/// latency. Returns the configuration with the best estimated throughput
/// that fits within the VGPR and LDS budgets.
///
/// Returns std::nullopt if no valid configuration exists (all candidates
/// exceed the VGPR or LDS budget).
std::optional<ScheduleConfig>
deriveScheduleConfig(int64_t workgroupM, int64_t workgroupN,
                     MMAIntrinsic intrinsic, TargetAttr target,
                     const ScheduleDerivationParams &params = {});

//===----------------------------------------------------------------------===//
// Latency Estimation (pure arithmetic, no dialect deps)
//===----------------------------------------------------------------------===//

/// Pipelined burst model: time for N instructions with issue/exec pipelining.
///
/// Total = (N-1) * issueCycles + execCycles.
/// Returns 0 if N <= 0.
int64_t computePipelinedBurst(int64_t count, int64_t issueCycles,
                              int64_t execCycles);

/// Per-phase instruction counts for the quarter-K pingpong schedule.
struct PhaseInstructionCounts {
  int64_t numWMMA = 0;        /// WMMA instructions.
  int64_t numVALUAddr = 0;    /// VALU address computation ops.
  int64_t numLDSLoads = 0;    /// DS_LOAD instructions.
  int64_t numLDSStores = 0;   /// DS_STORE instructions.
  int64_t numGlobalLoads = 0; /// GLOBAL_LOAD instructions.
  bool hasBarrier = true;     /// Phase ends with a barrier.
};

/// Timing for a single phase.
struct PhaseTiming {
  int64_t valuPipelineCycles; /// WMMA + VALU (serialized on gfx1201).
  int64_t ldsBusCycles;       /// LDS read + write (partially overlapped).
  int64_t vmemIssueCycles;    /// Global load issue overhead.
  int64_t barrierCycles;      /// Barrier overhead.
  int64_t totalCycles;        /// max(valu, lds, vmem) + barrier.
};

/// Compute execution time for a single phase given instruction counts and
/// target timing data.
///
/// Overlap model (gfx1201):
///   WMMA + VALU → serialized (same pipeline)
///   LDS read + write → partially overlapped (2-port banks)
///   Global loads → async, overlapped with VALU and LDS
///   Phase total = max(valuPipeline, ldsBus, vmemIssue) + barrier
PhaseTiming computePhaseTiming(const PhaseInstructionCounts &counts,
                               InstructionTiming mmaTiming,
                               InstructionTiming ldsReadTiming,
                               InstructionTiming ldsWriteTiming,
                               InstructionTiming globalLoadTiming,
                               int64_t barrierCycles);

/// Compute total cycles for one K iteration of the quarter-K schedule.
///
/// Builds per-phase instruction counts for the given configuration, computes
/// per-phase timing, and sums to get the total iteration time.
///
/// |workgroupM|, |workgroupN|: Full workgroup tile dimensions. Used for global
///   load instruction counts (loads are cooperative across all threads).
/// |subgroupM|, |subgroupN|: Per-subgroup tile dimensions. Used for MMA tile
///   counts and LDS read counts.
int64_t computeIterationCycles(int64_t workgroupM, int64_t workgroupN,
                               int64_t subgroupM, int64_t subgroupN,
                               int64_t kTile, int64_t mmaM, int64_t mmaN,
                               int64_t mmaK, int64_t numThreads,
                               int64_t inputBits, bool earlyWrite,
                               InstructionTiming mmaTiming,
                               InstructionTiming ldsReadTiming,
                               InstructionTiming ldsWriteTiming,
                               InstructionTiming globalLoadTiming,
                               int64_t barrierCycles);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_SCHEDULECONFIG_H_
