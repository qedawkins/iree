// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pure arithmetic cost model functions. These have no link-time dependency on
// IREEGPUDialect (no calls to getSingleSubgroupLayout or TargetAttr methods),
// allowing them to be tested in isolation without the full dialect link chain.

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/CostModel.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ScheduleConfig.h"

#include "llvm/Support/MathExtras.h"

#include <cassert>

namespace mlir::iree_compiler::IREE::GPU {

//===----------------------------------------------------------------------===//
// MMA Operand VGPRs (layout-based, no intrinsic lookup)
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
  int64_t elements = computeElementsPerThread(layout);
  return llvm::divideCeil(elements * elementBits, kVGPRBitWidth);
}

//===----------------------------------------------------------------------===//
// Global Load Staging Budget
//===----------------------------------------------------------------------===//

GlobalLoadVGPRs computeGlobalLoadVGPRs(int64_t lhsTileElements,
                                        int64_t rhsTileElements,
                                        int64_t numThreads,
                                        int64_t elementBits) {
  assert(numThreads > 0 && "numThreads must be positive");
  int64_t lhsPerThread = llvm::divideCeil(lhsTileElements, numThreads);
  int64_t rhsPerThread = llvm::divideCeil(rhsTileElements, numThreads);

  int64_t lhsVGPRs = llvm::divideCeil(lhsPerThread * elementBits, kVGPRBitWidth);
  int64_t rhsVGPRs = llvm::divideCeil(rhsPerThread * elementBits, kVGPRBitWidth);

  return GlobalLoadVGPRs{lhsVGPRs, rhsVGPRs};
}

//===----------------------------------------------------------------------===//
// Index/Address Overhead
//===----------------------------------------------------------------------===//

/// Default compiler overhead for LLVM register allocator inefficiency.
/// Empirically observed: total ISA VGPRs - structured overhead gives ~24-31
/// VGPRs attributable to compiler decisions (suboptimal live ranges, spill
/// temporaries, ABI requirements). We use 24 as a conservative estimate.
static constexpr int64_t kDefaultCompilerOverheadVGPRs = 24;

int64_t computeIndexOverheadVGPRs(int64_t numLoadOperands, int64_t numKQuarters,
                                  bool splitCopy) {
  // Thread identification: linear thread ID.
  int64_t threadId = 1;

  // Global load base addresses: row and column bases per operand.
  // buffer_load_b128 has a 12-bit offen field, so nearby loads share a base.
  int64_t globalAddr = 2 * numLoadOperands;

  // LDS base addresses: one per operand, plus quarter subview offsets if the
  // schedule uses multiple K quarters.
  int64_t ldsAddr = 2 + (numKQuarters > 1 ? 2 : 0);

  // Loop control: K iteration index and comparison value.
  int64_t loopControl = 2;

  // Copy phase: row/col decomposition for distributing elements to threads.
  // Split copy (per-dimension distribution) needs more index temporaries.
  int64_t copyDecomp = splitCopy ? 4 : 2;

  // Coordinate decomposition from delinearize_index for subgroup ID mapping.
  int64_t delinearize = 2;

  return threadId + globalAddr + ldsAddr + loopControl + copyDecomp +
         delinearize + kDefaultCompilerOverheadVGPRs;
}

//===----------------------------------------------------------------------===//
// Peak VGPR Usage
//===----------------------------------------------------------------------===//

PeakVGPRUsage computePeakVGPRUsage(int64_t accumulatorVGPRs,
                                    GlobalLoadVGPRs globalLoadVGPRs,
                                    LDSQuarterReadVGPRs quarterReadVGPRs,
                                    int64_t indexOverheadVGPRs,
                                    int64_t availableVGPRs, bool earlyWrite) {
  // Global load staging VGPRs are live at peak for both schedule variants.
  int64_t glVGPRs = globalLoadVGPRs.lhsVGPRs + globalLoadVGPRs.rhsVGPRs;

  // Quarter read VGPRs per quarter (one LHS + RHS pair).
  int64_t quarterVGPRs = quarterReadVGPRs.lhsVGPRs + quarterReadVGPRs.rhsVGPRs;

  // In the original schedule, the peak phase reads 2 quarters simultaneously
  // (q_i and q_{i+1}). In the early-write schedule, global load staging is
  // freed before the second quarter, so only 1 quarter is live at peak.
  int64_t simultaneousQuarters = earlyWrite ? 1 : 2;

  int64_t totalVGPRs = accumulatorVGPRs + glVGPRs +
                        simultaneousQuarters * quarterVGPRs +
                        indexOverheadVGPRs;

  int64_t headroom = availableVGPRs - totalVGPRs;
  bool spills = totalVGPRs > availableVGPRs;

  return PeakVGPRUsage{totalVGPRs, headroom, spills};
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

//===----------------------------------------------------------------------===//
// Pipelined Burst Model
//===----------------------------------------------------------------------===//

int64_t computePipelinedBurst(int64_t count, int64_t issueCycles,
                              int64_t execCycles) {
  if (count <= 0)
    return 0;
  return (count - 1) * issueCycles + execCycles;
}

//===----------------------------------------------------------------------===//
// Phase Timing
//===----------------------------------------------------------------------===//

PhaseTiming computePhaseTiming(const PhaseInstructionCounts &counts,
                               InstructionTiming mmaTiming,
                               InstructionTiming ldsReadTiming,
                               InstructionTiming ldsWriteTiming,
                               InstructionTiming globalLoadTiming,
                               int64_t barrierCycles) {
  // VALU pipeline: WMMA + VALU address ops serialized (same pipeline on
  // gfx1201). WMMAs pipeline: (N-1)*issue + exec.
  int64_t wmmaCycles = computePipelinedBurst(
      counts.numWMMA, mmaTiming.issueCycles, mmaTiming.execCycles);
  // VALU address ops are single-cycle each.
  int64_t valuAddrCycles = counts.numVALUAddr;
  int64_t valuPipeline = wmmaCycles + valuAddrCycles;

  // LDS bus: reads and writes are pipelined bursts. With 2-port banks,
  // reads and writes to different addresses can partially overlap.
  int64_t ldsReadTime = computePipelinedBurst(
      counts.numLDSLoads, ldsReadTiming.issueCycles, ldsReadTiming.execCycles);
  int64_t ldsWriteTime = computePipelinedBurst(
      counts.numLDSStores, ldsWriteTiming.issueCycles,
      ldsWriteTiming.execCycles);
  // 2-port overlap: read and write can happen simultaneously.
  int64_t ldsBus;
  if (counts.numLDSLoads > 0 && counts.numLDSStores > 0)
    ldsBus = std::max(ldsReadTime, ldsWriteTime);
  else
    ldsBus = ldsReadTime + ldsWriteTime;

  // VMEM issue: global loads are async, only counting issue overhead.
  int64_t vmemIssue = counts.numGlobalLoads * globalLoadTiming.issueCycles;

  // Barrier cost.
  int64_t barrier = counts.hasBarrier ? barrierCycles : 0;

  // Phase total: max of overlapping units + barrier.
  int64_t total = std::max({valuPipeline, ldsBus, vmemIssue}) + barrier;

  return PhaseTiming{valuPipeline, ldsBus, vmemIssue, barrier, total};
}

//===----------------------------------------------------------------------===//
// Iteration Latency
//===----------------------------------------------------------------------===//

/// Build per-phase instruction counts for the 4-quarter early-write schedule.
static SmallVector<PhaseInstructionCounts>
build4QEarlyWrite(int64_t wmmaPerQuarter, int64_t glLHS, int64_t glRHS,
                  int64_t ldsReadsPerQuarter, int64_t ldsWriteLHS,
                  int64_t ldsWriteRHS, int64_t valuPerMemPhase) {
  return {
      // P1: GL-LHS + LDS-read q0
      {0, valuPerMemPhase, ldsReadsPerQuarter, 0, glLHS, true},
      // P2: MFMA q0
      {wmmaPerQuarter, 0, 0, 0, 0, true},
      // P3: GL-RHS + LDS-read q1
      {0, valuPerMemPhase, ldsReadsPerQuarter, 0, glRHS, true},
      // P4: MFMA q1
      {wmmaPerQuarter, 0, 0, 0, 0, true},
      // P5: LDS-write-LHS + LDS-read q2
      {0, ldsReadsPerQuarter + 2 * ldsWriteLHS + 4, ldsReadsPerQuarter,
       ldsWriteLHS, 0, true},
      // P6: MFMA q2 + LDS-write-RHS + LDS-read q3 (addr precomputed in P5)
      {wmmaPerQuarter, 0, ldsReadsPerQuarter, ldsWriteRHS, 0, true},
      // P7: MFMA q3
      {wmmaPerQuarter, 0, 0, 0, 0, true},
      // P8: structural
      {0, 0, 0, 0, 0, true},
  };
}

/// Build per-phase instruction counts for the 4-quarter original schedule.
static SmallVector<PhaseInstructionCounts>
build4QOriginal(int64_t wmmaPerQuarter, int64_t glLHS, int64_t glRHS,
                int64_t ldsReadsPerQuarter, int64_t ldsWriteLHS,
                int64_t ldsWriteRHS, int64_t valuPerMemPhase) {
  return {
      // P1: GL-LHS + LDS-read q0
      {0, valuPerMemPhase, ldsReadsPerQuarter, 0, glLHS, true},
      // P2: MFMA q0
      {wmmaPerQuarter, 0, 0, 0, 0, true},
      // P3: GL-RHS + LDS-read q1
      {0, valuPerMemPhase, ldsReadsPerQuarter, 0, glRHS, true},
      // P4: MFMA q1
      {wmmaPerQuarter, 0, 0, 0, 0, true},
      // P5: LDS-read q2+q3
      {0, 2 * ldsReadsPerQuarter + 4, 2 * ldsReadsPerQuarter, 0, 0, true},
      // P6: MFMA q2
      {wmmaPerQuarter, 0, 0, 0, 0, true},
      // P7: LDS-write
      {0, 2 * (ldsWriteLHS + ldsWriteRHS) + 4, 0,
       ldsWriteLHS + ldsWriteRHS, 0, true},
      // P8: MFMA q3
      {wmmaPerQuarter, 0, 0, 0, 0, true},
  };
}

/// Build per-phase instruction counts for the 2-quarter schedule (K=32).
static SmallVector<PhaseInstructionCounts>
build2QSchedule(int64_t wmmaPerQuarter, int64_t glLHS, int64_t glRHS,
                int64_t ldsReadsPerQuarter, int64_t ldsWriteLHS,
                int64_t ldsWriteRHS, int64_t valuPerMemPhase) {
  return {
      // P1: GL-LHS+RHS + LDS-read q0
      {0, valuPerMemPhase, ldsReadsPerQuarter, 0, glLHS + glRHS, true},
      // P2: MFMA q0
      {wmmaPerQuarter, 0, 0, 0, 0, true},
      // P3: LDS-write + LDS-read q1
      {0, ldsReadsPerQuarter + 2 * (ldsWriteLHS + ldsWriteRHS) + 4,
       ldsReadsPerQuarter, ldsWriteLHS + ldsWriteRHS, 0, true},
      // P4: MFMA q1
      {wmmaPerQuarter, 0, 0, 0, 0, true},
  };
}

int64_t computeIterationCycles(int64_t subgroupM, int64_t subgroupN,
                               int64_t kTile, int64_t mmaM, int64_t mmaN,
                               int64_t mmaK, int64_t numThreads,
                               int64_t inputBits, bool earlyWrite,
                               InstructionTiming mmaTiming,
                               InstructionTiming ldsReadTiming,
                               InstructionTiming ldsWriteTiming,
                               InstructionTiming globalLoadTiming,
                               int64_t barrierCycles) {
  // Number of quarters.
  int64_t numQuarters = std::min(int64_t(4), kTile / mmaK);
  if (numQuarters < 1)
    return 0;
  int64_t quarterK = kTile / numQuarters;

  // WMMAs per quarter.
  int64_t wmmaPerQuarter =
      (subgroupM / mmaM) * (subgroupN / mmaN) * (quarterK / mmaK);

  // Global loads per operand: distributed across all threads.
  // M*K elements for LHS, K*N elements for RHS.
  // GLOBAL_LOAD_B128 = 16 bytes = 128 bits.
  int64_t lhsTileElements = subgroupM * kTile;
  int64_t rhsTileElements = kTile * subgroupN;
  int64_t bytesPerLoad = 16;
  int64_t elemBytes = inputBits / 8;
  int64_t glLHS =
      llvm::divideCeil(lhsTileElements * elemBytes, numThreads * bytesPerLoad);
  int64_t glRHS =
      llvm::divideCeil(rhsTileElements * elemBytes, numThreads * bytesPerLoad);

  // LDS reads per quarter per subgroup.
  int64_t ldsReadAPerQ = (subgroupM / mmaM) * (quarterK / mmaK);
  int64_t ldsReadBPerQ = (subgroupN / mmaN) * (quarterK / mmaK);
  int64_t ldsReadsPerQuarter = ldsReadAPerQ + ldsReadBPerQ;

  // LDS writes per operand (same count as global loads).
  int64_t ldsWriteLHS = glLHS;
  int64_t ldsWriteRHS = glRHS;

  // VALU address computation estimate per memory phase.
  int64_t valuPerMemPhase = 2 * (glLHS + glRHS) + ldsReadsPerQuarter + 4;

  // Build phase instruction counts.
  SmallVector<PhaseInstructionCounts> phases;
  if (numQuarters == 4) {
    if (earlyWrite) {
      phases = build4QEarlyWrite(wmmaPerQuarter, glLHS, glRHS,
                                 ldsReadsPerQuarter, ldsWriteLHS, ldsWriteRHS,
                                 valuPerMemPhase);
    } else {
      phases = build4QOriginal(wmmaPerQuarter, glLHS, glRHS,
                               ldsReadsPerQuarter, ldsWriteLHS, ldsWriteRHS,
                               valuPerMemPhase);
    }
  } else if (numQuarters == 2) {
    phases = build2QSchedule(wmmaPerQuarter, glLHS, glRHS, ldsReadsPerQuarter,
                             ldsWriteLHS, ldsWriteRHS, valuPerMemPhase);
  } else {
    // Fallback: simple estimate for other quarter counts.
    int64_t totalWMMA = numQuarters * wmmaPerQuarter;
    int64_t wmmaCycles = computePipelinedBurst(
        totalWMMA, mmaTiming.issueCycles, mmaTiming.execCycles);
    return wmmaCycles + numQuarters * barrierCycles * 2;
  }

  // Sum phase timings.
  int64_t total = 0;
  for (const PhaseInstructionCounts &counts : phases) {
    PhaseTiming timing = computePhaseTiming(counts, mmaTiming, ldsReadTiming,
                                            ldsWriteTiming, globalLoadTiming,
                                            barrierCycles);
    total += timing.totalCycles;
  }
  return total;
}

} // namespace mlir::iree_compiler::IREE::GPU
