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
// VGPR Budget Analysis (phase-by-phase liveness)
//===----------------------------------------------------------------------===//

const PhaseVGPRLiveness &VGPRBudgetAnalysis::peakPhase() const {
  assert(!phases.empty() && "no phases in budget analysis");
  const PhaseVGPRLiveness *peak = &phases[0];
  for (const PhaseVGPRLiveness &phase : phases) {
    if (phase.total() > peak->total())
      peak = &phase;
  }
  return *peak;
}

int64_t VGPRBudgetAnalysis::peakVGPRs() const { return peakPhase().total(); }

int64_t VGPRBudgetAnalysis::peakArchVGPRs() const {
  if (hasAccVGPR) {
    // On CDNA: accumulators use AccVGPRs, not ArchVGPRs.
    const PhaseVGPRLiveness &peak = peakPhase();
    return peak.total() - peak.accumulator;
  }
  return peakVGPRs();
}

int64_t VGPRBudgetAnalysis::headroom() const {
  if (hasAccVGPR)
    return availableVGPRs - peakArchVGPRs();
  return availableVGPRs - peakVGPRs();
}

int64_t VGPRBudgetAnalysis::accHeadroom() const {
  if (hasAccVGPR)
    return availableAccVGPRs - accumulatorVGPRs;
  return 0;
}

bool VGPRBudgetAnalysis::willSpill() const {
  if (hasAccVGPR)
    return headroom() < 0 || accHeadroom() < 0;
  return headroom() < 0;
}

/// Build the 8-phase liveness table for the 4-quarter original schedule.
static SmallVector<PhaseVGPRLiveness>
buildPhaseLiveness4QOriginal(int64_t acc, int64_t glL, int64_t glR,
                             int64_t qL, int64_t qR, int64_t idx,
                             int64_t simultaneousQuarters) {
  return {
      // P1: GL-LHS + LDS-read q0.
      {"P1: GL-LHS + LDS-read q0", acc, glL, 0, qL, qR, idx},
      // P2: MFMA q0 (q0 reads consumed).
      {"P2: MFMA q0", acc, glL, 0, 0, 0, idx},
      // P3: GL-RHS + LDS-read q1.
      {"P3: GL-RHS + LDS-read q1", acc, glL, glR, qL, qR, idx},
      // P4: MFMA q1 (q1 reads consumed).
      {"P4: MFMA q1", acc, glL, glR, 0, 0, idx},
      // P5: LDS-read q2+q3 simultaneously (PEAK).
      {"P5: LDS-read q2+q3 (PEAK)", acc, glL, glR,
       qL * simultaneousQuarters, qR * simultaneousQuarters, idx},
      // P6: MFMA q2 (q3 still live).
      {"P6: MFMA q2 (q3 live)", acc, glL, glR, qL, qR, idx},
      // P7: LDS-write (GL regs consumed).
      {"P7: LDS-write (GL consumed)", acc, 0, 0, qL, qR, idx},
      // P8: MFMA q3 (q3 consumed).
      {"P8: MFMA q3", acc, 0, 0, 0, 0, idx},
  };
}

/// Build the 8-phase liveness table for the 4-quarter early-write schedule.
static SmallVector<PhaseVGPRLiveness>
buildPhaseLiveness4QEarlyWrite(int64_t acc, int64_t glL, int64_t glR,
                               int64_t qL, int64_t qR, int64_t idx) {
  return {
      // P1: GL-LHS + LDS-read q0.
      {"P1: GL-LHS + LDS-read q0", acc, glL, 0, qL, qR, idx},
      // P2: MFMA q0.
      {"P2: MFMA q0", acc, glL, 0, 0, 0, idx},
      // P3: GL-RHS + LDS-read q1.
      {"P3: GL-RHS + LDS-read q1", acc, glL, glR, qL, qR, idx},
      // P4: MFMA q1.
      {"P4: MFMA q1", acc, glL, glR, 0, 0, idx},
      // P5: LDS-write-LHS + LDS-read q2 (GL-LHS consumed).
      {"P5: LDS-write-LHS + LDS-read q2", acc, 0, glR, qL, qR, idx},
      // P6: MFMA q2 + LDS-write-RHS + LDS-read q3 (GL-RHS consumed).
      {"P6: MFMA q2 + LDS-write-RHS + LDS-read q3", acc, 0, 0, qL, qR, idx},
      // P7: MFMA q3.
      {"P7: MFMA q3", acc, 0, 0, 0, 0, idx},
      // P8: structural.
      {"P8: structural", acc, 0, 0, 0, 0, idx},
  };
}

/// Build the 4-phase liveness table for the 2-quarter schedule.
static SmallVector<PhaseVGPRLiveness>
buildPhaseLiveness2Q(int64_t acc, int64_t glL, int64_t glR, int64_t qL,
                     int64_t qR, int64_t idx) {
  return {
      // P1: GL-LHS+RHS + LDS-read q0.
      {"P1: GL + LDS-read q0", acc, glL, glR, qL, qR, idx},
      // P2: MFMA q0 (q0 consumed, GL still live).
      {"P2: MFMA q0", acc, glL, glR, 0, 0, idx},
      // P3: LDS-write + LDS-read q1 (GL consumed).
      {"P3: LDS-write + LDS-read q1", acc, 0, 0, qL, qR, idx},
      // P4: MFMA q1 (q1 consumed).
      {"P4: MFMA q1", acc, 0, 0, 0, 0, idx},
  };
}

/// Build the 2-phase liveness table for a 1-quarter (no-pingpong) schedule.
static SmallVector<PhaseVGPRLiveness>
buildPhaseLiveness1Q(int64_t acc, int64_t glL, int64_t glR, int64_t qL,
                     int64_t qR, int64_t idx) {
  return {
      // P1: GL + LDS-read q0.
      {"P1: GL + LDS-read q0", acc, glL, glR, qL, qR, idx},
      // P2: MFMA q0 + LDS-write.
      {"P2: MFMA q0 + LDS-write", acc, 0, 0, 0, 0, idx},
  };
}

VGPRBudgetAnalysis buildVGPRBudgetAnalysis(
    int64_t accumulatorVGPRs, GlobalLoadVGPRs globalLoadVGPRs,
    LDSQuarterReadVGPRs quarterReadVGPRs, int64_t indexOverheadVGPRs,
    int64_t availableVGPRs, int64_t numQuarters, bool earlyWrite,
    bool hasAccVGPR, int64_t availableAccVGPRs) {
  int64_t glL = globalLoadVGPRs.lhsVGPRs;
  int64_t glR = globalLoadVGPRs.rhsVGPRs;
  int64_t qL = quarterReadVGPRs.lhsVGPRs;
  int64_t qR = quarterReadVGPRs.rhsVGPRs;
  int64_t acc = accumulatorVGPRs;
  int64_t idx = indexOverheadVGPRs;

  // Build the phase-by-phase liveness table.
  // numQuarters is expected to be 1, 2, or 4 (capped by callers).
  SmallVector<PhaseVGPRLiveness> phases;
  if (numQuarters == 4) {
    if (earlyWrite) {
      phases = buildPhaseLiveness4QEarlyWrite(acc, glL, glR, qL, qR, idx);
    } else {
      // Default: 2 simultaneous quarter reads at peak.
      phases =
          buildPhaseLiveness4QOriginal(acc, glL, glR, qL, qR, idx, /*n=*/2);
    }
  } else if (numQuarters == 2) {
    phases = buildPhaseLiveness2Q(acc, glL, glR, qL, qR, idx);
  } else {
    // 1 quarter or fallback.
    phases = buildPhaseLiveness1Q(acc, glL, glR, qL, qR, idx);
  }

  VGPRBudgetAnalysis result;
  result.accumulatorVGPRs = accumulatorVGPRs;
  result.globalLoadLHSVGPRs = glL;
  result.globalLoadRHSVGPRs = glR;
  result.ldsReadLHSPerQuarter = qL;
  result.ldsReadRHSPerQuarter = qR;
  result.indexOverheadVGPRs = idx;
  result.phases = std::move(phases);
  result.availableVGPRs = availableVGPRs;
  result.availableAccVGPRs = availableAccVGPRs;
  result.hasAccVGPR = hasAccVGPR;
  return result;
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

int64_t computeIterationCycles(int64_t workgroupM, int64_t workgroupN,
                               int64_t subgroupM, int64_t subgroupN,
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

  // WMMAs per quarter (per subgroup).
  int64_t wmmaPerQuarter =
      (subgroupM / mmaM) * (subgroupN / mmaN) * (quarterK / mmaK);

  // Global loads per thread: cooperative across all threads in the workgroup.
  // Uses workgroup-level tile dimensions, NOT subgroup.
  // GLOBAL_LOAD_B128 = 16 bytes = 128 bits.
  int64_t lhsTileElements = workgroupM * kTile;
  int64_t rhsTileElements = kTile * workgroupN;
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
