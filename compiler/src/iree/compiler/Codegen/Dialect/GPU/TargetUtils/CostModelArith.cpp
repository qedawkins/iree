// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pure arithmetic cost model functions. These have no link-time dependency on
// IREEGPUDialect (no calls to getSingleSubgroupLayout or TargetAttr methods),
// allowing them to be tested in isolation without the full dialect link chain.

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/CostModel.h"

#include "llvm/Support/MathExtras.h"

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
  int64_t elts = computeElementsPerThread(layout);
  return llvm::divideCeil(elts * elementBits, 32);
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

} // namespace mlir::iree_compiler::IREE::GPU
