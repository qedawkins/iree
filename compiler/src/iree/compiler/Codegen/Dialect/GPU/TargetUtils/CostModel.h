// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_COSTMODEL_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_COSTMODEL_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler::IREE::GPU {

/// Bitwidth of a single VGPR on AMD GPUs (32-bit scalar registers).
static constexpr int64_t kVGPRBitWidth = 32;

//===----------------------------------------------------------------------===//
// MMA Register Cost
//===----------------------------------------------------------------------===//

/// Compute the number of 32-bit VGPRs each thread needs to hold one MMA
/// operand, given the operand's subgroup layout and element bitwidth.
///
/// VGPRs = ceil(elementsPerThread * elementBits / kVGPRBitWidth), where
/// elementsPerThread = product(layout.outer) * product(layout.element).
int64_t computeMMAOperandVGPRs(const MMASingleSubgroupLayout &layout,
                               int64_t elementBits);

/// Per-MMA-tile register cost for a single subgroup (wave). Each field is the
/// number of 32-bit VGPRs needed per thread per MMA tile.
struct MMARegisterCost {
  int64_t lhsVGPRs;
  int64_t rhsVGPRs;
  int64_t accVGPRs;
};

/// Compute the per-tile VGPR cost for an MMA intrinsic.
///
/// |lhsBits|, |rhsBits|, |accBits| are the element bitwidths of the LHS (A),
/// RHS (B), and accumulator (C/D) operands respectively. For f16xf16->f32
/// matmul: lhsBits=16, rhsBits=16, accBits=32.
MMARegisterCost computeMMARegisterCost(MMAIntrinsic intrinsic,
                                       int64_t lhsBits, int64_t rhsBits,
                                       int64_t accBits);

//===----------------------------------------------------------------------===//
// VGPR Budget
//===----------------------------------------------------------------------===//

/// VGPR budget for a single wave at a given occupancy level.
struct VGPRBudget {
  int64_t totalVGPRs;  /// 32-bit VGPRs available per thread at this occupancy.
  int64_t occupancy;    /// Occupancy level (waves per SIMD).
};

/// Compute the VGPR budget per wave for the given target and occupancy.
///
/// Formula: totalVGPRs = vgpr_space_bits / (occupancy * kVGPRBitWidth)
///
/// Returns std::nullopt if the target doesn't have vgpr_space_bits populated.
std::optional<VGPRBudget> computeVGPRBudget(TargetAttr target,
                                            int64_t occupancy);

/// Compute the maximum occupancy (waves per SIMD) that can be achieved with
/// the given VGPR usage per wave.
///
/// Returns std::nullopt if the target doesn't have vgpr_space_bits populated
/// or if vgprsUsed is zero.
std::optional<int64_t> computeMaxOccupancyFromVGPRs(TargetAttr target,
                                                     int64_t vgprsUsed);

//===----------------------------------------------------------------------===//
// Accumulator Budget
//===----------------------------------------------------------------------===//

/// Compute the total accumulator VGPRs for a subgroup tile.
///
/// Given a subgroup that computes a (sgM x sgN) tile using a (mmaM x mmaN x
/// mmaK) MMA intrinsic:
///   numTiles = (sgM / mmaM) * (sgN / mmaN)
///   totalAccVGPRs = numTiles * accVGPRsPerTile
///
/// |accBits| is the accumulator element bitwidth (e.g., 32 for f32).
///
/// Returns std::nullopt if the subgroup tile is not evenly divisible by the
/// MMA tile dimensions.
std::optional<int64_t> computeAccumulatorVGPRs(MMAIntrinsic intrinsic,
                                               int64_t subgroupM,
                                               int64_t subgroupN,
                                               int64_t accBits);

//===----------------------------------------------------------------------===//
// Global Load Staging Budget
//===----------------------------------------------------------------------===//

/// VGPR cost of staging global loads into registers before writing to LDS.
struct GlobalLoadVGPRs {
  int64_t lhsVGPRs; /// VGPRs per thread for staging LHS global loads.
  int64_t rhsVGPRs; /// VGPRs per thread for staging RHS global loads.
};

/// Compute the number of VGPRs needed to stage global loads.
///
/// For each operand tile:
///   elementsPerThread = ceil(tileElements / numThreads)
///   vgprs = ceil(elementsPerThread * elementBits / kVGPRBitWidth)
///
/// |lhsTileElements|: M * K elements in the LHS tile.
/// |rhsTileElements|: K * N elements in the RHS tile.
/// |numThreads|: total threads per workgroup. Must be > 0.
/// |elementBits|: bitwidth of input data type (e.g., 16 for f16).
GlobalLoadVGPRs computeGlobalLoadVGPRs(int64_t lhsTileElements,
                                        int64_t rhsTileElements,
                                        int64_t numThreads,
                                        int64_t elementBits);

//===----------------------------------------------------------------------===//
// LDS Quarter Read Budget
//===----------------------------------------------------------------------===//

/// VGPR cost of reading one quarter of the K tile from LDS.
///
/// During the quarter-K pingpong schedule, each quarter reads a slice of the
/// LHS and RHS operands from LDS into registers before feeding them to MMA
/// instructions. These registers are live only during the quarter's compute
/// phase.
struct LDSQuarterReadVGPRs {
  int64_t lhsVGPRs; /// VGPRs per thread for one quarter's LHS reads.
  int64_t rhsVGPRs; /// VGPRs per thread for one quarter's RHS reads.
};

/// Compute the VGPRs needed for LDS quarter reads of MMA operands.
///
/// For each quarter of the K tile:
///   lhsTiles = (subgroupM / mmaM) * (quarterK / mmaK)
///   rhsTiles = (quarterK / mmaK) * (subgroupN / mmaN)
///   lhsVGPRs = lhsTiles * computeMMAOperandVGPRs(lhsLayout, lhsBits)
///   rhsVGPRs = rhsTiles * computeMMAOperandVGPRs(rhsLayout, rhsBits)
///
/// Returns std::nullopt if subgroup or quarter dimensions are not evenly
/// divisible by the MMA tile dimensions.
std::optional<LDSQuarterReadVGPRs>
computeLDSQuarterReadVGPRs(MMAIntrinsic intrinsic, int64_t subgroupM,
                            int64_t subgroupN, int64_t quarterK,
                            int64_t lhsBits, int64_t rhsBits);

//===----------------------------------------------------------------------===//
// Index/Address Overhead
//===----------------------------------------------------------------------===//

/// Estimate the VGPR overhead from index/address computations.
///
/// This is an approximate model based on empirical observation. The overhead
/// includes thread identification, global load base addresses, LDS base
/// addresses, loop control variables, copy phase decomposition indices, and
/// an empirical compiler overhead for LLVM register allocator inefficiency.
///
/// The model produces ~41 VGPRs for a reference RDNA4 configuration (64x64
/// subgroup, K=64, 16 subgroups). Empirical ISA measurement shows ~48. The
/// gap is attributed to compiler register allocation decisions not captured
/// by the structural model.
///
/// |numLoadOperands|: number of distinct load operands (typically 2: LHS+RHS).
/// |numKQuarters|: number of K quarters in the pingpong schedule (K/mmaK/4).
/// |splitCopy|: whether the copy phase uses per-dimension index decomposition.
int64_t computeIndexOverheadVGPRs(int64_t numLoadOperands, int64_t numKQuarters,
                                  bool splitCopy);

//===----------------------------------------------------------------------===//
// Peak VGPR Usage
//===----------------------------------------------------------------------===//

/// Peak VGPR usage summary for a quarter-K pingpong schedule.
struct PeakVGPRUsage {
  int64_t totalVGPRs; /// Peak VGPR count across all schedule phases.
  int64_t headroom;    /// Available - peak (positive = fits, negative = spills).
  bool spills;         /// Whether totalVGPRs exceeds available budget.
};

/// Compute peak VGPR usage for the quarter-K pingpong schedule.
///
/// The peak depends on the schedule variant:
///
/// **Original schedule**: The peak phase has global load staging and 2
/// simultaneous quarter reads (q_i and q_{i+1}) all live at once:
///   peak = acc + GL_LHS + GL_RHS + 2*(qLHS + qRHS) + index
///
/// **Early-write schedule**: Global load staging VGPRs are consumed (written
/// to LDS) before the second quarter read, so only 1 quarter is live at peak:
///   peak = acc + GL_LHS + GL_RHS + 1*(qLHS + qRHS) + index
///
/// |earlyWrite|: true for the early-write schedule variant.
PeakVGPRUsage computePeakVGPRUsage(int64_t accumulatorVGPRs,
                                    GlobalLoadVGPRs globalLoadVGPRs,
                                    LDSQuarterReadVGPRs quarterReadVGPRs,
                                    int64_t indexOverheadVGPRs,
                                    int64_t availableVGPRs, bool earlyWrite);

//===----------------------------------------------------------------------===//
// LDS Allocation
//===----------------------------------------------------------------------===//

/// LDS memory allocation for matmul tiles.
struct LDSAllocation {
  int64_t lhsBytes;     /// Bytes for one LHS tile buffer.
  int64_t rhsBytes;     /// Bytes for one RHS tile buffer.
  int64_t totalBytes;   /// Total LDS bytes (all buffers).
  int64_t bufferDepth;  /// Multi-buffering depth.
};

/// Compute LDS allocation for the given tile sizes and multi-buffering depth.
///
/// Returns std::nullopt if the allocation exceeds |maxLDSBytes|.
std::optional<LDSAllocation>
computeLDSAllocation(int64_t lhsTileElements, int64_t rhsTileElements,
                     int64_t elementBytes, int64_t bufferDepth,
                     int64_t maxLDSBytes);

/// Compute the maximum multi-buffering depth that fits in the given LDS
/// budget.
int64_t computeMaxBufferDepth(int64_t lhsTileElements,
                              int64_t rhsTileElements, int64_t elementBytes,
                              int64_t maxLDSBytes);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_COSTMODEL_H_
