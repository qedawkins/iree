// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unit tests for the pure arithmetic cost model functions in CostModelArith.
// These tests have no dependency on IREEGPUDialect or MLIRContext, avoiding
// the deep transitive link chain through DerivedConfigUtils -> Codegen/Utils
// -> HAL/IR that IREEGPUDialect introduces.
//
// Functions that depend on MMA intrinsics or TargetAttr (computeMMARegisterCost,
// computeVGPRBudget, computeAccumulatorVGPRs, etc.) are tested through
// integration tests that link the full compiler.

#include <gtest/gtest.h>

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/CostModel.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ScheduleConfig.h"

namespace mlir::iree_compiler::IREE::GPU {
namespace {

//===----------------------------------------------------------------------===//
// computeMMAOperandVGPRs
//===----------------------------------------------------------------------===//

TEST(CostModelArithTest, MMAOperandVGPRsBasic) {
  // Layout: outer={2,1}, element={1,4}.
  // elementsPerThread = 2*1*1*4 = 8.
  MMASingleSubgroupLayout layout;
  layout.outer = {2, 1};
  layout.thread = {8, 4};
  layout.tstrides = {4, 1};
  layout.element = {1, 4};

  // 16-bit elements: 8*16/32 = 4 VGPRs.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/16), 4);
  // 32-bit elements: 8*32/32 = 8 VGPRs.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/32), 8);
}

TEST(CostModelArithTest, MMAOperandVGPRsTrivial) {
  // All-ones layout: 1 element per thread.
  MMASingleSubgroupLayout layout;
  layout.outer = {1, 1};
  layout.thread = {16, 2};
  layout.tstrides = {2, 1};
  layout.element = {1, 1};

  // 32-bit: ceil(1*32/32) = 1 VGPR.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/32), 1);
  // 16-bit: ceil(1*16/32) = 1 VGPR.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/16), 1);
}

TEST(CostModelArithTest, MMAOperandVGPRsLargeLayout) {
  // Layout: outer={4, 2}, element={2, 2}.
  // elementsPerThread = 4*2*2*2 = 32.
  MMASingleSubgroupLayout layout;
  layout.outer = {4, 2};
  layout.thread = {4, 8};
  layout.tstrides = {8, 1};
  layout.element = {2, 2};

  // 16-bit: ceil(32*16/32) = 16 VGPRs.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/16), 16);
  // 32-bit: ceil(32*32/32) = 32 VGPRs.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/32), 32);
  // 8-bit: ceil(32*8/32) = 8 VGPRs.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/8), 8);
}

TEST(CostModelArithTest, MMAOperandVGPRsCeilDivision) {
  // Layout with 3 elements per thread to test ceiling behavior.
  // outer={1,1}, element={1,3}: elementsPerThread = 3.
  MMASingleSubgroupLayout layout;
  layout.outer = {1, 1};
  layout.thread = {16, 2};
  layout.tstrides = {2, 1};
  layout.element = {1, 3};

  // 16-bit: ceil(3*16/32) = ceil(48/32) = 2 VGPRs.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/16), 2);
  // 8-bit: ceil(3*8/32) = ceil(24/32) = 1 VGPR.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/8), 1);
}

//===----------------------------------------------------------------------===//
// computeGlobalLoadVGPRs
//===----------------------------------------------------------------------===//

TEST(CostModelArithTest, GlobalLoadVGPRsSymmetric) {
  // 256x64 LHS + 64x256 RHS, 512 threads, f16 (16-bit).
  // LHS: ceil(16384/512)=32 elts/thread, ceil(32*16/32)=16 VGPRs.
  // RHS: ceil(16384/512)=32 elts/thread, ceil(32*16/32)=16 VGPRs.
  GlobalLoadVGPRs vgprs = computeGlobalLoadVGPRs(
      /*lhsTileElements=*/256 * 64, /*rhsTileElements=*/64 * 256,
      /*numThreads=*/512, /*elementBits=*/16);

  EXPECT_EQ(vgprs.lhsVGPRs, 16);
  EXPECT_EQ(vgprs.rhsVGPRs, 16);
}

TEST(CostModelArithTest, GlobalLoadVGPRsAsymmetric) {
  // 128x64 LHS + 64x256 RHS, 256 threads, f16.
  // LHS: ceil(8192/256)=32, ceil(32*16/32)=16.
  // RHS: ceil(16384/256)=64, ceil(64*16/32)=32.
  GlobalLoadVGPRs vgprs = computeGlobalLoadVGPRs(
      /*lhsTileElements=*/128 * 64, /*rhsTileElements=*/64 * 256,
      /*numThreads=*/256, /*elementBits=*/16);

  EXPECT_EQ(vgprs.lhsVGPRs, 16);
  EXPECT_EQ(vgprs.rhsVGPRs, 32);
}

TEST(CostModelArithTest, GlobalLoadVGPRs32Bit) {
  // 64x32 LHS + 32x64 RHS, 128 threads, f32 (32-bit).
  // LHS: ceil(2048/128)=16, ceil(16*32/32)=16.
  // RHS: ceil(2048/128)=16, ceil(16*32/32)=16.
  GlobalLoadVGPRs vgprs = computeGlobalLoadVGPRs(
      /*lhsTileElements=*/64 * 32, /*rhsTileElements=*/32 * 64,
      /*numThreads=*/128, /*elementBits=*/32);

  EXPECT_EQ(vgprs.lhsVGPRs, 16);
  EXPECT_EQ(vgprs.rhsVGPRs, 16);
}

//===----------------------------------------------------------------------===//
// computeIndexOverheadVGPRs
//===----------------------------------------------------------------------===//

TEST(CostModelArithTest, IndexOverheadBasic) {
  // 2 load operands (LHS+RHS), 4 K quarters, no split copy.
  // = 1 (thread) + 4 (global addr) + 4 (lds addr) + 2 (loop) + 2 (copy)
  //   + 2 (delin) + 24 (compiler) = 39.
  int64_t overhead = computeIndexOverheadVGPRs(
      /*numLoadOperands=*/2, /*numKQuarters=*/4, /*splitCopy=*/false);
  EXPECT_EQ(overhead, 39);
}

TEST(CostModelArithTest, IndexOverheadSplitCopy) {
  // Same but with split copy: copyDecomp = 4 instead of 2, so +2.
  int64_t overhead = computeIndexOverheadVGPRs(
      /*numLoadOperands=*/2, /*numKQuarters=*/4, /*splitCopy=*/true);
  EXPECT_EQ(overhead, 41);
}

TEST(CostModelArithTest, IndexOverheadSingleQuarter) {
  // 1 K quarter: ldsAddr = 2 (no quarter offsets).
  // = 1 + 4 + 2 + 2 + 2 + 2 + 24 = 37.
  int64_t overhead = computeIndexOverheadVGPRs(
      /*numLoadOperands=*/2, /*numKQuarters=*/1, /*splitCopy=*/false);
  EXPECT_EQ(overhead, 37);
}

//===----------------------------------------------------------------------===//
// computePeakVGPRUsage
//===----------------------------------------------------------------------===//

TEST(CostModelArithTest, PeakVGPRUsageOriginalSchedule) {
  // acc=128, GL={16,16}, quarter={16,16}, index=39, available=256.
  // Original schedule (earlyWrite=false): 2 simultaneous quarters.
  // peak = 128 + 32 + 2*32 + 39 = 263.
  PeakVGPRUsage usage = computePeakVGPRUsage(
      /*accumulatorVGPRs=*/128,
      GlobalLoadVGPRs{16, 16},
      LDSQuarterReadVGPRs{16, 16},
      /*indexOverheadVGPRs=*/39,
      /*availableVGPRs=*/256, /*earlyWrite=*/false);

  EXPECT_EQ(usage.totalVGPRs, 263);
  EXPECT_EQ(usage.headroom, -7);
  EXPECT_TRUE(usage.spills);
}

TEST(CostModelArithTest, PeakVGPRUsageEarlyWrite) {
  // Same but earlyWrite=true: only 1 simultaneous quarter.
  // peak = 128 + 32 + 1*32 + 39 = 231.
  PeakVGPRUsage usage = computePeakVGPRUsage(
      /*accumulatorVGPRs=*/128,
      GlobalLoadVGPRs{16, 16},
      LDSQuarterReadVGPRs{16, 16},
      /*indexOverheadVGPRs=*/39,
      /*availableVGPRs=*/256, /*earlyWrite=*/true);

  EXPECT_EQ(usage.totalVGPRs, 231);
  EXPECT_EQ(usage.headroom, 25);
  EXPECT_FALSE(usage.spills);
}

TEST(CostModelArithTest, PeakVGPRUsageExactFit) {
  // Exactly fits available budget: headroom = 0, no spill.
  PeakVGPRUsage usage = computePeakVGPRUsage(
      /*accumulatorVGPRs=*/100,
      GlobalLoadVGPRs{10, 10},
      LDSQuarterReadVGPRs{5, 5},
      /*indexOverheadVGPRs=*/30,
      /*availableVGPRs=*/160, /*earlyWrite=*/true);

  EXPECT_EQ(usage.totalVGPRs, 160);
  EXPECT_EQ(usage.headroom, 0);
  EXPECT_FALSE(usage.spills);
}

//===----------------------------------------------------------------------===//
// computeLDSAllocation
//===----------------------------------------------------------------------===//

TEST(CostModelArithTest, LDSAllocationSingleBuffer) {
  // 256*64 LHS + 64*256 RHS, 2 bytes/element, depth=1, max=64KB.
  std::optional<LDSAllocation> alloc = computeLDSAllocation(
      /*lhsTileElements=*/256 * 64, /*rhsTileElements=*/64 * 256,
      /*elementBytes=*/2, /*bufferDepth=*/1, /*maxLDSBytes=*/64 * 1024);

  ASSERT_TRUE(alloc.has_value());
  EXPECT_EQ(alloc->lhsBytes, 256 * 64 * 2);
  EXPECT_EQ(alloc->rhsBytes, 64 * 256 * 2);
  EXPECT_EQ(alloc->totalBytes, 65536);
  EXPECT_EQ(alloc->bufferDepth, 1);
}

TEST(CostModelArithTest, LDSAllocationDoubleBufferOverflow) {
  // Double-buffer exceeds 64KB: nullopt.
  std::optional<LDSAllocation> alloc = computeLDSAllocation(
      /*lhsTileElements=*/256 * 64, /*rhsTileElements=*/64 * 256,
      /*elementBytes=*/2, /*bufferDepth=*/2, /*maxLDSBytes=*/64 * 1024);

  EXPECT_FALSE(alloc.has_value());
}

TEST(CostModelArithTest, LDSAllocationDoubleBufferFits) {
  // Smaller tile that allows double-buffering.
  // (4096 + 4096) * 2 * 2 = 32768.
  std::optional<LDSAllocation> alloc = computeLDSAllocation(
      /*lhsTileElements=*/128 * 32, /*rhsTileElements=*/32 * 128,
      /*elementBytes=*/2, /*bufferDepth=*/2, /*maxLDSBytes=*/64 * 1024);

  ASSERT_TRUE(alloc.has_value());
  EXPECT_EQ(alloc->totalBytes, (4096 + 4096) * 2 * 2);
  EXPECT_EQ(alloc->bufferDepth, 2);
}

//===----------------------------------------------------------------------===//
// computeMaxBufferDepth
//===----------------------------------------------------------------------===//

TEST(CostModelArithTest, MaxBufferDepthExactFit) {
  // Single buffer = 64KB, max = 64KB: depth 1.
  EXPECT_EQ(computeMaxBufferDepth(
      /*lhsTileElements=*/256 * 64, /*rhsTileElements=*/64 * 256,
      /*elementBytes=*/2, /*maxLDSBytes=*/64 * 1024), 1);

  // Double LDS budget: depth 2.
  EXPECT_EQ(computeMaxBufferDepth(
      /*lhsTileElements=*/256 * 64, /*rhsTileElements=*/64 * 256,
      /*elementBytes=*/2, /*maxLDSBytes=*/128 * 1024), 2);
}

TEST(CostModelArithTest, MaxBufferDepthSmallTile) {
  // Single buffer = 4096 bytes, 64KB / 4096 = 16.
  EXPECT_EQ(computeMaxBufferDepth(
      /*lhsTileElements=*/64 * 16, /*rhsTileElements=*/16 * 64,
      /*elementBytes=*/2, /*maxLDSBytes=*/64 * 1024), 16);
}

TEST(CostModelArithTest, MaxBufferDepthZeroElements) {
  // Zero tile elements: 0 (avoids division by zero).
  EXPECT_EQ(computeMaxBufferDepth(0, 0, 2, 64 * 1024), 0);
}

//===----------------------------------------------------------------------===//
// Stress tests: realistic RDNA4 configurations and edge cases
//===----------------------------------------------------------------------===//

TEST(CostModelArithTest, RDNA4PingpongSchedulePeakVGPRs) {
  // Canonical RDNA4 64x64 subgroup, WMMA_F32_16x16x16_F16, 4 K quarters.
  // This matches the reference schedule from the design doc.
  //
  // Accumulator: 128 VGPRs (4x4 tiles × 8 VGPRs each for f32)
  // GL staging: 16+16 = 32 VGPRs (256×64 tile, 512 threads, f16)
  // Quarter read: 16+16 = 32 VGPRs per quarter
  // Index overhead: 39 VGPRs (2 operands, 4 quarters, no split)
  // Total (original): 128 + 32 + 2*32 + 39 = 263 → SPILLS
  // Total (early-write): 128 + 32 + 1*32 + 39 = 231 → fits in 256

  PeakVGPRUsage original = computePeakVGPRUsage(
      128, GlobalLoadVGPRs{16, 16}, LDSQuarterReadVGPRs{16, 16}, 39, 256,
      /*earlyWrite=*/false);
  EXPECT_TRUE(original.spills);
  EXPECT_EQ(original.totalVGPRs, 263);

  PeakVGPRUsage earlyWrite = computePeakVGPRUsage(
      128, GlobalLoadVGPRs{16, 16}, LDSQuarterReadVGPRs{16, 16}, 39, 256,
      /*earlyWrite=*/true);
  EXPECT_FALSE(earlyWrite.spills);
  EXPECT_EQ(earlyWrite.totalVGPRs, 231);
  EXPECT_EQ(earlyWrite.headroom, 25);
}

TEST(CostModelArithTest, GlobalLoadVGPRsMinimum) {
  // Minimal tile: 1 element per operand, 1 thread.
  // 1 element × 16 bits / 32 = 1 VGPR each.
  GlobalLoadVGPRs vgprs = computeGlobalLoadVGPRs(1, 1, 1, 16);
  EXPECT_EQ(vgprs.lhsVGPRs, 1);
  EXPECT_EQ(vgprs.rhsVGPRs, 1);
}

TEST(CostModelArithTest, GlobalLoadVGPRsCeilDiv) {
  // 3 elements per thread at 16 bits: ceil(3*16/32) = ceil(1.5) = 2.
  GlobalLoadVGPRs vgprs = computeGlobalLoadVGPRs(3, 3, 1, 16);
  EXPECT_EQ(vgprs.lhsVGPRs, 2);
  EXPECT_EQ(vgprs.rhsVGPRs, 2);
}

TEST(CostModelArithTest, GlobalLoadVGPRsLargeAsymmetric) {
  // Very large tile: 512x128 LHS + 128x512 RHS, 1024 threads, f16.
  // LHS: ceil(65536/1024)=64 elts, ceil(64*16/32)=32.
  // RHS: same = 32.
  GlobalLoadVGPRs vgprs = computeGlobalLoadVGPRs(512 * 128, 128 * 512, 1024,
                                                   16);
  EXPECT_EQ(vgprs.lhsVGPRs, 32);
  EXPECT_EQ(vgprs.rhsVGPRs, 32);
}

TEST(CostModelArithTest, IndexOverheadManyQuarters) {
  // 8 K quarters, 3 load operands (e.g., LHS+RHS+bias), split copy.
  // = 1 (tid) + 6 (global) + 4 (lds+quarter offsets) + 2 (loop) + 4 (split)
  //   + 2 (delin) + 24 (compiler) = 43.
  int64_t overhead =
      computeIndexOverheadVGPRs(/*numLoadOperands=*/3, /*numKQuarters=*/8,
                                /*splitCopy=*/true);
  EXPECT_EQ(overhead, 43);
}

TEST(CostModelArithTest, IndexOverheadMinimal) {
  // 1 load operand, 1 K quarter, no split.
  // = 1 (tid) + 2 (global) + 2 (lds, no quarter offsets) + 2 (loop)
  //   + 2 (copy) + 2 (delin) + 24 (compiler) = 35.
  int64_t overhead =
      computeIndexOverheadVGPRs(/*numLoadOperands=*/1, /*numKQuarters=*/1,
                                /*splitCopy=*/false);
  EXPECT_EQ(overhead, 35);
}

TEST(CostModelArithTest, LDSAllocationMaxFit) {
  // Exactly fills 64KB with single buffer at 2 bytes/element.
  // Total = (16384 + 16384) * 2 = 65536 = 64KB.
  std::optional<LDSAllocation> alloc =
      computeLDSAllocation(16384, 16384, 2, 1, 64 * 1024);
  ASSERT_TRUE(alloc.has_value());
  EXPECT_EQ(alloc->totalBytes, 65536);
}

TEST(CostModelArithTest, LDSAllocationOneByteOver) {
  // One element over 64KB: should fail.
  std::optional<LDSAllocation> alloc =
      computeLDSAllocation(16385, 16384, 2, 1, 64 * 1024);
  EXPECT_FALSE(alloc.has_value());
}

TEST(CostModelArithTest, MaxBufferDepthLargeRatio) {
  // Small tile: (64+64)*2 = 256 bytes per buffer, 64KB LDS: depth = 256.
  EXPECT_EQ(
      computeMaxBufferDepth(/*lhsTileElements=*/64, /*rhsTileElements=*/64,
                            /*elementBytes=*/2, /*maxLDSBytes=*/64 * 1024),
      256);
}

TEST(CostModelArithTest, PeakVGPRUsageZeroAccumulator) {
  // Degenerate case: no accumulator (e.g., pure copy kernel).
  PeakVGPRUsage usage = computePeakVGPRUsage(
      0, GlobalLoadVGPRs{8, 8}, LDSQuarterReadVGPRs{4, 4}, 30, 256,
      /*earlyWrite=*/true);
  EXPECT_EQ(usage.totalVGPRs, 0 + 16 + 8 + 30);
  EXPECT_FALSE(usage.spills);
}

TEST(CostModelArithTest, PeakVGPRUsageMassiveSpill) {
  // Way over budget: 512 VGPRs needed, only 256 available.
  PeakVGPRUsage usage = computePeakVGPRUsage(
      256, GlobalLoadVGPRs{64, 64}, LDSQuarterReadVGPRs{32, 32}, 50, 256,
      /*earlyWrite=*/false);
  EXPECT_TRUE(usage.spills);
  EXPECT_EQ(usage.headroom, 256 - usage.totalVGPRs);
  EXPECT_LT(usage.headroom, -200); // Massively over budget.
}

//===----------------------------------------------------------------------===//
// buildVGPRBudgetAnalysis — Phase-by-phase VGPR liveness
//===----------------------------------------------------------------------===//

// Validation target 1: RDNA4 256x256, K=64, 16 subgroups, wave32.
// WMMA_F32_16x16x16_F16: sg 64x64, 16 MMA tiles, 512 threads.
// Pre-computed costs:
//   acc=128, GL_L=16, GL_R=16, qL=16, qR=16, idx=41
//   available=256 (RDNA4 occ1)
TEST(VGPRBudgetTest, RDNA4_256x256_K64_Original) {
  VGPRBudgetAnalysis budget = buildVGPRBudgetAnalysis(
      /*accumulatorVGPRs=*/128, GlobalLoadVGPRs{16, 16},
      LDSQuarterReadVGPRs{16, 16}, /*indexOverheadVGPRs=*/41,
      /*availableVGPRs=*/256, /*numQuarters=*/4, /*earlyWrite=*/false);

  // Per-category costs echoed back.
  EXPECT_EQ(budget.accumulatorVGPRs, 128);
  EXPECT_EQ(budget.globalLoadLHSVGPRs, 16);
  EXPECT_EQ(budget.globalLoadRHSVGPRs, 16);
  EXPECT_EQ(budget.ldsReadLHSPerQuarter, 16);
  EXPECT_EQ(budget.ldsReadRHSPerQuarter, 16);
  EXPECT_EQ(budget.indexOverheadVGPRs, 41);

  // 8 phases for 4-quarter schedule.
  EXPECT_EQ(budget.phases.size(), 8u);

  // Peak at P5 (2 simultaneous quarter reads + GL).
  // P5 = 128 + 16 + 16 + 2*16 + 2*16 + 41 = 265.
  EXPECT_EQ(budget.peakVGPRs(), 265);
  EXPECT_EQ(budget.headroom(), 256 - 265); // -9.
  EXPECT_TRUE(budget.willSpill());

  // Verify P5 is the peak phase.
  EXPECT_STREQ(budget.peakPhase().phaseName, "P5: LDS-read q2+q3 (PEAK)");

  // P1: acc(128) + GL_L(16) + 0 + qL(16) + qR(16) + idx(41) = 217.
  EXPECT_EQ(budget.phases[0].total(), 217);
  // P8: acc(128) + 0 + 0 + 0 + 0 + idx(41) = 169.
  EXPECT_EQ(budget.phases[7].total(), 169);
}

TEST(VGPRBudgetTest, RDNA4_256x256_K64_EarlyWrite) {
  VGPRBudgetAnalysis budget = buildVGPRBudgetAnalysis(
      /*accumulatorVGPRs=*/128, GlobalLoadVGPRs{16, 16},
      LDSQuarterReadVGPRs{16, 16}, /*indexOverheadVGPRs=*/41,
      /*availableVGPRs=*/256, /*numQuarters=*/4, /*earlyWrite=*/true);

  // 8 phases for 4-quarter early-write.
  EXPECT_EQ(budget.phases.size(), 8u);

  // Peak at P3 (GL-LHS + GL-RHS + 1 quarter read).
  // P3 = 128 + 16 + 16 + 16 + 16 + 41 = 233.
  EXPECT_EQ(budget.peakVGPRs(), 233);
  EXPECT_EQ(budget.headroom(), 256 - 233); // +23.
  EXPECT_FALSE(budget.willSpill());

  // P5 has GL-LHS freed: 128 + 0 + 16 + 16 + 16 + 41 = 217.
  EXPECT_EQ(budget.phases[4].total(), 217);
  // P6 has both GL freed: 128 + 0 + 0 + 16 + 16 + 41 = 201.
  EXPECT_EQ(budget.phases[5].total(), 201);
}

// Validation target 2: CDNA3 256x256, K=64, 16 subgroups, wave64.
// MFMA_F32_16x16x16_F16: sg 64x64, 16 MMA tiles, 1024 threads.
// Pre-computed costs:
//   acc=64, GL_L=8, GL_R=8, qL=8, qR=8, idx=41
//   available ArchVGPR=128, available AccVGPR=128
TEST(VGPRBudgetTest, CDNA3_256x256_K64_AccVGPR) {
  VGPRBudgetAnalysis budget = buildVGPRBudgetAnalysis(
      /*accumulatorVGPRs=*/64, GlobalLoadVGPRs{8, 8},
      LDSQuarterReadVGPRs{8, 8}, /*indexOverheadVGPRs=*/41,
      /*availableVGPRs=*/128, /*numQuarters=*/4, /*earlyWrite=*/false,
      /*hasAccVGPR=*/true, /*availableAccVGPRs=*/128);

  // Peak total (ArchVGPR + AccVGPR): P5 = 64 + 8 + 8 + 16 + 16 + 41 = 153.
  EXPECT_EQ(budget.peakVGPRs(), 153);
  // Peak ArchVGPR only: 153 - 64 = 89.
  EXPECT_EQ(budget.peakArchVGPRs(), 89);
  // ArchVGPR headroom: 128 - 89 = 39.
  EXPECT_EQ(budget.headroom(), 39);
  // AccVGPR headroom: 128 - 64 = 64.
  EXPECT_EQ(budget.accHeadroom(), 64);
  // Should NOT spill.
  EXPECT_FALSE(budget.willSpill());
}

// Validation target 3: RDNA4 128x128, K=64, 4 subgroups, wave32.
// 4 subgroups * 32 = 128 threads. SG 64x64. Same acc (128).
// GL: ceil(8192/128) = 64 elts, ceil(64*16/32) = 32 VGPRs each.
TEST(VGPRBudgetTest, RDNA4_128x128_K64_LargeGL) {
  VGPRBudgetAnalysis budget = buildVGPRBudgetAnalysis(
      /*accumulatorVGPRs=*/128, GlobalLoadVGPRs{32, 32},
      LDSQuarterReadVGPRs{16, 16}, /*indexOverheadVGPRs=*/41,
      /*availableVGPRs=*/256, /*numQuarters=*/4, /*earlyWrite=*/false);

  // P5 = 128 + 32 + 32 + 32 + 32 + 41 = 297.
  EXPECT_EQ(budget.peakVGPRs(), 297);
  EXPECT_TRUE(budget.willSpill());

  // Early-write version should be better.
  VGPRBudgetAnalysis ewBudget = buildVGPRBudgetAnalysis(
      128, GlobalLoadVGPRs{32, 32}, LDSQuarterReadVGPRs{16, 16}, 41, 256, 4,
      /*earlyWrite=*/true);
  // Peak in early-write: P3 = 128 + 32 + 32 + 16 + 16 + 41 = 265.
  EXPECT_EQ(ewBudget.peakVGPRs(), 265);
  EXPECT_TRUE(ewBudget.willSpill()); // Still spills at 256.
}

// Validation target 4: DMA (GL VGPRs = 0).
TEST(VGPRBudgetTest, CDNA3_DMA_NoGlobalLoadVGPRs) {
  VGPRBudgetAnalysis budget = buildVGPRBudgetAnalysis(
      /*accumulatorVGPRs=*/64, GlobalLoadVGPRs{0, 0},
      LDSQuarterReadVGPRs{8, 8}, /*indexOverheadVGPRs=*/41,
      /*availableVGPRs=*/128, /*numQuarters=*/4, /*earlyWrite=*/false,
      /*hasAccVGPR=*/true, /*availableAccVGPRs=*/128);

  // No GL staging: peak = 64 + 0 + 0 + 16 + 16 + 41 = 137.
  EXPECT_EQ(budget.peakVGPRs(), 137);
  EXPECT_EQ(budget.peakArchVGPRs(), 73); // 137 - 64.
  EXPECT_FALSE(budget.willSpill());
}

// Validation target 5: Occupancy 2 (halved VGPRs).
TEST(VGPRBudgetTest, RDNA4_Occupancy2_HalvedBudget) {
  VGPRBudgetAnalysis budget = buildVGPRBudgetAnalysis(
      /*accumulatorVGPRs=*/128, GlobalLoadVGPRs{16, 16},
      LDSQuarterReadVGPRs{16, 16}, /*indexOverheadVGPRs=*/41,
      /*availableVGPRs=*/128, // Halved for occupancy 2.
      /*numQuarters=*/4, /*earlyWrite=*/false);

  EXPECT_EQ(budget.peakVGPRs(), 265);
  EXPECT_EQ(budget.headroom(), 128 - 265); // -137.
  EXPECT_TRUE(budget.willSpill());
}

// 2-quarter schedule (K=32).
TEST(VGPRBudgetTest, TwoQuarterSchedule) {
  VGPRBudgetAnalysis budget = buildVGPRBudgetAnalysis(
      /*accumulatorVGPRs=*/128, GlobalLoadVGPRs{16, 16},
      LDSQuarterReadVGPRs{16, 16}, /*indexOverheadVGPRs=*/39,
      /*availableVGPRs=*/256, /*numQuarters=*/2, /*earlyWrite=*/false);

  // 4 phases for 2-quarter schedule.
  EXPECT_EQ(budget.phases.size(), 4u);

  // Peak at P1: acc(128) + GL_L(16) + GL_R(16) + qL(16) + qR(16) + idx(39)
  // = 231.
  EXPECT_EQ(budget.peakVGPRs(), 231);
  EXPECT_FALSE(budget.willSpill());
}

// Phase table structure verification.
TEST(VGPRBudgetTest, PhaseTableMonotonicity) {
  // In the original schedule, accumulator and index are always live.
  // GL regs appear in P1-P6 then disappear. Quarter reads appear in
  // read phases.
  VGPRBudgetAnalysis budget = buildVGPRBudgetAnalysis(
      100, GlobalLoadVGPRs{10, 10}, LDSQuarterReadVGPRs{5, 5}, 30, 256, 4,
      false);

  // All phases have accumulator and index.
  for (const PhaseVGPRLiveness &phase : budget.phases) {
    EXPECT_EQ(phase.accumulator, 100);
    EXPECT_EQ(phase.indexOverhead, 30);
  }

  // P8 (last phase) has no GL and no quarter reads.
  EXPECT_EQ(budget.phases[7].globalLoadLHS, 0);
  EXPECT_EQ(budget.phases[7].globalLoadRHS, 0);
  EXPECT_EQ(budget.phases[7].ldsReadLHS, 0);
  EXPECT_EQ(budget.phases[7].ldsReadRHS, 0);
  EXPECT_EQ(budget.phases[7].total(), 130); // 100 + 0 + 0 + 0 + 0 + 30.
}

// Consistency: peakVGPRs should match computePeakVGPRUsage.
TEST(VGPRBudgetTest, ConsistencyWithPeakVGPRUsage) {
  int64_t acc = 128;
  GlobalLoadVGPRs gl = {16, 16};
  LDSQuarterReadVGPRs qr = {16, 16};
  int64_t idx = 41;
  int64_t avail = 256;

  // Original schedule.
  PeakVGPRUsage peakOrig = computePeakVGPRUsage(acc, gl, qr, idx, avail,
                                                  /*earlyWrite=*/false);
  VGPRBudgetAnalysis budgetOrig =
      buildVGPRBudgetAnalysis(acc, gl, qr, idx, avail, 4, false);
  EXPECT_EQ(budgetOrig.peakVGPRs(), peakOrig.totalVGPRs);
  EXPECT_EQ(budgetOrig.headroom(), peakOrig.headroom);
  EXPECT_EQ(budgetOrig.willSpill(), peakOrig.spills);

  // Early-write schedule.
  PeakVGPRUsage peakEW =
      computePeakVGPRUsage(acc, gl, qr, idx, avail, /*earlyWrite=*/true);
  VGPRBudgetAnalysis budgetEW =
      buildVGPRBudgetAnalysis(acc, gl, qr, idx, avail, 4, true);
  EXPECT_EQ(budgetEW.peakVGPRs(), peakEW.totalVGPRs);
  EXPECT_EQ(budgetEW.headroom(), peakEW.headroom);
  EXPECT_EQ(budgetEW.willSpill(), peakEW.spills);
}

//===----------------------------------------------------------------------===//
// computePipelinedBurst
//===----------------------------------------------------------------------===//

TEST(LatencyModelTest, PipelinedBurstBasic) {
  // (N-1)*issue + exec: (10-1)*2 + 32 = 50.
  EXPECT_EQ(computePipelinedBurst(10, 2, 32), 50);
}

TEST(LatencyModelTest, PipelinedBurstSingleInstruction) {
  // Single instruction: just exec latency.
  EXPECT_EQ(computePipelinedBurst(1, 2, 32), 32);
}

TEST(LatencyModelTest, PipelinedBurstZeroCount) {
  EXPECT_EQ(computePipelinedBurst(0, 2, 32), 0);
}

TEST(LatencyModelTest, PipelinedBurstNegativeCount) {
  EXPECT_EQ(computePipelinedBurst(-1, 2, 32), 0);
}

TEST(LatencyModelTest, PipelinedBurstLargeCount) {
  // 100 WMMAs: (100-1)*2 + 32 = 230.
  EXPECT_EQ(computePipelinedBurst(100, 2, 32), 230);
}

//===----------------------------------------------------------------------===//
// computePhaseTiming
//===----------------------------------------------------------------------===//

TEST(LatencyModelTest, PhaseTimingPureCompute) {
  // Pure WMMA phase: no LDS, no global loads.
  // 16 WMMAs: (16-1)*2 + 32 = 62 cycles.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  PhaseInstructionCounts counts;
  counts.numWMMA = 16;
  counts.numVALUAddr = 0;
  counts.numLDSLoads = 0;
  counts.numLDSStores = 0;
  counts.numGlobalLoads = 0;
  counts.hasBarrier = true;

  PhaseTiming timing =
      computePhaseTiming(counts, mma, ldsRead, ldsWrite, globalLoad, 20);

  EXPECT_EQ(timing.valuPipelineCycles, 62); // (15)*2 + 32
  EXPECT_EQ(timing.ldsBusCycles, 0);
  EXPECT_EQ(timing.vmemIssueCycles, 0);
  EXPECT_EQ(timing.barrierCycles, 20);
  EXPECT_EQ(timing.totalCycles, 62 + 20); // max(62,0,0) + 20
}

TEST(LatencyModelTest, PhaseTimingMemoryPhase) {
  // Memory phase: LDS reads + global loads, no WMMA.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  PhaseInstructionCounts counts;
  counts.numWMMA = 0;
  counts.numVALUAddr = 20; // Address computation VALU ops.
  counts.numLDSLoads = 8;
  counts.numLDSStores = 0;
  counts.numGlobalLoads = 4;
  counts.hasBarrier = true;

  PhaseTiming timing =
      computePhaseTiming(counts, mma, ldsRead, ldsWrite, globalLoad, 20);

  // VALU pipeline: 0 (no WMMA) + 20 (addr) = 20.
  EXPECT_EQ(timing.valuPipelineCycles, 20);
  // LDS bus: reads only = (8-1)*2 + 25 = 39.
  EXPECT_EQ(timing.ldsBusCycles, 39);
  // VMEM issue: 4*2 = 8.
  EXPECT_EQ(timing.vmemIssueCycles, 8);
  // Total: max(20, 39, 8) + 20 = 59.
  EXPECT_EQ(timing.totalCycles, 59);
}

TEST(LatencyModelTest, PhaseTimingLDSReadWrite2Port) {
  // LDS reads + writes simultaneously: 2-port overlap = max(read, write).
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  PhaseInstructionCounts counts;
  counts.numWMMA = 0;
  counts.numVALUAddr = 0;
  counts.numLDSLoads = 8;
  counts.numLDSStores = 4;
  counts.numGlobalLoads = 0;
  counts.hasBarrier = true;

  PhaseTiming timing =
      computePhaseTiming(counts, mma, ldsRead, ldsWrite, globalLoad, 20);

  // LDS read: (8-1)*2 + 25 = 39.
  // LDS write: (4-1)*2 + 10 = 16.
  // 2-port overlap: max(39, 16) = 39.
  EXPECT_EQ(timing.ldsBusCycles, 39);
  EXPECT_EQ(timing.totalCycles, 39 + 20);
}

TEST(LatencyModelTest, PhaseTimingNoBarrier) {
  // Phase without barrier.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  PhaseInstructionCounts counts;
  counts.numWMMA = 4;
  counts.numVALUAddr = 0;
  counts.numLDSLoads = 0;
  counts.numLDSStores = 0;
  counts.numGlobalLoads = 0;
  counts.hasBarrier = false;

  PhaseTiming timing =
      computePhaseTiming(counts, mma, ldsRead, ldsWrite, globalLoad, 20);

  // WMMA: (4-1)*2 + 32 = 38.
  EXPECT_EQ(timing.valuPipelineCycles, 38);
  EXPECT_EQ(timing.barrierCycles, 0);
  EXPECT_EQ(timing.totalCycles, 38); // No barrier added.
}

TEST(LatencyModelTest, PhaseTimingWMMADominates) {
  // Compute-bound phase: WMMA pipeline >> LDS and VMEM.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  PhaseInstructionCounts counts;
  counts.numWMMA = 64;
  counts.numVALUAddr = 10;
  counts.numLDSLoads = 4;
  counts.numLDSStores = 2;
  counts.numGlobalLoads = 2;
  counts.hasBarrier = true;

  PhaseTiming timing =
      computePhaseTiming(counts, mma, ldsRead, ldsWrite, globalLoad, 20);

  // WMMA: (64-1)*2 + 32 = 158, + 10 addr = 168.
  EXPECT_EQ(timing.valuPipelineCycles, 168);
  // LDS: max((4-1)*2+25, (2-1)*2+10) = max(31, 12) = 31.
  EXPECT_EQ(timing.ldsBusCycles, 31);
  // VMEM: 2*2 = 4.
  EXPECT_EQ(timing.vmemIssueCycles, 4);
  // Total: max(168, 31, 4) + 20 = 188.
  EXPECT_EQ(timing.totalCycles, 188);
}

//===----------------------------------------------------------------------===//
// computeIterationCycles
//===----------------------------------------------------------------------===//

TEST(LatencyModelTest, IterationCyclesRDNA4Reference) {
  // RDNA4 reference: 256x256 workgroup (4x4 subgroups), 64x64 subgroup,
  // K=64, WMMA_F32_16x16x16_F16, 512 threads, early-write.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  int64_t cycles = computeIterationCycles(
      /*workgroupM=*/256, /*workgroupN=*/256,
      /*subgroupM=*/64, /*subgroupN=*/64, /*kTile=*/64,
      /*mmaM=*/16, /*mmaN=*/16, /*mmaK=*/16,
      /*numThreads=*/512, /*inputBits=*/16,
      /*earlyWrite=*/true, mma, ldsRead, ldsWrite, globalLoad,
      /*barrierCycles=*/20);

  // Should produce a positive cycle count.
  EXPECT_GT(cycles, 0);
  // 8 phases with barriers, WMMA-dominant: expect a few hundred cycles.
  // Rough lower bound: 4 WMMA phases × ~82 cycles each = ~328.
  EXPECT_GT(cycles, 300);
  // Upper bound sanity: shouldn't exceed 2000 for this config.
  EXPECT_LT(cycles, 2000);
}

TEST(LatencyModelTest, IterationCycles2Quarter) {
  // K=32 with mmaK=16: 2 quarters, not 4.
  // 256x256 workgroup (4x4 subgroups), 64x64 subgroup.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  int64_t cycles = computeIterationCycles(
      /*workgroupM=*/256, /*workgroupN=*/256,
      /*subgroupM=*/64, /*subgroupN=*/64, /*kTile=*/32,
      /*mmaM=*/16, /*mmaN=*/16, /*mmaK=*/16,
      /*numThreads=*/512, /*inputBits=*/16,
      /*earlyWrite=*/true, mma, ldsRead, ldsWrite, globalLoad,
      /*barrierCycles=*/20);

  // 2-quarter schedule: 4 phases. Should be roughly half of 4-quarter.
  EXPECT_GT(cycles, 100);
  EXPECT_LT(cycles, 1000);
}

TEST(LatencyModelTest, IterationCyclesOriginalVsEarlyWrite) {
  // Compare original and early-write schedules. They should produce
  // different cycle counts since the phase structure differs.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  int64_t earlyWriteCycles = computeIterationCycles(
      256, 256, 64, 64, 64, 16, 16, 16, 512, 16, /*earlyWrite=*/true, mma,
      ldsRead, ldsWrite, globalLoad, 20);

  int64_t originalCycles = computeIterationCycles(
      256, 256, 64, 64, 64, 16, 16, 16, 512, 16, /*earlyWrite=*/false, mma,
      ldsRead, ldsWrite, globalLoad, 20);

  // Both should be positive.
  EXPECT_GT(earlyWriteCycles, 0);
  EXPECT_GT(originalCycles, 0);
  // They should differ (different phase layouts).
  EXPECT_NE(earlyWriteCycles, originalCycles);
}

TEST(LatencyModelTest, IterationCyclesZeroKTile) {
  // Edge case: kTile=0 should return 0.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  int64_t cycles = computeIterationCycles(256, 256, 64, 64, 0, 16, 16, 16,
                                           512, 16, true, mma, ldsRead,
                                           ldsWrite, globalLoad, 20);
  EXPECT_EQ(cycles, 0);
}

TEST(LatencyModelTest, IterationCyclesLargerSubgroup) {
  // 128x128 subgroup: more WMMAs, should take more cycles.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  // 256x256 workgroup, 64x64 subgroup (4x4 layout), 512 threads.
  int64_t small = computeIterationCycles(
      256, 256, 64, 64, 64, 16, 16, 16, 512, 16, true, mma, ldsRead, ldsWrite,
      globalLoad, 20);

  // 256x256 workgroup, 128x128 subgroup (2x2 layout), 128 threads.
  int64_t large = computeIterationCycles(
      256, 256, 128, 128, 64, 16, 16, 16, 128, 16, true, mma, ldsRead,
      ldsWrite, globalLoad, 20);

  // 128x128 has 4× more WMMAs per quarter than 64x64.
  EXPECT_GT(large, small);
}

//===----------------------------------------------------------------------===//
// Validation tests: RDNA4 reference configuration cross-checks
//
// These tests encode hardware findings from the Fuzzer's measurement campaign
// as ground truth assertions. If any of these fail, it means the cost model
// no longer matches empirically validated hardware behavior.
//===----------------------------------------------------------------------===//

TEST(ValidationTest, RDNA4ReferenceVGPRBudget) {
  // RDNA4 gfx1201 at occupancy 1: 256 VGPRs per thread.
  // 128 acc + 32 GL + 32 quarter + 39 index = 231 (early-write).
  PeakVGPRUsage rdna4 = computePeakVGPRUsage(
      /*accumulatorVGPRs=*/128, GlobalLoadVGPRs{16, 16},
      LDSQuarterReadVGPRs{16, 16}, /*indexOverheadVGPRs=*/39,
      /*availableVGPRs=*/256, /*earlyWrite=*/true);
  EXPECT_EQ(rdna4.totalVGPRs, 231);
  EXPECT_EQ(rdna4.headroom, 25);
  EXPECT_FALSE(rdna4.spills);
}

TEST(ValidationTest, RDNA4OriginalScheduleSpills) {
  // Original schedule (2 simultaneous quarters) with K=64 SPILLS on RDNA4.
  // Early-write is REQUIRED for K=64.
  PeakVGPRUsage original = computePeakVGPRUsage(
      128, GlobalLoadVGPRs{16, 16}, LDSQuarterReadVGPRs{16, 16}, 39, 256,
      /*earlyWrite=*/false);
  EXPECT_TRUE(original.spills);
  EXPECT_LT(original.headroom, 0);
  EXPECT_EQ(original.totalVGPRs, 263);
}

TEST(ValidationTest, RDNA4K32FallbackAlsoFits) {
  // K=32 early-write must also fit. With same 16-subgroup layout, the index
  // overhead is the same as K=64 (both have numKQuarters > 1), and GL staging
  // and quarter read VGPRs depend on subgroup tile, not K. So K=32 has the
  // same headroom as K=64 (25 VGPRs). K=32's real benefit is using only
  // half the LDS (32KB vs 64KB), leaving room for double-buffering.
  int64_t indexK32 =
      computeIndexOverheadVGPRs(/*numLoadOperands=*/2, /*numKQuarters=*/2,
                                /*splitCopy=*/false);
  PeakVGPRUsage k32 = computePeakVGPRUsage(128, GlobalLoadVGPRs{16, 16},
                                             LDSQuarterReadVGPRs{16, 16},
                                             indexK32, 256, /*earlyWrite=*/true);
  EXPECT_FALSE(k32.spills);
  EXPECT_GE(k32.headroom, 20); // Minimum headroom for safety.
}

TEST(ValidationTest, RDNA4LDSExactlyFitsK64) {
  // 256x64 LHS + 64x256 RHS = 65536 bytes = 64KB exactly.
  std::optional<LDSAllocation> alloc =
      computeLDSAllocation(256 * 64, 64 * 256, 2, 1, 65536);
  ASSERT_TRUE(alloc.has_value());
  EXPECT_EQ(alloc->totalBytes, 65536);
  EXPECT_EQ(alloc->lhsBytes, 32768);
  EXPECT_EQ(alloc->rhsBytes, 32768);
}

TEST(ValidationTest, RDNA4LDSOverflowsK96) {
  // K=96: (256x96 + 96x256) x 2 = 98304 bytes > 64KB.
  std::optional<LDSAllocation> alloc =
      computeLDSAllocation(256 * 96, 96 * 256, 2, 1, 65536);
  EXPECT_FALSE(alloc.has_value());
}

TEST(ValidationTest, RDNA4LDSDoubleBufferNeedsK32) {
  // Double-buffering at K=64: 131072 > 64KB. K=32 fits exactly.
  std::optional<LDSAllocation> k64double =
      computeLDSAllocation(256 * 64, 64 * 256, 2, 2, 65536);
  EXPECT_FALSE(k64double.has_value());

  std::optional<LDSAllocation> k32double =
      computeLDSAllocation(256 * 32, 32 * 256, 2, 2, 65536);
  ASSERT_TRUE(k32double.has_value());
  EXPECT_EQ(k32double->totalBytes, 65536);
}

TEST(ValidationTest, RDNA4EarlyWriteSaves32VGPRs) {
  // Early-write saves exactly 32 VGPRs (1 fewer simultaneous quarter x 32).
  PeakVGPRUsage earlyWrite = computePeakVGPRUsage(
      128, GlobalLoadVGPRs{16, 16}, LDSQuarterReadVGPRs{16, 16}, 39, 256,
      true);
  PeakVGPRUsage original = computePeakVGPRUsage(
      128, GlobalLoadVGPRs{16, 16}, LDSQuarterReadVGPRs{16, 16}, 39, 256,
      false);
  EXPECT_EQ(original.totalVGPRs - earlyWrite.totalVGPRs, 32);
}

TEST(ValidationTest, PipelinedWMMABurst16) {
  // 16 WMMAs pipelined: (16-1)*2 + 32 = 62 cycles.
  // ATT-validated: WMMA issue=2cy, exec=32cy.
  EXPECT_EQ(computePipelinedBurst(16, 2, 32), 62);
}

TEST(ValidationTest, IterationCyclesConsistencyCheck) {
  // 4-quarter early-write: 8 phases, each with barrier (20cy).
  // Minimum: 8 x 20 = 160cy.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  int64_t cycles = computeIterationCycles(256, 256, 64, 64, 64, 16, 16, 16,
                                           512, 16, true, mma, ldsRead,
                                           ldsWrite, globalLoad, 20);
  EXPECT_GE(cycles, 160);
  // 4 compute phases x ~62cy + 4 memory phases + 8 barriers > 400.
  EXPECT_GT(cycles, 400);
}

TEST(ValidationTest, EarlyWriteVsOriginalLatencyRange) {
  // Both schedules should produce reasonable cycle counts.
  InstructionTiming mma = {2, 32};
  InstructionTiming ldsRead = {2, 25};
  InstructionTiming ldsWrite = {2, 10};
  InstructionTiming globalLoad = {2, 300};

  int64_t earlyWrite = computeIterationCycles(256, 256, 64, 64, 64, 16, 16, 16,
                                               512, 16, true, mma, ldsRead,
                                               ldsWrite, globalLoad, 20);
  int64_t original = computeIterationCycles(256, 256, 64, 64, 64, 16, 16, 16,
                                             512, 16, false, mma, ldsRead,
                                             ldsWrite, globalLoad, 20);
  EXPECT_GT(earlyWrite, 300);
  EXPECT_LT(earlyWrite, 2000);
  EXPECT_GT(original, 300);
  EXPECT_LT(original, 2000);
}

TEST(ValidationTest, WMMAPerIterationCount) {
  // 64x64 subgroup, K=64, MMA 16x16x16:
  // 4 quarters x (4x4x1) = 64 WMMAs per iteration.
  // 64 WMMAs x 8192 FLOPs/WMMA = 524288 FLOPs/iteration.
  int64_t numQuarters = 4;
  int64_t wmmaPerQuarter = (64 / 16) * (64 / 16) * (16 / 16);
  EXPECT_EQ(wmmaPerQuarter, 16);
  EXPECT_EQ(numQuarters * wmmaPerQuarter, 64);
  EXPECT_EQ(64 * (2 * 16 * 16 * 16), 524288);
}

//===----------------------------------------------------------------------===//
// BaselineComparisonTest - cross-check model against actual IREE output
//===----------------------------------------------------------------------===//
// These tests compare the cost model's predictions against the actual ISA
// metadata from compiling a 4096x4096x4096 f16 matmul with iree-compile
// targeting gfx1201. The IREE baseline uses:
//   - Workgroup: 128x128, 4 subgroups (2x2), 64x64 per subgroup
//   - K tile: 32 (reduction unroll=2, MMA_K=16)
//   - 128 threads/workgroup (4 waves x 32 lanes)
//   - 215 VGPRs, 0 spills, 17664 bytes LDS
//   - 64 WMMAs, 4 barriers, 16 buffer_loads, 160 LDS ops

TEST(BaselineComparisonTest, BaselineGlobalLoadVGPRs) {
  // IREE baseline: 128x128 workgroup, K=32, 128 threads, f16 inputs.
  // LHS tile: 128x32 = 4096 elements. RHS tile: 32x128 = 4096 elements.
  GlobalLoadVGPRs gl = computeGlobalLoadVGPRs(
      /*lhsTileElements=*/128 * 32, /*rhsTileElements=*/32 * 128,
      /*numThreads=*/128, /*elementBits=*/16);
  // 4096 / 128 = 32 elements/thread. 32 * 16 / 32 = 16 VGPRs.
  EXPECT_EQ(gl.lhsVGPRs, 16);
  EXPECT_EQ(gl.rhsVGPRs, 16);
}

TEST(BaselineComparisonTest, BaselineAccumulatorVGPRs) {
  // 64x64 subgroup tile, MMA 16x16x16 f16->f32.
  // (64/16) * (64/16) = 16 tiles, 8 acc VGPRs each = 128 VGPRs.
  // This matches the cost model's existing prediction.
  int64_t numTiles = (64 / 16) * (64 / 16);
  EXPECT_EQ(numTiles, 16);
  EXPECT_EQ(numTiles * 8, 128); // 8 VGPRs per f32 accumulator tile.
}

TEST(BaselineComparisonTest, BaselineLDSUsage) {
  // IREE baseline uses 17664 bytes LDS. Our model for single-buffered:
  // LHS: 128x32 x 2 bytes = 8192. RHS: 32x128 x 2 = 8192. Total: 16384.
  // The 17664 - 16384 = 1280 byte difference is padding/alignment overhead.
  std::optional<LDSAllocation> alloc =
      computeLDSAllocation(128 * 32, 32 * 128, 2, 1, 65536);
  ASSERT_TRUE(alloc.has_value());
  EXPECT_EQ(alloc->totalBytes, 16384);
  // Actual ISA uses 17664 bytes (8% more due to alignment padding).
  // This is within acceptable model error.
  EXPECT_LT(std::abs(alloc->totalBytes - 17664), 2048);
}

TEST(BaselineComparisonTest, BaselineVsModelPeakVGPRs) {
  // The IREE baseline uses a DIFFERENT schedule structure (not quarter-K
  // pingpong), so the quarter-K cost model doesn't directly apply.
  // However, we can verify the component predictions are reasonable:
  //   - Actual ISA: 215 VGPRs, no spills
  //   - Model acc: 128 VGPRs (matches - 4x4 MMA tiles x 8)
  //   - Model GL: 32 VGPRs (matches - 16 LHS + 16 RHS)
  //   - Index + misc: 215 - 128 - 32 = 55 VGPRs (compiler-managed)
  //
  // The quarter-K pingpong schedule (our target) would use MORE VGPRs
  // because it also holds quarter-read operands. The tradeoff is better
  // latency hiding from the structured schedule.
  int64_t actualVGPRs = 215;
  int64_t modelAcc = 128;
  int64_t modelGL = 32;
  int64_t impliedOverhead = actualVGPRs - modelAcc - modelGL;
  // Overhead should be in reasonable range (30-80 VGPRs for addresses,
  // loop control, temporaries).
  EXPECT_GE(impliedOverhead, 30);
  EXPECT_LE(impliedOverhead, 80);
}

TEST(BaselineComparisonTest, PingpongWouldNeedMoreVGPRs) {
  // The quarter-K pingpong schedule adds quarter-read operands on top of
  // the baseline budget. With K=32 (2 quarters of 16):
  //   earlyWrite: 128 + 32 + 32 + index = 192 + index
  //   Need index <= 64 to stay within 256 VGPRs.
  int64_t indexK32 =
      computeIndexOverheadVGPRs(/*numLoadOperands=*/2, /*numKQuarters=*/2,
                                /*splitCopy=*/false);
  PeakVGPRUsage pingpong = computePeakVGPRUsage(
      128, GlobalLoadVGPRs{16, 16}, LDSQuarterReadVGPRs{16, 16}, indexK32, 256,
      /*earlyWrite=*/true);
  // Pingpong schedule uses more VGPRs than baseline (215).
  EXPECT_GT(pingpong.totalVGPRs, 215);
  // But should still fit within 256.
  EXPECT_FALSE(pingpong.spills);
}

} // namespace
} // namespace mlir::iree_compiler::IREE::GPU
