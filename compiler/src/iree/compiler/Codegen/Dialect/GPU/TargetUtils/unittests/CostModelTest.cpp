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

} // namespace
} // namespace mlir::iree_compiler::IREE::GPU
