// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/CostModel.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler::IREE::GPU {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture with MLIR context for constructing GPU attributes.
//===----------------------------------------------------------------------===//

class CostModelTest : public ::testing::Test {
protected:
  CostModelTest() {
    DialectRegistry registry;
    registry.insert<IREEGPUDialect>();
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }

  /// Construct a minimal TargetAttr with the given vgpr_space_bits.
  /// This avoids depending on KnownTargets which has heavy link deps.
  TargetAttr makeTarget(int32_t vgprSpaceBits) {
    Builder b(&ctx);
    ComputeBitwidthsAttr compute =
        ComputeBitwidthsAttr::get(&ctx, ComputeBitwidths::FP16);
    StorageBitwidthsAttr storage =
        StorageBitwidthsAttr::get(&ctx, StorageBitwidths::B32);
    SubgroupOpsAttr subgroup =
        SubgroupOpsAttr::get(&ctx, SubgroupOps::Shuffle);
    DotProductOpsAttr dot =
        DotProductOpsAttr::get(&ctx, DotProductOps::None);
    DenseI32ArrayAttr subgroupSizes = b.getDenseI32ArrayAttr({32});
    DenseI32ArrayAttr maxWgSizes = b.getDenseI32ArrayAttr({1024, 1024, 1024});
    DenseI32ArrayAttr maxWgCounts =
        b.getDenseI32ArrayAttr({0x7fffffff, 0x7fffffff, 0x7fffffff});

    TargetWgpAttr wgp = TargetWgpAttr::get(
        &ctx, compute, storage, subgroup, dot,
        MMAOpsArrayAttr::get(&ctx, {}),
        ScaledMMAOpsArrayAttr::get(&ctx, {}),
        subgroupSizes, maxWgSizes,
        /*maxThreadCountPerWorkgroup=*/1024,
        /*maxWorkgroupMemoryBytes=*/65536,
        maxWgCounts,
        /*maxLoadInstructionBits=*/128,
        /*simdsPerWgp=*/4,
        /*vgprSpaceBits=*/vgprSpaceBits,
        /*dmaSizes=*/DenseI64ArrayAttr{},
        /*extra=*/DictionaryAttr{});

    return TargetAttr::get(&ctx, "gfx1201", "", wgp, TargetChipAttr{});
  }

  MLIRContext ctx;
};

//===----------------------------------------------------------------------===//
// MMA Register Cost
//===----------------------------------------------------------------------===//

// WMMAR4_F32_16x16x16_F16 on RDNA4 (wave32).
TEST_F(CostModelTest, MMARegisterCostRDNA4_F16) {
  MMARegisterCost cost = computeMMARegisterCost(
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      /*lhsBits=*/16, /*rhsBits=*/16, /*accBits=*/32);

  // Accumulator is f32, so each element is one VGPR.
  // LHS/RHS are f16, so two elements fit per VGPR.
  EXPECT_GT(cost.accVGPRs, 0);
  EXPECT_GT(cost.lhsVGPRs, 0);
  EXPECT_GT(cost.rhsVGPRs, 0);

  // Accumulator should use more registers than input operands for a matmul
  // intrinsic since acc is f32 and inputs are f16.
  EXPECT_GE(cost.accVGPRs, cost.lhsVGPRs);
  EXPECT_GE(cost.accVGPRs, cost.rhsVGPRs);
}

// MFMA_F32_16x16x16_F16 on CDNA (wave64).
TEST_F(CostModelTest, MMARegisterCostCDNA_F16) {
  MMARegisterCost cost = computeMMARegisterCost(
      MMAIntrinsic::MFMA_F32_16x16x16_F16,
      /*lhsBits=*/16, /*rhsBits=*/16, /*accBits=*/32);

  EXPECT_GT(cost.accVGPRs, 0);
  EXPECT_GT(cost.lhsVGPRs, 0);
  EXPECT_GT(cost.rhsVGPRs, 0);
}

// Direct operand VGPR computation with known layout.
TEST_F(CostModelTest, MMAOperandVGPRsBasic) {
  // Construct a simple layout: outer={2,1}, element={1,4}.
  // elements_per_thread = 2*1*1*4 = 8. For 16-bit elements: 8*16/32 = 4 VGPRs.
  MMASingleSubgroupLayout layout;
  layout.outer = {2, 1};
  layout.thread = {8, 4};
  layout.tstrides = {4, 1};
  layout.element = {1, 4};

  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/16), 4);
  // For 32-bit elements: 8*32/32 = 8 VGPRs.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/32), 8);
}

// Test with trivial layout (all ones).
TEST_F(CostModelTest, MMAOperandVGPRsTrivial) {
  MMASingleSubgroupLayout layout;
  layout.outer = {1, 1};
  layout.thread = {16, 2};
  layout.tstrides = {2, 1};
  layout.element = {1, 1};

  // 1 element per thread, 32-bit: 1 VGPR.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/32), 1);
  // 1 element per thread, 16-bit: ceil(16/32) = 1 VGPR.
  EXPECT_EQ(computeMMAOperandVGPRs(layout, /*elementBits=*/16), 1);
}

//===----------------------------------------------------------------------===//
// VGPR Budget
//===----------------------------------------------------------------------===//

TEST_F(CostModelTest, VGPRBudgetRDNA4) {
  // RDNA4: vgpr_space_bits = 256*32 = 8192 (per-wave at occupancy 1).
  TargetAttr target = makeTarget(/*vgprSpaceBits=*/256 * 32);

  // At occupancy 1: totalVGPRs = 8192/32 = 256.
  std::optional<VGPRBudget> budget = computeVGPRBudget(target, /*occupancy=*/1);
  ASSERT_TRUE(budget.has_value());
  EXPECT_EQ(budget->totalVGPRs, 256);
  EXPECT_EQ(budget->occupancy, 1);

  // At occupancy 2: totalVGPRs = 8192/(2*32) = 128.
  budget = computeVGPRBudget(target, /*occupancy=*/2);
  ASSERT_TRUE(budget.has_value());
  EXPECT_EQ(budget->totalVGPRs, 128);
  EXPECT_EQ(budget->occupancy, 2);

  // At occupancy 4: totalVGPRs = 8192/(4*32) = 64.
  budget = computeVGPRBudget(target, /*occupancy=*/4);
  ASSERT_TRUE(budget.has_value());
  EXPECT_EQ(budget->totalVGPRs, 64);
  EXPECT_EQ(budget->occupancy, 4);
}

TEST_F(CostModelTest, VGPRBudgetCDNA3) {
  // CDNA3 (gfx942): vgpr_space_bits = 512*32 = 16384.
  TargetAttr target = makeTarget(/*vgprSpaceBits=*/512 * 32);

  std::optional<VGPRBudget> budget = computeVGPRBudget(target, /*occupancy=*/1);
  ASSERT_TRUE(budget.has_value());
  EXPECT_EQ(budget->totalVGPRs, 512);

  budget = computeVGPRBudget(target, /*occupancy=*/2);
  ASSERT_TRUE(budget.has_value());
  EXPECT_EQ(budget->totalVGPRs, 256);
}

TEST_F(CostModelTest, VGPRBudgetInvalidOccupancy) {
  TargetAttr target = makeTarget(/*vgprSpaceBits=*/256 * 32);

  // Occupancy 0: nullopt.
  std::optional<VGPRBudget> budget = computeVGPRBudget(target, /*occupancy=*/0);
  EXPECT_FALSE(budget.has_value());

  // Negative occupancy: nullopt.
  budget = computeVGPRBudget(target, /*occupancy=*/-1);
  EXPECT_FALSE(budget.has_value());
}

//===----------------------------------------------------------------------===//
// Max Occupancy from VGPRs
//===----------------------------------------------------------------------===//

TEST_F(CostModelTest, MaxOccupancyFromVGPRsRDNA4) {
  TargetAttr target = makeTarget(/*vgprSpaceBits=*/256 * 32);

  // 256 VGPRs used: max occupancy 1 (256/256 = 1).
  EXPECT_EQ(computeMaxOccupancyFromVGPRs(target, /*vgprsUsed=*/256), 1);

  // 128 VGPRs used: max occupancy 2 (256/128 = 2).
  EXPECT_EQ(computeMaxOccupancyFromVGPRs(target, /*vgprsUsed=*/128), 2);

  // 64 VGPRs: max occupancy 4.
  EXPECT_EQ(computeMaxOccupancyFromVGPRs(target, /*vgprsUsed=*/64), 4);

  // 200 VGPRs: max occupancy 1 (256/200 = 1 truncated).
  EXPECT_EQ(computeMaxOccupancyFromVGPRs(target, /*vgprsUsed=*/200), 1);
}

TEST_F(CostModelTest, MaxOccupancyFromVGPRsZero) {
  TargetAttr target = makeTarget(/*vgprSpaceBits=*/256 * 32);

  // Zero VGPRs used: returns 0 (invalid input).
  EXPECT_EQ(computeMaxOccupancyFromVGPRs(target, /*vgprsUsed=*/0), 0);
}

//===----------------------------------------------------------------------===//
// Accumulator VGPRs
//===----------------------------------------------------------------------===//

// RDNA4 WMMA: 64x64 subgroup tile with 16x16x16 MMA = 4*4=16 tiles.
TEST_F(CostModelTest, AccumulatorVGPRsRDNA4_64x64) {
  std::optional<int64_t> accVGPRs = computeAccumulatorVGPRs(
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      /*subgroupM=*/64, /*subgroupN=*/64, /*accBits=*/32);
  ASSERT_TRUE(accVGPRs.has_value());
  EXPECT_GT(*accVGPRs, 0);
  // The hub issue says 128 VGPRs for 64x64 subgroup with WMMA f16->f32.
  EXPECT_EQ(*accVGPRs, 128);
}

// 32x32 subgroup tile = 2*2=4 tiles. Should be 1/4 of 64x64.
TEST_F(CostModelTest, AccumulatorVGPRsRDNA4_32x32) {
  std::optional<int64_t> vgprs64 = computeAccumulatorVGPRs(
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      /*subgroupM=*/64, /*subgroupN=*/64, /*accBits=*/32);
  std::optional<int64_t> vgprs32 = computeAccumulatorVGPRs(
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      /*subgroupM=*/32, /*subgroupN=*/32, /*accBits=*/32);
  ASSERT_TRUE(vgprs64.has_value());
  ASSERT_TRUE(vgprs32.has_value());
  // 4 tiles vs 16 tiles: exactly 1/4.
  EXPECT_EQ(*vgprs32 * 4, *vgprs64);
}

// Non-divisible subgroup tile: nullopt.
TEST_F(CostModelTest, AccumulatorVGPRsNonDivisible) {
  // 48 % 16 == 0, so this should work (3*4=12 tiles).
  std::optional<int64_t> accVGPRs = computeAccumulatorVGPRs(
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      /*subgroupM=*/48, /*subgroupN=*/64, /*accBits=*/32);
  ASSERT_TRUE(accVGPRs.has_value());

  // But 33 % 16 != 0: nullopt.
  accVGPRs = computeAccumulatorVGPRs(
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      /*subgroupM=*/33, /*subgroupN=*/64, /*accBits=*/32);
  EXPECT_FALSE(accVGPRs.has_value());
}

//===----------------------------------------------------------------------===//
// Global Load Staging VGPRs
//===----------------------------------------------------------------------===//

// RDNA4: 256x256 tile, K=64, 512 threads, f16 (16-bit).
// LHS tile: 256*64 = 16384 elements, per thread = ceil(16384/512) = 32.
// 32 f16 elements = 32*16/32 = 16 VGPRs.
TEST_F(CostModelTest, GlobalLoadVGPRsRDNA4) {
  int64_t lhsTileElements = 256 * 64;
  int64_t rhsTileElements = 64 * 256;
  int64_t numThreads = 512;
  int64_t elementBits = 16;

  GlobalLoadVGPRs vgprs =
      computeGlobalLoadVGPRs(lhsTileElements, rhsTileElements,
                             numThreads, elementBits);

  EXPECT_EQ(vgprs.lhsVGPRs, 16);
  EXPECT_EQ(vgprs.rhsVGPRs, 16);
}

// Asymmetric tile sizes.
TEST_F(CostModelTest, GlobalLoadVGPRsAsymmetric) {
  int64_t lhsTileElements = 128 * 64;
  int64_t rhsTileElements = 64 * 256;
  int64_t numThreads = 256;
  int64_t elementBits = 16;

  GlobalLoadVGPRs vgprs =
      computeGlobalLoadVGPRs(lhsTileElements, rhsTileElements,
                             numThreads, elementBits);

  // LHS: ceil(8192/256) = 32 elements/thread, 32*16/32 = 16 VGPRs.
  EXPECT_EQ(vgprs.lhsVGPRs, 16);
  // RHS: ceil(16384/256) = 64 elements/thread, 64*16/32 = 32 VGPRs.
  EXPECT_EQ(vgprs.rhsVGPRs, 32);
}

//===----------------------------------------------------------------------===//
// LDS Quarter Read VGPRs
//===----------------------------------------------------------------------===//

// RDNA4 WMMA: 64x64 subgroup, quarterK=16 with 16x16x16 MMA.
// LHS: (64/16) * (16/16) = 4 tiles, each 4 VGPRs (8 f16 elts/thread) = 16.
// RHS: (16/16) * (64/16) = 4 tiles, each 4 VGPRs = 16.
TEST_F(CostModelTest, LDSQuarterReadVGPRsRDNA4_64x64_Q16) {
  std::optional<LDSQuarterReadVGPRs> vgprs = computeLDSQuarterReadVGPRs(
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      /*subgroupM=*/64, /*subgroupN=*/64, /*quarterK=*/16,
      /*lhsBits=*/16, /*rhsBits=*/16);
  ASSERT_TRUE(vgprs.has_value());
  EXPECT_EQ(vgprs->lhsVGPRs, 16);
  EXPECT_EQ(vgprs->rhsVGPRs, 16);
}

// CDNA3 MFMA: 64x64 subgroup, quarterK=16 with 16x16x16 MMA.
// LHS: (64/16) * (16/16) = 4 tiles, each 2 VGPRs (4 f16 elts/thread) = 8.
// RHS: (16/16) * (64/16) = 4 tiles, each 2 VGPRs = 8.
TEST_F(CostModelTest, LDSQuarterReadVGPRsCDNA3_64x64_Q16) {
  std::optional<LDSQuarterReadVGPRs> vgprs = computeLDSQuarterReadVGPRs(
      MMAIntrinsic::MFMA_F32_16x16x16_F16,
      /*subgroupM=*/64, /*subgroupN=*/64, /*quarterK=*/16,
      /*lhsBits=*/16, /*rhsBits=*/16);
  ASSERT_TRUE(vgprs.has_value());
  EXPECT_EQ(vgprs->lhsVGPRs, 8);
  EXPECT_EQ(vgprs->rhsVGPRs, 8);
}

// Asymmetric 32x64 subgroup tile.
// LHS: (32/16) * (16/16) = 2 tiles = 8 VGPRs.
// RHS: (16/16) * (64/16) = 4 tiles = 16 VGPRs.
TEST_F(CostModelTest, LDSQuarterReadVGPRsAsymmetric) {
  std::optional<LDSQuarterReadVGPRs> vgprs = computeLDSQuarterReadVGPRs(
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      /*subgroupM=*/32, /*subgroupN=*/64, /*quarterK=*/16,
      /*lhsBits=*/16, /*rhsBits=*/16);
  ASSERT_TRUE(vgprs.has_value());
  EXPECT_EQ(vgprs->lhsVGPRs, 8);
  EXPECT_EQ(vgprs->rhsVGPRs, 16);
}

// Larger quarterK (32) doubles K tiles.
// LHS: (64/16) * (32/16) = 8 tiles = 32 VGPRs.
// RHS: (32/16) * (64/16) = 8 tiles = 32 VGPRs.
TEST_F(CostModelTest, LDSQuarterReadVGPRsLargerQuarterK) {
  std::optional<LDSQuarterReadVGPRs> vgprs = computeLDSQuarterReadVGPRs(
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      /*subgroupM=*/64, /*subgroupN=*/64, /*quarterK=*/32,
      /*lhsBits=*/16, /*rhsBits=*/16);
  ASSERT_TRUE(vgprs.has_value());
  EXPECT_EQ(vgprs->lhsVGPRs, 32);
  EXPECT_EQ(vgprs->rhsVGPRs, 32);
}

// Non-divisible quarterK: nullopt.
TEST_F(CostModelTest, LDSQuarterReadVGPRsNonDivisible) {
  // quarterK=12 not divisible by mmaK=16.
  std::optional<LDSQuarterReadVGPRs> vgprs = computeLDSQuarterReadVGPRs(
      MMAIntrinsic::WMMAR4_F32_16x16x16_F16,
      /*subgroupM=*/64, /*subgroupN=*/64, /*quarterK=*/12,
      /*lhsBits=*/16, /*rhsBits=*/16);
  EXPECT_FALSE(vgprs.has_value());
}

//===----------------------------------------------------------------------===//
// LDS Allocation
//===----------------------------------------------------------------------===//

// Single-buffered: 256*64 LHS + 64*256 RHS, 2 bytes per element.
TEST_F(CostModelTest, LDSAllocationSingleBuffer) {
  int64_t lhsElements = 256 * 64;
  int64_t rhsElements = 64 * 256;
  int64_t elementBytes = 2;
  int64_t bufferDepth = 1;
  int64_t maxLDS = 64 * 1024;

  std::optional<LDSAllocation> alloc =
      computeLDSAllocation(lhsElements, rhsElements, elementBytes,
                           bufferDepth, maxLDS);
  ASSERT_TRUE(alloc.has_value());
  EXPECT_EQ(alloc->lhsBytes, 256 * 64 * 2);
  EXPECT_EQ(alloc->rhsBytes, 64 * 256 * 2);
  EXPECT_EQ(alloc->totalBytes, 65536);
  EXPECT_EQ(alloc->bufferDepth, 1);
}

// Double-buffered exceeds LDS: nullopt.
TEST_F(CostModelTest, LDSAllocationDoubleBufferOverflow) {
  int64_t lhsElements = 256 * 64;
  int64_t rhsElements = 64 * 256;
  int64_t elementBytes = 2;
  int64_t maxLDS = 64 * 1024;

  std::optional<LDSAllocation> alloc =
      computeLDSAllocation(lhsElements, rhsElements, elementBytes,
                           /*bufferDepth=*/2, maxLDS);
  EXPECT_FALSE(alloc.has_value());
}

// Smaller tile that allows double-buffering.
TEST_F(CostModelTest, LDSAllocationDoubleBufferFits) {
  int64_t lhsElements = 128 * 32;
  int64_t rhsElements = 32 * 128;
  int64_t elementBytes = 2;
  int64_t maxLDS = 64 * 1024;

  std::optional<LDSAllocation> alloc =
      computeLDSAllocation(lhsElements, rhsElements, elementBytes,
                           /*bufferDepth=*/2, maxLDS);
  ASSERT_TRUE(alloc.has_value());
  EXPECT_EQ(alloc->totalBytes, (4096 + 4096) * 2 * 2);
  EXPECT_EQ(alloc->bufferDepth, 2);
}

//===----------------------------------------------------------------------===//
// Max Buffer Depth
//===----------------------------------------------------------------------===//

TEST_F(CostModelTest, MaxBufferDepth) {
  int64_t lhsElements = 256 * 64;
  int64_t rhsElements = 64 * 256;
  int64_t elementBytes = 2;
  int64_t maxLDS = 64 * 1024;

  // Single buffer = 64KB, max LDS = 64KB: depth 1.
  EXPECT_EQ(computeMaxBufferDepth(lhsElements, rhsElements,
                                  elementBytes, maxLDS), 1);

  // With 128KB LDS budget: depth 2.
  EXPECT_EQ(computeMaxBufferDepth(lhsElements, rhsElements,
                                  elementBytes, 128 * 1024), 2);
}

TEST_F(CostModelTest, MaxBufferDepthSmallTile) {
  int64_t lhsElements = 64 * 16;
  int64_t rhsElements = 16 * 64;
  int64_t elementBytes = 2;
  int64_t maxLDS = 64 * 1024;

  // Single buffer = 4096 bytes, 64KB/4096 = 16.
  EXPECT_EQ(computeMaxBufferDepth(lhsElements, rhsElements,
                                  elementBytes, maxLDS), 16);
}

TEST_F(CostModelTest, MaxBufferDepthZeroElements) {
  // Edge: zero tile elements returns 0.
  EXPECT_EQ(computeMaxBufferDepth(0, 0, 2, 64 * 1024), 0);
}

} // namespace
} // namespace mlir::iree_compiler::IREE::GPU
