// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TensorDynamicDimAnalysis.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/Template/IR/Template.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "iree-gpu-pingpong-config"

namespace mlir::iree_compiler::IREE::GPU {

namespace {

//===----------------------------------------------------------------------===//
// Pingpong Template Function Builder
//===----------------------------------------------------------------------===//
//
// Creates a template.func with the pingpong double-buffer loop structure.
// Even subgroups load while odd subgroups compute, then swap roles.
//
// Template types (filled in by process_inner_tile):
//   type<0>: Per-subgroup distributed accumulator.
//   type<1>: Per-subgroup distributed LHS input (from LDS).
//   type<2>: Per-subgroup distributed RHS input (from LDS).
//
// Implementation blocks (7 total):
//   Block 0: Init accumulators (sg_id, lane_id, dest) -> type<0>
//   Block 1: Copy LHS to shared (buf_idx, k_idx, sg_id, lane_id, lhs, alloc)
//   Block 2: Copy RHS to shared (buf_idx, k_idx, sg_id, lane_id, rhs, alloc)
//   Block 3: Read LHS from LDS (buf_idx, sg_id, lane_id, alloc) -> type<1>
//   Block 4: Read RHS from LDS (buf_idx, sg_id, lane_id, alloc) -> type<2>
//   Block 5: Compute MMA (acc, lhs_local, rhs_local) -> type<0>
//   Block 6: Write results (sg_id, lane_id, result, dest)

/// Build the main region of the pingpong template function. This creates the
/// double-buffered loop structure with barriers and template.branch calls.
static IREE::Template::FuncOp
createPingpongTemplateFunc(OpBuilder &builder, Location loc, StringRef name,
                           Type outputTensorType, Type lhsTensorType,
                           Type rhsTensorType, ArrayRef<int64_t> lhsTileShape,
                           ArrayRef<int64_t> rhsTileShape, Type elemType,
                           int64_t lhsKDimIdx) {
  MLIRContext *context = builder.getContext();

  // Template types.
  IREE::Template::TypeType type0 =
      IREE::Template::TypeType::get(context, 0); // acc
  IREE::Template::TypeType type1 =
      IREE::Template::TypeType::get(context, 1); // lhs local
  IREE::Template::TypeType type2 =
      IREE::Template::TypeType::get(context, 2); // rhs local
  Type indexType = builder.getIndexType();

  // Shared memory allocation types: sref<2 x tileM x tileK x elemType>.
  PCF::ScopeAttrInterface subgroupScope =
      cast<PCF::ScopeAttrInterface>(
          IREE::GPU::SubgroupScopeAttr::get(context));
  SmallVector<int64_t> lhsAllocShape = {2};
  lhsAllocShape.append(lhsTileShape.begin(), lhsTileShape.end());
  PCF::ShapedRefType lhsAllocType =
      PCF::ShapedRefType::get(context, lhsAllocShape, elemType, subgroupScope);
  SmallVector<int64_t> rhsAllocShape = {2};
  rhsAllocShape.append(rhsTileShape.begin(), rhsTileShape.end());
  PCF::ShapedRefType rhsAllocType =
      PCF::ShapedRefType::get(context, rhsAllocShape, elemType, subgroupScope);

  // Concrete sref type for outputs (blocks 0 and 6 use this instead of a
  // template type).
  auto outputRankedType = cast<RankedTensorType>(outputTensorType);
  PCF::ShapedRefType outputSrefType = PCF::ShapedRefType::get(
      context, outputRankedType.getShape(), outputRankedType.getElementType(),
      subgroupScope);

  // Function signature:
  // (%inits: tensor<MxNxT>, %k: index, %lhs: tensor<?x?xT>, %rhs: tensor<?x?xT>)
  //   -> tensor<MxNxT>
  FunctionType funcType = FunctionType::get(
      context, {outputTensorType, indexType, lhsTensorType, rhsTensorType},
      {outputTensorType});
  auto funcOp = IREE::Template::FuncOp::create(builder, loc, name, funcType);

  // Create entry block in main region.
  Region &mainRegion = funcOp.getMain();
  Block *mainBlock = builder.createBlock(&mainRegion);
  mainBlock->addArguments(
      {outputTensorType, indexType, lhsTensorType, rhsTensorType},
      {loc, loc, loc, loc});
  OpBuilder::InsertionGuard topGuard(builder);
  builder.setInsertionPointToStart(mainBlock);

  Value initsArg = mainBlock->getArgument(0);
  Value kTileSizeArg = mainBlock->getArgument(1);
  Value lhsArg = mainBlock->getArgument(2);
  Value rhsArg = mainBlock->getArgument(3);

  // Compute the number of K tiles from the LHS tensor's K dimension.
  // kTileSizeArg is the K tile size (e.g., 64), not the number of tiles.
  // numKTiles = ceildiv(totalK, kTileSize).
  Value kDimIdxConst =
      arith::ConstantIndexOp::create(builder, loc, lhsKDimIdx);
  Value totalK =
      tensor::DimOp::create(builder, loc, lhsArg, kDimIdxConst);
  Value numKTiles =
      arith::CeilDivUIOp::create(builder, loc, totalK, kTileSizeArg);

  // Constants.
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(builder, loc, 2);

  // ===== Outer pcf.generic: subgroup scope =====
  auto outerGeneric = IREE::PCF::GenericOp::create(
      builder, loc,
      /*resultTypes=*/TypeRange{outputTensorType},
      /*scope=*/subgroupScope,
      /*inits=*/ValueRange{initsArg},
      /*dynamicSizes=*/ValueRange{},
      /*isTied=*/ArrayRef<bool>{true},
      /*numIterators=*/1,
      /*syncOnReturn=*/false);

  // The builder creates execute region with args: [dest(type<3>), sg_id, sg_n].
  // We need leading args for allocs from initializer.
  Block &outerExecBlock = outerGeneric.getRegion().front();
  // Insert leading args for allocs (lhs_alloc, rhs_alloc).
  outerExecBlock.insertArgument(/*index=*/0u, lhsAllocType, loc);
  outerExecBlock.insertArgument(/*index=*/1u, rhsAllocType, loc);
  outerGeneric.setNumLeadingArgs(2);

  // Populate initializer region.
  {
    Region &initRegion = outerGeneric.getInitializer();
    Block *initBlock = builder.createBlock(&initRegion);
    builder.setInsertionPointToStart(initBlock);

    Value lhsAlloc =
        IREE::PCF::AllocOp::create(builder, loc, lhsAllocType).getResult();
    Value rhsAlloc =
        IREE::PCF::AllocOp::create(builder, loc, rhsAllocType).getResult();
    IREE::PCF::YieldOp::create(builder, loc, ValueRange{lhsAlloc, rhsAlloc});
  }

  // Populate outer execute block.
  {
    builder.setInsertionPointToStart(&outerExecBlock);

    Value lhsAllocArg = outerExecBlock.getArgument(0);
    Value rhsAllocArg = outerExecBlock.getArgument(1);
    Value dest = outerExecBlock.getArgument(2);       // type<3>
    Value subgroupId = outerExecBlock.getArgument(3);  // index

    // Helper to emit pcf.fence on LHS and RHS shared memory allocations.
    auto emitFence = [&](bool isRelease) {
      IREE::PCF::FenceOp::create(builder, loc, isRelease,
                                  ValueRange{lhsAllocArg, rhsAllocArg});
    };

    // scf.if (sg_id % 2 == 0) -- even vs odd waves.
    Value rem = arith::RemUIOp::create(builder, loc, subgroupId, c2);
    Value isEven = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, rem, c0);

    auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, isEven,
                                   /*withElseRegion=*/true);

    // Lambda to build the lane-scope generic for one wave group.
    PCF::ScopeAttrInterface laneScope =
        cast<PCF::ScopeAttrInterface>(
            IREE::GPU::LaneScopeAttr::get(context));

    // Skewed double-buffer schedule. Even subgroups copy first while odd
    // subgroups compute, then swap roles. This enables overlapping memory
    // loads with MMA compute across the two wave groups.
    //
    // EVEN: copy(buf0,k=0), A, B, loop(copy(buf0,k+1), read(buf1), C,
    //       compute, D), write
    // ODD:  copy(buf1,k=0), A, read(buf0), B, loop(compute, C,
    //       copy(buf1,k), read(buf0), D), epilogue_compute, E, F, write
    //
    // Barrier counts: 2 + 2*K per path (even), 2 + 2*(K-1) + 2 per path
    //   (odd), both equal 2*K + 2 total.
    //
    // Consecutive barriers A+B (even) and E+F (odd) are preserved because
    // pcf.barrier lowers to iree_gpu.global_subgroup_barrier (no canonicalizer).
    // Memory fencing is handled explicitly by pcf.fence ops.

    // ===== EVEN waves: copy-first schedule =====
    //   copy(buf0, k=0) → A → B → loop(copy, read, C, compute, D) → write
    {
      Block &thenBlock = ifOp.getThenRegion().front();
      builder.setInsertionPoint(thenBlock.getTerminator());

      auto innerGeneric = IREE::PCF::GenericOp::create(
          builder, loc, /*scope=*/laneScope, /*numIterators=*/1,
          /*syncOnReturn=*/false);

      Block &innerExec = innerGeneric.getRegion().front();
      builder.setInsertionPointToStart(&innerExec);
      Value laneId = innerExec.getArgument(0);

      // Init accumulators.
      auto initBranch = IREE::Template::BranchOp::create(
          builder, loc, TypeRange{type0}, builder.getI64IntegerAttr(0),
          ValueRange{subgroupId, laneId, dest});
      Value acc = initBranch.getResults()[0];

      // Copy LHS and RHS prologue (k=0) into buf0.
      IREE::Template::BranchOp::create(
          builder, loc, TypeRange{}, builder.getI64IntegerAttr(1),
          ValueRange{c0, c0, subgroupId, laneId, lhsArg, lhsAllocArg});
      IREE::Template::BranchOp::create(
          builder, loc, TypeRange{}, builder.getI64IntegerAttr(2),
          ValueRange{c0, c0, subgroupId, laneId, rhsArg, rhsAllocArg});

      // Fence release: prologue wrote to LDS.
      emitFence(/*isRelease=*/true);

      // Barrier A: prologue copy done (pairs with odd's A).
      IREE::PCF::BarrierOp::create(builder, loc, subgroupScope);
      // Barrier B: vacuous for even (pairs with odd's "done reading buf0").
      IREE::PCF::BarrierOp::create(builder, loc, subgroupScope);

      // Fence acquire: prepare for loop reads from LDS.
      emitFence(/*isRelease=*/false);

      // Loop k=0..K-1: copy(buf0,k+1), read(buf1), C, compute, D.
      auto forOp =
          scf::ForOp::create(builder, loc, c0, numKTiles, c1, ValueRange{acc});
      {
        Block *forBody = forOp.getBody();
        if (!forBody->empty() &&
            forBody->back().hasTrait<OpTrait::IsTerminator>())
          forBody->back().erase();
        builder.setInsertionPointToEnd(forBody);

        Value iv = forOp.getInductionVar();
        Value loopAcc = forOp.getRegionIterArg(0);

        // Copy next tile (k+1) into buf0 while odd computes.
        Value kNext = arith::AddIOp::create(builder, loc, iv, c1);
        IREE::Template::BranchOp::create(
            builder, loc, TypeRange{}, builder.getI64IntegerAttr(1),
            ValueRange{c0, kNext, subgroupId, laneId, lhsArg, lhsAllocArg});
        IREE::Template::BranchOp::create(
            builder, loc, TypeRange{}, builder.getI64IntegerAttr(2),
            ValueRange{c0, kNext, subgroupId, laneId, rhsArg, rhsAllocArg});

        // Read from buf1 (odd's buffer, current k-tile data).
        auto readLhs = IREE::Template::BranchOp::create(
            builder, loc, TypeRange{type1}, builder.getI64IntegerAttr(3),
            ValueRange{c1, subgroupId, laneId, lhsAllocArg});
        auto readRhs = IREE::Template::BranchOp::create(
            builder, loc, TypeRange{type2}, builder.getI64IntegerAttr(4),
            ValueRange{c1, subgroupId, laneId, rhsAllocArg});

        // Fence release: copied to LDS + read from LDS.
        emitFence(/*isRelease=*/true);

        // Barrier C: done copying buf0 + done reading buf1.
        IREE::PCF::BarrierOp::create(builder, loc, subgroupScope);

        // Compute while odd copies+reads.
        auto computeBranch = IREE::Template::BranchOp::create(
            builder, loc, TypeRange{type0}, builder.getI64IntegerAttr(5),
            ValueRange{loopAcc, readLhs.getResults()[0],
                       readRhs.getResults()[0]});

        // Barrier D: done computing.
        IREE::PCF::BarrierOp::create(builder, loc, subgroupScope);

        // Fence acquire: see odd's writes for next iteration.
        emitFence(/*isRelease=*/false);

        scf::YieldOp::create(builder, loc,
                             ValueRange{computeBranch.getResults()[0]});
      }

      // Write results to output.
      builder.setInsertionPointAfter(forOp);
      IREE::Template::BranchOp::create(
          builder, loc, TypeRange{}, builder.getI64IntegerAttr(6),
          ValueRange{subgroupId, laneId, forOp.getResult(0), dest});

      IREE::PCF::ReturnOp::create(builder, loc);
    }

    // ===== ODD waves: compute-first schedule =====
    //   copy(buf1, k=0) → A → read(buf0) → B →
    //     loop(compute, C, copy, read, D) → epilogue → E → F → write
    {
      Block &elseBlock = ifOp.getElseRegion().front();
      builder.setInsertionPoint(elseBlock.getTerminator());

      auto innerGeneric = IREE::PCF::GenericOp::create(
          builder, loc, /*scope=*/laneScope, /*numIterators=*/1,
          /*syncOnReturn=*/false);

      Block &innerExec = innerGeneric.getRegion().front();
      builder.setInsertionPointToStart(&innerExec);
      Value laneId = innerExec.getArgument(0);

      // Init accumulators.
      auto initBranch = IREE::Template::BranchOp::create(
          builder, loc, TypeRange{type0}, builder.getI64IntegerAttr(0),
          ValueRange{subgroupId, laneId, dest});
      Value acc = initBranch.getResults()[0];

      // Copy LHS and RHS prologue (k=0) into buf1.
      IREE::Template::BranchOp::create(
          builder, loc, TypeRange{}, builder.getI64IntegerAttr(1),
          ValueRange{c1, c0, subgroupId, laneId, lhsArg, lhsAllocArg});
      IREE::Template::BranchOp::create(
          builder, loc, TypeRange{}, builder.getI64IntegerAttr(2),
          ValueRange{c1, c0, subgroupId, laneId, rhsArg, rhsAllocArg});

      // Fence release: prologue wrote to LDS.
      emitFence(/*isRelease=*/true);

      // Barrier A: prologue copy done (pairs with even's A).
      IREE::PCF::BarrierOp::create(builder, loc, subgroupScope);

      // Fence acquire: prepare to read buf0.
      emitFence(/*isRelease=*/false);

      // Read initial data from buf0 (even's buffer, k=0).
      auto firstLhs = IREE::Template::BranchOp::create(
          builder, loc, TypeRange{type1}, builder.getI64IntegerAttr(3),
          ValueRange{c0, subgroupId, laneId, lhsAllocArg});
      auto firstRhs = IREE::Template::BranchOp::create(
          builder, loc, TypeRange{type2}, builder.getI64IntegerAttr(4),
          ValueRange{c0, subgroupId, laneId, rhsAllocArg});

      // Barrier B: done reading buf0 (pairs with even's B).
      IREE::PCF::BarrierOp::create(builder, loc, subgroupScope);

      // Loop k=1..K-1: compute, C, copy(buf1,k), read(buf0), D.
      auto forOp = scf::ForOp::create(
          builder, loc, c1, numKTiles, c1,
          ValueRange{acc, firstLhs.getResults()[0],
                     firstRhs.getResults()[0]});
      {
        Block *forBody = forOp.getBody();
        if (!forBody->empty() &&
            forBody->back().hasTrait<OpTrait::IsTerminator>())
          forBody->back().erase();
        builder.setInsertionPointToEnd(forBody);

        Value iv = forOp.getInductionVar();
        Value loopAcc = forOp.getRegionIterArg(0);
        Value loopLhs = forOp.getRegionIterArg(1);
        Value loopRhs = forOp.getRegionIterArg(2);

        // Compute with previous iteration's read data while even copies.
        auto computeBranch = IREE::Template::BranchOp::create(
            builder, loc, TypeRange{type0}, builder.getI64IntegerAttr(5),
            ValueRange{loopAcc, loopLhs, loopRhs});

        // Barrier C: done computing (safe for even to overwrite buf1).
        IREE::PCF::BarrierOp::create(builder, loc, subgroupScope);

        // Fence acquire: see even's writes.
        emitFence(/*isRelease=*/false);

        // Copy current tile (k) into buf1 while even computes.
        IREE::Template::BranchOp::create(
            builder, loc, TypeRange{}, builder.getI64IntegerAttr(1),
            ValueRange{c1, iv, subgroupId, laneId, lhsArg, lhsAllocArg});
        IREE::Template::BranchOp::create(
            builder, loc, TypeRange{}, builder.getI64IntegerAttr(2),
            ValueRange{c1, iv, subgroupId, laneId, rhsArg, rhsAllocArg});

        // Read from buf0 (even's buffer, current k-tile data).
        auto readLhs = IREE::Template::BranchOp::create(
            builder, loc, TypeRange{type1}, builder.getI64IntegerAttr(3),
            ValueRange{c0, subgroupId, laneId, lhsAllocArg});
        auto readRhs = IREE::Template::BranchOp::create(
            builder, loc, TypeRange{type2}, builder.getI64IntegerAttr(4),
            ValueRange{c0, subgroupId, laneId, rhsAllocArg});

        // Fence release: copied to LDS + read from LDS.
        emitFence(/*isRelease=*/true);

        // Barrier D: done copying buf1 + done reading buf0.
        IREE::PCF::BarrierOp::create(builder, loc, subgroupScope);

        scf::YieldOp::create(
            builder, loc,
            ValueRange{computeBranch.getResults()[0],
                       readLhs.getResults()[0], readRhs.getResults()[0]});
      }

      // Epilogue: compute with the last iteration's read data.
      builder.setInsertionPointAfter(forOp);
      auto finalCompute = IREE::Template::BranchOp::create(
          builder, loc, TypeRange{type0}, builder.getI64IntegerAttr(5),
          ValueRange{forOp.getResult(0), forOp.getResult(1),
                     forOp.getResult(2)});

      // Barriers E+F: structural (match even's last loop C+D).
      IREE::PCF::BarrierOp::create(builder, loc, subgroupScope);
      IREE::PCF::BarrierOp::create(builder, loc, subgroupScope);

      // Write results to output.
      IREE::Template::BranchOp::create(
          builder, loc, TypeRange{}, builder.getI64IntegerAttr(6),
          ValueRange{subgroupId, laneId, finalCompute.getResults()[0], dest});

      IREE::PCF::ReturnOp::create(builder, loc);
    }

    // pcf.return (subgroup scope).
    builder.setInsertionPointAfter(ifOp);
    IREE::PCF::ReturnOp::create(builder, loc);
  }

  // template.return %res.
  builder.setInsertionPointAfter(outerGeneric);
  IREE::Template::ReturnOp::create(builder, loc,
                                   ValueRange{outerGeneric.getResult(0)});

  // ===== Create implementation blocks =====
  Region &implRegion = funcOp.getImplementations();

  // Block 0: (sg_id, lane_id, dest: sref) -> type<0>
  {
    Block *block = builder.createBlock(&implRegion);
    block->addArguments({indexType, indexType, outputSrefType},
                        {loc, loc, loc});
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{type0},
                                            ValueRange{});
  }

  // Block 1: Copy LHS (buf_idx, k_idx, sg_id, lane_id, lhs, lhs_alloc)
  {
    Block *block = builder.createBlock(&implRegion);
    block->addArguments(
        {indexType, indexType, indexType, indexType, lhsTensorType,
         lhsAllocType},
        {loc, loc, loc, loc, loc, loc});
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{},
                                            ValueRange{});
  }

  // Block 2: Copy RHS (buf_idx, k_idx, sg_id, lane_id, rhs, rhs_alloc)
  {
    Block *block = builder.createBlock(&implRegion);
    block->addArguments(
        {indexType, indexType, indexType, indexType, rhsTensorType,
         rhsAllocType},
        {loc, loc, loc, loc, loc, loc});
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{},
                                            ValueRange{});
  }

  // Block 3: Read LHS from LDS (buf_idx, sg_id, lane_id, lhs_alloc) -> type<1>
  {
    Block *block = builder.createBlock(&implRegion);
    block->addArguments({indexType, indexType, indexType, lhsAllocType},
                        {loc, loc, loc, loc});
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{type1},
                                            ValueRange{});
  }

  // Block 4: Read RHS from LDS (buf_idx, sg_id, lane_id, rhs_alloc) -> type<2>
  {
    Block *block = builder.createBlock(&implRegion);
    block->addArguments({indexType, indexType, indexType, rhsAllocType},
                        {loc, loc, loc, loc});
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{type2},
                                            ValueRange{});
  }

  // Block 5: Compute MMA (acc: type<0>, lhs: type<1>, rhs: type<2>) -> type<0>
  {
    Block *block = builder.createBlock(&implRegion);
    block->addArguments({type0, type1, type2}, {loc, loc, loc});
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{type0},
                                            ValueRange{});
  }

  // Block 6: Write results (sg_id, lane_id, result: type<0>, dest: sref)
  {
    Block *block = builder.createBlock(&implRegion);
    block->addArguments({indexType, indexType, type0, outputSrefType},
                        {loc, loc, loc, loc});
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{},
                                            ValueRange{});
  }

  return funcOp;
}

/// Build a copy block implementation that copies a tile from a dynamic input
/// tensor to shared memory using coalesced 128-bit vector reads.
///
/// Block args: (buf_idx, k_idx, sg_id, lane_id, input_tensor, shared_alloc)
///
/// The copy distributes work across threads (sg_id * subgroupSize + lane_id)
/// using the maximum vector size (128 bits). A scf.for loop handles the case
/// where there are more vector loads than threads.
///
/// Padding is handled via dynamic tensor.extract_slice + tensor.pad to ensure
/// no out-of-bounds reads.
static void populateCopyBlock(OpBuilder &builder, Block *block, Location loc,
                              int64_t tileRows, int64_t tileCols,
                              int64_t numSubgroups, int64_t subgroupSize,
                              Type elemType, int64_t rowDimIdx,
                              int64_t colDimIdx, int64_t kDimIdx) {
  // Block args: (buf_idx, k_idx, sg_id, lane_id, input_tensor, shared_alloc).
  Value bufIdx = block->getArgument(0);
  Value kIdx = block->getArgument(1);
  Value sgId = block->getArgument(2);
  Value laneId = block->getArgument(3);
  Value inputTensor = block->getArgument(4);
  Value sharedAlloc = block->getArgument(5);

  // Find and erase the unimplemented terminator.
  Operation *terminator = block->getTerminator();

  builder.setInsertionPoint(terminator);

  int64_t elemBits = elemType.getIntOrFloatBitWidth();
  int64_t vecSize = 128 / elemBits; // 8 for f16, 16 for i8, 4 for f32.
  // Each copy block runs with only half the subgroups (even or odd group).
  // Remap global sg_id to a local copy-group index for contiguous thread IDs.
  int64_t numCopySubgroups = numSubgroups / 2;
  int64_t numCopyThreads = numCopySubgroups * subgroupSize;
  int64_t numColVecs = tileCols / vecSize;
  int64_t totalVecs = tileRows * numColVecs;

  Value cTwo = arith::ConstantIndexOp::create(builder, loc, 2);
  Value cSubgroupSize =
      arith::ConstantIndexOp::create(builder, loc, subgroupSize);
  Value cNumThreads =
      arith::ConstantIndexOp::create(builder, loc, numCopyThreads);
  Value cTotalVecs =
      arith::ConstantIndexOp::create(builder, loc, totalVecs);
  Value cNumColVecs =
      arith::ConstantIndexOp::create(builder, loc, numColVecs);
  Value cVecSize = arith::ConstantIndexOp::create(builder, loc, vecSize);

  // Remap sg_id to local copy-group index: even sgs (0,2) → (0,1),
  // odd sgs (1,3) → (0,1). Integer division by 2 works for both.
  Value localSgId = arith::DivUIOp::create(builder, loc, sgId, cTwo);
  // thread_id = local_sg_id * subgroup_size + lane_id.
  Value threadId = arith::AddIOp::create(
      builder, loc,
      arith::MulIOp::create(builder, loc, localSgId, cSubgroupSize), laneId);

  // Compute the base offset along the K dimension in the source tensor.
  // For LHS (kDimIdx==colDimIdx): K advances along columns.
  // For RHS (kDimIdx==rowDimIdx): K advances along rows.
  int64_t kTileSize = (kDimIdx == rowDimIdx) ? tileRows : tileCols;
  Value cKTileSize = arith::ConstantIndexOp::create(builder, loc, kTileSize);
  Value kBase = arith::MulIOp::create(builder, loc, kIdx, cKTileSize);

  // Distribute work: for i = thread_id to total_vecs step num_threads.
  auto forOp = scf::ForOp::create(builder, loc, threadId, cTotalVecs,
                                   cNumThreads, ValueRange{});
  {
    Block *forBody = forOp.getBody();
    if (!forBody->empty() &&
        forBody->back().hasTrait<OpTrait::IsTerminator>())
      forBody->back().erase();
    builder.setInsertionPointToEnd(forBody);

    Value iv = forOp.getInductionVar();

    // Decompose iv into (row, col_vec).
    Value row = arith::DivUIOp::create(builder, loc, iv, cNumColVecs);
    Value colVec = arith::RemUIOp::create(builder, loc, iv, cNumColVecs);
    Value col = arith::MulIOp::create(builder, loc, colVec, cVecSize);

    // Source coordinates in the input tensor.
    // Advance the K dimension by kBase.
    Value srcRow = (kDimIdx == rowDimIdx)
                       ? arith::AddIOp::create(builder, loc, kBase, row)
                       : row;
    Value srcCol = (kDimIdx == colDimIdx)
                       ? arith::AddIOp::create(builder, loc, kBase, col)
                       : col;

    // Read a 1D vector along the column dimension using transfer_read.
    // This replaces extract_slice + pad: transfer_read handles OOB via padding.
    VectorType vecType = VectorType::get({vecSize}, elemType);

    // Zero padding value for OOB reads.
    Value padValue;
    if (isa<FloatType>(elemType)) {
      padValue = arith::ConstantOp::create(builder, loc, elemType,
                                           builder.getFloatAttr(elemType, 0.0));
    } else {
      padValue = arith::ConstantOp::create(
          builder, loc, elemType, builder.getIntegerAttr(elemType, 0));
    }

    // Indices: [srcRow, srcCol] for the input tensor.
    int64_t inputRank = cast<RankedTensorType>(inputTensor.getType()).getRank();
    SmallVector<Value> readIndices(inputRank,
                                  arith::ConstantIndexOp::create(builder, loc, 0));
    readIndices[rowDimIdx] = srcRow;
    readIndices[colDimIdx] = srcCol;

    // Permutation map: (d0, d1) -> (d_col) reads vecSize elements along col.
    AffineMap permMap = AffineMap::get(
        inputRank, /*symbolCount=*/0,
        {builder.getAffineDimExpr(colDimIdx)}, builder.getContext());

    // in_bounds=[false]: col access may go OOB, transfer_read pads with zero.
    Value readVec = vector::TransferReadOp::create(
        builder, loc, vecType, inputTensor, readIndices, padValue, permMap,
        SmallVector<bool>{false});

    // Shape cast vector<vecSize> -> vector<1x1xvecSize> for rank-3 sref.
    VectorType writeVecType = VectorType::get({1, 1, vecSize}, elemType);
    Value shapedVec =
        vector::ShapeCastOp::create(builder, loc, writeVecType, readVec);

    // Write to shared memory.
    // pcf.write_slice %shapedVec into %alloc[buf_idx, row, col][1, 1, vecSize].
    SmallVector<OpFoldResult> writeOffsets = {bufIdx, row, col};
    SmallVector<OpFoldResult> writeSizes = {
        builder.getIndexAttr(1), builder.getIndexAttr(1),
        builder.getIndexAttr(vecSize)};
    SmallVector<OpFoldResult> writeStrides(3, builder.getIndexAttr(1));
    PCF::WriteSliceOp::create(builder, loc, shapedVec, sharedAlloc,
                              writeOffsets, writeSizes, writeStrides);

    scf::YieldOp::create(builder, loc, ValueRange{});
  }

  // Replace unimplemented with template.return.
  builder.setInsertionPoint(terminator);
  IREE::Template::ReturnOp::create(builder, loc, ValueRange{});
  terminator->erase();
}

} // namespace

//===----------------------------------------------------------------------===//
// setPingpongLoweringConfig
//===----------------------------------------------------------------------===//

LogicalResult setPingpongLoweringConfig(IREE::GPU::TargetAttr target,
                                        FunctionOpInterface entryPoint,
                                        Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) {
    return failure();
  }

  // Restrict to standard matmul: exactly 1 M, 1 N, 1 K dim.
  FailureOr<linalg::ContractionDimensions> contractionDims =
      linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return failure();
  }
  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      contractionDims->k.size() != 1) {
    return failure();
  }
  unsigned mDim = contractionDims->m[0];
  unsigned nDim = contractionDims->n[0];
  unsigned kDim = contractionDims->k[0];

  // Load dynamic dim analysis for bounds info.
  TensorDynamicDimAnalysis dynamicDimAnalysis(entryPoint);
  if (failed(dynamicDimAnalysis.run())) {
    return failure();
  }

  ModuleOp moduleOp = entryPoint->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    return failure();
  }

  // Step 1: Get valid MMA instructions.
  ArrayRef<IREE::GPU::MMAAttr> mmaAttrs = target.getWgp().getMma();
  if (mmaAttrs.empty()) {
    return failure();
  }
  // HACK: Just pick the first MMA kind.
  IREE::GPU::MMAAttr mmaKind = mmaAttrs[0];

  // Step 2: Pick tile sizes.
  Type lhsElemType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType())
          .getElementType();
  int64_t elemBitWidth = lhsElemType.getIntOrFloatBitWidth();
  int64_t kTileSize = (128 * 8) / elemBitWidth; // K elements for 128 bytes.
  int64_t subgroupSize = target.getPreferredSubgroupSize();
  // Select tile sizes and subgroup count based on subgroup size.
  // LDS usage: 2 * (mTile*kTile + kTile*nTile) * elemBytes.
  int64_t mTileSize, nTileSize, numSubgroups;
  if (subgroupSize == 64) {
    // Wave64 (CDNA): 256x256, 8 subgroups (4M x 2N), 512 threads.
    mTileSize = 256;
    nTileSize = 256;
    numSubgroups = 8;
  } else if (subgroupSize == 32) {
    // Wave32 (RDNA): 128x128, 4 subgroups (2M x 2N), 128 threads.
    // LDS: 2*(128*64 + 64*128)*2 = 65536 bytes = 64 KB.
    mTileSize = 128;
    nTileSize = 128;
    numSubgroups = 4;
  } else {
    return failure();
  }
  int64_t flatWorkgroupSize = numSubgroups * subgroupSize;

  // Step 3: LDS layout (HACK: default, no change).

  // Step 4: Copy operators (HACK: naive roundtrip through registers).

  // Step 5: Build the pingpong loop structure.
  // Determine indexing map positions for LHS and RHS.
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  // LHS map: find which dim maps to M and which to K.
  AffineMap lhsMap = maps[0];
  AffineMap rhsMap = maps[1];

  // For LHS tensor: find row/col dims.
  // Standard matmul indexing: LHS has M and K dims.
  int64_t lhsRowDim = -1, lhsColDim = -1;
  for (unsigned i = 0; i < lhsMap.getNumResults(); ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(lhsMap.getResult(i));
    if (!dimExpr)
      return failure();
    if (dimExpr.getPosition() == mDim)
      lhsRowDim = i;
    else if (dimExpr.getPosition() == kDim)
      lhsColDim = i;
  }
  if (lhsRowDim < 0 || lhsColDim < 0) {
    return failure();
  }

  // For RHS tensor: find row/col dims.
  int64_t rhsRowDim = -1, rhsColDim = -1;
  for (unsigned i = 0; i < rhsMap.getNumResults(); ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(rhsMap.getResult(i));
    if (!dimExpr)
      return failure();
    if (dimExpr.getPosition() == kDim)
      rhsRowDim = i;
    else if (dimExpr.getPosition() == nDim)
      rhsColDim = i;
  }
  if (rhsRowDim < 0 || rhsColDim < 0) {
    return failure();
  }

  // LHS tile shape: [mTileSize, kTileSize] in the indexing map order.
  SmallVector<int64_t, 2> lhsTileShape(2);
  lhsTileShape[lhsRowDim] = mTileSize;
  lhsTileShape[lhsColDim] = kTileSize;
  // RHS tile shape: [kTileSize, nTileSize] in the indexing map order.
  SmallVector<int64_t, 2> rhsTileShape(2);
  rhsTileShape[rhsRowDim] = kTileSize;
  rhsTileShape[rhsColDim] = nTileSize;

  // Input tensor types: compute the shapes after workgroup tiling.
  // Parallel dims (M, N) get replaced with tile sizes; reduction dim (K) keeps
  // the original shape (may be static or dynamic).
  RankedTensorType origLhsType =
      cast<RankedTensorType>(linalgOp.getDpsInputOperand(0)->get().getType());
  RankedTensorType origRhsType =
      cast<RankedTensorType>(linalgOp.getDpsInputOperand(1)->get().getType());
  SmallVector<int64_t> tiledLhsShape(origLhsType.getShape());
  tiledLhsShape[lhsRowDim] = mTileSize; // M dim gets tiled.
  // K dim (lhsColDim) keeps original size.
  RankedTensorType lhsTensorType =
      RankedTensorType::get(tiledLhsShape, lhsElemType);
  SmallVector<int64_t> tiledRhsShape(origRhsType.getShape());
  tiledRhsShape[rhsColDim] = nTileSize; // N dim gets tiled.
  // K dim (rhsRowDim) keeps original size.
  RankedTensorType rhsTensorType =
      RankedTensorType::get(tiledRhsShape, lhsElemType);
  // Output tensor type: use the workgroup-tiled shape (not the original full
  // tensor) since the template will be instantiated after workgroup tiling.
  Type outputElemType =
      cast<RankedTensorType>(linalgOp.getDpsInits()[0].getType())
          .getElementType();
  AffineMap accMap = maps.back();
  SmallVector<int64_t> tiledOutputShape;
  for (unsigned i = 0; i < accMap.getNumResults(); ++i) {
    auto dimExpr = cast<AffineDimExpr>(accMap.getResult(i));
    unsigned iterDim = dimExpr.getPosition();
    if (iterDim == mDim) {
      tiledOutputShape.push_back(mTileSize);
    } else if (iterDim == nDim) {
      tiledOutputShape.push_back(nTileSize);
    } else {
      return failure();
    }
  }
  RankedTensorType outputTensorType =
      RankedTensorType::get(tiledOutputShape, outputElemType);

  // Generate unique name.
  static unsigned pingpongCounter = 0;
  std::string templateName =
      ("__pingpong_template_" + Twine(pingpongCounter++)).str();

  MLIRContext *context = op->getContext();
  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToStart(moduleOp.getBody());

  auto templateFunc = createPingpongTemplateFunc(
      moduleBuilder, op->getLoc(), templateName, outputTensorType,
      lhsTensorType, rhsTensorType, lhsTileShape, rhsTileShape, lhsElemType,
      lhsColDim);

  SymbolTable symbolTable(moduleOp);
  symbolTable.insert(templateFunc);

  // Step 6: Implement copy blocks (1 and 2).
  Region &implRegion = templateFunc.getImplementations();
  auto blockIt = implRegion.begin();
  std::advance(blockIt, 1); // Block 1: copy LHS.
  Block *lhsCopyBlock = &*blockIt;
  populateCopyBlock(moduleBuilder, lhsCopyBlock, op->getLoc(),
                    lhsTileShape[0], lhsTileShape[1], numSubgroups,
                    subgroupSize, lhsElemType,
                    /*rowDimIdx=*/0, /*colDimIdx=*/1, /*kDimIdx=*/1);

  std::advance(blockIt, 1); // Block 2: copy RHS.
  Block *rhsCopyBlock = &*blockIt;
  populateCopyBlock(moduleBuilder, rhsCopyBlock, op->getLoc(),
                    rhsTileShape[0], rhsTileShape[1], numSubgroups,
                    subgroupSize, lhsElemType,
                    /*rowDimIdx=*/0, /*colDimIdx=*/1, /*kDimIdx=*/0);

  LLVM_DEBUG(llvm::dbgs() << "Created pingpong template: " << templateName
                          << "\n");

  // Step 7: Attach lowering config to root op.
  Builder b(context);
  SmallVector<NamedAttribute> attrs;

  SmallVector<int64_t> workgroupTileSizes(linalgOp.getNumLoops(), 0);
  workgroupTileSizes[mDim] = mTileSize;
  workgroupTileSizes[nDim] = nTileSize;
  attrs.emplace_back("workgroup", b.getI64ArrayAttr(workgroupTileSizes));

  // Subgroup tile sizes encode the outer dimension distribution and K tile.
  // Parallel dims: number of subgroups along each dimension (distribution).
  // Reduction dim: K tile size.
  int64_t numSubgroupsM, numSubgroupsN;
  if (numSubgroups == 8) {
    numSubgroupsM = 4;
    numSubgroupsN = 2;
  } else {
    // 4 subgroups: 2M x 2N.
    numSubgroupsM = 2;
    numSubgroupsN = 2;
  }
  SmallVector<int64_t> subgroupTileSizes(linalgOp.getNumLoops(), 0);
  subgroupTileSizes[mDim] = numSubgroupsM;
  subgroupTileSizes[nDim] = numSubgroupsN;
  subgroupTileSizes[kDim] = kTileSize;
  attrs.emplace_back("subgroup", b.getI64ArrayAttr(subgroupTileSizes));

  // Reduction tile sizes for K dimension.
  SmallVector<int64_t> reductionTileSizes(linalgOp.getNumLoops(), 0);
  reductionTileSizes[kDim] = kTileSize;
  attrs.emplace_back("reduction", b.getI64ArrayAttr(reductionTileSizes));

  GPU::setMmaKind(context, attrs, mmaKind);
  GPU::setTemplateCall(context, attrs,
                       FlatSymbolRefAttr::get(context, templateName));

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig =
      IREE::GPU::LoweringConfigAttr::get(context, configDict);

  // Step 8: Attach translation info to entryPoint.
  std::array<int64_t, 3> workgroupSize = {flatWorkgroupSize, 1, 1};
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, subgroupSize, DictionaryAttr());
}

} // namespace mlir::iree_compiler::IREE::GPU
