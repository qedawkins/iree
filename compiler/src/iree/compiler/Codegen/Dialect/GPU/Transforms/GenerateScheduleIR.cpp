// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- GenerateScheduleIR.cpp - Structured schedule IR generation ---------===//
//
// This pass generates structured schedule IR for GPU matmul operations.
// It takes a function containing a contraction op (linalg.generic with matmul
// semantics) and replaces it with a structured schedule using PCF ops,
// barriers, and MMA operations following the quarter-K pingpong pattern.
//
// The generated IR structure mirrors the 8-phase early-write schedule:
//   P1: Global load LHS + LDS read q0
//   P2: WMMA compute q0
//   P3: Global load RHS + LDS read q1
//   P4: WMMA compute q1
//   P5: LDS write LHS + LDS read q2
//   P6: WMMA compute q2 + LDS write RHS + LDS read q3
//   P7: WMMA compute q3
//   P8: Barrier + loop control
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_GENERATESCHEDULEIRPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Schedule Configuration (hardcoded reference for initial implementation)
//===----------------------------------------------------------------------===//

// Reference RDNA4 schedule configuration.
// TODO: Replace with ScheduleConfig derived from target.
struct HardcodedConfig {
  int64_t kTile = 64;
  int64_t numPhases = 8;     // 4 compute + 4 memory phases.
  int64_t numQuarters = 4;   // Quarter-K segments per K tile.
  int64_t mmaM = 16;         // WMMA_F32_16x16x16_F16 tile dimensions.
  int64_t mmaN = 16;
  int64_t mmaK = 16;

  int64_t quarterK() const { return kTile / numQuarters; }
};

//===----------------------------------------------------------------------===//
// Schedule IR Generation
//===----------------------------------------------------------------------===//

/// Generate a single WMMA compute phase using vector.contract.
///
/// Creates a matmul contraction: acc = A * B + acc, where A and B are
/// placeholder zero vectors (will be replaced with LDS read results).
/// The accumulator is updated in-place and returned.
static Value generateComputePhase(OpBuilder &builder, Location loc,
                                  Value acc, const HardcodedConfig &config,
                                  Type inputElementType, Type accElementType) {
  auto aType = VectorType::get({config.mmaM, config.mmaK}, inputElementType);
  auto bType = VectorType::get({config.mmaK, config.mmaN}, inputElementType);

  // Placeholder zero operands (will be replaced with LDS reads).
  Value zeroInput = arith::ConstantOp::create(
      builder, loc, DenseElementsAttr::get(aType, builder.getZeroAttr(inputElementType)));
  Value zeroB = arith::ConstantOp::create(
      builder, loc, DenseElementsAttr::get(bType, builder.getZeroAttr(inputElementType)));

  // Contraction maps: C[m,n] += A[m,k] * B[k,n].
  AffineMap mapA = AffineMap::get(
      3, 0, {builder.getAffineDimExpr(0), builder.getAffineDimExpr(2)},
      builder.getContext());
  AffineMap mapB = AffineMap::get(
      3, 0, {builder.getAffineDimExpr(2), builder.getAffineDimExpr(1)},
      builder.getContext());
  AffineMap mapC = AffineMap::get(
      3, 0, {builder.getAffineDimExpr(0), builder.getAffineDimExpr(1)},
      builder.getContext());

  SmallVector<Attribute> iteratorTypes = {
      vector::IteratorTypeAttr::get(builder.getContext(),
                                    vector::IteratorType::parallel),
      vector::IteratorTypeAttr::get(builder.getContext(),
                                    vector::IteratorType::parallel),
      vector::IteratorTypeAttr::get(builder.getContext(),
                                    vector::IteratorType::reduction),
  };

  return vector::ContractionOp::create(
      builder, loc, zeroInput, zeroB, acc,
      builder.getAffineMapArrayAttr({mapA, mapB, mapC}),
      builder.getArrayAttr(iteratorTypes));
}

/// Generate the 8-phase K-loop body with barriers and compute ops.
///
/// For each K iteration, the early-write schedule generates 8 phases:
///   P1: Memory  - Global load LHS(k+1) + LDS read q0
///   P2: Compute - WMMA q0
///   P3: Memory  - Global load RHS(k+1) + LDS read q1
///   P4: Compute - WMMA q1
///   P5: Memory  - LDS write GL-LHS + LDS read q2
///   P6: Mixed   - WMMA q2 + LDS write GL-RHS + LDS read q3
///   P7: Compute - WMMA q3
///   P8: Sync    - Barrier for loop iteration boundary
///
/// Each phase ends with a pcf.barrier. Memory phase bodies are empty
/// placeholders for now. Compute phases contain vector.contract ops
/// with zero placeholder operands.
static Value generatePhaseBody(OpBuilder &builder, Location loc,
                               PCF::ScopeAttrInterface barrierScope,
                               Value acc, const HardcodedConfig &config,
                               Type inputElementType, Type accElementType) {
  // P1: Memory phase (placeholder).
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P2: Compute WMMA q0.
  acc = generateComputePhase(builder, loc, acc, config, inputElementType,
                             accElementType);
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P3: Memory phase (placeholder).
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P4: Compute WMMA q1.
  acc = generateComputePhase(builder, loc, acc, config, inputElementType,
                             accElementType);
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P5: Memory phase (placeholder).
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P6: Mixed — compute WMMA q2 (+ memory placeholder).
  acc = generateComputePhase(builder, loc, acc, config, inputElementType,
                             accElementType);
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P7: Compute WMMA q3.
  acc = generateComputePhase(builder, loc, acc, config, inputElementType,
                             accElementType);
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P8: Sync barrier (loop iteration boundary).
  PCF::BarrierOp::create(builder, loc, barrierScope);

  return acc;
}

/// Generate the schedule IR for a contraction op.
///
/// Replaces the contraction with:
///   pcf.generic scope(subgroup) {
///     pcf.generic scope(lane) {
///       scf.for K {
///         <8 phase barriers>
///       }
///       pcf.return
///     }
///     pcf.return
///   }
static LogicalResult
generateScheduleForContraction(IRRewriter &rewriter,
                               linalg::LinalgOp contractionOp) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = contractionOp.getLoc();

  // Verify this is a contraction (matmul-like).
  if (!linalg::isaContractionOpInterface(contractionOp)) {
    return contractionOp.emitError("expected a contraction op");
  }

  // Get operands.
  Value lhs = contractionOp.getDpsInputs()[0];
  Value out = contractionOp.getDpsInits()[0];

  // Extract tensor types and dimensions.
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto outType = cast<RankedTensorType>(out.getType());
  int64_t kDim = lhsType.getDimSize(lhsType.getRank() - 1);
  if (kDim == ShapedType::kDynamic) {
    return contractionOp.emitError("dynamic K dimension not yet supported");
  }
  int64_t workgroupM = outType.getDimSize(0);
  int64_t workgroupN = outType.getDimSize(1);
  Type inputElementType = lhsType.getElementType();
  Type accElementType = outType.getElementType();

  // Schedule configuration.
  HardcodedConfig config;

  // Create scope attributes.
  PCF::ScopeAttrInterface sgScope =
      cast<PCF::ScopeAttrInterface>(SubgroupScopeAttr::get(ctx));
  PCF::ScopeAttrInterface laneScope =
      cast<PCF::ScopeAttrInterface>(LaneScopeAttr::get(ctx));

  // Create outer pcf.generic at subgroup scope.
  // The output tensor is passed as a tied init.
  rewriter.setInsertionPoint(contractionOp);
  PCF::GenericOp sgGeneric = PCF::GenericOp::create(
      rewriter, loc,
      /*resultTypes=*/contractionOp->getResultTypes(),
      /*scope=*/sgScope,
      /*inits=*/ValueRange{out},
      /*dynamicSizes=*/ValueRange{},
      /*isTied=*/SmallVector<bool>{true},
      /*numIterators=*/1,
      /*syncOnReturn=*/false);

  // Get subgroup execute body. Block args: [dest_ref, sg_id, sg_count].
  Block *sgBody = &sgGeneric.getRegion().front();
  rewriter.setInsertionPointToStart(sgBody);

  // Allocate LDS for LHS (workgroupM x kTile) and RHS (kTile x workgroupN).
  // These are shared across all lanes within a subgroup.
  PCF::ShapedRefType ldsLhsType = PCF::ShapedRefType::get(
      ctx, {workgroupM, config.kTile}, inputElementType, sgScope);
  PCF::ShapedRefType ldsRhsType = PCF::ShapedRefType::get(
      ctx, {config.kTile, workgroupN}, inputElementType, sgScope);
  PCF::AllocOp::create(rewriter, loc, ldsLhsType);
  PCF::AllocOp::create(rewriter, loc, ldsRhsType);

  // Create inner pcf.generic at lane scope (no tied results).
  PCF::GenericOp laneGeneric = PCF::GenericOp::create(
      rewriter, loc,
      /*resultTypes=*/TypeRange{},
      /*scope=*/laneScope,
      /*inits=*/ValueRange{},
      /*dynamicSizes=*/ValueRange{},
      /*isTied=*/SmallVector<bool>{},
      /*numIterators=*/1,
      /*syncOnReturn=*/false);

  // Get lane execute body. Block args: [lane_id, lane_count].
  Block *laneBody = &laneGeneric.getRegion().front();
  rewriter.setInsertionPointToStart(laneBody);

  // Initialize accumulator (zero vector at MMA tile granularity).
  auto accVecType = VectorType::get({config.mmaM, config.mmaN}, accElementType);
  Value accInit = arith::ConstantOp::create(
      rewriter, loc,
      DenseElementsAttr::get(accVecType, rewriter.getZeroAttr(accElementType)));

  // Create K loop constants.
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value kStep = arith::ConstantIndexOp::create(rewriter, loc, config.kTile);
  Value kBound = arith::ConstantIndexOp::create(rewriter, loc, kDim);

  // Create scf.for K loop carrying the accumulator.
  scf::ForOp kLoop = scf::ForOp::create(
      rewriter, loc, c0, kBound, kStep, /*initArgs=*/ValueRange{accInit},
      [&](OpBuilder &bodyBuilder, Location bodyLoc, Value /*iv*/,
          ValueRange iterArgs) {
        Value acc = iterArgs[0];
        acc = generatePhaseBody(bodyBuilder, bodyLoc, sgScope, acc, config,
                                inputElementType, accElementType);
        scf::YieldOp::create(bodyBuilder, bodyLoc, ValueRange{acc});
      });

  // Write the final accumulator back to the output sref.
  // destRef is the subgroup body's first block arg (tied init → sref).
  Value destRef = sgBody->getArgument(0);
  Value kLoopResult = kLoop.getResult(0);
  SmallVector<OpFoldResult> writeOffsets = {rewriter.getIndexAttr(0),
                                            rewriter.getIndexAttr(0)};
  SmallVector<OpFoldResult> writeSizes = {rewriter.getIndexAttr(config.mmaM),
                                          rewriter.getIndexAttr(config.mmaN)};
  SmallVector<OpFoldResult> writeStrides = {rewriter.getIndexAttr(1),
                                            rewriter.getIndexAttr(1)};
  PCF::WriteSliceOp::create(rewriter, loc, kLoopResult, destRef, writeOffsets,
                            writeSizes, writeStrides);

  // Terminate lane body with pcf.return.
  PCF::ReturnOp::create(rewriter, loc);

  // Terminate subgroup body with pcf.return.
  // Results are produced by snapshotting tied srefs, not by returning values.
  rewriter.setInsertionPointToEnd(sgBody);
  PCF::ReturnOp::create(rewriter, loc);

  // Replace contraction op results with generic results.
  rewriter.replaceOp(contractionOp, sgGeneric.getResults());

  return success();
}

//===----------------------------------------------------------------------===//
// GenerateScheduleIRPass
//===----------------------------------------------------------------------===//

struct GenerateScheduleIRPass final
    : impl::GenerateScheduleIRPassBase<GenerateScheduleIRPass> {
  void runOnOperation() override;
};

} // namespace

void GenerateScheduleIRPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  MLIRContext *ctx = &getContext();

  // Find contraction ops.
  SmallVector<linalg::LinalgOp> contractions;
  funcOp.walk([&](linalg::LinalgOp op) {
    if (linalg::isaContractionOpInterface(op)) {
      contractions.push_back(op);
    }
  });

  if (contractions.empty()) {
    return; // Nothing to do.
  }

  IRRewriter rewriter(ctx);
  for (linalg::LinalgOp contractionOp : contractions) {
    if (failed(generateScheduleForContraction(rewriter, contractionOp))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
