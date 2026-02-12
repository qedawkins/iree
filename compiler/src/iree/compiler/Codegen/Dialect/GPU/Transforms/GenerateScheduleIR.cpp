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
/// Creates a matmul contraction: acc = A * B + acc.
/// The A and B operands are provided by the caller (typically from LDS reads).
/// The accumulator is updated in-place and returned.
static Value generateComputePhase(OpBuilder &builder, Location loc, Value lhs,
                                  Value rhs, Value acc) {
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
      builder, loc, lhs, rhs, acc,
      builder.getAffineMapArrayAttr({mapA, mapB, mapC}),
      builder.getArrayAttr(iteratorTypes));
}

/// Read a quarter-K tile from LDS shared memory.
///
/// For LHS: reads [0, quarterIdx*quarterK] with size [mmaM, quarterK].
/// For RHS: reads [quarterIdx*quarterK, 0] with size [quarterK, mmaN].
/// Returns a vector value suitable for vector.contract.
static Value readQuarterFromLDS(OpBuilder &builder, Location loc,
                                Value ldsSref, bool isLHS,
                                int64_t quarterIdx,
                                const HardcodedConfig &config,
                                Type inputElementType) {
  int64_t quarterK = config.quarterK();
  int64_t kOffset = quarterIdx * quarterK;

  SmallVector<OpFoldResult> offsets, sizes, strides;
  VectorType resultType;
  if (isLHS) {
    // LHS: [0, kOffset] size [mmaM, quarterK].
    offsets = {builder.getIndexAttr(0), builder.getIndexAttr(kOffset)};
    sizes = {builder.getIndexAttr(config.mmaM),
             builder.getIndexAttr(quarterK)};
    resultType = VectorType::get({config.mmaM, quarterK}, inputElementType);
  } else {
    // RHS: [kOffset, 0] size [quarterK, mmaN].
    offsets = {builder.getIndexAttr(kOffset), builder.getIndexAttr(0)};
    sizes = {builder.getIndexAttr(quarterK),
             builder.getIndexAttr(config.mmaN)};
    resultType = VectorType::get({quarterK, config.mmaN}, inputElementType);
  }
  strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};

  return PCF::ReadSliceOp::create(builder, loc, resultType, ldsSref, offsets,
                                  sizes, strides);
}

/// Load a full workgroup tile from a global tensor using tensor.extract_slice.
///
/// For LHS: extracts [0, kOffset] with size [workgroupM, kTile].
/// For RHS: extracts [kOffset, 0] with size [kTile, workgroupN].
static Value loadGlobalTile(OpBuilder &builder, Location loc, Value globalTensor,
                            Value kOffset, bool isLHS,
                            int64_t workgroupM, int64_t workgroupN,
                            const HardcodedConfig &config) {
  auto tensorType = cast<RankedTensorType>(globalTensor.getType());
  Type elementType = tensorType.getElementType();

  SmallVector<OpFoldResult> offsets, sizes, strides;
  RankedTensorType resultType;
  if (isLHS) {
    // LHS: [0, kOffset] size [workgroupM, kTile].
    offsets = {builder.getIndexAttr(0), kOffset};
    sizes = {builder.getIndexAttr(workgroupM),
             builder.getIndexAttr(config.kTile)};
    strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};
    resultType = RankedTensorType::get({workgroupM, config.kTile}, elementType);
  } else {
    // RHS: [kOffset, 0] size [kTile, workgroupN].
    offsets = {kOffset, builder.getIndexAttr(0)};
    sizes = {builder.getIndexAttr(config.kTile),
             builder.getIndexAttr(workgroupN)};
    strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};
    resultType = RankedTensorType::get({config.kTile, workgroupN}, elementType);
  }

  return tensor::ExtractSliceOp::create(builder, loc, resultType, globalTensor,
                                        offsets, sizes, strides);
}

/// Write a staged tile into LDS shared memory using pcf.write_slice.
static void writeToLDS(OpBuilder &builder, Location loc, Value source,
                       Value ldsSref) {
  auto sourceType = cast<ShapedType>(source.getType());
  int64_t rank = sourceType.getRank();
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  for (int64_t i = 0; i < rank; ++i) {
    sizes.push_back(builder.getIndexAttr(sourceType.getDimSize(i)));
  }
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

  PCF::WriteSliceOp::create(builder, loc, source, ldsSref, offsets, sizes,
                            strides);
}

/// Generate the 8-phase K-loop body with all memory and compute operations.
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
/// Each phase ends with a pcf.barrier. Memory phases read quarter-K tiles
/// from LDS and stage global loads for the next iteration. Compute phases
/// use the LDS-read data in vector.contract ops.
static Value generatePhaseBody(OpBuilder &builder, Location loc,
                               PCF::ScopeAttrInterface barrierScope,
                               Value acc, Value ldsLhs, Value ldsRhs,
                               Value globalLhs, Value globalRhs, Value kIV,
                               int64_t workgroupM, int64_t workgroupN,
                               const HardcodedConfig &config,
                               Type inputElementType) {
  // Compute next iteration's K offset for prefetching.
  Value kStep = arith::ConstantIndexOp::create(builder, loc, config.kTile);
  Value kNext = arith::AddIOp::create(builder, loc, kIV, kStep);

  // P1: Global load LHS(k+1) + LDS read q0.
  Value lhsStage = loadGlobalTile(builder, loc, globalLhs, kNext,
                                  /*isLHS=*/true, workgroupM, workgroupN,
                                  config);
  Value lhsQ0 =
      readQuarterFromLDS(builder, loc, ldsLhs, /*isLHS=*/true,
                         /*quarterIdx=*/0, config, inputElementType);
  Value rhsQ0 =
      readQuarterFromLDS(builder, loc, ldsRhs, /*isLHS=*/false,
                         /*quarterIdx=*/0, config, inputElementType);
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P2: Compute WMMA q0.
  acc = generateComputePhase(builder, loc, lhsQ0, rhsQ0, acc);
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P3: Global load RHS(k+1) + LDS read q1.
  Value rhsStage = loadGlobalTile(builder, loc, globalRhs, kNext,
                                  /*isLHS=*/false, workgroupM, workgroupN,
                                  config);
  Value lhsQ1 =
      readQuarterFromLDS(builder, loc, ldsLhs, /*isLHS=*/true,
                         /*quarterIdx=*/1, config, inputElementType);
  Value rhsQ1 =
      readQuarterFromLDS(builder, loc, ldsRhs, /*isLHS=*/false,
                         /*quarterIdx=*/1, config, inputElementType);
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P4: Compute WMMA q1.
  acc = generateComputePhase(builder, loc, lhsQ1, rhsQ1, acc);
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P5: LDS write staged LHS + LDS read q2.
  writeToLDS(builder, loc, lhsStage, ldsLhs);
  Value lhsQ2 =
      readQuarterFromLDS(builder, loc, ldsLhs, /*isLHS=*/true,
                         /*quarterIdx=*/2, config, inputElementType);
  Value rhsQ2 =
      readQuarterFromLDS(builder, loc, ldsRhs, /*isLHS=*/false,
                         /*quarterIdx=*/2, config, inputElementType);
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P6: Compute WMMA q2 + LDS write staged RHS + LDS read q3.
  acc = generateComputePhase(builder, loc, lhsQ2, rhsQ2, acc);
  writeToLDS(builder, loc, rhsStage, ldsRhs);
  Value lhsQ3 =
      readQuarterFromLDS(builder, loc, ldsLhs, /*isLHS=*/true,
                         /*quarterIdx=*/3, config, inputElementType);
  Value rhsQ3 =
      readQuarterFromLDS(builder, loc, ldsRhs, /*isLHS=*/false,
                         /*quarterIdx=*/3, config, inputElementType);
  PCF::BarrierOp::create(builder, loc, barrierScope);

  // P7: Compute WMMA q3.
  acc = generateComputePhase(builder, loc, lhsQ3, rhsQ3, acc);
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
  Value rhs = contractionOp.getDpsInputs()[1];
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
  Value ldsLhs = PCF::AllocOp::create(rewriter, loc, ldsLhsType);
  Value ldsRhs = PCF::AllocOp::create(rewriter, loc, ldsRhsType);

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

  // Prologue: load the first K tile (k=0) into LDS.
  // Global tensors (lhs, rhs) are captured from enclosing func scope.
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value lhsInit = loadGlobalTile(rewriter, loc, lhs, c0, /*isLHS=*/true,
                                 workgroupM, workgroupN, config);
  Value rhsInit = loadGlobalTile(rewriter, loc, rhs, c0, /*isLHS=*/false,
                                 workgroupM, workgroupN, config);
  writeToLDS(rewriter, loc, lhsInit, ldsLhs);
  writeToLDS(rewriter, loc, rhsInit, ldsRhs);
  PCF::BarrierOp::create(rewriter, loc, sgScope);

  // Create K loop constants.
  Value kStep = arith::ConstantIndexOp::create(rewriter, loc, config.kTile);
  Value kBound = arith::ConstantIndexOp::create(rewriter, loc, kDim);

  // Create scf.for K loop carrying the accumulator.
  // LDS srefs and global tensors are captured from enclosing scopes.
  scf::ForOp kLoop = scf::ForOp::create(
      rewriter, loc, c0, kBound, kStep, /*initArgs=*/ValueRange{accInit},
      [&](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
          ValueRange iterArgs) {
        Value acc = iterArgs[0];
        acc = generatePhaseBody(bodyBuilder, bodyLoc, sgScope, acc, ldsLhs,
                                ldsRhs, lhs, rhs, iv, workgroupM, workgroupN,
                                config, inputElementType);
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
