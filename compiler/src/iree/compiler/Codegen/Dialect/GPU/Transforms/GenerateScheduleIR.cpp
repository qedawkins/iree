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
// barriers, and vector.contract operations.
//
// The generated IR uses a simple load-barrier-compute-barrier pattern:
//   1. Collaboratively load a K-tile from global to LDS
//   2. Barrier (ensure all threads see the LDS data)
//   3. Each thread reads its sub-tile and accumulates via vector.contract
//   4. Barrier (ensure all threads are done reading before next write)
//
// Thread distribution:
//   - Subgroups tile the M dimension.
//   - Lanes within a subgroup tile the N dimension.
//   - Each thread computes a mmaM x mmaN output tile.
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
  int64_t numQuarters = 4;   // Quarter-K segments per K tile.
  int64_t mmaM = 16;         // Compute tile dimensions per thread.
  int64_t mmaN = 16;
  int64_t mmaK = 16;

  int64_t quarterK() const { return kTile / numQuarters; }
};

//===----------------------------------------------------------------------===//
// Schedule IR Generation
//===----------------------------------------------------------------------===//

/// Generate a single compute phase using vector.contract.
///
/// Creates a matmul contraction: acc = A * B + acc.
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

/// Read a quarter-K tile from LDS shared memory at thread-dependent offsets.
///
/// For LHS: reads [mOffset, quarterIdx*quarterK] with size [mmaM, quarterK].
/// For RHS: reads [quarterIdx*quarterK, nOffset] with size [quarterK, mmaN].
static Value readQuarterFromLDS(OpBuilder &builder, Location loc,
                                Value ldsSref, bool isLHS,
                                int64_t quarterIdx, Value mOffset,
                                Value nOffset, const HardcodedConfig &config,
                                Type inputElementType) {
  int64_t quarterK = config.quarterK();
  int64_t kOffset = quarterIdx * quarterK;

  SmallVector<OpFoldResult> offsets, sizes, strides;
  VectorType resultType;
  if (isLHS) {
    // LHS: [mOffset, kOffset] size [mmaM, quarterK].
    offsets = {OpFoldResult(mOffset), builder.getIndexAttr(kOffset)};
    sizes = {builder.getIndexAttr(config.mmaM),
             builder.getIndexAttr(quarterK)};
    resultType = VectorType::get({config.mmaM, quarterK}, inputElementType);
  } else {
    // RHS: [kOffset, nOffset] size [quarterK, mmaN].
    offsets = {builder.getIndexAttr(kOffset), OpFoldResult(nOffset)};
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
static Value loadGlobalTile(OpBuilder &builder, Location loc,
                            Value globalTensor, Value kOffset, bool isLHS,
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
  for (int64_t i = 0, e = rank; i < e; ++i) {
    sizes.push_back(builder.getIndexAttr(sourceType.getDimSize(i)));
  }
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

  PCF::WriteSliceOp::create(builder, loc, source, ldsSref, offsets, sizes,
                            strides);
}

/// Generate the schedule IR for a contraction op.
///
/// Replaces the contraction with a PCF schedule that:
///   1. Distributes output tiles across subgroups (M dim) and lanes (N dim).
///   2. Loops over K in kTile-sized chunks.
///   3. Each iteration: load global->LDS, barrier, read/compute quarters,
///      barrier.
///   4. Writes the per-thread accumulator to the output at the thread's offset.
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

  // LDS allocation types.
  PCF::ShapedRefType ldsLhsType = PCF::ShapedRefType::get(
      ctx, {workgroupM, config.kTile}, inputElementType, sgScope);
  PCF::ShapedRefType ldsRhsType = PCF::ShapedRefType::get(
      ctx, {config.kTile, workgroupN}, inputElementType, sgScope);

  PCF::GenericOp sgGeneric = PCF::GenericOp::create(
      rewriter, loc,
      /*resultTypes=*/contractionOp->getResultTypes(),
      /*scope=*/sgScope,
      /*inits=*/ValueRange{out},
      /*dynamicSizes=*/ValueRange{},
      /*isTied=*/SmallVector<bool>{true},
      /*numIterators=*/1,
      /*syncOnReturn=*/false);

  // Populate the initializer region with LDS allocations.
  // Allocations must be in the initialize region, not the execute body,
  // so that ConvertSRefToMemRef can properly lower them to memref.alloc.
  {
    Region &initRegion = sgGeneric.getInitializer();
    Block *initBlock = rewriter.createBlock(&initRegion);
    rewriter.setInsertionPointToStart(initBlock);
    Value ldsLhsAlloc = PCF::AllocOp::create(rewriter, loc, ldsLhsType);
    Value ldsRhsAlloc = PCF::AllocOp::create(rewriter, loc, ldsRhsType);
    PCF::YieldOp::create(rewriter, loc,
                         ValueRange{ldsLhsAlloc, ldsRhsAlloc});
  }

  // Add leading block args in the execute region for the initialized allocs.
  // Execute block arg layout: [leading_args, ref_args, id_args, count_args].
  // Before: [dest_ref, sg_id, sg_count].
  // After:  [lds_lhs, lds_rhs, dest_ref, sg_id, sg_count].
  Block *sgBody = &sgGeneric.getRegion().front();
  Value ldsLhs = sgBody->insertArgument(0u, ldsLhsType, loc);
  Value ldsRhs = sgBody->insertArgument(1u, ldsRhsType, loc);
  sgGeneric.setNumLeadingArgs(2);

  rewriter.setInsertionPointToStart(sgBody);

  // Get subgroup ID from execute block args.
  // Block arg layout: [ldsLhs, ldsRhs, destRef, sgId, sgCount].
  Value sgId = sgBody->getArgument(3);

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

  Value laneId = laneBody->getArgument(0);
  Value laneCount = laneBody->getArgument(1);

  // Compute per-thread output tile offsets.
  // Subgroups tile the M dimension, lanes tile the N dimension.
  // mOffset = sgId * (workgroupM / sgCount) + (laneId / nTilesPerSg) * mmaM
  // nOffset = (laneId % nTilesPerSg) * mmaN
  //
  // For simplicity, compute a linear thread ID and map to 2D tile grid:
  //   threadId = sgId * laneCount + laneId
  //   numNTiles = workgroupN / mmaN
  //   tileRow = threadId / numNTiles
  //   tileCol = threadId % numNTiles
  //   mOffset = tileRow * mmaM
  //   nOffset = tileCol * mmaN
  Value threadId = arith::AddIOp::create(
      rewriter, loc,
      arith::MulIOp::create(rewriter, loc, sgId, laneCount), laneId);

  int64_t numNTiles = workgroupN / config.mmaN;
  Value numNTilesVal =
      arith::ConstantIndexOp::create(rewriter, loc, numNTiles);
  Value mmaMVal = arith::ConstantIndexOp::create(rewriter, loc, config.mmaM);
  Value mmaNVal = arith::ConstantIndexOp::create(rewriter, loc, config.mmaN);

  Value tileRow = arith::DivUIOp::create(rewriter, loc, threadId, numNTilesVal);
  Value tileCol = arith::RemUIOp::create(rewriter, loc, threadId, numNTilesVal);
  Value mOffset = arith::MulIOp::create(rewriter, loc, tileRow, mmaMVal);
  Value nOffset = arith::MulIOp::create(rewriter, loc, tileCol, mmaNVal);

  // Initialize accumulator (zero vector at MMA tile granularity).
  auto accVecType =
      VectorType::get({config.mmaM, config.mmaN}, accElementType);
  Value accInit = arith::ConstantOp::create(
      rewriter, loc,
      DenseElementsAttr::get(accVecType,
                             rewriter.getZeroAttr(accElementType)));

  // K loop constants.
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value kStep = arith::ConstantIndexOp::create(rewriter, loc, config.kTile);
  Value kBound = arith::ConstantIndexOp::create(rewriter, loc, kDim);

  // Create scf.for K loop carrying the accumulator.
  // Pattern: load global->LDS, barrier, read quarters + compute, barrier.
  scf::ForOp kLoop = scf::ForOp::create(
      rewriter, loc, c0, kBound, kStep, /*initArgs=*/ValueRange{accInit},
      [&](OpBuilder &b, Location bodyLoc, Value kIV, ValueRange iterArgs) {
        Value acc = iterArgs[0];

        // Collaborative global-to-LDS load for this K tile.
        Value lhsTile = loadGlobalTile(b, bodyLoc, lhs, kIV,
                                       /*isLHS=*/true, workgroupM,
                                       workgroupN, config);
        Value rhsTile = loadGlobalTile(b, bodyLoc, rhs, kIV,
                                       /*isLHS=*/false, workgroupM,
                                       workgroupN, config);
        writeToLDS(b, bodyLoc, lhsTile, ldsLhs);
        writeToLDS(b, bodyLoc, rhsTile, ldsRhs);
        PCF::BarrierOp::create(b, bodyLoc, sgScope);

        // Per-thread: read quarters from LDS and accumulate.
        for (int64_t q = 0, e = config.numQuarters; q < e; ++q) {
          Value lhsQ = readQuarterFromLDS(b, bodyLoc, ldsLhs, /*isLHS=*/true,
                                          q, mOffset, nOffset, config,
                                          inputElementType);
          Value rhsQ = readQuarterFromLDS(b, bodyLoc, ldsRhs, /*isLHS=*/false,
                                          q, mOffset, nOffset, config,
                                          inputElementType);
          acc = generateComputePhase(b, bodyLoc, lhsQ, rhsQ, acc);
        }

        // Barrier before next iteration's LDS write.
        PCF::BarrierOp::create(b, bodyLoc, sgScope);

        scf::YieldOp::create(b, bodyLoc, ValueRange{acc});
      });

  // Write the final accumulator back to the output sref at thread offset.
  // destRef is after the 2 leading args: [ldsLhs, ldsRhs, destRef, ...].
  Value destRef = sgBody->getArgument(2);
  Value kLoopResult = kLoop.getResult(0);
  SmallVector<OpFoldResult> writeOffsets = {OpFoldResult(mOffset),
                                            OpFoldResult(nOffset)};
  SmallVector<OpFoldResult> writeSizes = {rewriter.getIndexAttr(config.mmaM),
                                          rewriter.getIndexAttr(config.mmaN)};
  SmallVector<OpFoldResult> writeStrides = {rewriter.getIndexAttr(1),
                                            rewriter.getIndexAttr(1)};
  PCF::WriteSliceOp::create(rewriter, loc, kLoopResult, destRef, writeOffsets,
                            writeSizes, writeStrides);

  // Terminate lane body with pcf.return.
  PCF::ReturnOp::create(rewriter, loc);

  // Terminate subgroup body with pcf.return.
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
