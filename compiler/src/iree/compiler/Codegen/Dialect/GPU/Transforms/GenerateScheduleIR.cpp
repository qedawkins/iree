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
  int64_t numPhases = 8; // 4 compute + 4 memory phases.
};

//===----------------------------------------------------------------------===//
// Schedule IR Generation
//===----------------------------------------------------------------------===//

/// Generate the 8-phase K-loop body with barriers between phases.
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
/// Each phase is separated by a pcf.barrier.
///
/// Currently generates the structural skeleton with barriers only.
/// Phase-specific operations (loads, stores, WMMAs) are TODO.
static void generatePhaseBarriers(OpBuilder &builder, Location loc,
                                  PCF::ScopeAttrInterface barrierScope,
                                  int64_t numPhases) {
  for (int64_t i = 0; i < numPhases; ++i) {
    PCF::BarrierOp::create(builder, loc, barrierScope);
  }
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
///     pcf.return %dest_ref
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

  // Get K dimension from LHS shape (M*K).
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  int64_t kDim = lhsType.getDimSize(lhsType.getRank() - 1);
  if (kDim == ShapedType::kDynamic) {
    return contractionOp.emitError("dynamic K dimension not yet supported");
  }

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
  Value destRef = sgBody->getArgument(0);
  rewriter.setInsertionPointToStart(sgBody);

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

  // Create K loop constants.
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value kStep = arith::ConstantIndexOp::create(rewriter, loc, config.kTile);
  Value kBound = arith::ConstantIndexOp::create(rewriter, loc, kDim);

  // Create scf.for K loop with phase barriers in the body.
  scf::ForOp::create(
      rewriter, loc, c0, kBound, kStep, /*initArgs=*/ValueRange{},
      [&](OpBuilder &bodyBuilder, Location bodyLoc, Value /*iv*/,
          ValueRange /*iterArgs*/) {
        generatePhaseBarriers(bodyBuilder, bodyLoc, sgScope,
                              config.numPhases);
        scf::YieldOp::create(bodyBuilder, bodyLoc);
      });

  // Terminate lane body with pcf.return.
  PCF::ReturnOp::create(rewriter, loc);

  // Terminate subgroup body with pcf.return returning dest_ref.
  rewriter.setInsertionPointToEnd(sgBody);
  PCF::ReturnOp::create(rewriter, loc, ValueRange{destRef});

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
