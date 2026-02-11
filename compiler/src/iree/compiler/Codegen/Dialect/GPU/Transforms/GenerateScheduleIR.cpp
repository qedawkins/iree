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
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFDialect.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
// Schedule IR Generation Helpers
//===----------------------------------------------------------------------===//

/// Create a pcf.barrier at the given scope.
static void emitBarrier(OpBuilder &builder, Location loc,
                        PCF::ScopeAttrInterface scope) {
  PCF::BarrierOp::create(builder, loc, scope);
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
  // Stub implementation. The pass is registered and can be invoked, but
  // does not yet transform the IR. Full implementation will:
  //
  // 1. Find contraction ops (linalg.generic with matmul semantics).
  // 2. Derive ScheduleConfig from target and tile sizes.
  // 3. Generate the outer pcf.generic nest (subgroup + lane scopes).
  // 4. Generate pcf.alloc for LDS buffers.
  // 5. Generate the K-loop (scf.for) with 8-phase schedule.
  // 6. Generate phase-specific IR (global loads, LDS ops, WMMAs, barriers).
  //
  // For now, this is a no-op pass that demonstrates the infrastructure.
}

} // namespace mlir::iree_compiler::IREE::GPU
