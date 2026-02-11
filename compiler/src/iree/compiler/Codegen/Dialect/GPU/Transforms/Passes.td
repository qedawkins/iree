// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def CombineBarrierRegionsPass :
    Pass<"iree-gpu-combine-barrier-regions", ""> {
  let summary = "Combines `iree_gpu.barrier_region` ops";
  let dependentDialects = ["::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"];
}

def DistributeInnerTiledToLanesPass :
    InterfacePass<"iree-gpu-distribute-inner-tiled-to-lanes", "mlir::FunctionOpInterface"> {
  let summary = "Distributes iree_codegen.inner_tiled ops to lanes";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def ExpandUndistributedInnerTilesPass :
    InterfacePass<"iree-gpu-expand-undistributed-inner-tiles", "mlir::FunctionOpInterface"> {
  let summary = "Expands the inner dimensions of iree_codegen.inner_tiled ops to match the thread layout";
  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
  let options = [
    Option<"expandInputs", "expand-inputs",
      "bool", /*default=*/"true",
      "Expand the inner dimensions for the input operands of the inner_tiled ops.">,
    Option<"expandOutputs", "expand-outputs",
      "bool", /*default=*/"true",
      "Expand the inner dimensions for the output operands and results of the inner_tiled ops.">,
  ];
}

def LowerIREEGPUOpsPass :
    InterfacePass<"iree-gpu-lower-ops", "mlir::FunctionOpInterface"> {
  let summary = "Post bufferization lowerings of iree_gpu ops before late lowerings";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
  ];
}

def UnrollToIntrinsicsPass :
    InterfacePass<"iree-gpu-unroll-to-intrinsics", "mlir::FunctionOpInterface"> {
  let summary = "Unrolls iree_gpu.multi_mma ops to their inner vector size.";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::vector::VectorDialect",
  ];
}

def VectorizeIREEGPUOpsPass :
    InterfacePass<"iree-gpu-vectorize-ops", "mlir::FunctionOpInterface"> {
  let summary = "Vectorizes then lowers a few iree_gpu ops before vectorization.";
  let dependentDialects = [
    "::mlir::vector::VectorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
  ];
}

def ConvertForallToGenericNestGPUPass :
    InterfacePass<"iree-gpu-convert-forall-to-generic-nest",
                  "mlir::FunctionOpInterface"> {
  let summary = "Converts scf.forall ops with GPU mapping to pcf.generic";
  let description = [{
    Converts scf.forall ops with gpu.thread mapping to nested pcf.generic ops
    using subgroup (outer) and lane (inner) scopes.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::PCF::PCFDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::scf::SCFDialect",
  ];
}

def GenerateScheduleIRPass :
    InterfacePass<"iree-gpu-generate-schedule-ir",
                  "mlir::FunctionOpInterface"> {
  let summary = "Generates structured schedule IR for GPU matmul.";
  let description = [{
    Takes a function containing a contraction op (linalg.generic with matmul
    semantics) and replaces it with a structured schedule using PCF ops,
    barriers, and MMA operations.

    The schedule follows the quarter-K pingpong pattern with configurable
    early-write optimization. It generates:
      - Outer pcf.generic at subgroup scope.
      - Inner pcf.generic at lane scope.
      - LDS allocation via pcf.alloc.
      - K-loop (scf.for) with phased computation:
        P1: Global load LHS + LDS read q0.
        P2: WMMA compute q0.
        P3: Global load RHS + LDS read q1.
        P4: WMMA compute q1.
        P5: LDS write LHS + LDS read q2.
        P6: WMMA compute q2 + LDS write RHS + LDS read q3.
        P7: WMMA compute q3.
        P8: Barrier + loop control.
      - Barriers (pcf.barrier) between phases.

    Configuration is derived from a ScheduleConfig, which can be computed
    by deriveScheduleConfig() from the cost model.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::PCF::PCFDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect",
  ];
}

#endif // IREE_CODEGEN_DIALECt_GPU_TRANSFORMS_PASSES
