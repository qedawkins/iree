// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the VMVX related Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_VMVX_PASSES_H_
#define IREE_COMPILER_CODEGEN_VMVX_PASSES_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//------------------------------------------------------------------------------
// VMVX Pass Pipelines
//------------------------------------------------------------------------------

/// Populates the passes to lower to tiled/distributed/bufferized ops,
/// suitable for library call dispatch and lowering to loops.
void addVMVXDefaultPassPipeline(OpPassManager &funcPassManager,
                                bool enableUKernels);

/// Builds a function-level pass pipeline for the given dispatch lowering
/// pipeline enum value. Returns failure if the pipeline is not a VMVX pipeline
/// or requires per-operation information not available at this level.
LogicalResult buildVMVXDispatchPassPipeline(
    IREE::Codegen::DispatchLoweringPassPipeline pipeline,
    IREE::HAL::ExecutableTargetAttr target, OpPassManager &pm);

//----------------------------------------------------------------------------//
// VMVX Linking Passes and Pipelines
//----------------------------------------------------------------------------//

/// Populates passes needed to link HAL executables across VMVX targets.
void buildVMVXLinkingPassPipeline(OpPassManager &variantPassManager);

//----------------------------------------------------------------------------//
// Register VMVX Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/VMVX/Passes.h.inc" // IWYU pragma: keep

void registerCodegenVMVXPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_VMVX_PASSES_H_
