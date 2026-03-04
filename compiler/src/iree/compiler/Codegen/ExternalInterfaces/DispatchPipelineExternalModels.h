// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_DISPATCHPIPELINEEXTERNALMODELS_H_
#define IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_DISPATCHPIPELINEEXTERNALMODELS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::Codegen {

/// Callback type for per-backend dispatch pipeline builders.
/// Returns success if the pipeline was handled; failure if it should be
/// tried by another backend.
using DispatchPipelineBuilder = llvm::function_ref<LogicalResult(
    DispatchLoweringPassPipeline pipeline,
    IREE::HAL::ExecutableTargetAttr target, OpPassManager &pm)>;

/// Registers a backend-specific dispatch pipeline builder.
/// Backends call this during their pass registration to make their
/// pipelines available through PipelineAttrInterface on
/// DispatchLoweringPassPipelineAttr.
void registerDispatchPipelineBuilder(
    llvm::StringRef name,
    LogicalResult (*builder)(DispatchLoweringPassPipeline,
                             IREE::HAL::ExecutableTargetAttr,
                             OpPassManager &));

/// Registers the ExternalModel that makes DispatchLoweringPassPipelineAttr
/// implement PipelineAttrInterface, dispatching to registered builders.
void registerDispatchPipelineExternalModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler::IREE::Codegen

#endif // IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_DISPATCHPIPELINEEXTERNALMODELS_H_
