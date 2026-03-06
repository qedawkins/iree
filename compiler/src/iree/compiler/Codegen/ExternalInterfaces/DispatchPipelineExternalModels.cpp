// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/DispatchPipelineExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// Dispatch pipeline builder registry.
//===----------------------------------------------------------------------===//

namespace {

struct BuilderEntry {
  llvm::StringRef name;
  LogicalResult (*builder)(DispatchLoweringPassPipeline,
                           IREE::HAL::ExecutableTargetAttr, OpPassManager &,
                           const CodegenPipelineOptions *);
};

/// Returns the global mutable list of registered builders.
/// Thread safety: registration happens during static initialization
/// (registerCodegen*Passes), before any concurrent use.
SmallVector<BuilderEntry> &getBuilderRegistry() {
  static SmallVector<BuilderEntry> registry;
  return registry;
}

} // namespace

void registerDispatchPipelineBuilder(
    llvm::StringRef name,
    LogicalResult (*builder)(DispatchLoweringPassPipeline,
                             IREE::HAL::ExecutableTargetAttr, OpPassManager &,
                             const CodegenPipelineOptions *)) {
  getBuilderRegistry().push_back({name, builder});
}

//===----------------------------------------------------------------------===//
// ExternalModel implementation.
//===----------------------------------------------------------------------===//

namespace {

struct DispatchLoweringPipelineExternalModel final
    : PipelineAttrInterface::ExternalModel<
          DispatchLoweringPipelineExternalModel,
          DispatchLoweringPassPipelineAttr> {
  LogicalResult buildPipeline(Attribute attr, OpPassManager &pm,
                              IREE::HAL::ExecutableTargetAttr target,
                              const CodegenPipelineOptions *options) const {
    auto enumAttr = cast<DispatchLoweringPassPipelineAttr>(attr);
    DispatchLoweringPassPipeline pipeline = enumAttr.getValue();

    // Pipeline::None means no codegen pipeline — return success with empty PM.
    if (pipeline == DispatchLoweringPassPipeline::None) {
      return success();
    }

    // Try each registered backend builder.
    for (const BuilderEntry &entry : getBuilderRegistry()) {
      if (succeeded(entry.builder(pipeline, target, pm, options))) {
        return success();
      }
    }

    // No backend handled this pipeline.
    return failure();
  }
};

} // namespace

void registerDispatchPipelineExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREECodegenDialect *dialect) {
        DispatchLoweringPassPipelineAttr::attachInterface<
            DispatchLoweringPipelineExternalModel>(*ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::Codegen
