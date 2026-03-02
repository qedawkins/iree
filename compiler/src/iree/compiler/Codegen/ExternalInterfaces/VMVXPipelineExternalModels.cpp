// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/VMVXPipelineExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"

namespace mlir::iree_compiler {
namespace {

struct VMVXPipelineExternalModel final
    : IREE::Codegen::PipelineAttrInterface::ExternalModel<
          VMVXPipelineExternalModel,
          IREE::Codegen::VMVXDispatchLoweringPipelineAttr> {
  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<IREE::Codegen::VMVXDispatchLoweringPipelineAttr>(attr)
        .getConfiguration();
  }

  LogicalResult buildPipeline(Attribute attr, OpPassManager &pm) const {
    auto pipelineAttr =
        cast<IREE::Codegen::VMVXDispatchLoweringPipelineAttr>(attr);

    // Read options from pre-computed configuration.
    DictionaryAttr config = pipelineAttr.getConfiguration();
    bool enableUKernels = false;
    if (config) {
      if (auto enableAttr = config.getAs<BoolAttr>("enable_ukernels")) {
        enableUKernels = enableAttr.getValue();
      }
    }

    using Pipeline = IREE::Codegen::VMVXPipeline;
    switch (pipelineAttr.getPipeline()) {
    case Pipeline::VMVXDefault: {
      addVMVXDefaultPassPipeline(pm, enableUKernels);
      break;
    }
    }
    return success();
  }
};

} // namespace

void registerVMVXPipelineExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Codegen::IREECodegenDialect *dialect) {
        IREE::Codegen::VMVXDispatchLoweringPipelineAttr::attachInterface<
            VMVXPipelineExternalModel>(*ctx);
      });
}

} // namespace mlir::iree_compiler
