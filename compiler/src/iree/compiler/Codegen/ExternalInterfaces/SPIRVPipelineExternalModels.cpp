// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/SPIRVPipelineExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {
namespace {

struct SPIRVPipelineExternalModel final
    : IREE::Codegen::PipelineAttrInterface::ExternalModel<
          SPIRVPipelineExternalModel,
          IREE::Codegen::SPIRVDispatchLoweringPipelineAttr> {
  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<IREE::Codegen::SPIRVDispatchLoweringPipelineAttr>(attr)
        .getConfiguration();
  }

  Attribute materializeConfiguration(Attribute attr, Operation *funcOp) const {
    auto pipelineAttr =
        cast<IREE::Codegen::SPIRVDispatchLoweringPipelineAttr>(attr);
    auto funcIfaceOp = cast<FunctionOpInterface>(funcOp);
    MLIRContext *ctx = attr.getContext();

    // Read software pipelining config from TranslationInfo's configuration.
    IREE::Codegen::TranslationInfoAttr translationInfo =
        getTranslationInfo(funcIfaceOp);
    DictionaryAttr translationConfig =
        translationInfo ? translationInfo.getConfiguration() : DictionaryAttr();

    // If there is a translation config, embed it in the pipeline attr.
    if (translationConfig) {
      return IREE::Codegen::SPIRVDispatchLoweringPipelineAttr::get(
          ctx, pipelineAttr.getPipeline(), translationConfig);
    }
    return attr;
  }

  LogicalResult buildPipeline(Attribute attr, OpPassManager &pm) const {
    auto pipelineAttr =
        cast<IREE::Codegen::SPIRVDispatchLoweringPipelineAttr>(attr);

    // Read software pipelining config from the pipeline attr's configuration.
    DictionaryAttr config = pipelineAttr.getConfiguration();

    using Pipeline = IREE::Codegen::SPIRVPipeline;
    switch (pipelineAttr.getPipeline()) {
    case Pipeline::SPIRVBaseLowering: {
      addSPIRVBaseLoweringPassPipeline(pm);
      break;
    }
    case Pipeline::SPIRVBaseDistribute: {
      addSPIRVBaseDistributePassPipeline(pm);
      break;
    }
    case Pipeline::SPIRVBaseVectorize: {
      addSPIRVBaseVectorizePassPipeline(pm);
      break;
    }
    case Pipeline::SPIRVSubgroupReduce: {
      addSPIRVSubgroupReducePassPipeline(pm);
      break;
    }
    case Pipeline::SPIRVCooperativeMatrixVectorize: {
      FailureOr<int64_t> maybeDepth = getSoftwarePipelineDepth(config);
      FailureOr<int64_t> maybeStage = getSoftwarePipelineStoreStage(config);
      if (failed(maybeDepth) || failed(maybeStage)) {
        return failure();
      }
      addSPIRVCooperativeMatrixVectorizePassPipeline(pm, *maybeDepth,
                                                     *maybeStage);
      break;
    }
    case Pipeline::SPIRVMatmulPromoteVectorize: {
      FailureOr<int64_t> maybeDepth = getSoftwarePipelineDepth(config);
      FailureOr<int64_t> maybeStage = getSoftwarePipelineStoreStage(config);
      if (failed(maybeDepth) || failed(maybeStage)) {
        return failure();
      }
      addSPIRVMatmulPromoteVectorizePassPipeline(pm, *maybeDepth, *maybeStage);
      break;
    }
    case Pipeline::SPIRVWinogradVectorize: {
      addSPIRVWinogradVectorizePassPipeline(pm);
      break;
    }
    }
    return success();
  }
};

} // namespace

void registerSPIRVPipelineExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Codegen::IREECodegenDialect *dialect) {
        IREE::Codegen::SPIRVDispatchLoweringPipelineAttr::attachInterface<
            SPIRVPipelineExternalModel>(*ctx);
      });
}

} // namespace mlir::iree_compiler
