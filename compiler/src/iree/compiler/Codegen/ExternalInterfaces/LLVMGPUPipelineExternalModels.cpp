// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/LLVMGPUPipelineExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {
namespace {

/// Reads GPU pipeline options from the pipeline attr's configuration dict.
static IREE::GPU::GPUPipelineOptions
getGPUPipelineOptionsFromConfig(DictionaryAttr config) {
  IREE::GPU::GPUPipelineOptions pipelineOptions = {};
  if (!config) {
    return pipelineOptions;
  }

  // Read GPUPipelineOptionsAttr from the config dict.
  std::optional<NamedAttribute> maybePipelineOptionsAttr =
      config.getNamed(IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName());
  if (maybePipelineOptionsAttr.has_value()) {
    auto pipelineOptionsAttr = cast<IREE::GPU::GPUPipelineOptionsAttr>(
        maybePipelineOptionsAttr->getValue());
    std::optional<int64_t> prefetchNumStages =
        pipelineOptionsAttr.getPrefetchNumStages();
    if (prefetchNumStages) {
      pipelineOptions.prefetchNumStages = *prefetchNumStages;
    }
    BoolAttr noReduceBankConflicts =
        pipelineOptionsAttr.getNoReduceSharedMemoryBankConflicts();
    if (noReduceBankConflicts) {
      pipelineOptions.enableReduceSharedMemoryBankConflicts =
          !noReduceBankConflicts.getValue();
    }
    BoolAttr useIgemmConvolution = pipelineOptionsAttr.getUseIgemmConvolution();
    if (useIgemmConvolution) {
      pipelineOptions.useIgemmConvolution = useIgemmConvolution.getValue();
    }
    IREE::GPU::ReorderWorkgroupsStrategyAttr reorderWorkgroupsStrategy =
        pipelineOptionsAttr.getReorderWorkgroupsStrategy();
    if (reorderWorkgroupsStrategy) {
      pipelineOptions.reorderStrategy = reorderWorkgroupsStrategy.getValue();
    }
  }

  // Read enable_ukernels from pre-computed config.
  if (auto enableAttr = config.getAs<BoolAttr>("enable_ukernels")) {
    pipelineOptions.enableUkernels = enableAttr.getValue();
  }

  return pipelineOptions;
}

struct LLVMGPUPipelineExternalModel final
    : IREE::Codegen::PipelineAttrInterface::ExternalModel<
          LLVMGPUPipelineExternalModel,
          IREE::Codegen::LLVMGPUDispatchLoweringPipelineAttr> {
  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<IREE::Codegen::LLVMGPUDispatchLoweringPipelineAttr>(attr)
        .getConfiguration();
  }

  Attribute materializeConfiguration(Attribute attr, Operation *funcOp) const {
    auto pipelineAttr =
        cast<IREE::Codegen::LLVMGPUDispatchLoweringPipelineAttr>(attr);
    auto funcIfaceOp = cast<FunctionOpInterface>(funcOp);

    // Gather target-dependent options.
    IREE::HAL::ExecutableTargetAttr target =
        IREE::HAL::ExecutableTargetAttr::lookup(funcIfaceOp);
    bool forROCDL = target && target.getBackend().getValue() == "rocm";
    bool enableUKernels = target && hasUkernel(target.getConfiguration());

    // Start with existing TranslationInfo configuration (has
    // GPUPipelineOptionsAttr).
    IREE::Codegen::TranslationInfoAttr translationInfo =
        getTranslationInfo(funcIfaceOp);
    DictionaryAttr existingConfig =
        translationInfo ? translationInfo.getConfiguration() : DictionaryAttr();

    // Build augmented config with target-specific options.
    MLIRContext *ctx = attr.getContext();
    Builder b(ctx);
    SmallVector<NamedAttribute> items;
    if (existingConfig) {
      llvm::append_range(items, existingConfig.getValue());
    }
    items.emplace_back(b.getStringAttr("for_rocdl"), b.getBoolAttr(forROCDL));
    items.emplace_back(b.getStringAttr("enable_ukernels"),
                       b.getBoolAttr(enableUKernels));
    DictionaryAttr augmentedConfig = b.getDictionaryAttr(items);

    return IREE::Codegen::LLVMGPUDispatchLoweringPipelineAttr::get(
        ctx, pipelineAttr.getPipeline(), augmentedConfig);
  }

  LogicalResult buildPipeline(Attribute attr, OpPassManager &pm) const {
    auto pipelineAttr =
        cast<IREE::Codegen::LLVMGPUDispatchLoweringPipelineAttr>(attr);
    DictionaryAttr config = pipelineAttr.getConfiguration();

    // Read pre-computed options from the pipeline attr's configuration.
    bool forROCDL = false;
    if (config) {
      if (auto forROCDLAttr = config.getAs<BoolAttr>("for_rocdl")) {
        forROCDL = forROCDLAttr.getValue();
      }
    }
    IREE::GPU::GPUPipelineOptions pipelineOptions =
        getGPUPipelineOptionsFromConfig(config);

    using Pipeline = IREE::Codegen::LLVMGPUPipeline;
    switch (pipelineAttr.getPipeline()) {
    case Pipeline::LLVMGPUDefault: {
      addGPUDefaultPassPipeline(pm, pipelineOptions);
      break;
    }
    case Pipeline::LLVMGPUBaseLowering: {
      addGPUBaseLoweringPassPipeline(pm);
      break;
    }
    case Pipeline::LLVMGPUDistribute: {
      addGPUSimpleDistributePassPipeline(pm);
      break;
    }
    case Pipeline::LLVMGPUVectorize: {
      addGPUVectorizationPassPipeline(pm);
      break;
    }
    case Pipeline::LLVMGPUWinogradVectorize: {
      addGPUWinogradVectorizePassPipeline(pm);
      break;
    }
    case Pipeline::LLVMGPUVectorDistribute: {
      addGPUVectorDistributePassPipeline(pm, pipelineOptions, forROCDL);
      break;
    }
    case Pipeline::LLVMGPUTileAndFuse: {
      addGPUTileAndFusePassPipeline(pm, pipelineOptions, forROCDL);
      break;
    }
    }
    return success();
  }
};

} // namespace

void registerLLVMGPUPipelineExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Codegen::IREECodegenDialect *dialect) {
        IREE::Codegen::LLVMGPUDispatchLoweringPipelineAttr::attachInterface<
            LLVMGPUPipelineExternalModel>(*ctx);
      });
}

} // namespace mlir::iree_compiler
