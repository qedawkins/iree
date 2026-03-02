// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/LLVMCPUPipelineExternalModels.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/CodegenOptions.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {
namespace {

/// Returns a new lowering config with distribution tile sizes set to zeros.
static IREE::CPU::LoweringConfigAttr
getConfigWithZeroDistributionTiles(IREE::CPU::LoweringConfigAttr config) {
  using TilingLevel = IREE::CPU::TilingLevel;
  MLIRContext *ctx = config.getContext();
  SmallVector<NamedAttribute> items;
  for (int i : IREE::CPU::getTilingLevelsAsInts()) {
    if (!config.hasTilingLevel(i)) {
      continue;
    }
    auto level = static_cast<TilingLevel>(i);
    if (level != TilingLevel::DistributionTiles) {
      items.emplace_back(IREE::CPU::getTilingLevelName(level),
                         config.getTilingLevelAttr(i));
      continue;
    }
    // Reset distribution tiles to zeros.
    SmallVector<int64_t> origSizes = config.getWorkgroupTileSizes();
    SmallVector<int64_t> zeroSizes(origSizes.size(), 0);
    auto newLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
        ctx, zeroSizes, /*interchange=*/{}, /*scalableFlags=*/{});
    items.emplace_back(IREE::CPU::getTilingLevelName(level), newLevel);
  }
  return IREE::CPU::LoweringConfigAttr::get(ctx, items);
}

/// Serializes LLVMCPUPipelineOptions into a DictionaryAttr for embedding
/// in the pipeline attr's configuration.
static DictionaryAttr
serializeLLVMCPUPipelineOptions(MLIRContext *ctx,
                                const LLVMCPUPipelineOptions &opts) {
  Builder b(ctx);
  SmallVector<NamedAttribute> items;
  items.emplace_back(b.getStringAttr("decompose_pack_unpack_ops"),
                     b.getBoolAttr(opts.decomposePackUnPackOps));
  items.emplace_back(b.getStringAttr("enable_aarch64_i8mm"),
                     b.getBoolAttr(opts.enableAArch64I8mm));
  items.emplace_back(b.getStringAttr("enable_aarch64_sme"),
                     b.getBoolAttr(opts.enableAArch64SME));
  items.emplace_back(b.getStringAttr("enable_peeling"),
                     b.getBoolAttr(opts.enablePeeling));
  items.emplace_back(b.getStringAttr("enable_vector_masking"),
                     b.getBoolAttr(opts.enableVectorMasking));
  items.emplace_back(b.getStringAttr("lower_to_avx2"),
                     b.getBoolAttr(opts.lowerToAVX2));
  items.emplace_back(b.getStringAttr("use_configured_vector_sizes"),
                     b.getBoolAttr(opts.useConfiguredVectorSizes));
  return b.getDictionaryAttr(items);
}

/// Deserializes LLVMCPUPipelineOptions from a DictionaryAttr.
static LLVMCPUPipelineOptions
deserializeLLVMCPUPipelineOptions(DictionaryAttr config) {
  LLVMCPUPipelineOptions opts;
  // CPU codegen options always come from global flags.
  opts.cpuOpts = CPUCodegenOptions::FromFlags::get();
  if (!config) {
    return opts;
  }

  auto getBool = [&](StringRef key, bool defaultVal) -> bool {
    if (auto attr = config.getAs<BoolAttr>(key)) {
      return attr.getValue();
    }
    return defaultVal;
  };

  opts.decomposePackUnPackOps =
      getBool("decompose_pack_unpack_ops", opts.decomposePackUnPackOps);
  opts.enableAArch64I8mm =
      getBool("enable_aarch64_i8mm", opts.enableAArch64I8mm);
  opts.enableAArch64SME = getBool("enable_aarch64_sme", opts.enableAArch64SME);
  opts.enablePeeling = getBool("enable_peeling", opts.enablePeeling);
  opts.enableVectorMasking =
      getBool("enable_vector_masking", opts.enableVectorMasking);
  opts.lowerToAVX2 = getBool("lower_to_avx2", opts.lowerToAVX2);
  opts.useConfiguredVectorSizes =
      getBool("use_configured_vector_sizes", opts.useConfiguredVectorSizes);
  return opts;
}

/// Derives LLVMCPUPipelineOptions from a function's target information.
/// Also applies side effects like resetting distribution tile sizes.
static LLVMCPUPipelineOptions
getLLVMCPUPipelineOptionsFromFunc(FunctionOpInterface funcOp) {
  LLVMCPUPipelineOptions pipelineOpts;
  // Get session-level CPU options from global flags.
  pipelineOpts.cpuOpts = CPUCodegenOptions::FromFlags::get();

  IREE::HAL::ExecutableTargetAttr target =
      IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!target) {
    return pipelineOpts;
  }
  DictionaryAttr targetConfig = target.getConfiguration();

  // If distribution is disabled, reset distribution tile sizes to zeros in the
  // lowering config so the IR reflects the actual behavior.
  if (pipelineOpts.cpuOpts.disableDistribution) {
    funcOp.walk([&](Operation *op) {
      auto config = getLoweringConfig<IREE::CPU::LoweringConfigAttr>(op);
      if (config && config.hasWorkgroupTilingLevel()) {
        setLoweringConfig(op, getConfigWithZeroDistributionTiles(config));
      }
    });
  }

  if (isX86(targetConfig) || isRISCV(targetConfig)) {
    pipelineOpts.useConfiguredVectorSizes = false;
  }
  pipelineOpts.decomposePackUnPackOps =
      isOptEnabled(funcOp, getEnableDecompositionStr());
  pipelineOpts.lowerToAVX2 = hasAVX2Feature(targetConfig);
  pipelineOpts.enableVectorMasking =
      isX86(targetConfig) || isRISCV(targetConfig) ||
      (isAArch64(targetConfig) && hasAnySVEFeature(targetConfig));

  // TODO(#16956): The decomposition of attention op leads to complex control
  // flow, which leads to non-trivial stack allocation. Enforcing masking for
  // targets that do not support native vector masking enables the
  // functionality, though the performance may be suboptimal.
  if (isAArch64(targetConfig)) {
    bool hasAttention = false;
    funcOp.walk([&](IREE::LinalgExt::AttentionOp) {
      hasAttention = true;
      return WalkResult::interrupt();
    });
    if (hasAttention) {
      pipelineOpts.enableVectorMasking = true;
    }
  }
  pipelineOpts.enableAArch64SME = isAArch64(targetConfig) &&
                                  hasAnySVEFeature(targetConfig) &&
                                  hasSMEFeature(targetConfig);
  pipelineOpts.enableAArch64I8mm =
      isAArch64(targetConfig) && hasI8mmFeature(targetConfig);
  pipelineOpts.enablePeeling = isOptEnabled(funcOp, getEnableLoopPeelingStr());
  return pipelineOpts;
}

/// Finds the root lowering config among the compute ops in the function.
static IREE::Codegen::LoweringConfigAttrInterface
getRootLoweringConfig(FunctionOpInterface funcOp) {
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  for (Operation *op : computeOps) {
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
        getLoweringConfig(op);
    if (loweringConfig && loweringConfig.hasWorkgroupTilingLevel()) {
      return loweringConfig;
    }
  }
  return nullptr;
}

struct LLVMCPUPipelineExternalModel final
    : IREE::Codegen::PipelineAttrInterface::ExternalModel<
          LLVMCPUPipelineExternalModel,
          IREE::Codegen::LLVMCPUDispatchLoweringPipelineAttr> {
  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<IREE::Codegen::LLVMCPUDispatchLoweringPipelineAttr>(attr)
        .getConfiguration();
  }

  Attribute materializeConfiguration(Attribute attr, Operation *funcOp) const {
    auto pipelineAttr =
        cast<IREE::Codegen::LLVMCPUDispatchLoweringPipelineAttr>(attr);
    auto funcIfaceOp = cast<FunctionOpInterface>(funcOp);
    MLIRContext *ctx = attr.getContext();

    // Compute all target-dependent options from funcOp.
    LLVMCPUPipelineOptions pipelineOpts =
        getLLVMCPUPipelineOptionsFromFunc(funcIfaceOp);
    DictionaryAttr serializedOpts =
        serializeLLVMCPUPipelineOptions(ctx, pipelineOpts);

    // For CPUDoubleTilingExpert, also store the root lowering config.
    Builder b(ctx);
    SmallVector<NamedAttribute> configItems(serializedOpts.getValue().begin(),
                                            serializedOpts.getValue().end());
    if (pipelineAttr.getPipeline() ==
        IREE::Codegen::LLVMCPUPipeline::CPUDoubleTilingExpert) {
      IREE::Codegen::LoweringConfigAttrInterface rootConfig =
          getRootLoweringConfig(funcIfaceOp);
      if (rootConfig) {
        configItems.emplace_back(b.getStringAttr("root_lowering_config"),
                                 cast<Attribute>(rootConfig));
      }
    }

    DictionaryAttr augmentedConfig = b.getDictionaryAttr(configItems);
    return IREE::Codegen::LLVMCPUDispatchLoweringPipelineAttr::get(
        ctx, pipelineAttr.getPipeline(), augmentedConfig);
  }

  LogicalResult buildPipeline(Attribute attr, OpPassManager &pm) const {
    auto pipelineAttr =
        cast<IREE::Codegen::LLVMCPUDispatchLoweringPipelineAttr>(attr);
    DictionaryAttr config = pipelineAttr.getConfiguration();
    LLVMCPUPipelineOptions pipelineOpts =
        deserializeLLVMCPUPipelineOptions(config);

    using Pipeline = IREE::Codegen::LLVMCPUPipeline;
    switch (pipelineAttr.getPipeline()) {
    case Pipeline::CPUDefault: {
      addCPUDefaultPassPipeline(pm, pipelineOpts);
      break;
    }
    case Pipeline::CPUDoubleTilingExpert: {
      // Read root lowering config from the pre-computed configuration.
      IREE::Codegen::LoweringConfigAttrInterface loweringConfig;
      if (config) {
        if (auto rootConfigAttr = config.get("root_lowering_config")) {
          loweringConfig = dyn_cast<IREE::Codegen::LoweringConfigAttrInterface>(
              rootConfigAttr);
        }
      }
      addMultiTilingExpertPassPipeline(pm, loweringConfig, pipelineOpts);
      break;
    }
    case Pipeline::CPUConvTileAndDecomposeExpert: {
      addConvTileAndDecomposeExpertPassPipeline(pm, pipelineOpts);
      break;
    }
    case Pipeline::Mmt4dTilingExpert: {
      addMmt4dTilingExpertPassPipeline(pm, pipelineOpts);
      break;
    }
    case Pipeline::CPUBufferOpsTileAndVectorize: {
      addCPUBufferOpsTileAndVectorizePipeline(pm, pipelineOpts);
      break;
    }
    case Pipeline::CPUDataTiling: {
      addCPUDataTilingPipeline(pm, pipelineOpts);
      break;
    }
    case Pipeline::CPULinalgExtTileAndVectorize: {
      addCPULinalgExtTileAndVectorizePipeline(pm, pipelineOpts);
      break;
    }
    }
    return success();
  }
};

} // namespace

void registerLLVMCPUPipelineExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Codegen::IREECodegenDialect *dialect) {
        IREE::Codegen::LLVMCPUDispatchLoweringPipelineAttr::attachInterface<
            LLVMCPUPipelineExternalModel>(*ctx);
      });
}

} // namespace mlir::iree_compiler
