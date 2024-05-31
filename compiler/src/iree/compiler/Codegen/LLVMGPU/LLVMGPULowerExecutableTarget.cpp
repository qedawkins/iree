// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvmgpu-lower-executable-target"

namespace mlir::iree_compiler {

namespace {
/// Lowers an hal.executable.variant operation to scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code
/// - then convert to NVVM/ROCDL dialect.
/// This should be merged with the equivalent pass in LinalgToLLVM. Fo
/// simplicity it is currently a separate pass.
class LLVMGPULowerExecutableTargetPass
    : public LLVMGPULowerExecutableTargetBase<
          LLVMGPULowerExecutableTargetPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<IREE::HAL::HALDialect,
                IREE::GPU::IREEGPUDialect,
                IREE::LinalgExt::IREELinalgExtDialect,
                IREE::VectorExt::IREEVectorExtDialect,
                linalg::LinalgDialect,
                gpu::GPUDialect,
                nvgpu::NVGPUDialect,
                pdl::PDLDialect,
                pdl_interp::PDLInterpDialect,
                scf::SCFDialect,
                tensor::TensorDialect,
                transform::TransformDialect,
                vector::VectorDialect>();
    // clang-format on
  }

  LLVMGPULowerExecutableTargetPass() = default;
  LLVMGPULowerExecutableTargetPass(
      const LLVMGPULowerExecutableTargetPass &pass) {}

  void runOnOperation() override;
};

static LLVMGPUPipelineOptions
getPipelineOptions(FunctionOpInterface funcOp,
                   IREE::Codegen::TranslationInfoAttr translationInfo) {
  LLVMGPUPipelineOptions pipelineOptions = {};
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);

  LLVM_DEBUG(llvm::dbgs() << "Translation Info: " << translationInfo << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Target Attr: " << targetAttr << "\n");

  if (DictionaryAttr config = translationInfo.getConfiguration()) {
    if (config.contains(LLVMGPUAttrNames::kNoReduceSharedMemoryBankConflicts))
      pipelineOptions.enableReduceSharedMemoryBankConflicts = false;
    if (config.contains(LLVMGPUAttrNames::kNoReorderWorkgroups))
      pipelineOptions.enableReorderWorkgroups = false;
  }

  pipelineOptions.enableUkernels = targetAttr && hasUkernel(targetAttr);

  LLVM_DEBUG(llvm::dbgs() << "LLVMGPU Pipeline Options: " << pipelineOptions
                          << "\n");
  return pipelineOptions;
}
} // namespace

void LLVMGPULowerExecutableTargetPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo)
    return;

  std::optional<OpPassManager> maybePipeline =
      getFunctionOpInterfacePassManager(funcOp);
  if (!maybePipeline) {
    funcOp.emitOpError(
        "unhandled function-like container during executable lowering");
    return signalPassFailure();
  }
  OpPassManager &pipeline = maybePipeline.value();

  LLVMGPUPipelineOptions pipelineOptions =
      getPipelineOptions(funcOp, translationInfo);

  switch (translationInfo.getDispatchLoweringPassPipeline()) {
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDefault:
    addGPUDefaultPassPipeline(pipeline, pipelineOptions);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUBaseLowering:
    addGPUBaseLoweringPassPipeline(pipeline);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute:
    addGPUSimpleDistributePassPipeline(pipeline);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorize:
    addGPUVectorizationPassPipeline(pipeline);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUWinogradVectorize:
    addGPUWinogradVectorizePassPipeline(pipeline);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt:
    addGPUMatmulSimtPassPipeline(pipeline, pipelineOptions);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulTensorCore: {
    FailureOr<int64_t> maybeDepth =
        getSoftwarePipelineDepth(translationInfo.getConfiguration());
    if (failed(maybeDepth)) {
      funcOp.emitOpError(
          "invalid matmul configuration without software pipelining config");
      return signalPassFailure();
    }
    addGPUMatmulTensorCorePassPipeline(pipeline, pipelineOptions, *maybeDepth);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      LLVMGPUMatmulTensorCoreMmaSync: {
    FailureOr<int64_t> maybeDepth =
        getSoftwarePipelineDepth(translationInfo.getConfiguration());
    if (failed(maybeDepth)) {
      funcOp.emitOpError(
          "invalid matmul configuration without software pipelining config");
      return signalPassFailure();
    }
    addGPUMatmulTensorCoreMmaSyncPassPipeline(pipeline, pipelineOptions,
                                              *maybeDepth);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTransposeSharedMem:
    addGPUTransposePassPipeline(pipeline, pipelineOptions);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorDistribute:
    addGPUVectorDistributePassPipeline(pipeline, pipelineOptions,
                                       /*usePadToModelSharedMemcpy=*/false);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::
      LLVMGPUPadAndVectorDistribute:
    addGPUVectorDistributePassPipeline(pipeline, pipelineOptions,
                                       /*usePadToModelSharedMemcpy=*/true);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUWarpReduction:
    addGPUWarpReductionPassPipeline(pipeline);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUPackUnPack:
    addGPUPackUnPackPasses(pipeline);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse:
    addGPUTileAndFusePassPipeline(pipeline);
    break;
  // no pipeline specified, nothing to do.
  case IREE::Codegen::DispatchLoweringPassPipeline::None:
    return;
  default:
    funcOp.emitOpError("unsupported pipeline on GPU target.");
    return signalPassFailure();
  }

  if (failed(runPipeline(pipeline, funcOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPULowerExecutableTargetPass() {
  return std::make_unique<LLVMGPULowerExecutableTargetPass>();
}

} // namespace mlir::iree_compiler
