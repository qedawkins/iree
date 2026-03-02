// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "iree-codegen-lower-dispatch-using-pipeline-attr"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LOWERDISPATCHUSINGPIPELINEATTRPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
/// Function-level pass that reads the function's TranslationInfoAttr,
/// materializes configuration, and dynamically builds+runs the lowering
/// pipeline via PipelineAttrInterface::buildPipeline().
class LowerDispatchUsingPipelineAttrPass final
    : public impl::LowerDispatchUsingPipelineAttrPassBase<
          LowerDispatchUsingPipelineAttrPass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    // This pass dynamically constructs and runs backend-specific pipelines,
    // so it must pre-register the union of dialects any backend might load.
    // Only common MLIR dialects and IREE dialects available in Common/ are
    // registered here. Backend-specific dialects (SPIRV, AMDGPU, NVGPU) are
    // registered by the backend's pipeline setup or tool initialization.
    // clang-format off
    registry.insert<
        IREE::Codegen::IREECodegenDialect,
        IREE::GPU::IREEGPUDialect,
        IREE::HAL::HALDialect,
        IREE::LinalgExt::IREELinalgExtDialect,
        IREE::VectorExt::IREEVectorExtDialect,
        affine::AffineDialect,
        bufferization::BufferizationDialect,
        gpu::GPUDialect,
        linalg::LinalgDialect,
        LLVM::LLVMDialect,
        memref::MemRefDialect,
        scf::SCFDialect,
        tensor::TensorDialect,
        transform::TransformDialect,
        vector::VectorDialect>();
    // clang-format on
  }

  void runOnOperation() override;
};
} // namespace

void LowerDispatchUsingPipelineAttrPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo) {
    return;
  }

  Attribute pipelineAttr = translationInfo.getPassPipeline();
  auto pipeline = dyn_cast<IREE::Codegen::PipelineAttrInterface>(pipelineAttr);
  if (!pipeline) {
    funcOp.emitOpError(
        "pass pipeline attribute does not implement PipelineAttrInterface");
    return signalPassFailure();
  }

  // Pre-compute target-dependent configuration into the pipeline attr.
  // This allows buildPipeline() to be independent of the function operation.
  Attribute materializedAttr = pipeline.materializeConfiguration(funcOp);
  auto materializedPipeline =
      dyn_cast<IREE::Codegen::PipelineAttrInterface>(materializedAttr);
  if (!materializedPipeline) {
    funcOp.emitOpError("materializeConfiguration returned invalid attribute");
    return signalPassFailure();
  }

  OpPassManager pm(funcOp->getName());

  if (failed(materializedPipeline.buildPipeline(pm))) {
    funcOp.emitOpError("failed to build pass pipeline");
    return signalPassFailure();
  }

  LDBG() << "Using pass pipeline for " << funcOp.getName() << ":";
  LLVM_DEBUG(pm.dump());

  if (failed(runPipeline(pm, funcOp))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
