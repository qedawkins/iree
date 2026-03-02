// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_PASSUTILS_H_
#define IREE_COMPILER_CODEGEN_COMMON_PASSUTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

/// Pass manager nesting for `FunctionOpInterface` ops.
using FunctionLikeNest = MultiOpNest<func::FuncOp>;

/// Helper method to get a pass manager nested at `FunctionOpInterface`.
std::optional<OpPassManager>
getFunctionOpInterfacePassManager(FunctionOpInterface funcOp);

//===----------------------------------------------------------------------===//
// Condition helpers for MultiPipelineNest
//===----------------------------------------------------------------------===//

/// Returns a condition that matches any function whose TranslationInfoAttr
/// has a pass pipeline implementing PipelineAttrInterface. Used as a catch-all
/// fallback for dynamic pipeline dispatch in MultiPipelineNest.
std::function<bool(Operation *)> hasPipelineAttrInterface();

/// Returns a condition that matches a function whose TranslationInfoAttr has
/// a specific pipeline attr type with a specific enum value. Used for static
/// pipelines in MultiPipelineNest.
///
/// Usage:
///   nest.nestIf(matchesPipeline<LLVMGPUDispatchLoweringPipelineAttr>(
///       LLVMGPUPipeline::LLVMGPUBaseLowering));
template <typename PipelineAttrT, typename EnumT>
std::function<bool(Operation *)> matchesPipeline(EnumT pipeline) {
  return [pipeline](Operation *op) -> bool {
    auto funcOp = dyn_cast<FunctionOpInterface>(op);
    if (!funcOp) {
      return false;
    }
    IREE::Codegen::TranslationInfoAttr translationInfo =
        getTranslationInfo(funcOp);
    if (!translationInfo) {
      return false;
    }
    auto pipelineAttr =
        dyn_cast<PipelineAttrT>(translationInfo.getPassPipeline());
    if (!pipelineAttr) {
      return false;
    }
    return pipelineAttr.getPipeline() == pipeline;
  };
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_PASSUTILS_H_
