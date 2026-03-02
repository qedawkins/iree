// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassUtils.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::iree_compiler {

std::optional<OpPassManager>
getFunctionOpInterfacePassManager(FunctionOpInterface interfaceOp) {
  return TypeSwitch<Operation *, std::optional<OpPassManager>>(
             interfaceOp.getOperation())
      .Case<func::FuncOp, IREE::Util::FuncOp>(
          [&](auto funcOp) { return OpPassManager(funcOp.getOperationName()); })
      .Default([&](Operation *op) { return std::nullopt; });
}

std::function<bool(Operation *)> hasPipelineAttrInterface() {
  return [](Operation *op) -> bool {
    auto funcOp = dyn_cast<FunctionOpInterface>(op);
    if (!funcOp) {
      return false;
    }
    IREE::Codegen::TranslationInfoAttr translationInfo =
        getTranslationInfo(funcOp);
    if (!translationInfo) {
      return false;
    }
    return isa<IREE::Codegen::PipelineAttrInterface>(
        translationInfo.getPassPipeline());
  };
}

} // namespace mlir::iree_compiler
