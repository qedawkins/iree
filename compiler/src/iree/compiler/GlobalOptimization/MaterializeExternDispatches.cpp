// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/ExternBuildingUtils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/Utils/CustomPatternApplicatorPassBase.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class MaterializeExternDispatchesPass
    : public iree_compiler::PatternApplicatorPassBase<
          MaterializeExternDispatchesPass,
          iree_compiler::GlobalOptimization::
              MaterializeExternDispatchesPassBase> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::iree_compiler::IREE::HAL::HALDialect, pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect>();
  }

  LogicalResult initializePatterns(MLIRContext *context,
                                   RewritePatternSet &tmpPatterns) {
    iree_compiler::IREE::HAL::registerExternDispatchRewriteFunction(
        tmpPatterns.getPDLPatterns());
    for (auto fileName : this->pdlModuleFileNames) {
      if (failed(iree_compiler::detail::populatePDLModuleFromFileName(
          context, tmpPatterns, fileName))) {
        return failure();
      }
    }
    return success();
  }

  MaterializeExternDispatchesPass(ArrayRef<std::string> pdlModuleFileNames) {
    this->pdlModuleFileNames = pdlModuleFileNames;
  }
  MaterializeExternDispatchesPass(const MaterializeExternDispatchesPass &pass) =
      default;
};
} // namespace

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {
std::unique_ptr<Pass>
createMaterializeExternDispatchesPass(ArrayRef<std::string> pdlModuleFileNames) {
  return std::make_unique<MaterializeExternDispatchesPass>(pdlModuleFileNames);
}
} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
