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
    return iree_compiler::detail::populatePDLModuleFromFileName(
        context, tmpPatterns, this->pdlModuleFileName);
  }

  MaterializeExternDispatchesPass(StringRef pdlModuleFileName = StringRef()) {
    this->pdlModuleFileName = pdlModuleFileName.str();
  }
  MaterializeExternDispatchesPass(const MaterializeExternDispatchesPass &pass) =
      default;
};
} // namespace

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {
std::unique_ptr<Pass>
createMaterializeExternDispatchesPass(std::string pdlModuleFileName) {
  return std::make_unique<MaterializeExternDispatchesPass>(pdlModuleFileName);
}
} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
