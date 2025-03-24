// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LOWERINGCONFIGINTERPRETERPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class LoweringConfigInterpreterPass final
    : public impl::LoweringConfigInterpreterPassBase<
          LoweringConfigInterpreterPass> {
public:
  using impl::LoweringConfigInterpreterPassBase<
      LoweringConfigInterpreterPass>::LoweringConfigInterpreterPassBase;
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    auto *symbolTableOp = SymbolTable::getNearestSymbolTable(rootOp);
    MLIRContext *ctx = &getContext();
    auto dialect = ctx->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
    std::optional<ModuleOp> originalSpec =
        dialect->getLoneTransformLibraryModule();

    SmallVector<std::pair<Operation *, transform::NamedSequenceOp>>
        targetStrategyPairs;
    rootOp->walk([&](Operation *op) {
      IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
          getLoweringConfig(op);
      if (!loweringConfig) {
        return;
      }

      std::optional<StringRef> maybeSymName =
          loweringConfig.getLoweringStrategy();
      if (!maybeSymName) {
        return;
      }

      auto strategy = dyn_cast_or_null<transform::NamedSequenceOp>(
          SymbolTable::lookupSymbolIn(symbolTableOp, *maybeSymName));
      if (!strategy) {
        if (originalSpec) {
          strategy = dyn_cast_or_null<transform::NamedSequenceOp>(
              SymbolTable::lookupSymbolIn(originalSpec.value(), *maybeSymName));
        }
      }

      if (!strategy) {
        return;
      }

      targetStrategyPairs.push_back(std::make_pair(op, strategy));
    });

    transform::TransformOptions options;
    options.enableExpensiveChecks(true);
    for (auto [target, strategy] : targetStrategyPairs) {
      if (failed(transform::applyTransformNamedSequence(
              target, strategy, /*transformModule=*/nullptr, options))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
