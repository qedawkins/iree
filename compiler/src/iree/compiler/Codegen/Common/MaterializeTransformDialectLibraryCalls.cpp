// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-materialize-library-calls"

namespace mlir {
namespace iree_compiler {

// TODO(qedawkins): Change to a pass option.
llvm::cl::opt<std::string> clCodegenTransformDialectTestName(
    "iree-codegen-test-transform-dialect-strategy",
    llvm::cl::desc(
        "Broadcasts the given transform dialect strategy specification to all"
        "dispatches. Supports two modes; a path to the MLIR file containing a"
        "transform dialect specification to apply, and a symbol reference to"
        "load from a library of transform specs (@library_call)"),
    llvm::cl::init(""));

namespace {

static void createIncludeTransformStrategy(func::FuncOp entryPoint,
                                           SymbolRefAttr strategySymbol) {
  MLIRContext *ctx = entryPoint.getContext();
  Location loc = entryPoint.getLoc();
  OpBuilder b(ctx);
  b.setInsertionPointAfter(entryPoint);
  auto topLevelTransformModule = b.create<ModuleOp>(loc);
  topLevelTransformModule->setAttr(
      transform::TransformDialect::kWithNamedSequenceAttrName, b.getUnitAttr());
  Region &topLevelTransformRegion = topLevelTransformModule.getBodyRegion();
  b.setInsertionPointToStart(&topLevelTransformRegion.front());
  auto anyOpType = transform::AnyOpType::get(b.getContext());

  // Create the internal named sequence op to be linked against.
  auto funcTypeAttr =
      TypeAttr::get(FunctionType::get(ctx, anyOpType, TypeRange{}));
  auto symVisibility = StringAttr::get(ctx, "public");
  // Pessimistically assume the handle is consumed as we don't yet know the
  // contents of the strategy.
  auto consumedName =
      StringAttr::get(ctx, transform::TransformDialect::kArgConsumedAttrName);
  auto argAttrs =
      ArrayAttr::get(ctx, ArrayRef<Attribute>{b.getDictionaryAttr(
                              b.getNamedAttr(consumedName, b.getUnitAttr()))});
  auto resAttrs = ArrayAttr::get(ctx, ArrayRef<Attribute>{});
  auto namedSequence = b.create<transform::NamedSequenceOp>(
      loc, TypeRange{}, strategySymbol.getRootReference(), funcTypeAttr,
      symVisibility, argAttrs, resAttrs);
  (void)namedSequence;

  // Create the include for the named sequence with the expectation that the
  // external definition will be linked in later.
  auto propMode = transform::FailurePropagationMode::Propagate;
  auto sequence = b.create<transform::SequenceOp>(
      loc, TypeRange{}, transform::FailurePropagationMode::Propagate, anyOpType,
      [&](OpBuilder &b, Location loc, Value variantH) {
        b.create<transform::IncludeOp>(loc, TypeRange{}, strategySymbol,
                                       propMode, variantH);
        b.create<transform::YieldOp>(loc);
      });
  (void)sequence;
}


struct MaterializeTransformDialectLibraryCallsPass
    : public MaterializeTransformDialectLibraryCallsBase<MaterializeTransformDialectLibraryCallsPass> {
  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp moduleOp = variantOp.getInnerModule();
    llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);

    std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      auto exportOp = exportOps.lookup(funcOp.getName());

      /// First, apply all user configs.
      funcOp.walk([&](Operation *op)) {
        if (auto compilationInfo = getCompilationInfo(op)) {
          setUserConfig(funcOp, op, compilationInfo);
        }
      }

      if (IREE::Codegen::TranslationInfoAttr exportedTranslationInfo =
              getTranslationInfo(exportOp)) {
        if (translationInfo) {
          if (exportedTranslationInfo != translationInfo.value()) {
            moduleOp.emitOpError(
                "unhandled compilation of entry point functions with different "
                "translation info");
          }
        } else {
          translationInfo = exportedTranslationInfo;
        }
      }
    }

    /// We only need to resolve symbols for transform dialect based strategies.
    if (!translationInfo || translationInfo.value().getDispatchLoweringPassPipeline() == IREE::CodegenDispatchLoweringPassPipeline::TransformDialectCodegen) {
      return;
    }

    SymbolRefAttr libraryFunc = translationInfo.
  }
};

} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createMaterializeTransformDialectLibraryCallsPass() {
  return std::make_unique<MaterializeTransformDialectLibraryCallsPass>();
}

} // namespace iree_compiler
} // namespace mlir
