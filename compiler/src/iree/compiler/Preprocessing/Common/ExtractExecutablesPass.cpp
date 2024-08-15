// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_EXTRACTEXECUTABLESPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

namespace {

struct ExtractExecutablesPass
    : public iree_compiler::Preprocessing::impl::ExtractExecutablesPassBase<
          ExtractExecutablesPass> {
  void runOnOperation() override;
};
} // namespace

void ExtractExecutablesPass::runOnOperation() {
  auto module = getOperation();

  SmallVector<IREE::HAL::ExecutableOp> executables;
  SmallVector<func::FuncOp> functions;
  for (auto executable : module.getOps<IREE::HAL::ExecutableOp>()) {
    func::FuncOp function;
    WalkResult res = executable.walk([&](func::FuncOp func) {
      if (function) {
        return WalkResult::interrupt();
      }
      function = func;
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) {
      executable->emitError("unimplemented: multiple function executables");
      return signalPassFailure();
    }
    executables.push_back(executable);
    functions.push_back(function);
  }

  IRRewriter rewriter(module);
  for (auto [executable, function] : llvm::zip_equal(executables, functions)) {
    rewriter.setInsertionPoint(executable);
    SmallVector<Type> operandTypes;
    SmallVector<Value> replacementTargets;
    function.walk([&](IREE::Flow::DispatchTensorLoadOp load) {
      replacementTargets.push_back(load.getResult());
      operandTypes.push_back(load.getResult().getType());
    });

    SmallVector<Type> resultTypes;
    SmallVector<Value> returnVals;
    function.walk([&](IREE::Flow::DispatchTensorStoreOp store) {
      resultTypes.push_back(store.getValueType());
      returnVals.push_back(store.getValue());
      rewriter.eraseOp(store);
    });

    FunctionType funcType =
        FunctionType::get(rewriter.getContext(), operandTypes, resultTypes);
    auto funcOp = rewriter.create<func::FuncOp>(function.getLoc(),
                                                function.getName(), funcType);

    Block *entryBlock = funcOp.addEntryBlock();

    rewriter.inlineBlockBefore(&function.getFunctionBody().getBlocks().front(),
                               entryBlock, entryBlock->begin());

    for (auto [v, replacement] :
         llvm::zip_equal(replacementTargets, entryBlock->getArguments())) {
      rewriter.replaceAllUsesWith(v, replacement);
    }

    rewriter.eraseOp(entryBlock->getTerminator());
    rewriter.setInsertionPointToEnd(entryBlock);
    rewriter.create<func::ReturnOp>(function.getLoc(), returnVals);
    rewriter.eraseOp(executable);

    funcOp.walk(
        [&](IREE::Flow::DispatchTensorLoadOp load) { rewriter.eraseOp(load); });
    funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp binding) {
      rewriter.eraseOp(binding);
    });
    SmallVector<Operation *> opsToErase;
    funcOp.walk([&](IREE::HAL::InterfaceConstantLoadOp load) {
      SmallVector<Operation *> users(load->getUsers());
      opsToErase.append(users);
      opsToErase.push_back(load);
    });
    for (auto op : opsToErase) {
      rewriter.eraseOp(op);
    }
  }
}

} // namespace mlir::iree_compiler::Preprocessing
