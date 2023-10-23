// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static SmallVector<Value> filterTensorValues(ValueRange &&range) {
  SmallVector<Value> result;
  for (auto value : range) {
    if (llvm::isa<TensorType>(value.getType()))
      result.push_back(value);
  }
  return result;
}

class InjectDispatchTracingPass
    : public InjectDispatchTracingBase<InjectDispatchTracingPass> {
public:
  InjectDispatchTracingPass() = default;

  void runOnOperation() override {
    auto funcOp = getOperation();
    for (auto dispatchOp : funcOp.getFunctionBody().getOps<DispatchOp>()) {
      // TODO: Figure out a way to accurately trace which entry point
      // gets executed.
      SymbolRefAttr firstEntry = *dispatchOp.getEntryPointRefs().begin();
      std::string entryPointName =
          firstEntry.getRootReference().getValue().str();
      for (FlatSymbolRefAttr nestedRef : firstEntry.getNestedReferences()) {
        entryPointName = (entryPointName + "::" + nestedRef.getValue()).str();
      }

      // Input tensors:
      OpBuilder builder(dispatchOp);
      builder.create<IREE::Flow::TensorTraceOp>(
          dispatchOp.getLoc(),
          builder.getStringAttr(entryPointName + " inputs"),
          filterTensorValues(dispatchOp.getArguments()));

      // Output tensors:
      builder.setInsertionPointAfter(dispatchOp);
      builder.create<IREE::Flow::TensorTraceOp>(
          dispatchOp.getLoc(),
          builder.getStringAttr(entryPointName + " outputs"),
          filterTensorValues(dispatchOp.getResults()));
    }
  }
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createInjectDispatchTracingPass() {
  return std::make_unique<InjectDispatchTracingPass>();
}

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
