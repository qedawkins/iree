// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFNamespace.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_TESTNAMESPACERESOLUTIONPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {
struct TestNamespaceResolutionPass final
    : impl::TestNamespaceResolutionPassBase<TestNamespaceResolutionPass> {
  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      // Resolve cluster IDs from operand/result types.
      auto tryResolve = [&](ClusterType ct) {
        NamespacedSymbolAttr id = ct.getId();
        std::optional<OpFoldResult> result = resolveNamespacedSymbol(op, id);
        if (result) {
          if (Attribute attr = dyn_cast<Attribute>(*result)) {
            op->emitRemark("resolved '") << id << "' to " << attr;
          } else {
            op->emitRemark("resolved '") << id << "' to a value";
          }
        }
        // Failures emit their own diagnostics via resolveNamespacedSymbol.
      };

      for (Type type : op->getOperandTypes()) {
        if (ClusterType ct = dyn_cast<ClusterType>(type)) {
          tryResolve(ct);
        }
      }
      for (Type type : op->getResultTypes()) {
        if (ClusterType ct = dyn_cast<ClusterType>(type)) {
          tryResolve(ct);
        }
      }

      // Also resolve explicit test attributes.
      if (StringAttr testAttr = op->getAttrOfType<StringAttr>("test.resolve")) {
        SmallVector<StringAttr> path;
        SmallVector<StringRef> segments;
        testAttr.getValue().split(segments, '.');
        for (StringRef seg : segments) {
          path.push_back(StringAttr::get(op->getContext(), seg));
        }
        NamespacedSymbolAttr testSym =
            NamespacedSymbolAttr::get(op->getContext(), path);
        std::optional<OpFoldResult> result =
            resolveNamespacedSymbol(op, testSym);
        if (result) {
          if (Attribute attr = dyn_cast<Attribute>(*result)) {
            op->emitRemark("resolved '") << testSym << "' to " << attr;
          } else {
            op->emitRemark("resolved '") << testSym << "' to a value";
          }
        }
      }
    });
  }
};
} // namespace
} // namespace mlir::iree_compiler::IREE::PCF
