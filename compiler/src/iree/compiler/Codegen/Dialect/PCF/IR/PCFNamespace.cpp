// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFNamespace.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"

namespace mlir::iree_compiler::IREE::PCF {

std::optional<OpFoldResult> resolveNamespacedSymbol(Operation *from,
                                                    NamespacedSymbolAttr sym) {
  ArrayRef<StringAttr> path = sym.getPath();
  assert(!path.empty() && "symbol path must not be empty");

  // Collect ancestor namespace ops (innermost first).
  SmallVector<NamespaceOpInterface> ancestors;
  Operation *current = from->getParentOp();
  while (current) {
    if (NamespaceOpInterface ns = dyn_cast<NamespaceOpInterface>(current)) {
      ancestors.push_back(ns);
    }
    current = current->getParentOp();
  }

  if (ancestors.empty()) {
    from->emitError("no enclosing namespace found for symbol '") << sym << "'";
    return std::nullopt;
  }

  StringAttr leaf = path.back();

  if (path.size() == 1) {
    // Leaf-only: scan innermost to outermost.
    for (NamespaceOpInterface ns : ancestors) {
      for (auto [defSym, defVal] : ns.getDefinedSymbols()) {
        NamespacedSymbolAttr defNsSym = cast<NamespacedSymbolAttr>(defSym);
        if (defNsSym.getLeaf() == leaf) {
          return defVal;
        }
      }
    }
    from->emitError("failed to resolve leaf symbol '")
        << leaf.getValue() << "'";
    return std::nullopt;
  }

  // Qualified path: find first matching namespace for seg0.
  StringAttr seg0 = path.front();
  int64_t outerIdx = -1;
  for (int64_t i = 0, e = static_cast<int64_t>(ancestors.size()); i < e; ++i) {
    std::optional<StringAttr> nsName = ancestors[i].getNamespaceName();
    if (nsName && *nsName == seg0) {
      outerIdx = i;
      break;
    }
  }

  if (outerIdx < 0) {
    from->emitError("no namespace named '")
        << seg0.getValue() << "' found in ancestor chain";
    return std::nullopt;
  }

  // Walk strictly contiguous inward from outerIdx, consuming intermediate
  // namespace segments.
  int64_t currentIdx = outerIdx;
  for (int64_t segI = 1, numNsSegs = static_cast<int64_t>(path.size()) - 1;
       segI < numNsSegs; ++segI) {
    int64_t nextIdx = currentIdx - 1;
    if (nextIdx < 0) {
      from->emitError("namespace path segment '")
          << path[segI].getValue() << "' has no matching inner namespace";
      return std::nullopt;
    }
    std::optional<StringAttr> nextName = ancestors[nextIdx].getNamespaceName();
    if (!nextName || *nextName != path[segI]) {
      from->emitError("expected namespace '")
          << path[segI].getValue() << "' at ancestor position but found "
          << (nextName ? ("'" + nextName->getValue() + "'").str()
                       : std::string("anonymous namespace"));
      return std::nullopt;
    }
    currentIdx = nextIdx;
  }

  // Look up the leaf in the namespace at currentIdx.
  for (auto [defSym, defVal] : ancestors[currentIdx].getDefinedSymbols()) {
    NamespacedSymbolAttr defNsSym = cast<NamespacedSymbolAttr>(defSym);
    if (defNsSym.getLeaf() == leaf) {
      return defVal;
    }
  }

  from->emitError("symbol '") << leaf.getValue() << "' not found in namespace";
  return std::nullopt;
}

} // namespace mlir::iree_compiler::IREE::PCF
