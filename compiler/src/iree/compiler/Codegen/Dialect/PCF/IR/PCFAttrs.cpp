// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "llvm/ADT/StringExtras.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.cpp.inc" // IWYU pragma: keep

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void PCFDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

//===----------------------------------------------------------------------===//
// NamespacedSymbolAttr
//===----------------------------------------------------------------------===//

Attribute NamespacedSymbolAttr::parse(AsmParser &parser, Type type) {
  // Inside type/attribute angle brackets, the MLIR lexer treats '.' as part
  // of a keyword token. This lets us parse "tg.left" as a single keyword
  // and split on '.' ourselves.
  StringRef keyword;
  SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseKeyword(&keyword))) {
    parser.emitError(loc, "expected namespaced symbol keyword");
    return {};
  }

  SmallVector<StringAttr> path;
  SmallVector<StringRef> segments;
  keyword.split(segments, '.');
  for (StringRef seg : segments) {
    if (seg.empty()) {
      parser.emitError(loc, "empty segment in namespaced symbol");
      return {};
    }
    path.push_back(StringAttr::get(parser.getContext(), seg));
  }

  return NamespacedSymbolAttr::get(parser.getContext(), path);
}

void NamespacedSymbolAttr::print(AsmPrinter &printer) const {
  llvm::interleave(
      getPath(), printer.getStream(),
      [&](StringAttr seg) { printer << seg.getValue(); }, ".");
}

} // namespace mlir::iree_compiler::IREE::PCF
