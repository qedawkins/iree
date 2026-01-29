// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateTypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"

#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateTypes.cpp.inc" // IWYU pragma: keep

namespace mlir::iree_compiler::IREE::Template {

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void TemplateDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateTypes.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::Template
