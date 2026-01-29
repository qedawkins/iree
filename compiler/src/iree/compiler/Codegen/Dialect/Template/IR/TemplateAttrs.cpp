// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateAttrs.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateAttrs.cpp.inc" // IWYU pragma: keep

namespace mlir::iree_compiler::IREE::Template {

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void TemplateDialect::registerAttributes() {
  // No attributes defined yet.
}

} // namespace mlir::iree_compiler::IREE::Template
