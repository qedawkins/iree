// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void PCFDialect::registerOperations() {
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::PCF

//===----------------------------------------------------------------------===//
// TableGen definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp.inc" // IWYU pragma: keep
