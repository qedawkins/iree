// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFNAMESPACE_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFNAMESPACE_H_

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::PCF {

/// Resolves a namespaced symbol starting from the given operation by walking
/// the parent chain. Returns the symbol's definition as an OpFoldResult, or
/// nullopt if resolution fails (with a diagnostic emitted on `from`).
std::optional<OpFoldResult> resolveNamespacedSymbol(Operation *from,
                                                    NamespacedSymbolAttr sym);

} // namespace mlir::iree_compiler::IREE::PCF

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFNAMESPACE_H_
