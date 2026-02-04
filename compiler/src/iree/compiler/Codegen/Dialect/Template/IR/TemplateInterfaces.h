// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_IR_TEMPLATEINTERFACES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_IR_TEMPLATEINTERFACES_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::Template {

/// Helper to create unrealized_conversion_cast ops for type conversion at
/// block boundaries during template instantiation.
///
/// When inlining implementation blocks, template types need to be converted
/// to concrete types. This helper creates the necessary cast operations.
Value createTypeCast(OpBuilder &builder, Location loc, Type targetType,
                     ValueRange inputs);

/// Helper to create casts for multiple values.
SmallVector<Value> createTypeCasts(OpBuilder &builder, Location loc,
                                   TypeRange targetTypes, ValueRange inputs);

} // namespace mlir::iree_compiler::IREE::Template

// Include generated interface declarations.
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateInterfaces.h.inc"

#endif // IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_IR_TEMPLATEINTERFACES_H_
