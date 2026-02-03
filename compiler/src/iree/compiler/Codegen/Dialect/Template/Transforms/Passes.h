// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Template {

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/Dialect/Template/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Dialect/Template/Transforms/Passes.h.inc"

} // namespace mlir::iree_compiler::IREE::Template

namespace mlir::iree_compiler {

/// Registration for all Template transform passes.
void registerTemplateTransformsPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_TRANSFORMS_PASSES_H_
