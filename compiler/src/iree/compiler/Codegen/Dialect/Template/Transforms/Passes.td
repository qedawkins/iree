// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_TEMPLATE_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_TEMPLATE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ConcretizeTemplateCallsPass :
    Pass<"iree-template-concretize-calls", "ModuleOp"> {
  let summary = "Concretize template.call ops into template.instance ops";
  let description = [{
    Converts template.call operations into concrete template.instance
    operations by:
    1. Resolving type bindings (!template.type<N> -> concrete types)
    2. Cloning the template.func body
    3. Populating unimplemented blocks from call-provided implementations
    4. Running type conversion on the resulting instance

    Uses a worklist algorithm with recursion detection to handle
    nested template calls.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::Template::TemplateDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
  ];
}

#endif // IREE_CODEGEN_DIALECT_TEMPLATE_TRANSFORMS_PASSES
