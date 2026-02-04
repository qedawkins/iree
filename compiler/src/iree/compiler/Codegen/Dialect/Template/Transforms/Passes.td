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
    "::mlir::iree_compiler::IREE::Codegen::IREECodegenDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect",
  ];
}

def InlineTemplateInstancesPass :
    Pass<"iree-template-inline-instances"> {
  let summary = "Inline template.instance ops by expanding branch operations";
  let description = [{
    Inlines template.instance operations by:
    1. Finding all template.branch ops within the instance
    2. Inlining the corresponding implementation block at each branch site
    3. Inlining the main region content in place of the instance op

    Uses walkAndApplyPatterns with post-order traversal to ensure nested
    instances are inlined before their parents.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::Template::TemplateDialect",
  ];
}

#endif // IREE_CODEGEN_DIALECT_TEMPLATE_TRANSFORMS_PASSES
