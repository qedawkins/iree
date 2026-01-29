// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_TEMPLATE_DIALECT
#define IREE_CODEGEN_DIALECT_TEMPLATE_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Template dialect
//===----------------------------------------------------------------------===//

def Template_Dialect : Dialect {
  let name = "template";
  let cppNamespace = "::mlir::iree_compiler::IREE::Template";

  let summary = [{
    A dialect for template-like IR structures with progressive specialization.
  }];
  let description = [{
    The Template dialect represents template-like IR structures designed to work
    in three stages:

    1. **Template definition**: Representing functions with templated types
       (`!template.type<N>`) and unimplemented blocks (`template.unimplemented`).

    2. **Progressive filling**: Filling in blocks of callees based on the
       semantics of the caller, progressively specializing abstract templates.

    3. **Full instantiation**: Cloning into fully specialized instances where
       all template types are concrete and all blocks are implemented.

    The goal is to enable the most complex individual op/DAG rewrites to happen
    iteratively rather than in a single monolithic transformation.

    Example workflow:
    ```mlir
    // Stage 1: Template definition with abstract blocks
    template.func @matmul_template
    implementations {
    ^load_a:
      template.unimplemented -> !template.type<0>
    ^compute(%a: !template.type<0>, %b: !template.type<1>):
      template.unimplemented -> !template.type<2>
    }
    main { ... }

    // Stage 2: Caller instantiates with concrete types and implementations
    %result = template.call @matmul_template
        types(!template.type<0> = tensor<4x4xf16>, ...)
        implementations {
          ^load_a: %tile = ... ; template.return %tile
        }

    // Stage 3: Fully specialized instance (after inlining/cloning)
    %result = template.instance implementations {
    ^load_a:
      %tile = vector.load ... : vector<4x4xf16>
      template.return %tile : vector<4x4xf16>
    ^compute(%a: vector<4x4xf16>, %b: vector<4x8xf16>):
      %c = vector.contract ... : vector<4x8xf32>
      template.return %c : vector<4x8xf32>
    } main { ... }
    ```
  }];

  let dependentDialects = [];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base template dialect op classes
//===----------------------------------------------------------------------===//

class Template_Op<string mnemonic, list<Trait> traits = []>
    : Op<Template_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Template types
//===----------------------------------------------------------------------===//

def Template_TypeType : TypeDef<Template_Dialect, "Type"> {
  let mnemonic = "type";

  let summary = [{A template type parameter reference.}];
  let description = [{
    A reference to a type parameter in a template. The `id` identifies which
    type parameter this refers to. During template instantiation, this type
    is replaced with the concrete type provided at the callsite.

    Example:
    ```mlir
    // In a template definition
    ^bb0(%arg: !template.type<0>):
      ...

    // At instantiation, !template.type<0> might become tensor<4x4xf32>
    ```
  }];

  let parameters = (ins "int64_t":$id);
  let assemblyFormat = "`<` $id `>`";
}

#endif  // IREE_CODEGEN_DIALECT_TEMPLATE_DIALECT
