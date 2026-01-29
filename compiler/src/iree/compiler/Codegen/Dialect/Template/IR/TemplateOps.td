// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_TEMPLATE_OPS
#define IREE_CODEGEN_DIALECT_TEMPLATE_OPS

include "iree/compiler/Codegen/Dialect/Template/IR/TemplateBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// template.instance
//===----------------------------------------------------------------------===//

def Template_InstanceOp : Template_Op<"instance", [
    RecursiveMemoryEffects
]> {
  let summary = [{A fully specialized template instance.}];
  let description = [{
    Encapsulates a computation with optional implementation blocks and a main
    entry region. This is the fully specialized form after template instantiation.

    The **main** region is the entry point when control flow reaches this op.
    Operations within the main region may branch into the implementation blocks
    using `template.branch`.

    The **implementations** region contains indexed blocks (0..N) that can be
    referenced from the main region. Each block is terminated with
    `template.return`.

    Example:
    ```mlir
    %result = template.instance -> tensor<4x8xf32> {
      %c0 = arith.constant 0 : index
      %loaded = template.branch 0(%c0) : (index) -> (vector<8xf32>)
      %computed = template.branch 1(%loaded, %loaded) : (...) -> (vector<8xf32>)
      template.return %computed : vector<8xf32>
    } {
    ^load(%offset: index):
      %tile = vector.load %src[%offset] : memref<256xf32>, vector<8xf32>
      template.return %tile : vector<8xf32>
    ^compute(%a: vector<8xf32>, %b: vector<8xf32>):
      %c = arith.addf %a, %b : vector<8xf32>
      template.return %c : vector<8xf32>
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region
    SizedRegion<1>:$main,
    AnyRegion:$implementations
  );

  let assemblyFormat = [{
    (`ins` `(` $inputs^ `:` type($inputs) `)`)? (`->` type($results)^)?
    $main ($implementations^)? attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// template.func
//===----------------------------------------------------------------------===//

def Template_FuncOp : Template_Op<"func", [
    Symbol,
    IsolatedFromAbove
]> {
  let summary = [{A template function that may have abstract blocks.}];
  let description = [{
    A symbol op representing a template function. Implementation blocks may be
    fully implemented, partially implemented, or abstract (terminated with
    `template.unimplemented`).

    During progressive specialization, callers provide concrete implementations
    for abstract blocks and concrete types for `!template.type<N>` references.

    Example:
    ```mlir
    template.func @matmul_template(%m: index, %n: index, %k: index) {
      // Loop structure that branches into implementation blocks.
      ...
      template.return %final : !template.type<2>
    } {
    ^load_a(%idx: index):
      // Abstract - caller must provide implementation.
      template.unimplemented -> !template.type<0>
    ^load_b(%idx: index):
      // Abstract - caller must provide implementation.
      template.unimplemented -> !template.type<1>
    ^compute(%a: !template.type<0>, %b: !template.type<1>, %acc: !template.type<2>):
      // Concrete implementation provided.
      %result = linalg.matmul ins(%a, %b) outs(%acc)
      template.return %result : !template.type<2>
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);
  let regions = (region
    SizedRegion<1>:$main,
    AnyRegion:$implementations
  );

  let assemblyFormat = [{
    $sym_name
    custom<TemplateFuncMain>($main)
    ($implementations^)?
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// template.branch
//===----------------------------------------------------------------------===//

def Template_BranchOp : Template_Op<"branch", []> {
  let summary = [{Branch into an implementation block.}];
  let description = [{
    Branches into block N of an enclosing `template.instance` or `template.func`
    implementations region, passing arguments and receiving results.

    The branch transfers control to the target block, and when that block
    executes `template.return`, control returns here with the returned values.

    Example:
    ```mlir
    // Branch to block 0 with one argument, receiving one result.
    %result = template.branch 0(%offset) : (index) -> (tensor<4x4xf32>)
    ```
  }];

  let arguments = (ins I64Attr:$block_index, Variadic<AnyType>:$arguments);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $block_index (`(` $arguments^ `)`)? (`:` functional-type($arguments, $results)^)?
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// template.return
//===----------------------------------------------------------------------===//

def Template_ReturnOp : Template_Op<"return", [
    Pure,
    ReturnLike,
    Terminator
]> {
  let summary = [{Return from a block or region.}];
  let description = [{
    Terminates a block and returns values. Used both in implementation blocks
    (returning to the `template.branch` callsite) and in the main region
    (returning the final results of the instance/func).

    Example:
    ```mlir
    // In an implementation block - returns to branch callsite.
    ^compute(%a: tensor<4x4xf32>):
      %result = ...
      template.return %result : tensor<4x4xf32>

    // In main region - returns instance results.
    main {
      ...
      template.return %final : tensor<8x8xf32>
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ValueRange{});
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// template.unimplemented
//===----------------------------------------------------------------------===//

def Template_UnimplementedOp : Template_Op<"unimplemented", [
    Pure,
    Terminator
]> {
  let summary = [{Marks an implementation block as abstract.}];
  let description = [{
    Terminates an implementation block to indicate it is abstract and must be
    provided by the caller during template instantiation.

    This op must be the only operation in its block (no predecessors).

    Example:
    ```mlir
    ^load_tile(%offset: index):
      // Caller must provide concrete load implementation.
      template.unimplemented -> tensor<4x4xf32>
    ```
  }];

  let arguments = (ins);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict (`->` type($results)^)?
  }];

  let hasVerifier = 1;
}

#endif  // IREE_CODEGEN_DIALECT_TEMPLATE_OPS
