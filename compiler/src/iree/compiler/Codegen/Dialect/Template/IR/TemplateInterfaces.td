// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_TEMPLATE_INTERFACES
#define IREE_CODEGEN_DIALECT_TEMPLATE_INTERFACES

include "mlir/IR/OpBase.td"

def TemplateCallOpInterface : OpInterface<"TemplateCallOpInterface"> {
  let description = [{
    Interface for operations that call template functions and can be resolved
    into template.instance ops by the ConcretizeTemplateCallsPass.

    Implementors provide:
    1. The symbol of the template.func being called.
    2. Type bindings for each template.type<N> parameter.
    3. Logic to populate implementation blocks with concrete IR.
  }];

  let cppNamespace = "::mlir::iree_compiler::IREE::Template";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the symbol of the template.func being called.",
      /*retTy=*/"::mlir::FlatSymbolRefAttr",
      /*methodName=*/"getCalledSymbol",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the type bindings for template.type<N> in order of id.
        Each element is a list of types (supporting 1:N expansion).
        For example, if template.type<0> expands to [tensor<4xf32>, tensor<8xf32>],
        the first element of the returned vector would be that list.
      }],
      /*retTy=*/"::llvm::SmallVector<::llvm::SmallVector<::mlir::Type>>",
      /*methodName=*/"getTemplateTypes",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the operands to pass to the template.func call.
        For template.call, this returns all operands.
        For ops like process_inner_tile, this returns only the operands
        that correspond to template.func arguments (e.g., outputs and
        reduction bounds, not inputs or other metadata operands).
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::Value>",
      /*methodName=*/"getCallOperands",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::llvm::SmallVector<::mlir::Value>(
            $_op->getOperands().begin(), $_op->getOperands().end());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Populates the provided implementation blocks with generated IR.
        For template.call, this clones from its implementations region.
        For other ops like process_inner_tile, this generates IR based on
        the op's semantics.

        The blocks are provided in order corresponding to the unimplemented
        blocks in the template.func being called. The implementation should:
        1. Set up block arguments as needed.
        2. Generate/clone the implementation IR.
        3. Add a template.return terminator.

        Returns failure if the blocks cannot be populated.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"inlineImplementationBlocks",
      /*args=*/(ins
        "::mlir::OpBuilder &":$builder,
        "::llvm::ArrayRef<::mlir::Block *>":$blocksToPopulate
      )
    >,
  ];
}

#endif // IREE_CODEGEN_DIALECT_TEMPLATE_INTERFACES
