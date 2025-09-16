// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_PCF_OPS
#define IREE_CODEGEN_DIALECT_PCF_OPS

include "iree/compiler/Codegen/Dialect/PCF/IR/PCFBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ParallelOps
//===----------------------------------------------------------------------===//

def OpGroupParallelOps : OpDocGroup {
  let summary = "Parallel execution ops";
  let description = "";
}

let opDocGroup = OpGroupParallelOps in {
} // OpGroupParallelOps

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

def PCF_YieldOp : PCF_Op<"yield", [
  // ParentOneOf<[]>,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = [{Yields results from a region.}];
  let description = [{
    The values returned are copied by-value.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = [{
    attr-dict
    $operands `:` type($operands)
  }];
}

def PCF_JoinYieldOp : PCF_Op<"join_yield", [
  // ParentOneOf<[
  //   "IREE::PCF::GenericOp",
  //   "IREE::PCF::ConcurrentOp",
  // ]>,
  Pure,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
  Terminator,
]> {
  let summary = [{Joins and yields values from a parallel execution region.}];
  let description = [{
    Joins workers on the specified tokens and yields from the parallel
    execution region.
  }];

  let arguments = (ins
    Variadic<PCF_AnyToken>:$tokens
  );

  let assemblyFormat = [{
    attr-dict ($tokens^ `:` type($tokens))?
  }];
}

#endif  // IREE_CODEGEN_DIALECT_PCF_OPS
