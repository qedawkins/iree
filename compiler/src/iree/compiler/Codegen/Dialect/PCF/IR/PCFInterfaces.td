// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_PCF_INTERFACES
#define IREE_CODEGEN_DIALECT_PCF_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE::PCF::ScopeAttr
//===----------------------------------------------------------------------===//

def PCF_ScopeAttr : AttrInterface<"ScopeAttr"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::PCF";

  let summary = [{Defines parallel execution scope.}];
  let description = [{
    Attributes implementing this interface are tied to ops and types to
    indicate scope. Loops carrying this scope are backed by physical resources
    described by the attribute, and tokens with a scope are visible to all
    workers within that scope. This enables tying a synchronizing operation
    to the parallelism level being synchronized when within multi-level parallel
    nests.

    In addition to the above, this interface is responsible for providing scope
    specific implementations of various primitives, such as worker ID and global
    barriers.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the worker IDs for this level of parallelism based on the
        requested basis. If the basis is empty, a linear ID is returned. No
        outer bound is implied by the basis, meaning the outer most returned
        ID (IDs[0]) is always bounded by the physical number of threads divided
        by the number of basis elements, and this method will return |basis| + 1
        values.
      }],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"getWorkerIDs",
      /*args=*/(ins
        "OpBuilder&":$builder,
        "ArrayRef<OpFoldResult>":$basis)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Creates a barrier synchronizing all workers at this scope.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"addBarrier",
      /*args=*/(ins "OpBuilder&":$builder),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return failure();
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::PCF::LoopOpInterface
//===----------------------------------------------------------------------===//

def PCF_ParallelOp : OpInterface<"ParallelOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::PCF";

  let description = [{
    TBD.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the scope for this parallel op.
      }],
      /*retTy=*/"PCF::ScopeAttr",
      /*methodName=*/"getScope",
      /*args=*/(ins)
    >,
  ];
}

#endif  // IREE_CODEGEN_DIALECT_PCF_INTERFACES
