// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_PCF_INTERFACES
#define IREE_CODEGEN_DIALECT_PCF_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/TilingInterface.td"

//===----------------------------------------------------------------------===//
// IREE::PCF::ScopeAttrInterface
//===----------------------------------------------------------------------===//

def PCF_ScopeAttrInterface : AttrInterface<"ScopeAttrInterface"> {
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
        Returns the maximum number of workers for this level of parallelism.
        `num_ids` indicates the number of id counts to return.

        If `num_ids` is less than the number of physical processor ids, the
        slowest varying id counts are linearized into the slowest varying
        returned value. If `num_ids` is greater than the number of actual ids,
        this returns `1` for the slowest varying id counts.

        Worker counts are ordered from fastest to slowest varying thread
        dimension. Concretely, `getWorkerCounts().front()` represents the
        maximum value of the fastest varying ID, while `.back()` is the
        slowest varying.
      }],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"getWorkerCounts",
      /*args=*/(ins
        "OpBuilder&":$builder,
        "Location":$loc,
        "int64_t":$num_ids)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the worker ids for this level of parallelism. `num_ids`
        indicates the number of processor ids to return.

        If `num_ids` is less than the number of physical processor ids, the
        slowest varying worker ids are linearized into the slowest varying
        returned value. If `num_ids` is greater than the number of actual ids,
        this returns `0` for the slowest varying ids.

        Worker ids are ordered from fastest to slowest varying thread
        dimension. Concretely, `getWorkerIds().front()` represents the
        the fastest varying ID, while `.back()` is the slowest varying.
      }],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"getWorkerIDs",
      /*args=*/(ins
        "OpBuilder&":$builder,
        "Location":$loc,
        "int64_t":$num_ids)
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
    InterfaceMethod<
      /*desc=*/[{
        Returns the memory space at which to allocate for this scope. Defaults
        to empty address space, however can be overridden to return failure if
        allocations are illegal at this scope.
      }],
      /*retTy=*/"FailureOr<Attribute>",
      /*methodName=*/"getAllocMemSpace",
      /*args=*/(ins "MLIRContext*":$context),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return Attribute();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the memory space at which to allocate for this scope. Defaults
        to empty address space, however can be overridden to return failure if
        allocations are illegal at this scope.
      }],
      /*retTy=*/"IntegerAttr",
      /*methodName=*/"getPreferredAllocAlignment",
      /*args=*/(ins "MLIRContext*":$context),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Builder b(context);
        return b.getI64IntegerAttr(16);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the number of physical processor IDs this scope natively
        supports. Iterating over fewer than the native number of IDs results
        in potentially duplicated execution over the unaccounted for
        dimension(s).

        For example, a workgroup scope typically returns 3 (x, y, z), while
        a sequential scope returns 1.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNativeNumProcessorIds",
      /*args=*/(ins)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::PCF::SyncScopeAttrInterface
//===----------------------------------------------------------------------===//

def PCF_SyncScopeAttrInterface : AttrInterface<"SyncScopeAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::PCF";

  let summary = [{Determines synchronization semantics.}];
  let description = [{
    Attributes implementing this interface are tied to shaped refs and define
    the synchronization semantics for that ref.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the expanded types for this attribute that are introduced when
        resolving pcf tokens.
      }],
      /*retTy=*/"TypeRange",
      /*methodName=*/"getConcreteTypes",
      /*args=*/(ins "MLIRContext*":$context)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Allocates the requisite data for this token. This must produce a range
        with types matching `getConcreteTypes`.
      }],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"allocate",
      /*args=*/(ins
        "OpBuilder&":$builder
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Default to nothing.
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Enqueues a write given the expanded types and the writing operation.
      }],
      /*retTy=*/"void",
      /*methodName=*/"enqueueWrite",
      /*args=*/(ins
        "OpBuilder&":$builder,
        "ValueRange":$sync_values,
        "Operation *":$writing_operation
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Default to nothing.
        return;
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::PCF::NamespaceOpInterface
//===----------------------------------------------------------------------===//

def PCF_NamespaceOpInterface : OpInterface<"NamespaceOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::PCF";

  let description = [{
    Defines private namespace boundaries for hierarchical symbols.

    Operations implementing this interface define a namespace containing
    privately-scoped symbols. Namespaces can be named (reachable by qualified
    path) or anonymous (symbols reachable only by leaf name).

    Symbols resolve upward through the parent chain -- never sideways to
    siblings. This is the key property distinguishing private namespaces
    from MLIR's standard symbol tables.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the namespace name, or nullopt for anonymous namespaces.
      }],
      /*retTy=*/"std::optional<StringAttr>",
      /*methodName=*/"getNamespaceName",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns all symbols defined in this namespace, each paired with its
        definition as an OpFoldResult (attribute or SSA value). The symbol
        attribute is a NamespacedSymbolAttr; callers should cast accordingly.
        (The return type uses Attribute rather than NamespacedSymbolAttr to
        avoid a circular include: PCFAttrs.h includes PCFInterfaces.h.)
      }],
      /*retTy=*/"SmallVector<std::pair<Attribute, OpFoldResult>>",
      /*methodName=*/"getDefinedSymbols",
      /*args=*/(ins)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::PCF::PCFTilingOpInterface
//===----------------------------------------------------------------------===//

def PCF_TilingOpInterface : OpInterface<"PCFTilingOpInterface",
    [TilingInterface]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::PCF";

  let description = [{
    Interface for distributing ops into PCF parallel constructs.

    Extends TilingInterface with a single `getDistributedImplementation`
    method that replaces the combination of `getTiledImplementation`,
    `generateResultTileValue`, and `getTiledImplementationFromOperandTiles`.

    The caller provides structured info about how operands are supplied
    (as tiles or full values, potentially from srefs) and how results should
    be produced (returned as tiles or written to destination srefs).
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the indices of operands that are tileable. Non-tileable
        operands are passed through as full values. All results are
        implicitly tileable (required for distribution into parallel
        constructs).
      }],
      /*retTy=*/"SmallVector<unsigned>",
      /*methodName=*/"getTileableOperandIndices",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the distributed/tiled implementation of this op.

        The `offsets` and `sizes` describe the current iteration tile in the
        iteration domain (same semantics as `getTiledImplementation`).

        `operandInfo` provides a `DistributedOperandInfo` for each operand
        (inputs + DPS inits), indexed by operand number. Each entry specifies
        the replacement value and whether it is a tile.

        `resultInfo` provides a `DistributedResultInfo` for each result,
        indexed by result number. When `destSref` is non-null, the
        implementation should write the result to that sref (via
        `pcf.write_slice` or similar) and leave the corresponding
        `tiledValues` entry empty. When null, a tensor tile should be
        returned in `tiledValues`.

        Returns a `TilingResult` containing the tiled ops and tile values.
      }],
      /*retTy=*/"FailureOr<mlir::TilingResult>",
      /*methodName=*/"getDistributedImplementation",
      /*args=*/(ins
        "OpBuilder &":$b,
        "ArrayRef<OpFoldResult>":$offsets,
        "ArrayRef<OpFoldResult>":$sizes,
        "ArrayRef<::mlir::iree_compiler::IREE::PCF::DistributedOperandInfo>":$operandInfo,
        "ArrayRef<::mlir::iree_compiler::IREE::PCF::DistributedResultInfo>":$resultInfo),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
  ];

  let extraClassDeclaration = [{
    /// Returns true if the operand at the given index is tileable.
    bool isOperandTileable(unsigned operandIndex) {
      SmallVector<unsigned> indices = getTileableOperandIndices();
      return llvm::is_contained(indices, operandIndex);
    }
  }];
}

#endif  // IREE_CODEGEN_DIALECT_PCF_INTERFACES
