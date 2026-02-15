// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_PCF_INTERFACES
#define IREE_CODEGEN_DIALECT_PCF_INTERFACES

include "mlir/IR/OpBase.td"

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
// IREE::PCF::LayoutAttrInterface
//===----------------------------------------------------------------------===//

def PCF_LayoutAttrInterface : AttrInterface<"LayoutAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::PCF";

  let summary = [{Describes how a tensor is distributed across workers.}];
  let description = [{
    Attributes implementing this interface describe the mapping from tensor
    elements to parallel workers (subgroups and threads). The PCF dialect uses
    this interface in layout constraint operations (`pcf.constrain_layout`,
    `pcf.redistribute`) without depending on any specific layout attribute
    implementation.

    The canonical implementation is VectorExt's `NestedLayoutAttr`, which
    provides a 5-level tiling hierarchy:

      subgroup_tile -> batch_tile -> outer_tile -> thread_tile -> element_tile

    with stride arrays controlling the virtual-to-physical ID mapping. Other
    implementations (e.g., block-cyclic layouts) can be added by implementing
    this interface.

    The interface supports the iterative layout analysis performed by the
    distribution pass:
      1. Anchor layouts at constrained values (MMA ops, explicit constraints).
      2. Propagate through the dataflow graph.
      3. Reach a fixed point.
      4. For unconstrained dimensions, fill with a greedy coalesced layout.
      5. Resume analysis until fully resolved.
  }];

  let methods = [
    //===----------------------------------------------------------------===//
    // Validation
    //===----------------------------------------------------------------===//

    InterfaceMethod<
      /*desc=*/[{
        Validates that this layout is compatible with the given tensor shape.
        Returns failure with a diagnostic if the layout rank does not match,
        or if the tile sizes do not multiply to the tensor dimensions.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"isValidLayout",
      /*args=*/(ins "ShapedType":$tensorType, "Location":$loc)
    >,

    //===----------------------------------------------------------------===//
    // Shape queries
    //===----------------------------------------------------------------===//

    InterfaceMethod<
      /*desc=*/[{
        Returns the rank of the undistributed tensor this layout describes.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getRank",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*desc=*/[{
        Returns the full undistributed tensor shape implied by this layout.
        For each dimension: subgroup * batch * outer * thread * element.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getUndistributedShape",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*desc=*/[{
        Returns the per-thread shape after distribution. This is the shape of
        the tensor slice each thread holds: batch * outer * element per dim.
        This determines the vector shape after vectorization.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getDistributedShape",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*desc=*/[{
        Returns the element tile sizes (the atomic unit each thread processes).
        This is the innermost level of the tiling hierarchy.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getElementTile",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*desc=*/[{
        Returns the thread tile sizes (how threads within a subgroup tile the
        data).
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getThreadTile",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*desc=*/[{
        Returns the subgroup tile sizes (how subgroups tile the data).
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getSubgroupTile",
      /*args=*/(ins)
    >,

    //===----------------------------------------------------------------===//
    // Distribution: computing per-thread slices
    //===----------------------------------------------------------------===//

    InterfaceMethod<
      /*desc=*/[{
        Given linearized thread and subgroup IDs, computes the offsets into the
        undistributed tensor for this thread's slice. Returns one Value per
        tensor dimension.

        `subgroupSize` is the number of threads per subgroup (needed to
        delinearize the thread ID into per-dimension virtual IDs).

        The returned offsets account for the full hierarchy:
          offset[i] = sg_offset[i] + batch_offset[i] + outer_offset[i]
                    + thread_offset[i]
        where each component is derived from the layout strides and tile sizes.

        Note: This returns the offset to the *first* element of the thread's
        tile. The thread's tile shape is `getElementTile()`.
      }],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"computeThreadSliceOffsets",
      /*args=*/(ins
        "OpBuilder&":$builder,
        "Location":$loc,
        "Value":$threadId,
        "Value":$subgroupId,
        "int64_t":$subgroupSize)
    >,

    //===----------------------------------------------------------------===//
    // Layout transformations (for dataflow propagation)
    //===----------------------------------------------------------------===//

    InterfaceMethod<
      /*desc=*/[{
        Permutes the layout dimensions according to the given permutation.
        Used when propagating layouts through transpose-like operations.
      }],
      /*retTy=*/"Attribute",
      /*methodName=*/"permute",
      /*args=*/(ins "ArrayRef<int64_t>":$permutation)
    >,

    InterfaceMethod<
      /*desc=*/[{
        Projects the layout by dropping the specified dimensions. Used when
        propagating layouts through reduction or slice operations that
        eliminate dimensions.
      }],
      /*retTy=*/"Attribute",
      /*methodName=*/"project",
      /*args=*/(ins "ArrayRef<bool>":$droppedDims)
    >,

    //===----------------------------------------------------------------===//
    // Iterative analysis support
    //===----------------------------------------------------------------===//

    InterfaceMethod<
      /*desc=*/[{
        Returns true if all dimensions have fully specified distribution
        (no stride-0 / undistributed dimensions that still need resolution).
        During iterative analysis, a layout may be partially constrained.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isFullyDistributed",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Default: assume fully distributed.
        return true;
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        Returns a bitmask of dimensions that are not yet distributed (stride
        is 0 or tile is 1 with no thread coverage). Used by the iterative
        analysis to identify dimensions that need greedy layout assignment.
      }],
      /*retTy=*/"SmallVector<bool>",
      /*methodName=*/"getUnconstrainedDims",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Default: all dims constrained.
        SmallVector<bool> result(
            mlir::cast<LayoutAttrInterface>($_attr).getRank(), false);
        return result;
      }]
    >,

    //===----------------------------------------------------------------===//
    // Compatibility checks
    //===----------------------------------------------------------------===//

    InterfaceMethod<
      /*desc=*/[{
        Returns true if this layout is compatible with `other` for the purpose
        of eliding a redistribution. Two layouts are compatible if they
        produce the same thread-to-element mapping (possibly with different
        batch/outer tiling that does not affect physical distribution).
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isCompatibleWith",
      /*args=*/(ins "Attribute":$other),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Default: compatible only if identical.
        return $_attr == other;
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::PCF::MMALayoutInterface
//===----------------------------------------------------------------------===//

def PCF_MMALayoutInterface : AttrInterface<"MMALayoutInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::PCF";

  let summary = [{Provides MMA-specific layout information for distribution.}];
  let description = [{
    Attributes implementing this interface describe the data layout requirements
    of a matrix multiply-accumulate instruction. The PCF dialect uses this to
    derive distribution layouts for MMA operands without depending on any
    specific GPU dialect.

    GPU dialects register their MMA attributes as external models of this
    interface, following the same pattern as ScopeAttrInterface.

    Each MMA instruction defines how its LHS, RHS, and accumulator operands
    must be distributed across threads. The `getOperandLayout` method returns
    an attribute implementing `PCF_LayoutAttrInterface` (concretely, a
    `NestedLayoutAttr`) that describes this distribution.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the layout attribute (implementing PCF_LayoutAttrInterface)
        for the given MMA operand role ("lhs", "rhs", or "acc").
        The layout describes how tensor elements map to threads for this
        operand of the MMA instruction.
      }],
      /*retTy=*/"Attribute",
      /*methodName=*/"getOperandLayout",
      /*args=*/(ins "StringRef":$operandRole)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the expected tensor shape for the given operand role.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getOperandShape",
      /*args=*/(ins "StringRef":$operandRole)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the expected element type for the given operand role.
      }],
      /*retTy=*/"Type",
      /*methodName=*/"getOperandElementType",
      /*args=*/(ins "MLIRContext*":$context, "StringRef":$operandRole)
    >,
  ];
}

#endif  // IREE_CODEGEN_DIALECT_PCF_INTERFACES
