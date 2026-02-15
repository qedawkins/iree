// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_PCF_DIALECT
#define IREE_CODEGEN_DIALECT_PCF_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Parallel control flow dialect
//===----------------------------------------------------------------------===//

def PCF_Dialect : Dialect {
  let name = "pcf";
  let cppNamespace = "::mlir::iree_compiler::IREE::PCF";

  let summary = [{
    A dialect designed to model parallel control flow.
  }];
  let description = [{
    The pcf dialect models parallelized control flow using structured operations
    akin to dialects like scf. It offers a set of core loop-like constructs
    alongside the glue necessary to represent splitting and joining parallel
    work.

    In contrast with scf whose scope is purely focused on representing common
    control flow, the pcf dialect includes type, interfaces, and operations that
    represent dataflow across parallel workers. This comprises two key
    conceptual types:

    1. Scoped memory. This is a reference to memory that carries information
       about its allocation scope as well as how to synchronize it. This allows
       for fencing at fine granularities (e.g. allocation).
    2. Tokens. Types capable of managing synchronization of resources between
       threads. This could be anything ranging from fences + (named) barriers to
       producer/consumer queues implemented with ringbuffers.

    PCF ops + types are designed to be lowered in three phases, starting from
    structural ops on scoped memory infused with synchronization tokens. Prior
    to each phase a different level of scheduling is implied.

    1. Tokens tied to resources are split and lowered to separate ops. Before
       this the compiler can perform coarse grain scheduling around resources
       according to their tied synchronization.
    2. Generic scoped memory is converted to memref. Since all tokens have been
       resolved by this point, this is just a matter of propagating layout and
       memory space.
    3. Wrapping structured ops are lowered to control flow (scf and/or cf).

    This diagram illustrates where the dialect fits in to executable
    lowering pipelines for typical GPUs. For CPUs and other accelerators, the
    same flow is intended to work modulo different levels of physical
    parallelism instead of thread/subgroup/lane.

    ```
            v----------+----------v
            | Executable Input    |
            | (Linalg on tensors) |
            +----------v----------+
                       |
     TileAndDistribute |
     to workgroups     |
                       |
            v----------+----------v
            |    PCF and/or SCF   |
            +----------v----------+
                       |
     SCF(.forall)ToPCF |
                       |
             v---------+---------v
    +--------|-------+   +-------|--------+
    |   Tile Op1 to  |   |   Tile OpN to  |
    |   Subgroups/   |   |   Subgroups/   |
    |   Threads/     |...|   Threads/     |
    |   Lanes        |   |   Lanes        |
    | pcf.concurrent |   | pcf.concurrent |
    +--------|-------+   +-------|--------+
             +---------v---------+
                       |
             Vectorize | // WriteOps vectorize
                       |
             Bufferize | // PCF tensor -> pcf.sref
                       | // becomes memref -> sref
                       |
         ResolveTokens |
          SRefToMemRef | // pcf.sref -> memref
              LowerPCF | // pcf -> scf/cf
                       |
           v-----------+-----------v
           | SCF+GPU+vector+memref |
           +-----------------------+
    ```

  }];

  let dependentDialects = [];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base pcf dialect op classes
//===----------------------------------------------------------------------===//

class PCF_Op<string mnemonic, list<Trait> traits = []>
    : Op<PCF_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Accessor mode for sref types
//===----------------------------------------------------------------------===//

def PCF_AM_ReadWrite : I32EnumAttrCase<"ReadWrite", 0, "readwrite">;
def PCF_AM_ReadOnly  : I32EnumAttrCase<"ReadOnly",  1, "readonly">;

def PCF_AccessorMode : I32EnumAttr<"AccessorMode",
    "Accessor mode for shaped references", [
      PCF_AM_ReadWrite,
      PCF_AM_ReadOnly,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::PCF";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Base pcf types
//===----------------------------------------------------------------------===//

def PCF_ShapedRef : TypeDef<PCF_Dialect, "ShapedRef", [
    ShapedTypeInterface
]> {
  let mnemonic = "sref";

  let summary = [{A shaped reference to a buffer.}];
  let description = [{
    A reference to a buffer with unspecified layout and physical storage.
    Carries the shape and element type of the referenced region. Elements can
    be accessed by index, though no assumptions about the physical relation
    between two coordinates can be made. Elements at different coordinates must
    not internally alias. For example, if `foo` is a `pcf.sref<2xi32>`, `foo[0]`
    and `foo[1]` must be distinct values.

    ```
    template<size_t rank, typename eltype, typename alloc_scope_ty, sync_scope>
    class ShapedRef {
      // Access is pointwise within the coordinate space implied by the shape.
      // Element type determines the minimum access bitwidth.
      eltype *getElementPtr(int a, ...) // |rank| operands.
      size_t shape[rank];

      // Scope this referenced memory was allocated at. Defines memory space.
      alloc_scope_ty alloc_scope;
      // Class defining synchronization for this reference.
      token_ty sync_scope;
    }
    ```

    When the `sync_scope` is of type `#pcf.sync_on_return`, then a special
    printer kicks in, i.e. the following two types are equivalent:
    ```
    !pcf.ref<?xi32, #pcf.test_scope, #pcf.sync_on_return>
    !pcf.ref<?xi32, sync(#pcf.test_scope)>
    ```

    An optional accessor mode (`readwrite` or `readonly`) restricts how the
    reference can be used. A `readonly` sref cannot be the target of
    `pcf.write_slice`. When omitted, no accessor restriction is enforced.
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
      "Type":$elementType, "PCF::ScopeAttrInterface":$scope,
      "Attribute":$sync_scope,
      OptionalParameter<"std::optional<PCF::AccessorMode>">:$accessorMode);
  let hasCustomAssemblyFormat = 1;
  let builders = [
      // Builder with no sync scope and no accessor mode.
      AttrOrTypeBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$element_type,
          "PCF::ScopeAttrInterface":$scope)>,
      // Builder with sync scope but no accessor mode (backward compat).
      AttrOrTypeBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$element_type,
          "PCF::ScopeAttrInterface":$scope, "Attribute":$sync_scope)>,
      // Builder with accessor mode but no sync scope.
      AttrOrTypeBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$element_type,
          "PCF::ScopeAttrInterface":$scope,
          "PCF::AccessorMode":$accessorMode)>,
  ];

  let extraClassDeclaration = [{
    ShapedRefType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const {
      return ShapedRefType::get(getContext(), shape.value_or(getShape()), elementType,
        getScope(), getSyncScope(), getAccessorMode());
    }

    bool hasRank() const {
      return true;
    }

    // Helper to check whether the only way accesses to this sref are
    // synchronized is on return from the worker of the same scope.
    bool isReturnOnlySync() const;

    /// Returns true if this sref has an explicit accessor mode.
    bool hasAccessorMode() const {
      return getAccessorMode().has_value();
    }

    /// Returns true if this sref is read-only.
    bool isReadOnly() const {
      return getAccessorMode() == PCF::AccessorMode::ReadOnly;
    }

    /// Returns true if this sref is read-write.
    bool isReadWrite() const {
      return getAccessorMode() == PCF::AccessorMode::ReadWrite;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Bundle type
//===----------------------------------------------------------------------===//

def PCF_BundleType : TypeDef<PCF_Dialect, "Bundle", []> {
  let mnemonic = "bundle";
  let summary = [{A worker bundle representing a subset of workers at a scope.}];
  let description = [{
    Represents a named subset of workers at a specific scope level. Bundles
    are created by `pcf.form_bundles` and consumed by `pcf.execute_as`.
    The integer ID identifies which bundle this is (0-based) within the
    parent `form_bundles` operation.
  }];

  let parameters = (ins
      "PCF::ScopeAttrInterface":$scope,
      "int64_t":$id);

  let assemblyFormat = [{
    `<` $scope `,` $id `>`
  }];

  let genVerifyDecl = 1;
}

def PCF_AnyBundle : AnyTypeOf<[PCF_BundleType]>;

//===----------------------------------------------------------------------===//
// Base pcf attributes
//===----------------------------------------------------------------------===//

def PCF_TestScopeAttr
    : AttrDef<
          PCF_Dialect,
          "TestScope", [DeclareAttrInterfaceMethods<PCF_ScopeAttrInterface>]> {
  let mnemonic = "test_scope";
  let summary = [{Test scope attribute used for testing.}];
  let description = [{
    Scope that fails on all interface uses. For use in testing where the scope
    is not relevant.
  }];
  let parameters = (ins);
  let assemblyFormat = [{}];

  let extraClassDeclaration = [{
    SmallVector<Value> getWorkerCounts(OpBuilder&, Location, int64_t) {
      assert(false && "why are you here?");
      return {};
    }
    SmallVector<Value> getWorkerIDs(OpBuilder&, Location, int64_t) {
      assert(false && "why are you here?");
      return {};
    }
    int64_t getNativeNumProcessorIds() {
      assert(false && "why are you here?");
      return 1;
    }
  }];
}

def PCF_SequentialAttr
    : AttrDef<
          PCF_Dialect,
          "Sequential", [DeclareAttrInterfaceMethods<PCF_ScopeAttrInterface>]> {
  let mnemonic = "sequential";
  let summary = [{Attribute representing sequential execution}];
  let description = [{
    Scope that reuses the current process as the sole executor of a parallel
    region.
  }];
  let parameters = (ins);
  let assemblyFormat = [{}];

  let extraClassDeclaration = [{
    SmallVector<Value> getWorkerCounts(OpBuilder& builder, Location loc, int64_t numIds) {
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      SmallVector<Value> counts(numIds, one);
      return counts;
    }
    SmallVector<Value> getWorkerIDs(OpBuilder& builder, Location loc, int64_t numIds) {
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      SmallVector<Value> ids(numIds, zero);
      return ids;
    }
    int64_t getNativeNumProcessorIds() {
      return 1;
    }
  }];
}

def PCF_SyncOnReturnAttr
    : AttrDef<PCF_Dialect, "SyncOnReturn",
              [DeclareAttrInterfaceMethods<PCF_SyncScopeAttrInterface>]> {
  let mnemonic = "sync_on_return";
  let summary = [{
    Synchronize when returning from the worker of the same scope
  }];
  let description = [{
    Attribute indicating that the shaped ref this attribute is tied to is only
    fenced when the parent of the same scope returns. This is akin to memory
    order acquire on scope entry and __syncthreads followed by a memory order
    release fence on scope exit.
  }];
  let parameters = (ins);
  let assemblyFormat = [{}];

  let extraClassDeclaration = [{
    TypeRange getConcreteTypes(MLIRContext *context) {
      // Expansion of this sync scope involves simply dropping it.
      return TypeRange();
    }
    void enqueueWrite(OpBuilder& builder, ValueRange syncValues, Operation *writingOperation) {
      // Nothing to do.
      return;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Layout constraint attributes
//===----------------------------------------------------------------------===//

def PCF_RedistributeMethod_SharedMemory
    : I32EnumAttrCase<"SharedMemory", 0, "shared_memory">;
def PCF_RedistributeMethod_Shuffle
    : I32EnumAttrCase<"Shuffle", 1, "shuffle">;
def PCF_RedistributeMethod_Registers
    : I32EnumAttrCase<"Registers", 2, "registers">;

def PCF_RedistributeMethodEnum
    : I32EnumAttr<"RedistributeMethod",
                  "Method for data redistribution between layouts.", [
      PCF_RedistributeMethod_SharedMemory,
      PCF_RedistributeMethod_Shuffle,
      PCF_RedistributeMethod_Registers,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::PCF";
  let genSpecializedAttr = 0;
}

def PCF_RedistributeMethodAttr
    : EnumAttr<PCF_Dialect, PCF_RedistributeMethodEnum,
               "redistribute_method">;

def PCF_SharedLayoutAttr : AttrDef<PCF_Dialect, "SharedLayout"> {
  let mnemonic = "shared_layout";
  let summary = [{Shared memory access layout for bank conflict avoidance.}];
  let description = [{
    Describes the physical layout of shared memory, including stride padding
    and swizzle patterns. Applied to `pcf.sref` values to control how threads
    access shared memory.

    Example:
    ```mlir
    #pcf.shared_layout<{stride = [36, 1], swizzle = none}>
    ```
  }];

  let parameters = (ins
      ArrayRefParameter<"int64_t">:$strides,
      DefaultValuedParameter<"StringAttr", "{}">:$swizzle);

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Misc aliases
//===----------------------------------------------------------------------===//

def PCF_AnyShapedRef : AnyTypeOf<[PCF_ShapedRef]>;

#endif  // IREE_CODEGEN_DIALECT_PCF_DIALECT
