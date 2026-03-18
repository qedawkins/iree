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
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
      "Type":$elementType, "PCF::ScopeAttrInterface":$scope,
      "Attribute":$sync_scope);
  let hasCustomAssemblyFormat = 1;
  let builders = [
      // Builder with no sync scope.
      AttrOrTypeBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$element_type,
          "PCF::ScopeAttrInterface":$scope)>,
  ];

  let extraClassDeclaration = [{
    ShapedRefType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const {
      return ShapedRefType::get(getContext(), shape.value_or(getShape()), elementType,
        getScope(), getSyncScope());
    }

    bool hasRank() const {
      return true;
    }

    // Helper to check whether the only way accesses to this sref are
    // synchronized is on return from the worker of the same scope.
    bool isReturnOnlySync() const;
  }];
}

def PCF_ThreadGroupType : TypeDef<PCF_Dialect, "ThreadGroup", []> {
  let mnemonic = "threadgroup";

  let summary = [{A handle representing a group of cooperating workers.}];
  let description = [{
    Represents the group of workers executing a `pcf.shared_executor` region
    at a specific scope level. The threadgroup carries optional struct element
    types for attaching arbitrary data (workspace descriptors, accumulators,
    etc.).

    The threadgroup does not directly expose worker IDs — it represents the
    collective group. Worker identity is accessed through scope-specific
    mechanisms when the shared_executor is lowered to `pcf.generic`.

    Examples:
    ```
    !pcf.threadgroup<#iree_gpu.subgroup_scope>
    !pcf.threadgroup<#iree_gpu.subgroup_scope, {index, tensor<64x128xf32>}>
    ```
  }];

  let parameters = (ins
      AttrParameter<"ScopeAttrInterface", "scope">:$scope,
      OptionalArrayRefParameter<"Type">:$structTypes);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns true if the threadgroup carries struct elements.
    bool hasStructElements() const { return !getStructTypes().empty(); }
  }];
}

def PCF_AnyThreadGroup : AnyTypeOf<[PCF_ThreadGroupType]>;

def PCF_ClusterType : TypeDef<PCF_Dialect, "Cluster", []> {
  let mnemonic = "cluster";

  let summary = [{A rectangular subset of a threadgroup's worker grid.}];
  let description = [{
    Represents a subset of workers within a scope's grid, defined by
    per-dimension half-open ranges [lower, upper). Range bounds are
    affine expressions over two kinds of variables:

    - d0, d1, ... -- dependent variables. Each distinct d variable adds
      one SSA index operand requirement to ops using the type.
    - s0, s1, ... -- scope size symbols, implicit from the scope's
      getWorkerCounts. Do NOT add SSA dependencies.

    The bounds are stored as a single AffineMap with 2 * rank results:
    [lower0, upper0, lower1, upper1, ...]. The map's dims are dependent
    variables and its symbols are scope grid sizes.

    Examples:
    ```
    !pcf.cluster<#scope, (0 -> d0), left>
    !pcf.cluster<#scope, (0 -> d0) x (d0 -> s1), tg.left>
    !pcf.cluster<#scope, (0 -> d0), private: {f32}, shared: {tensor<64xf32>}, uniform: {index}, c0>
    !pcf.cluster<#scope, (0 -> d0), uniform: {index}, c0>
    ```
  }];

  let parameters = (ins
      AttrParameter<"ScopeAttrInterface", "scope">:$scope,
      "AffineMap":$boundsMap,
      OptionalArrayRefParameter<"Type">:$privateTypes,
      OptionalArrayRefParameter<"Type">:$sharedTypes,
      OptionalArrayRefParameter<"Type">:$uniformTypes,
      "NamespacedSymbolAttr":$id);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the number of grid dimensions.
    int64_t getRank() const { return getBoundsMap().getNumResults() / 2; }

    /// Returns the number of dependent SSA values ops using this type need.
    int64_t getNumDependentValues() const { return getBoundsMap().getNumDims(); }

    /// Returns the lower bound expression for dimension i.
    AffineExpr getLowerBound(int64_t i) const {
      return getBoundsMap().getResult(2 * i);
    }

    /// Returns the upper bound expression for dimension i.
    AffineExpr getUpperBound(int64_t i) const {
      return getBoundsMap().getResult(2 * i + 1);
    }

    /// Returns true if the cluster carries any struct elements.
    bool hasStructElements() const {
      return !getPrivateTypes().empty() || !getSharedTypes().empty() ||
             !getUniformTypes().empty();
    }
  }];
}

def PCF_AnyCluster : AnyTypeOf<[PCF_ClusterType]>;

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
    SmallVector<Value> getWorkerCounts(OpBuilder& builder, Location loc, int64_t numIds) {
      SmallVector<Type> types(numIds, builder.getIndexType());
      auto castOp = UnrealizedConversionCastOp::create(builder, loc, types, ValueRange{});
      return SmallVector<Value>(castOp.getResults());
    }
    SmallVector<Value> getWorkerIDs(OpBuilder& builder, Location loc, int64_t numIds) {
      SmallVector<Type> types(numIds, builder.getIndexType());
      auto castOp = UnrealizedConversionCastOp::create(builder, loc, types, ValueRange{});
      return SmallVector<Value>(castOp.getResults());
    }
    int64_t getNativeNumIds() { return 1; }
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
    int64_t getNativeNumIds() { return 1; }
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

def PCF_NamespacedSymbolAttr
    : AttrDef<PCF_Dialect, "NamespacedSymbol", []> {
  let mnemonic = "ns_sym";
  let summary = [{A hierarchical symbol path for private namespace resolution.}];
  let description = [{
    Stores an ordered list of path segments forming a hierarchical symbol
    identifier. The last segment is the leaf symbol name; preceding segments
    are namespace qualifiers. Used as the `id` field on `!pcf.cluster` types
    to give clusters stable, symbolically resolvable identifiers.

    Printed as a dot-joined bare keyword:
    ```
    tg.left          // path: ["tg", "left"]
    left             // path: ["left"]
    outer.inner.leaf // path: ["outer", "inner", "leaf"]
    ```
  }];

  let parameters = (ins ArrayRefParameter<"StringAttr">:$path);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the last segment (the symbol name).
    StringAttr getLeaf() const {
      assert(!getPath().empty() && "leaf requires non-empty path");
      return getPath().back();
    }

    /// Returns all segments except the last (the namespace path).
    ArrayRef<StringAttr> getNamespacePath() const {
      return getPath().drop_back();
    }

    /// Returns true if the path has exactly one segment (leaf-only).
    bool isLeafOnly() const { return getPath().size() == 1; }
  }];
}

//===----------------------------------------------------------------------===//
// Misc aliases
//===----------------------------------------------------------------------===//

def PCF_AnyShapedRef : AnyTypeOf<[PCF_ShapedRef]>;

#endif  // IREE_CODEGEN_DIALECT_PCF_DIALECT
