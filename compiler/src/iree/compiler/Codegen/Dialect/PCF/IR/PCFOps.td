// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_PCF_OPS
#define IREE_CODEGEN_DIALECT_PCF_OPS

include "iree/compiler/Codegen/Dialect/PCF/IR/PCFBase.td"
include "iree/compiler/Utils/CommonTypeConstraints.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/Properties.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Alloc
//===----------------------------------------------------------------------===//

def OpGroupAllocOps : OpDocGroup {
  let summary = "Alloc ops";
  let description = "";
}

let opDocGroup = OpGroupAllocOps in {

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

def AllocOp : PCF_Op<"alloc", []> {
  let summary = [{
    Shaped ref allocation operation
  }];
  let description = [{
    Allocates a `pcf.sref` with the given element type and shape. Dynamic
    dimensions in the result type must have corresponding dynamic size
    operands. The allocation scope is determined by the scope attribute of
    the result type.

    Example:
    ```mlir
      %sref = pcf.alloc() : !pcf.sref<4x8xf32, #foo.scope>
      %sref_dyn = pcf.alloc(%d0, %d1) : !pcf.sref<?x?xf32, #foo.scope>
    ```
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes);
  let results = (outs PCF_AnyShapedRef:$result);

  let builders = [
    OpBuilder<(ins "ShapedRefType":$srefType), [{
      return build($_builder, $_state, srefType, {});
    }]>
  ];

  let assemblyFormat = [{
    `(`$dynamicSizes`)` attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    ShapedRefType getResultType() {
      return cast<ShapedRefType>(getResult().getType());
    }
    SmallVector<OpFoldResult> getMixedSizes();
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ToSrefOp
//===----------------------------------------------------------------------===//

def ToSrefOp : PCF_Op<"to_sref", [Pure]> {
  let summary = [{
    Create a readonly sref view of a tensor or memref.
  }];
  let description = [{
    Creates a `pcf.sref` that provides a readonly view of the input tensor or
    memref within a scope's execution context. The input is captured from the
    enclosing scope. The result sref's shape and element type must match the
    input's shape and element type.

    Example:
    ```mlir
      %sref = pcf.to_sref %input : tensor<128x256xf16> -> !pcf.sref<128x256xf16, #scope>
    ```
  }];

  let arguments = (ins AnyRankedTensorOrMemRef:$input);
  let results = (outs PCF_AnyShapedRef:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];

  let extraClassDeclaration = [{
    ShapedRefType getResultType() {
      return cast<ShapedRefType>(getResult().getType());
    }
  }];

  let hasVerifier = 1;
}

} // OpGroupAllocOps

//===----------------------------------------------------------------------===//
// ParallelOps
//===----------------------------------------------------------------------===//

def OpGroupParallelOps : OpDocGroup {
  let summary = "Parallel execution ops";
  let description = "";
}

let opDocGroup = OpGroupParallelOps in {

def GenericOp : PCF_Op<"generic", [
       AttrSizedOperandSegments,
       AutomaticAllocationScope,
       RecursiveMemoryEffects,
       DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
       DeclareOpInterfaceMethods<PCF_NamespaceOpInterface>,
     ]> {
  let summary = [{
    Execute a set of workers in parallel on a region.
  }];
  let description = [{
    Executes a region across a set of workers at a specified scope. When
    control flow reaches this op, `nproc` workers of the specified scope are
    spawned and begin executing the region. The scope is given by an attribute
    implementing the `ScopeAttrInterface` interface and is responsible for the semantics
    of all pcf primitives at the same scope. Further details about scopes are
    included in the docs for the interface.

    The optional `initialize` region is executed once when control flow first
    reaches the op. Values yielded from the initializer become block arguments
    available to the execute region. This is useful for setting up per-op
    state that persists across all worker invocations.

    Results are produced by snapshotting the value of each result's tied sref
    once all workers have returned. Results can either be:
    1. Tied to initial values (tensor or memref) - the init value provides the
       initial contents and the result captures the final state.
    2. Allocated by the op itself - dynamic sizes must be provided for
       untied results with dynamic dimensions.

    Basic example with tied results:
    ```mlir
      %0 = ... : tensor<4x8xf32>
      %1 = pcf.generic scope(#foo.scope)
        execute(%ref = %0)[%id: index, %num_workers: index]
             : (!pcf.sref<4x8xf32, #foo.scope>) -> (tensor<4x8xf32>) {
        // Each worker can read/write %ref.
        pcf.return
      }
    ```

    Example with initializer:
    ```mlir
      %result = pcf.generic scope(#foo.scope)
        initialize {
          %scratch = pcf.alloc() : !pcf.sref<16xf32, #foo.scope>
          pcf.yield %scratch : !pcf.sref<16xf32, #foo.scope>
        } -> (%scratch_arg: !pcf.sref<16xf32, #foo.scope>)
        execute(%ref = %init)[%id: index, %num_workers: index]
             : (!pcf.sref<4x8xf32, #foo.scope>) -> (tensor<4x8xf32>) {
        // %scratch_arg is available here, initialized once.
        pcf.return
      }
    ```

    Example with untied (allocated) results:
    ```mlir
      %d0, %d1 = ... : index
      %result = pcf.generic scope(#foo.scope)
        execute[%id: index, %num_workers: index]
             : () -> (tensor<?x?xf32>{%d0, %d1}) {
        // Result sref is allocated by the op, not tied to any init.
        pcf.return
      }
    ```
  }];
  let arguments = (ins PCF_ScopeAttrInterface:$scope,
      Variadic<AnyRankedTensorOrMemRef>:$readonlyInits,
      Variadic<AnyRankedTensorOrMemRef>:$inits, Variadic<Index>:$dynamic_sizes,
      ArrayProp<BoolProp>:$is_tied,
      DefaultValuedProp<BoolProp, "false">:$sync_on_return,
      IntProp<"int64_t">:$num_index_args, IntProp<"int64_t">:$num_leading_args,
      IntProp<"int64_t">:$num_readonly_refs);

  let results = (outs Variadic<AnyRankedTensorOrMemRef>:$results);
  let regions = (region
    MaxSizedRegion<1>:$initializer,
    MinSizedRegion<1>:$region
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  // The default builder does not add the proper body BBargs, roll our own.
  let skipDefaultBuilders = 1;
  let builders = [
      // Builder with no return values.
      OpBuilder<(ins "ScopeAttrInterface":$scope, "int64_t":$num_iterators,
          CArg<"bool", "false">:$sync_on_return)>,
      // Builder with return types inferred from inits.
      OpBuilder<(ins "ScopeAttrInterface":$scope, "ValueRange":$inits,
          "int64_t":$num_iterators, CArg<"bool", "false">:$sync_on_return)>,
      // Builder with no inits.
      OpBuilder<(ins "TypeRange":$result_types, "ScopeAttrInterface":$scope,
          "ValueRange":$dynamic_sizes, "int64_t":$num_iterators,
          CArg<"bool", "false">:$sync_on_return)>,
      // Builder with everything.
      OpBuilder<(ins "TypeRange":$result_types, "ScopeAttrInterface":$scope,
          "ValueRange":$inits, "ValueRange":$dynamic_sizes,
          "ArrayRef<bool>":$is_tied, "int64_t":$num_iterators,
          CArg<"bool", "false">:$sync_on_return)>,
      // Builder with readonly and readwrite inits.
      OpBuilder<(ins "ScopeAttrInterface":$scope,
          "ValueRange":$readonlyInits, "ValueRange":$inits,
          "int64_t":$num_iterators,
          CArg<"bool", "false">:$sync_on_return)>,
  ];

  let extraClassDeclaration = [{
    int64_t getNumIterators() {
      return getNumIndexArgs() / 2;
    }

    MutableArrayRef<BlockArgument> getLeadingArgsMutable() {
      return getRegion().getArguments().take_front(getNumLeadingArgs());
    }
    ArrayRef<BlockArgument> getLeadingArgs() {
      return getLeadingArgsMutable();
    }

    /// Returns the readonly sref block arguments.
    MutableArrayRef<BlockArgument> getReadonlyRefArgsMutable() {
      int64_t start = getNumLeadingArgs();
      return getRegion().getArguments().slice(start, getNumReadonlyRefs());
    }
    ArrayRef<BlockArgument> getReadonlyRefArgs() {
      return getReadonlyRefArgsMutable();
    }

    MutableArrayRef<BlockArgument> getIdArgsMutable() {
      return getRegion().getArguments().take_back(getNumIndexArgs()).take_front(getNumIterators());
    }
    ArrayRef<BlockArgument> getIdArgs() {
      return getIdArgsMutable();
    }
    MutableArrayRef<BlockArgument> getCountArgsMutable() {
      return getRegion().getArguments().take_back(getNumIterators());
    }
    ArrayRef<BlockArgument> getCountArgs() {
      return getCountArgsMutable();
    }
    MutableArrayRef<BlockArgument> getIdAndCountArgsMutable() {
      return getRegion().getArguments().take_back(getNumIndexArgs());
    }
    ArrayRef<BlockArgument> getIdAndCountArgs() {
      return getIdAndCountArgsMutable();
    }

    /// Returns the readwrite sref block arguments (skipping readonly refs).
    MutableArrayRef<BlockArgument> getRegionRefArgsMutable() {
      int64_t start = getNumLeadingArgs() + getNumReadonlyRefs();
      return getRegion().getArguments().slice(start, getNumResults());
    }
    ArrayRef<BlockArgument> getRegionRefArgs() {
      return getRegionRefArgsMutable();
    }

    bool isRegionRefArg(BlockArgument b);
    bool isInitializedArg(BlockArgument b) {
      assert(b.getOwner() == &getRegion().front() && "unexpected non-entry block arg");
      return b.getArgNumber() < getNumLeadingArgs();
    }
    SmallVector<int64_t> getInitTiedResultIndices();
    OpResult getTiedResult(OpOperand &operand);
    OpResult getTiedResult(BlockArgument b) {
      assert(isRegionRefArg(b) && "unexpected non region ref arg");
      int64_t readwriteStart = getNumLeadingArgs() + getNumReadonlyRefs();
      return (*this)->getOpResult(b.getArgNumber() - readwriteStart);
    }
    OpOperand *getTiedInit(int64_t i);
    ShapedType getResultType(int64_t i) {
      return cast<ShapedType>(getResults()[i].getType());
    }
    ValueRange getResultDims(int64_t i);
  }];
}

def LoopOp : PCF_Op<"loop", [
       AttrSizedOperandSegments,
       AutomaticAllocationScope,
       RecursiveMemoryEffects,
       DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
       DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
       SingleBlockImplicitTerminator<"mlir::iree_compiler::IREE::PCF::ReturnOp">
     ]> {
  let summary = [{
    Execute a set of workers in parallel on a region.
  }];
  let description = [{
    Executes a region for each point in the iteration space defined by the
    `count` operands. Unlike `pcf.generic` which spawns workers equal to the
    native parallelism of the scope, `pcf.loop` explicitly specifies the
    iteration count and maps iterations to workers according to the scope's
    scheduling policy.

    When control flow reaches this op, the scope determines how to distribute
    the iterations across available workers. The scope is given by an attribute
    implementing the `ScopeAttrInterface` interface. Further details about scopes are
    included in the docs for the interface.

    The execute region receives one index block argument per count operand,
    representing the current iteration's coordinates in the iteration space.

    Results are produced by snapshotting the value of each result's tied sref
    once all iterations have completed. Results can either be:
    1. Tied to initial values (tensor or memref) - the init value provides the
       initial contents and the result captures the final state.
    2. Allocated by the op itself - dynamic sizes must be provided for
       untied results with dynamic dimensions.

    Basic example with 1D iteration:
    ```mlir
      %n = ... : index
      %0 = ... : tensor<4x8xf32>
      %1 = pcf.loop scope(#foo.scope) count(%n)
        execute(%ref = %0)[%id: index]
             : (!pcf.sref<4x8xf32, #foo.scope>) -> (tensor<4x8xf32>) {
        // %id ranges from 0 to %n-1.
        pcf.return
      }
    ```

    Example with multi-dimensional iteration:
    ```mlir
      %m, %n = ... : index
      %result = pcf.loop scope(#foo.scope) count(%m, %n)
        execute(%ref = %init)[%i: index, %j: index]
             : (!pcf.sref<?x?xf32, #foo.scope>) -> (tensor<?x?xf32>) {
        // %i ranges from 0 to %m-1, %j ranges from 0 to %n-1.
        pcf.return
      }
    ```
  }];
  let arguments = (ins PCF_ScopeAttrInterface:$scope, Variadic<Index>:$count,
      Variadic<AnyRankedTensorOrMemRef>:$readonlyInits,
      Variadic<AnyRankedTensorOrMemRef>:$inits, Variadic<Index>:$dynamic_sizes,
      ArrayProp<BoolProp>:$is_tied,
      DefaultValuedProp<BoolProp, "false">:$sync_on_return,
      IntProp<"int64_t">:$num_readonly_refs);

  let results = (outs Variadic<AnyRankedTensorOrMemRef>:$results);
  let regions = (region AnyRegion:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  // The default builder does not add the proper body BBargs, roll our own.
  let skipDefaultBuilders = 1;
  let builders = [
      // Builder with no return values.
      OpBuilder<(ins "ScopeAttrInterface":$scope, "ValueRange":$count,
          CArg<"bool", "false">:$sync_on_return)>,
      // Builder with return types inferred from inits.
      OpBuilder<(ins "ScopeAttrInterface":$scope, "ValueRange":$count,
          "ValueRange":$inits, CArg<"bool", "false">:$sync_on_return)>,
      // Builder with no inits.
      OpBuilder<(ins "TypeRange":$result_types, "ScopeAttrInterface":$scope,
          "ValueRange":$count, "ValueRange":$dynamic_sizes,
          CArg<"bool", "false">:$sync_on_return)>,
      // Builder with everything (no readonly).
      OpBuilder<(ins "TypeRange":$result_types, "ScopeAttrInterface":$scope,
          "ValueRange":$count, "ValueRange":$inits, "ValueRange":$dynamic_sizes,
          "ArrayRef<bool>":$is_tied, CArg<"bool", "false">:$sync_on_return)>,
      // Builder with readonly and readwrite inits (types inferred).
      OpBuilder<(ins "ScopeAttrInterface":$scope, "ValueRange":$count,
          "ValueRange":$readonlyInits, "ValueRange":$inits,
          CArg<"bool", "false">:$sync_on_return)>,
      // Builder with everything including readonly.
      OpBuilder<(ins "TypeRange":$result_types, "ScopeAttrInterface":$scope,
          "ValueRange":$count, "ValueRange":$readonlyInits,
          "ValueRange":$inits, "ValueRange":$dynamic_sizes,
          "ArrayRef<bool>":$is_tied, CArg<"bool", "false">:$sync_on_return)>,
  ];

  let extraClassDeclaration = [{
    int64_t getNumIdArgs() {
      return getCount().size();
    }

    /// Returns the readonly sref block arguments.
    MutableArrayRef<BlockArgument> getReadonlyRefArgsMutable() {
      return getRegion().getArguments().take_front(getNumReadonlyRefs());
    }
    ArrayRef<BlockArgument> getReadonlyRefArgs() {
      return getReadonlyRefArgsMutable();
    }

    MutableArrayRef<BlockArgument> getIdArgsMutable() {
      return getRegion().getArguments().take_back(getNumIdArgs());
    }
    ArrayRef<BlockArgument> getIdArgs() {
      return getIdArgsMutable();
    }

    /// Returns the readwrite sref block arguments (skipping readonly refs).
    MutableArrayRef<BlockArgument> getRegionRefArgsMutable() {
      int64_t start = getNumReadonlyRefs();
      return getRegion().getArguments().slice(start, getNumResults());
    }
    ArrayRef<BlockArgument> getRegionRefArgs() {
      return getRegionRefArgsMutable();
    }

    SmallVector<int64_t> getInitTiedResultIndices();
    OpResult getTiedResult(OpOperand &operand);
    OpResult getTiedResult(BlockArgument b) {
      int64_t readwriteStart = getNumReadonlyRefs();
      return (*this)->getOpResult(b.getArgNumber() - readwriteStart);
    }
    OpOperand *getTiedInit(int64_t i);
    ShapedType getResultType(int64_t i) {
      return cast<ShapedType>(getResults()[i].getType());
    }
    ValueRange getResultDims(int64_t i);
  }];
}

def SharedExecutorOp : PCF_Op<"shared_executor", [
       AttrSizedOperandSegments,
       AutomaticAllocationScope,
       RecursiveMemoryEffects,
       DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
       DeclareOpInterfaceMethods<PCF_NamespaceOpInterface>,
     ]> {
  let summary = [{
    Execute a collective region where all workers cooperatively operate on
    shared memory.
  }];
  let description = [{
    Executes a region in collective/SIMD semantics at the specified scope.
    Unlike `pcf.generic` where each worker gets an ID and executes
    independently (SIMT), in `shared_executor` the body is conceptually
    executed by all workers cooperatively. Individual ops in the body must
    be distributed (e.g., via VectorDistribute) when lowering to SIMT.

    The op takes two kinds of sref operands:
    - **Readonly** (`<-`): Input tensor bound as a readonly sref. Does not
      produce a result.
    - **Readwrite** (`=`): Input tensor bound as a readwrite sref. Produces
      a result tensor capturing the final state.

    The execute region receives a `!pcf.threadgroup` block argument
    representing the collective worker group. The threadgroup carries the
    scope and optional struct elements.

    Results are produced by snapshotting each readwrite sref once all
    workers return, identical to `pcf.generic`.

    The optional `initialize` region allocates shared memory. Values yielded
    from the initializer become additional block arguments in the execute
    region.

    Example:
    ```mlir
      %result = pcf.shared_executor scope(#sg)
        initialize {
          %smem = pcf.alloc() : !pcf.sref<128x64xf16, #sg>
          pcf.yield %smem : !pcf.sref<128x64xf16, #sg>
        } -> (%smem: !pcf.sref<128x64xf16, #sg>)
        execute(%in_ref <- %input, %out_ref = %output)
            [%tg: !pcf.threadgroup<#sg>]
             : (!pcf.sref<128x256xf16, #sg>,
                !pcf.sref<128x128xf32, #sg>)
            -> (tensor<128x128xf32>) {
          pcf.return
        }
    ```
  }];

  let arguments = (ins PCF_ScopeAttrInterface:$scope,
      Variadic<AnyRankedTensorOrMemRef>:$readonlyInits,
      Variadic<AnyRankedTensorOrMemRef>:$readwriteInits,
      IntProp<"int64_t">:$num_leading_args,
      IntProp<"int64_t">:$num_readonly_refs);

  let results = (outs Variadic<AnyRankedTensorOrMemRef>:$results);
  let regions = (region
    MaxSizedRegion<1>:$initializer,
    MinSizedRegion<1>:$region
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
      // Builder with readwrite inits only (results inferred from inits).
      OpBuilder<(ins "ScopeAttrInterface":$scope,
          "ValueRange":$readwriteInits)>,
      // Builder with readonly and readwrite inits.
      OpBuilder<(ins "ScopeAttrInterface":$scope,
          "ValueRange":$readonlyInits,
          "ValueRange":$readwriteInits)>,
  ];

  let extraClassDeclaration = [{
    /// Returns the number of readwrite sref operands (equals num results).
    int64_t getNumReadwrite() { return getNumResults(); }

    MutableArrayRef<BlockArgument> getLeadingArgsMutable() {
      return getRegion().getArguments().take_front(getNumLeadingArgs());
    }
    ArrayRef<BlockArgument> getLeadingArgs() {
      return getLeadingArgsMutable();
    }

    /// Returns the threadgroup block argument (last arg in the region).
    BlockArgument getThreadGroup() {
      return getRegion().getArguments().back();
    }

    /// Returns the readonly sref block arguments.
    MutableArrayRef<BlockArgument> getReadonlyRefArgsMutable() {
      int64_t start = getNumLeadingArgs();
      return getRegion().getArguments().slice(start, getNumReadonlyRefs());
    }
    ArrayRef<BlockArgument> getReadonlyRefArgs() {
      return getReadonlyRefArgsMutable();
    }

    /// Returns the readwrite sref block arguments.
    MutableArrayRef<BlockArgument> getReadwriteRefArgsMutable() {
      int64_t start = getNumLeadingArgs() + getNumReadonlyRefs();
      return getRegion().getArguments().slice(start, getNumResults());
    }
    ArrayRef<BlockArgument> getReadwriteRefArgs() {
      return getReadwriteRefArgsMutable();
    }

    OpResult getTiedResult(BlockArgument b) {
      int64_t readwriteStart = getNumLeadingArgs() + getNumReadonlyRefs();
      assert(b.getArgNumber() >= readwriteStart && "not a readwrite arg");
      return (*this)->getOpResult(b.getArgNumber() - readwriteStart);
    }

    ShapedType getResultType(int64_t i) {
      return cast<ShapedType>(getResults()[i].getType());
    }
  }];
}

def TileGroupOp : PCF_Op<"shared_executor.tile_group", [
       RecursiveMemoryEffects,
       SingleBlock,
       DeclareOpInterfaceMethods<PCF_NamespaceOpInterface>,
     ]> {
  let summary = [{
    Partition a threadgroup or cluster into sub-clusters at split points.
  }];
  let description = [{
    Destructures a source threadgroup or cluster into sub-clusters that
    collectively cover the source's full worker grid. Split points are
    provided per dimension as a nested list of index values.

    The number of resulting clusters is:
      product(num_splits_in_dim_i + 1) for all dimensions i

    Block arguments are ordered in row-major order over the partition grid.

    The source must not have struct elements (destructuring with struct is
    currently illegal). All result clusters have no struct elements.

    An optional namespace name can be specified with `ns(name)`. When
    present, all cluster IDs must be qualified with the namespace name as
    the first path segment. When absent (anonymous namespace), all cluster
    IDs must be leaf-only.

    Example without namespace (anonymous):
    ```mlir
    pcf.shared_executor.tile_group %tg split [[%k]]
        (%left: !pcf.cluster<#scope, (0 -> d0), left>,
         %right: !pcf.cluster<#scope, (d0 -> s0), right>) {
      pcf.return
    } : !pcf.threadgroup<#scope>
    ```

    Example with named namespace:
    ```mlir
    pcf.shared_executor.tile_group %tg ns(tg) split [[%k]]
        (%left: !pcf.cluster<#scope, (0 -> d0), tg.left>,
         %right: !pcf.cluster<#scope, (d0 -> s0), tg.right>) {
      pcf.return
    } : !pcf.threadgroup<#scope>
    ```
  }];

  let arguments = (ins
      AnyType:$source,
      Variadic<Index>:$splitPoints,
      DenseI64ArrayAttr:$numSplitsPerDim,
      OptionalAttr<StrAttr>:$nsName);

  let regions = (region MinSizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;

  let builders = [
      OpBuilder<(ins
          "Value":$source,
          "ArrayRef<SmallVector<Value>>":$splitPointsPerDim,
          "ArrayRef<ClusterType>":$resultClusterTypes,
          CArg<"StringAttr", "{}">:$nsName)>,
  ];

  let extraClassDeclaration = [{
    /// Returns the scope from the source type.
    ScopeAttrInterface getScope();

    /// Returns the split points organized per dimension.
    SmallVector<SmallVector<Value>> getSplitPointsPerDim();

    /// Returns the cluster block arguments.
    ArrayRef<BlockArgument> getClusterArgs() {
      return getBody().getArguments();
    }
  }];
}

def ClusterYieldOp : PCF_Op<"cluster_yield", [
  ParentOneOf<[
    "IREE::PCF::RunClusterOp",
    "IREE::PCF::RunThreadOp",
  ]>,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = [{Yields values from a cluster execution region.}];
  let description = [{
    Terminates the body of `run_cluster` or `run_thread`. The operands
    populate the result cluster's `sharedTypes` (if parent is
    `run_cluster`) or `privateTypes` (if parent is `run_thread`).

    Example:
    ```mlir
    pcf.cluster_yield %v1, %v2 : type1, type2
    pcf.cluster_yield
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$values);

  let hasCustomAssemblyFormat = 1;
}

def RunClusterOp : PCF_Op<"shared_executor.run_cluster", [
       AttrSizedOperandSegments,
       RecursiveMemoryEffects,
       SingleBlock,
       SingleBlockImplicitTerminator<"mlir::iree_compiler::IREE::PCF::ClusterYieldOp">,
     ]> {
  let summary = [{
    Execute a cluster region collectively with shared values.
  }];
  let description = [{
    Executes a region where all workers in the cluster cooperate. The body
    receives shared struct values from the input clusters as block
    arguments, grouped per cluster in operand order.

    All input clusters must have identical `boundsMap` and `scope`. The
    dependent values for the range bounds are provided as `rangeValues`.

    Optionally produces a result cluster with the same scope and bounds
    but new shared struct elements. The result cluster cannot have
    private elements (collective execution cannot produce per-thread data).

    Example:
    ```mlir
    %result = pcf.shared_executor.run_cluster(%c0, %c1)[%k]
        (%s0: tensor<64xf32>, %s1: f32) {
      pcf.cluster_yield %new_s : tensor<64xf32>
    } : (!pcf.cluster<...>, !pcf.cluster<...>) -> !pcf.cluster<...>
    ```
  }];

  let arguments = (ins
      Variadic<PCF_AnyCluster>:$sources,
      Variadic<Index>:$rangeValues);

  let results = (outs Optional<PCF_AnyCluster>:$result);
  let regions = (region MinSizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;

  let builders = [
      OpBuilder<(ins
          "ValueRange":$sources,
          "ValueRange":$rangeValues,
          "ArrayRef<Type>":$bodyArgTypes,
          CArg<"ClusterType", "{}">:$resultType)>,
  ];

  let extraClassDeclaration = [{
    /// Returns the scope from the first source cluster.
    ScopeAttrInterface getScope() {
      return cast<ClusterType>(getSources().front().getType()).getScope();
    }

    /// Returns the boundsMap from the first source cluster.
    AffineMap getBoundsMap() {
      return cast<ClusterType>(getSources().front().getType()).getBoundsMap();
    }

    /// Returns the result cluster type, or nullptr if no result.
    ClusterType getResultClusterType() {
      if (getResult()) {
        return cast<ClusterType>(getResult().getType());
      }
      return {};
    }
  }];
}

def RunThreadOp : PCF_Op<"shared_executor.run_thread", [
       AttrSizedOperandSegments,
       RecursiveMemoryEffects,
       SingleBlock,
       SingleBlockImplicitTerminator<"mlir::iree_compiler::IREE::PCF::ClusterYieldOp">,
     ]> {
  let summary = [{
    Execute a cluster region per-thread with private values.
  }];
  let description = [{
    Executes a region from the perspective of a single worker in the
    cluster. The body receives private struct values from the input
    clusters as block arguments (grouped per cluster in operand order),
    followed by per-dimension thread IDs.

    Thread IDs are tile-relative: each ranges from 0 to tile_size-1,
    where tile_size = upper_bound - lower_bound. Thread ID 0 corresponds
    to the scope worker at the cluster's lower bound.

    All input clusters must have identical `boundsMap` and `scope`. The
    dependent values for the range bounds are provided as `rangeValues`.

    Optionally produces a result cluster with the same scope and bounds
    but new private struct elements. The result cluster cannot have
    shared elements (per-thread execution cannot produce collective
    data).

    Example:
    ```mlir
    %result = pcf.shared_executor.run_thread(%c0)[%k]
        (%priv: f32)[%tid0: index, %tid1: index] {
      pcf.cluster_yield %new_p : f32
    } : (!pcf.cluster<...>) -> !pcf.cluster<...>
    ```
  }];

  let arguments = (ins
      Variadic<PCF_AnyCluster>:$sources,
      Variadic<Index>:$rangeValues);

  let results = (outs Optional<PCF_AnyCluster>:$result);
  let regions = (region MinSizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;

  let builders = [
      OpBuilder<(ins
          "ValueRange":$sources,
          "ValueRange":$rangeValues,
          "ArrayRef<Type>":$bodyArgTypes,
          "int64_t":$numThreadIds,
          CArg<"ClusterType", "{}">:$resultType)>,
  ];

  let extraClassDeclaration = [{
    /// Returns the scope from the first source cluster.
    ScopeAttrInterface getScope() {
      return cast<ClusterType>(getSources().front().getType()).getScope();
    }

    /// Returns the boundsMap from the first source cluster.
    AffineMap getBoundsMap() {
      return cast<ClusterType>(getSources().front().getType()).getBoundsMap();
    }

    /// Returns the result cluster type, or nullptr if no result.
    ClusterType getResultClusterType() {
      if (getResult()) {
        return cast<ClusterType>(getResult().getType());
      }
      return {};
    }

    /// Returns the number of thread ID block arguments.
    int64_t getNumThreadIds() {
      return getScope().getNativeNumProcessorIds();
    }

    /// Returns the thread ID block arguments.
    MutableArrayRef<BlockArgument> getThreadIdArgs() {
      return getBody().getArguments().take_back(getNumThreadIds());
    }

    /// Returns the struct value block arguments (excluding thread IDs).
    MutableArrayRef<BlockArgument> getStructArgs() {
      return getBody().getArguments().drop_back(getNumThreadIds());
    }
  }];
}

} // OpGroupParallelOps


//===----------------------------------------------------------------------===//
// Control Flow Ops
//===----------------------------------------------------------------------===//

def OpGroupControlFlowOps : OpDocGroup {
  let summary = "Parallel execution ops";
  let description = "";
}

let opDocGroup = OpGroupControlFlowOps in {

//===----------------------------------------------------------------------===//
// BranchCondReturnOp
//===----------------------------------------------------------------------===//

def BranchCondReturnOp : PCF_Op<"br.cond_return", [
    ParentOneOf<[
      "IREE::PCF::GenericOp",
    ]>,
    DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
    Pure, Terminator
  ]> {
  let summary = "Branch operation with conditional return";
  let description = [{
    The `pcf.br.cond_return` operation represents a conditional branch operation
    to a given block, or return from the parent.

    Example:

    ```mlir
    pcf.<scoped op> #foo.scope {
      ^bb0(%0: !foo.type)
        %1 = ... %0 : !foo.type
        pcf.br.cond_return %cond ^bb0(%0: !foo.type)
    }
    ```
  }];

  let arguments = (ins
    I1:$condition,
    Variadic<AnyType>:$dest_operands
  );
  let successors = (successor AnySuccessor:$dest);

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let assemblyFormat = [{
    $condition $dest (`(` $dest_operands^ `:` type($dest_operands) `)`)? attr-dict
  }];
}

} // OpGroupControlFlowOps

//===----------------------------------------------------------------------===//
// WriteOps
//===----------------------------------------------------------------------===//

def OpGroupWriteOps : OpDocGroup {
  let summary = "Write ops";
  let description = "";
}

let opDocGroup = OpGroupWriteOps in {

  //===----------------------------------------------------------------------===//
  // WriteSliceOp
  //===----------------------------------------------------------------------===//

  def PCF_WriteSliceOp
      : PCF_Op<"write_slice", [AttrSizedOperandSegments,
                               OffsetSizeAndStrideOpInterface,
                               DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                               AllRanksMatch<["source", "dest"]>,
                               AllElementTypesMatch<["source", "dest"]>,
  ]> {
    let summary = [{
    Submit a write of a tensor, vector, or memref to a slice of a `pcf.sref`.
  }];
    let description = [{
    The `pcf.write_slice` operation supports the following arguments:

    * source: the shaped value that is written.
    * dest: the sref into which the source is written.
    * offsets: shaped-rank number of offsets into the `dest` into which
               the slice is inserted.
    * sizes: shaped-rank number of sizes which specify the sizes of the source
             tensor type.
    * strides: shaped-rank number of strides that specify subsampling in each
               dimension.
  }];

    let arguments = (ins AnyRankedTensorVectorOrMemRef:$source,
        PCF_AnyShapedRef:$dest, Variadic<Index>:$offsets,
        Variadic<Index>:$sizes, Variadic<Index>:$strides,
        DenseI64ArrayAttr:$static_offsets, DenseI64ArrayAttr:$static_sizes,
        DenseI64ArrayAttr:$static_strides);
    let assemblyFormat = [{
    $source `into` $dest ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `into` type($dest)
  }];

    let extraClassDeclaration = [{
    ShapedType getSourceType() {
      return ::llvm::cast<ShapedType>(getSource().getType());
    }

    PCF::ShapedRefType getDestType() {
      return ::llvm::cast<PCF::ShapedRefType>(getDest().getType());
    }

    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getDestType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before `offsets`, `sizes` and
    /// `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
  }];

    let builders = [
        // Build a WriteSliceOp with mixed static and dynamic entries.
        OpBuilder<(ins "Value":$source, "Value":$dest,
            "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
            "ArrayRef<OpFoldResult>":$strides,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build a WriteSliceOp with mixed static and dynamic entries
        // packed into a Range vector.
        OpBuilder<(ins "Value":$source, "Value":$dest,
            "ArrayRef<Range>":$ranges,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build a WriteSliceOp with dynamic entries.
        OpBuilder<(ins "Value":$source, "Value":$dest, "ValueRange":$offsets,
            "ValueRange":$sizes, "ValueRange":$strides,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

    let hasFolder = 1;
  }

} // OpGroupWriteOps

//===----------------------------------------------------------------------===//
// ReadOps
//===----------------------------------------------------------------------===//

def OpGroupReadOps : OpDocGroup {
  let summary = "Read ops";
  let description = "";
}

let opDocGroup = OpGroupReadOps in {

//===----------------------------------------------------------------------===//
// ReadSliceOp
//===----------------------------------------------------------------------===//

def PCF_ReadSliceOp : PCF_Op<"read_slice", [
       AttrSizedOperandSegments,
       OffsetSizeAndStrideOpInterface,
       AllRanksMatch<["source", "result"]>,
       AllElementTypesMatch<["source", "result"]>,
  ]> {
  let summary = [{
    Read a tensor or vector from a `pcf.sref` based on the provided slice
    parameters.
  }];
  let description = [{
    Read a slice from a `pcf.sref`. If this is reading a vector, the sizes
    may be smaller than the return vector type. In this case out of bounds
    elements have undefined value.

    The `pcf.read_slice` operation supports the following arguments:
    * source: the shaped value that is written.
    * dest: the sref into which the source is written.
    * offsets: shaped-rank number of offsets into the `dest` into which
               the slice is inserted.
    * sizes: shaped-rank number of sizes which specify the sizes of the source
             tensor type.
    * strides: shaped-rank number of strides that specify subsampling in each
               dimension.
  }];

  let arguments = (ins
    PCF_AnyShapedRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );

  let results = (outs AnyTypeOf<[AnyRankedTensor, AnyVectorOfAnyRank]>:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    PCF::ShapedRefType getSourceType() {
      return ::llvm::cast<PCF::ShapedRefType>(getSource().getType());
    }

    ShapedType getResultType() {
      return ::llvm::cast<ShapedType>(getResult().getType());
    }

    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before `offsets`, `sizes` and
    /// `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];

  let builders = [
    // Build a ReadSliceOp with mixed static and dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ReadSliceOp with mixed static and dynamic entries
    // packed into a Range vector.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<Range>":$ranges,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ReadSliceOp with dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// GetMemrefOp
//===----------------------------------------------------------------------===//

def PCF_GetMemrefOp : PCF_Op<"get_memref", [
       AttrSizedOperandSegments,
       OffsetSizeAndStrideOpInterface,
       AllRanksMatch<["source", "result"]>,
       AllElementTypesMatch<["source", "result"]>,
  ]> {
  let summary = [{
    Extract a memref view from a slice of a `pcf.sref`.
  }];
  let description = [{
    The `pcf.get_memref` operation extracts a memref view from a slice of a
    sref, breaking the synchronization guarantees of the source.

    The returned memref must have a maximally dynamic layout (all strides and
    offset dynamic) and no memory space. Layout and memory space information
    is determined by the ConvertSRefToMemRef analysis pass.

    The operation supports the following arguments:
    * source: the sref from which to extract a view.
    * offsets: shaped-rank number of offsets into the `source` from which
               the slice begins.
    * sizes: shaped-rank number of sizes which specify the sizes of the result
             memref type.
    * strides: shaped-rank number of strides that specify subsampling in each
               dimension.
  }];

  let arguments = (ins
    PCF_AnyShapedRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    PCF::ShapedRefType getSourceType() {
      return ::llvm::cast<PCF::ShapedRefType>(getSource().getType());
    }

    MemRefType getResultType() {
      return ::llvm::cast<MemRefType>(getResult().getType());
    }

    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before `offsets`, `sizes` and
    /// `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];

  let builders = [
    // Build a GetMemrefOp with mixed static and dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a GetMemrefOp with mixed static and dynamic entries
    // packed into a Range vector.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<Range>":$ranges,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a GetMemrefOp with dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasFolder = 1;
  let hasVerifier = 1;
}

} // OpGroupReadOps

//===----------------------------------------------------------------------===//
// Shape Ops
//===----------------------------------------------------------------------===//

def OpGroupShapeOps : OpDocGroup {
  let summary = "Shape manipulation ops";
  let description = "";
}

let opDocGroup = OpGroupShapeOps in {

//===----------------------------------------------------------------------===//
// ExpandShapeOp
//===----------------------------------------------------------------------===//

def PCF_ExpandShapeOp : PCF_Op<"expand_shape", [Pure]> {
  let summary = [{
    Reshape an sref by expanding dimensions according to a reassociation map.
  }];
  let description = [{
    The `pcf.expand_shape` operation reshapes a `pcf.sref` by expanding
    dimensions according to a reassociation map, analogous to
    `tensor.expand_shape`. The scope and element type of the result must
    match those of the source.

    Example:
    ```mlir
      %expanded = pcf.expand_shape %src [[0, 1], [2]]
        : !pcf.sref<128x4xf16, #scope> into !pcf.sref<8x16x4xf16, #scope>
    ```
  }];

  let arguments = (ins
    PCF_AnyShapedRef:$src,
    IndexListArrayAttr:$reassociation,
    Variadic<Index>:$output_shape
  );
  let results = (outs PCF_AnyShapedRef:$result);

  let assemblyFormat = [{
    $src $reassociation
    (`output_shape` `[` $output_shape^ `]`)?
    attr-dict `:` type($src) `into` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

def PCF_SubviewOp : PCF_Op<"subview", [
       Pure, AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface,
  ]> {
  let summary = [{
    Extract a sub-region of an sref as a new sref.
  }];
  let description = [{
    The `pcf.subview` operation extracts a sub-region of a `pcf.sref`,
    analogous to `tensor.extract_slice` or `memref.subview`. The scope
    and element type of the result must match those of the source.

    Example:
    ```mlir
      %sub = pcf.subview %src[%i, 0, 0] [1, 16, 4] [1, 1, 1]
        : !pcf.sref<8x16x4xf16, #scope> to !pcf.sref<16x4xf16, #scope>
    ```
  }];

  let arguments = (ins
    PCF_AnyShapedRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs PCF_AnyShapedRef:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    PCF::ShapedRefType getSourceType() {
      return ::llvm::cast<PCF::ShapedRefType>(getSource().getType());
    }

    PCF::ShapedRefType getResultType() {
      return ::llvm::cast<PCF::ShapedRefType>(getResult().getType());
    }

    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before `offsets`, `sizes` and
    /// `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];

  let builders = [
    // Build a SubviewOp with mixed static and dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubviewOp with dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasFolder = 1;
}

} // OpGroupShapeOps

//===----------------------------------------------------------------------===//
// Scope Ops
//===----------------------------------------------------------------------===//

def OpGroupScopeOps : OpDocGroup {
  let summary = "Scope ops";
  let description = "";
}

let opDocGroup = OpGroupScopeOps in {

//===----------------------------------------------------------------------===//
// InitSubscopeOp
//===----------------------------------------------------------------------===//

def InitSubscopeOp : PCF_Op<"init_subscope", [
    RecursiveMemoryEffects,
    SingleBlock,
    SingleBlockImplicitTerminator<"mlir::iree_compiler::IREE::PCF::YieldOp">,
  ]> {
  let summary = [{Allocate resources for a child scope and attach as struct fields.}];
  let description = [{
    Allocates memory for a child scope and attaches the allocations as
    struct fields to the threadgroup. The body contains allocation ops
    that produce values for the child scope. The yielded values become
    struct fields on the returned threadgroup.

    Semantics: run-once, uniform across all scope levels.

    The input threadgroup must have no struct fields. The output
    threadgroup has the same scope but with struct fields matching the
    yielded types.

    Example:
    ```mlir
    %tg2 = pcf.init_subscope %tg {
      %alloc = pcf.alloc() : !pcf.sref<128x64xf16, #child_scope>
      pcf.yield %alloc : !pcf.sref<128x64xf16, #child_scope>
    } -> !pcf.threadgroup<#parent_scope, {!pcf.sref<128x64xf16, #child_scope>}>
    ```
  }];

  let arguments = (ins PCF_AnyThreadGroup:$source);
  let results = (outs PCF_AnyThreadGroup:$result);
  let regions = (region MinSizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TelescopeOp
//===----------------------------------------------------------------------===//

def TelescopeOp : PCF_Op<"telescope", [Pure]> {
  let summary = [{Convert a threadgroup identity from parent to child scope.}];
  let description = [{
    A pure type conversion op that takes a threadgroup with optional
    struct fields and an index operand (the thread ID). Returns a
    child-scope threadgroup (no struct fields) plus the struct fields
    as separate SSA values.

    The thread ID operand is the parent scope's worker ID. Passing a
    value that is not the actual thread ID is undefined behavior.

    The child scope is determined by the result threadgroup type.

    Example with struct fields:
    ```mlir
    %child_tg, %sref = pcf.telescope %tg[%tid]
        : !pcf.threadgroup<#parent, {!pcf.sref<128x64xf16, #child>}>
       -> (!pcf.threadgroup<#child>, !pcf.sref<128x64xf16, #child>)
    ```

    Example without struct fields (pure scope conversion):
    ```mlir
    %child_tg = pcf.telescope %tg[%tid]
        : !pcf.threadgroup<#parent> -> !pcf.threadgroup<#child>
    ```
  }];

  let arguments = (ins PCF_AnyThreadGroup:$source, Index:$threadId);
  let results = (outs Variadic<AnyType>:$results);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

} // OpGroupScopeOps

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

def PCF_YieldOp : PCF_Op<"yield", [ParentOneOf<["IREE::PCF::GenericOp",
                                                 "IREE::PCF::SharedExecutorOp",
                                                 "IREE::PCF::InitSubscopeOp"]>, Pure,
                                   ReturnLike, Terminator,
]> {
  let summary = [{Yields results from a region.}];
  let description = [{
    The values returned are copied by-value.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let builders = [
    // Builder with no operands (used by SingleBlockImplicitTerminator).
    OpBuilder<(ins), [{
      return build($_builder, $_state, ValueRange{});
    }]>,
  ];

  let assemblyFormat = [{
    attr-dict
    $operands `:` type($operands)
  }];
}

//===----------------------------------------------------------------------===//
// IndexSymbolOp
//===----------------------------------------------------------------------===//

def PCF_IndexSymbolOp : PCF_Op<"index_symbol", [
    PCF_NamespaceSymbolOpInterface,
  ]> {
  let summary = [{Binds an index value to a symbol name in the enclosing namespace.}];
  let description = [{
    Defines a named symbol with an index value in the enclosing namespace.
    The symbol can be resolved by name from ops in the namespace's scope.

    Must be placed in the symbol-defining region (e.g., the initializer)
    of a NamespaceOpInterface op.

    Example:
    ```mlir
    pcf.generic scope(#pcf.sequential)
      initialize {
        %n = arith.constant 4 : index
        pcf.index_symbol "tile_count" = %n
        pcf.yield
      }
      execute(...) { ... }
    ```
  }];

  let arguments = (ins StrAttr:$sym_name, Index:$value);
  let results = (outs);

  let assemblyFormat = [{
    $sym_name `=` $value attr-dict
  }];

  let extraClassDeclaration = [{
    // NamespaceSymbolOpInterface methods.
    StringAttr getSymbolName() {
      return getSymNameAttr();
    }
    OpFoldResult getSymbolDefinition() {
      return getValue();
    }
  }];
}

def PCF_ReturnOp : PCF_Op<"return", [
  ParentOneOf<[
    "IREE::PCF::GenericOp",
    "IREE::PCF::LoopOp",
    "IREE::PCF::SharedExecutorOp",
    "IREE::PCF::TileGroupOp",
  ]>,
  Pure,
  Terminator,
  // Note: This op is not return-like because it does not produce the results
  // of the region. The parent is responsible for that.
]> {
  let summary = [{Returns from a thread.}];
  let description = [{
    Returns control flow to the parent without fencing memory. If the parent
    carries an implicit fence one may still occur after the parent has finished.
  }];

  let arguments = (ins);

  let assemblyFormat = [{attr-dict}];
}

#endif  // IREE_CODEGEN_DIALECT_PCF_OPS
