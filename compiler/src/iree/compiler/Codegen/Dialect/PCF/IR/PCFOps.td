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
       RecursiveMemoryEffects,
       DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       // DestinationStyleOpInterface
     ]> {
  let summary = "evaluate a region multiple times in parallel";
  let description = [{
    Example:
    ```mlir
      %0 = ... : tensor<?>
      %1 = ... : tensor<?>
      %d0, %d1 = ... : index
      %n = ... : index
      %2:4 = pcf.generic scope(#foo.scope) tripcount(%n) {attr-dict-with-keyword}
        initialize(%ref = %0, %ref_1[%token: !pcf.token<#foo.scope>], %ref_2, %ref_3[%token: !pcf.token<#foo.scope>] = %1)[%num_threads: index]
                : (!pcf.tensor<?>, !pcf.tensor<?>, !pcf.tensor<?>, !pcf.tensor<?>)
               -> (tensor<?>, tensor<?>{%d0}, tensor<?>{%d1}, tensor<?>) {
        ...
      }
    ```
  }];
  let arguments = (ins
    PCF_ScopeAttr:$scope,
    Optional<Index>:$tripcount,
    Variadic<PCF_AnyTensorOrMemRef>:$inits,
    Variadic<Index>:$dynamic_sizes,
    ArrayProp<BoolProp>:$is_tied,
    ArrayProp<BoolProp>:$has_token
  );

  let results = (outs Variadic<PCF_AnyTensorOrMemRef>:$results);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `scope` `(` $scope `)`
    (`tripcount` `(` $tripcount^ `)`)?
    custom<ParallelBody>($inits,
                         type($inits),
                         $dynamic_sizes,
                         type($results),
                         $is_tied,
                         $has_token,
                         $region)
    prop-dict attr-dict
  }];
  let hasVerifier = 1;

  // // The default builder does not add the proper body BBargs, roll our own.
  // // let skipDefaultBuilders = 1;
  // let builders = [
  //   // Builder with tripcount.
  //   OpBuilder<(ins "TypeRange":$result_types, "ScopeAttr":$scope, "Value":$tripcount,
  //      "ArrayRef<Value>":$inits, "ArrayRef<Value>":$dynamic_sizes)>,
  //   // Builder without tripcount.
  //   OpBuilder<(ins "TypeRange":$result_types, "ScopeAttr":$scope,
  //      "ArrayRef<Value>":$inits, "ArrayRef<Value>":$dynamic_sizes)>,
  //   // Builder with inferred result types from inits.
  //   OpBuilder<(ins "ScopeAttr":$scope, "ArrayRef<Value>":$inits)>,
  //   // Builder with no inits.
  //   OpBuilder<(ins "TypeRange":$result_types, "ScopeAttr":$scope,
  //       "ArrayRef<Value>":$dynamic_sizes)>,
  //   // Same builders with tripcount.
  //   OpBuilder<(ins "ScopeAttr":$scope, "Value":$tripcount, "ArrayRef<Value>":$inits)>,
  //   OpBuilder<(ins "TypeRange":$result_types, "ScopeAttr":$scope, "Value":$tripcount,
  //       "ArrayRef<Value>":$dynamic_sizes)>,
  // ];

  let extraClassDeclaration = [{
    BlockArgument getNumThreadsArg() {
      return getRegion().getArguments().front();
    }
    MutableArrayRef<BlockArgument> getRegionRefArgs() {
      return getRegion().getArguments().drop_front().take_front(getNumResults());
    }
    MutableArrayRef<BlockArgument> getRegionTokenArgs() {
      // 1 for the thread count. The rest is split between shaped refs and
      // tokens where each result has a single associated ref.
      return getRegion().getArguments().drop_front(1 + getNumResults());
    }

    SmallVector<int64_t> getTiedResults() {
      SmallVector<int64_t> tiedResults;
      for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
        if (isTied) {
          tiedResults.push_back(i);
        }
      }
      return tiedResults;
    }

    SmallVector<int64_t> getRegionRefBlockArgIndicesWithToken() {
      SmallVector<int64_t> tokenIndices;
      for (auto [i, hasToken] : llvm::enumerate(getHasToken())) {
        if (hasToken) {
          tokenIndices.push_back(i);
        }
      }
      return tokenIndices;
    }
  }];
}

} // OpGroupParallelOps

//===----------------------------------------------------------------------===//
// WriteOps
//===----------------------------------------------------------------------===//

def OpGroupWriteOps : OpDocGroup {
  let summary = "Write ops";
  let description = "";
}

let opDocGroup = OpGroupWriteOps in {

//===----------------------------------------------------------------------===//
// ParallelInsertSliceOp
//===----------------------------------------------------------------------===//

def PCF_WriteSliceOp : PCF_Op<"write_slice", [
       AttrSizedOperandSegments,
       OffsetSizeAndStrideOpInterface,
       AllRanksMatch<["source", "dest"]>,
       AllElementTypesMatch<["source", "dest"]>,
  ]> {
  let summary = [{
    Submit a write of a tensor, vector, or memref to a slice of a `pcf.shaped_ref`.
  }];
  let description = [{
    The `pcf.write_slice` operation supports the following arguments:

    * source: the shaped value that is written.
    * dest: the shaped_ref into which the source is written.
    * offsets: shaped-rank number of offsets into the `dest` into which
               the slice is inserted.
    * sizes: shaped-rank number of sizes which specify the sizes of the source
             tensor type.
    * strides: shaped-rank number of strides that specify subsampling in each
               dimension.
  }];

  let arguments = (ins
    PCF_AnyTensorVectorOrMemRef:$source,
    PCF_AnyShapedRef:$dest,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
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
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];
}

} // OpGroupWriteOps

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
  ParentOneOf<[
    "IREE::PCF::GenericOp",
    // "IREE::PCF::ConcurrentOp",
  ]>,
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

def PCF_ReturnOp : PCF_Op<"return", [
  ParentOneOf<[
    "IREE::PCF::GenericOp",
    // "IREE::PCF::ConcurrentOp",
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
