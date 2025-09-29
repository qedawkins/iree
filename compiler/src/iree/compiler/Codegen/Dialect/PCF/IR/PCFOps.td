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
  let summary = [{
    Execute a set of workers in parallel on a region.
  }];
  let description = [{
    Executes a region across a set of workers at a specified scope. The scope
    is given by an attribute implementing the `ScopeAttr` interface and is
    responsible for the semantics of all pcf primitives at the same scope.
    Further details about scopes are included in the docs for the interface.

// DO NOT SUBMIT

    Example:
    ```mlir
      %0 = ... : tensor<?>
      %1 = ... : tensor<?x?>
      %d0, %d1 = ... : index
      %n = ... : index
      %2:4 = pcf.generic scope(#foo.scope) count(%n) {attr-dict-with-keyword}
        initialize(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%num_threads: index]
                : (!pcf.sref<?, #foo.scope>, !pcf.sref<?, #foo.scope>, !pcf.sref<?x?, #foo.scope>, !pcf.sref<?x?, #foo.scope>)
               -> (tensor<?>, tensor<?>{%d0}, tensor<?x?>{%d0, %d1}, tensor<?x?>) {
        ...
      }
    ```
  }];
  let arguments = (ins
    PCF_ScopeAttr:$scope,
    Variadic<Index>:$count,
    Variadic<PCF_AnyTensorOrMemRef>:$inits,
    Variadic<Index>:$dynamic_sizes,
    ArrayProp<BoolProp>:$is_tied
  );

  let results = (outs Variadic<PCF_AnyTensorOrMemRef>:$results);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `scope` `(` $scope `)`
    (`count` `(` $count^ `)`)?
    custom<ParallelBody>($inits,
                         type($inits),
                         $dynamic_sizes,
                         type($results),
                         $is_tied,
                         $region)
    prop-dict attr-dict
  }];
  let hasVerifier = 1;

  // The default builder does not add the proper body BBargs, roll our own.
  // let skipDefaultBuilders = 1;
  let builders = [
    // Builder with return types inferred from inits.
    OpBuilder<(ins "ScopeAttr":$scope, "ArrayRef<Value>":$count,
                   "ArrayRef<Value>":$inits)>,
    // Builder with no inits.
    OpBuilder<(ins "TypeRange":$result_types, "ScopeAttr":$scope,
                   "ArrayRef<Value>":$count,
                   "ArrayRef<Value>":$dynamic_sizes)>,
    // Builder with everything.
    OpBuilder<(ins "TypeRange":$result_types, "ScopeAttr":$scope,
                   "ArrayRef<Value>":$count, "ArrayRef<Value>":$inits,
                   "ArrayRef<Value>":$dynamic_sizes, "ArrayRef<bool>":$is_tied)>,
  ];

  let extraClassDeclaration = [{
    int64_t getNumCountArgs() {
      int64_t numCountArgs = getCount().size();
      return numCountArgs == 0 ? 1 : numCountArgs;
    }
    MutableArrayRef<BlockArgument> getCountArgs() {
      return getRegion().getArguments().take_front(getNumCountArgs());
    }
    MutableArrayRef<BlockArgument> getRegionRefArgs() {
      return getRegion().getArguments().drop_front(getNumCountArgs()).take_front(getNumResults());
    }

    SmallVector<int64_t> getInitTiedResultIndices() {
      SmallVector<int64_t> tiedResults;
      for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
        if (isTied) {
          tiedResults.push_back(i);
        }
      }
      return tiedResults;
    }

    OpResult getTiedResult(OpOperand &operand) {
      int64_t beginIndex = getInits().getBeginOperandIndex();
      int64_t operandIndex = operand.getOperandNumber();
      if (operandIndex < beginIndex || operandIndex >= getInits().size() + beginIndex) {
        return OpResult();
      }

      int64_t initIndex = operandIndex - beginIndex;
      for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
        if (isTied) {
          if (initIndex == 0) {
            return (*this)->getOpResult(i);
          }
          --initIndex;
        }
      }

      return OpResult();
    }

    OpOperand *getTiedInit(int64_t i) {
      if (i < 0 || i >= getNumResults() || !getIsTied()[i]) {
        return nullptr;
      }

      int64_t initIndex = llvm::count(getIsTied().take_front(i), true);
      return &getInitsMutable()[initIndex];
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
// BranchOp
//===----------------------------------------------------------------------===//

def BranchIdOp : PCF_Op<"br.id", [
    ParentOneOf<[
      "IREE::PCF::GenericOp",
      // "IREE::PCF::ConcurrentOp",
    ]>,
    DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
    Pure, Terminator
  ]> {
  let summary = "Branch operation with worker id";
  let description = [{
    The `pcf.br.id` operation represents a direct branch operation to a given
    block, providing the current worker's id as the first successor operand.

    Example:

    ```mlir
    pcf.<scoped op> #foo.scope {
      ^entry_block:
        %0 = ... : !foo.type
        pcf.br.id ^bb1(%0 : !foo.type)
      ^bb1(%id: index, %arg0: !foo.type):
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let builders = [
    OpBuilder<(ins "Block *":$dest,
                   CArg<"ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>];

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

// DO NOT SUBMIT
// def BranchNextIdOp : PCF_Op<"br.next_id", [
//     ParentOneOf<[
//       "IREE::PCF::GenericOp",
//       // "IREE::PCF::ConcurrentOp",
//     ]>,
//     DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
//     Pure, Terminator
//   ]> {
//   let summary = "Branch operation with worker id";
//   let description = [{
//     The `pcf.br.next_id` operation represents a conditional branch with the next
//     tile id owned by the current worker. If the current id is the final
// 
//     Example:
// 
//     ```mlir
//     pcf.<scoped op> #foo.scope {
//       ^entry_block:
//         pcf.br.id ^bb1
//       ^bb1(%id: index):
//         pcf.br.next_id %id ^bb1
//       ^bb1(%id: index):
//     }
//     ```
// 
//     This operation also includes a returning mode if the successor block is
//     omitted.
// 
//     Example:
// 
//     ```mlir
//     pcf.<scoped op> #foo.scope {
//       ^entry_block:
//         pcf.br.id ^bb1
//       ^bb1(%id: index):
//         pcf.br.id ^bb1
//     }
//     ```
//   }];
// 
//   let arguments = (ins Variadic<AnyType>:$destOperands);
//   let successors = (successor AnySuccessor:$dest);
// 
//   let builders = [
//     OpBuilder<(ins "Block *":$dest,
//                    CArg<"ValueRange", "{}">:$destOperands), [{
//       $_state.addSuccessors(dest);
//       $_state.addOperands(destOperands);
//     }]>];
// 
//   let extraClassDeclaration = [{
//     void setDest(Block *block);
// 
//     /// Erase the operand at 'index' from the operand list.
//     void eraseOperand(unsigned index);
//   }];
// 
//   let assemblyFormat = [{
//     $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
//   }];
// }

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
// DO NOT SUBMIT
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

// def PCF_JoinYieldOp : PCF_Op<"join_yield", [
//   ParentOneOf<[
//     "IREE::PCF::GenericOp",
// // DO NOT SUBMIT
//     // "IREE::PCF::ConcurrentOp",
//   ]>,
//   Terminator,
// ]> {
//   let summary = [{Joins and yields values from a parallel execution region.}];
//   let description = [{
//     Joins workers on the specified tokens and yields from the parallel
//     execution region.
//   }];
// 
//   let arguments = (ins
//     Variadic<PCF_AnyToken>:$tokens
//   );
// 
//   let assemblyFormat = [{
//     attr-dict ($tokens^ `:` type($tokens))?
//   }];
// }

def PCF_ReturnOp : PCF_Op<"return", [
  ParentOneOf<[
    "IREE::PCF::GenericOp",
// DO NOT SUBMIT
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
