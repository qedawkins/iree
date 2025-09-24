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
    control flow, the pcf dialect includes types and operations that represent
    dataflow across parallel workers.

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
             Bufferize | // PCF on Memref
                       |
            ResolvePCF |
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

class PCF_Op<string mnemonic, list<Trait> traits = []> :
    Op<PCF_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Base pcf types
//===----------------------------------------------------------------------===//

def PCF_Token : TypeDef<PCF_Dialect, "Token", []> {
  let mnemonic = "token";

  let summary = [{A token handle used for synchronizing workers in the parent matching scope.}];
  let description = [{
    A token with a scope used to synchronize a resource.
  }];

  let parameters = (ins
    "PCF::ScopeAttr":$scope
  );
  let assemblyFormat = "`<` $scope `>`";
}

def PCF_ShapedRef : TypeDef<PCF_Dialect, "ShapedRef", [
    ShapedTypeInterface
]> {
  let mnemonic = "sref";

  let summary = [{A shaped reference to a buffer.}];
  let description = [{
    A reference to a buffer with unspecified layout and physical storage.
    Carries the shape and element type of the referenced region. Elements can
    be accessed by index, though no assumptions about the physical relation
    between two coordinates can be made. Elements referenced by an `sref` may
    not internally alias.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "PCF::ScopeAttr":$scope
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ShapedRefType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const {
      return ShapedRefType::get(getContext(), shape.value_or(getShape()), elementType, getScope());
    }

    bool hasRank() const {
      return true;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Base pcf attributes
//===----------------------------------------------------------------------===//

def PCF_SequentialAttr :
    AttrDef<PCF_Dialect, "Sequential", [
      DeclareAttrInterfaceMethods<PCF_ScopeAttr>
    ]> {
  let mnemonic = "sequential";
  let summary = [{Attribute representing sequential execution}];
  let description = [{
    Scope that reuses the current process as the sole executor of a parallel
    region.
  }];
  let parameters = (ins);
  let assemblyFormat = [{}];

  let extraClassDeclaration = [{
    SmallVector<Value> getWorkerIDs(OpBuilder& builder, Location loc, ArrayRef<OpFoldResult> basis) {
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      SmallVector<Value> ids(basis.size(), zero);
      return ids;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Misc aliases
//===----------------------------------------------------------------------===//

def PCF_AnyToken: AnyTypeOf<[PCF_Token]>;
def PCF_AnyRef: AnyTypeOf<[PCF_ShapedRef, AnyMemRef]>;
def PCF_AnyShapedRef: AnyTypeOf<[PCF_ShapedRef]>;
def PCF_AnyTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyMemRef]>;
def PCF_AnyTensorVectorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyVectorOfAnyRank, AnyMemRef]>;

#endif  // IREE_CODEGEN_DIALECT_PCF_DIALECT
