// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_TEMPLATE_DIALECT
#define IREE_CODEGEN_DIALECT_TEMPLATE_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Properties.td"

//===----------------------------------------------------------------------===//
// Template dialect
//===----------------------------------------------------------------------===//

def Template_Dialect : Dialect {
  let name = "template";
  let cppNamespace = "::mlir::iree_compiler::IREE::Template";

  let summary = [{
    A dialect for template-like IR structures with progressive specialization.
  }];
  let description = [{
    The Template dialect represents template-like IR structures designed to work
    in three stages:

    1. **Template definition**: Representing functions with templated types
       (`!template.type<N>`) and unimplemented blocks (`template.unimplemented`).

    2. **Progressive filling**: Filling in blocks of callees based on the
       semantics of the caller, progressively specializing abstract templates.

    3. **Full instantiation**: Cloning into fully specialized instances where
       all template types are concrete and all blocks are implemented.

    The goal is to enable the most complex individual op/DAG rewrites to happen
    iteratively rather than in a single monolithic transformation.

    Example workflow:
    ```mlir
    // Stage 1: Template definition with abstract blocks
    template.func @matmul_template
    implementations {
    ^load_a:
      template.unimplemented -> !template.type<0>
    ^compute(%a: !template.type<0>, %b: !template.type<1>):
      template.unimplemented -> !template.type<2>
    }
    main { ... }

    // Stage 2: Caller instantiates with concrete types and implementations
    %result = template.call @matmul_template
        types(!template.type<0> = tensor<4x4xf16>, ...)
        implementations {
          ^load_a: %tile = ... ; template.return %tile
        }

    // Stage 3: Fully specialized instance (after inlining/cloning)
    %result = template.instance implementations {
    ^load_a:
      %tile = vector.load ... : vector<4x4xf16>
      template.return %tile : vector<4x4xf16>
    ^compute(%a: vector<4x4xf16>, %b: vector<4x8xf16>):
      %c = vector.contract ... : vector<4x8xf32>
      template.return %c : vector<4x8xf32>
    } main { ... }
    ```
  }];

  let dependentDialects = [];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Template properties
//===----------------------------------------------------------------------===//

// Nested array of types for template type bindings.
// Stored as SmallVector<SmallVector<Type>>, serialized as ArrayAttr of ArrayAttr of TypeAttr.
def TypeBindingsProp : Property<
    "::llvm::SmallVector<::llvm::SmallVector<::mlir::Type>>",
    "type bindings"> {
  // Use storage type directly since ArrayRef<ArrayRef<T>> cannot be formed
  // from SmallVector<SmallVector<T>> (inner vectors need stable backing).
  let convertToAttribute = [{
    ::llvm::SmallVector<::mlir::Attribute> outerAttrs;
    for (const auto &inner : $_storage) {
      ::llvm::SmallVector<::mlir::Attribute> innerAttrs;
      for (::mlir::Type t : inner) {
        innerAttrs.push_back(::mlir::TypeAttr::get(t));
      }
      outerAttrs.push_back(::mlir::ArrayAttr::get($_ctxt, innerAttrs));
    }
    return ::mlir::ArrayAttr::get($_ctxt, outerAttrs);
  }];
  let convertFromAttribute = [{
    auto outerArray = ::llvm::dyn_cast_if_present<::mlir::ArrayAttr>($_attr);
    if (!outerArray) {
      return $_diag() << "expected array attribute for type bindings";
    }
    $_storage.clear();
    for (::mlir::Attribute outerElem : outerArray) {
      auto innerArray = ::llvm::dyn_cast<::mlir::ArrayAttr>(outerElem);
      if (!innerArray) {
        return $_diag() << "expected nested array attribute";
      }
      ::llvm::SmallVector<::mlir::Type> innerTypes;
      for (::mlir::Attribute innerElem : innerArray) {
        auto typeAttr = ::llvm::dyn_cast<::mlir::TypeAttr>(innerElem);
        if (!typeAttr) {
          return $_diag() << "expected type attribute";
        }
        innerTypes.push_back(typeAttr.getValue());
      }
      $_storage.push_back(std::move(innerTypes));
    }
    return ::mlir::success();
  }];
  let hashProperty = [{
    [&]() {
      size_t hash = 0;
      for (const auto &inner : $_storage) {
        for (::mlir::Type t : inner) {
          hash = ::llvm::hash_combine(hash, t);
        }
        hash = ::llvm::hash_combine(hash, inner.size());
      }
      return hash;
    }()
  }];
  let writeToMlirBytecode = [{
    writer.writeVarInt($_storage.size());
    for (const auto &inner : $_storage) {
      writer.writeVarInt(inner.size());
      for (::mlir::Type t : inner) {
        writer.writeType(t);
      }
    }
  }];
  let readFromMlirBytecode = [{
    uint64_t outerSize;
    if (::mlir::failed(reader.readVarInt(outerSize))) {
      return ::mlir::failure();
    }
    $_storage.clear();
    $_storage.reserve(outerSize);
    for (uint64_t i = 0; i < outerSize; ++i) {
      uint64_t innerSize;
      if (::mlir::failed(reader.readVarInt(innerSize))) {
        return ::mlir::failure();
      }
      ::llvm::SmallVector<::mlir::Type> innerTypes;
      innerTypes.reserve(innerSize);
      for (uint64_t j = 0; j < innerSize; ++j) {
        ::mlir::Type t;
        if (::mlir::failed(reader.readType(t))) {
          return ::mlir::failure();
        }
        innerTypes.push_back(t);
      }
      $_storage.push_back(std::move(innerTypes));
    }
    return ::mlir::success();
  }];
}

//===----------------------------------------------------------------------===//
// Base template dialect op classes
//===----------------------------------------------------------------------===//

class Template_Op<string mnemonic, list<Trait> traits = []>
    : Op<Template_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Template types
//===----------------------------------------------------------------------===//

def Template_TypeType : TypeDef<Template_Dialect, "Type"> {
  let mnemonic = "type";

  let summary = [{A template type parameter reference.}];
  let description = [{
    A reference to a type parameter in a template. The `id` identifies which
    type parameter this refers to. During template instantiation, this type
    is replaced with the concrete type provided at the callsite.

    Example:
    ```mlir
    // In a template definition
    ^bb0(%arg: !template.type<0>):
      ...

    // At instantiation, !template.type<0> might become tensor<4x4xf32>
    ```
  }];

  let parameters = (ins "int64_t":$id);
  let assemblyFormat = "`<` $id `>`";
}

#endif  // IREE_CODEGEN_DIALECT_TEMPLATE_DIALECT
