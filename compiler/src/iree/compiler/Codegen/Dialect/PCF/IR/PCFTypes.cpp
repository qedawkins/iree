// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.cpp.inc" // IWYU pragma: keep

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// #pcf.sref<...>
//===----------------------------------------------------------------------===//

Type ShapedRefType::parse(AsmParser &parser) {
  if (parser.parseLess()) {
    return {};
  }

  SmallVector<int64_t> shape;
  Type elementType;
  Attribute scope;

  SMLoc shapeLoc = parser.getCurrentLocation();
  if (failed(parser.parseDimensionList(shape))) {
    parser.emitError(shapeLoc, "failed to parse parameter 'shape'");
    return {};
  }

  SMLoc elemTypeLoc = parser.getCurrentLocation();
  if (failed(parser.parseType(elementType))) {
    parser.emitError(elemTypeLoc, "failed to parse parameter 'elementType'");
    return {};
  }

  if (failed(parser.parseComma())) {
    return {};
  }

  Attribute syncScope;
  if (succeeded(parser.parseOptionalKeyword("sync"))) {
    if (failed(parser.parseLParen())) {
      return {};
    }

    SMLoc scopeLoc = parser.getCurrentLocation();
    if (failed(parser.parseAttribute(scope))) {
      parser.emitError(scopeLoc, "failed to parse parameter 'scope'");
      return {};
    }

    // Special parsing for SyncOnReturnAttr sync scope.
    syncScope = SyncOnReturnAttr::get(parser.getContext());
    if (failed(parser.parseRParen())) {
      return {};
    }
  } else {
    SMLoc scopeLoc = parser.getCurrentLocation();
    if (failed(parser.parseAttribute(scope))) {
      parser.emitError(scopeLoc, "failed to parse parameter 'scope'");
      return {};
    }

    if (!isa<ScopeAttrInterface>(scope)) {
      parser.emitError(scopeLoc, "expected 'scope' parameter ")
          << scope << " to implement 'ScopeAttrInterface'";
      return {};
    }

    if (succeeded(parser.parseOptionalComma())) {
      SMLoc syncLoc = parser.getCurrentLocation();
      if (failed(parser.parseAttribute(syncScope))) {
        parser.emitError(syncLoc, "failed to parse parameter 'sync_scope'");
        return {};
      }
    }
  }

  if (parser.parseGreater()) {
    return {};
  }

  MLIRContext *context = parser.getContext();
  return ShapedRefType::get(context, shape, elementType,
                            cast<ScopeAttrInterface>(scope), syncScope);
}

void ShapedRefType::print(AsmPrinter &printer) const {
  printer << "<";

  ArrayRef<int64_t> shape = getShape();
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim)) {
      printer << '?';
    } else {
      printer << dim;
    }
    printer << 'x';
  }

  printer << getElementType();
  printer << ", ";
  if (isReturnOnlySync()) {
    // Special case printer for parent only sync for convenience.
    printer << "sync";
    printer << "(" << getScope() << ")";
  } else if (getSyncScope()) {
    // Default for other sync scopes.
    printer << getScope() << ", " << getSyncScope();
  } else {
    // printer case with no sync scope.
    printer << getScope();
  }
  printer << ">";
}

ShapedRefType ShapedRefType::get(MLIRContext *context, ArrayRef<int64_t> shape,
                                 Type elementType, ScopeAttrInterface scope) {
  return ShapedRefType::get(context, shape, elementType, scope, Attribute());
}

bool ShapedRefType::isReturnOnlySync() const {
  return isa_and_present<SyncOnReturnAttr>(getSyncScope());
}

//===----------------------------------------------------------------------===//
// !pcf.group<...>
//===----------------------------------------------------------------------===//

Type GroupType::parse(AsmParser &parser) {
  if (parser.parseLess()) {
    return {};
  }

  // Parse IDs: either `[id0, id1, ...]` or a single integer.
  SmallVector<int64_t> ids;
  SMLoc idsLoc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalLSquare())) {
    // Grouped group: parse comma-separated integers.
    if (failed(parser.parseCommaSeparatedList([&]() -> ParseResult {
          int64_t id;
          if (failed(parser.parseInteger(id))) {
            return failure();
          }
          ids.push_back(id);
          return success();
        }))) {
      return {};
    }
    if (parser.parseRSquare()) {
      return {};
    }
  } else {
    // Single ID.
    int64_t id;
    SMLoc idLoc = parser.getCurrentLocation();
    if (failed(parser.parseInteger(id))) {
      parser.emitError(idLoc, "expected group ID");
      return {};
    }
    ids.push_back(id);
  }

  // Validate IDs are non-negative and unique.
  DenseSet<int64_t> seenIds;
  for (int64_t id : ids) {
    if (id < 0) {
      parser.emitError(idsLoc, "group IDs must be non-negative");
      return {};
    }
    if (!seenIds.insert(id).second) {
      parser.emitError(idsLoc, "duplicate group ID: ") << id;
      return {};
    }
  }

  // Parse comma and scope attribute.
  if (parser.parseComma()) {
    return {};
  }

  Attribute scopeAttr;
  SMLoc scopeLoc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(scopeAttr))) {
    parser.emitError(scopeLoc, "failed to parse scope attribute");
    return {};
  }

  ScopeAttrInterface scope = dyn_cast<ScopeAttrInterface>(scopeAttr);
  if (!scope) {
    parser.emitError(scopeLoc, "expected scope attribute to implement "
                               "ScopeAttrInterface");
    return {};
  }

  // Optionally parse struct elements: `, {type0, type1, ...}`.
  SmallVector<Type> structTypes;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseLBrace()) {
      return {};
    }
    if (failed(parser.parseCommaSeparatedList([&]() -> ParseResult {
          Type ty;
          if (failed(parser.parseType(ty))) {
            return failure();
          }
          structTypes.push_back(ty);
          return success();
        }))) {
      return {};
    }
    if (parser.parseRBrace()) {
      return {};
    }
  }

  if (parser.parseGreater()) {
    return {};
  }

  return GroupType::get(parser.getContext(), ids, scope, structTypes);
}

void GroupType::print(AsmPrinter &printer) const {
  printer << "<";

  // Print IDs.
  ArrayRef<int64_t> ids = getIds();
  if (ids.size() == 1) {
    printer << ids.front();
  } else {
    printer << "[";
    llvm::interleaveComma(ids, printer);
    printer << "]";
  }

  // Print scope.
  printer << ", " << getScope();

  // Print struct elements if present.
  ArrayRef<Type> structTypes = getStructTypes();
  if (!structTypes.empty()) {
    printer << ", {";
    llvm::interleaveComma(structTypes, printer);
    printer << "}";
  }

  printer << ">";
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void PCFDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::PCF
