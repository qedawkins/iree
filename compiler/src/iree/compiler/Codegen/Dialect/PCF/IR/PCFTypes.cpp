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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

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

  SMLoc commaLoc = parser.getCurrentLocation();
  if (failed(parser.parseComma())) {
    parser.emitError(commaLoc, "expected comma after 'elementType'");
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
          << scope << " to implement 'ScopeAttrInterfaceInterface'";
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
// #pcf.threadgroup<...>
//===----------------------------------------------------------------------===//

Type ThreadGroupType::parse(AsmParser &parser) {
  if (parser.parseLess()) {
    return {};
  }

  Attribute scopeAttr;
  SMLoc scopeLoc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(scopeAttr))) {
    parser.emitError(scopeLoc, "failed to parse parameter 'scope'");
    return {};
  }

  ScopeAttrInterface scope = dyn_cast<ScopeAttrInterface>(scopeAttr);
  if (!scope) {
    parser.emitError(scopeLoc, "expected 'scope' parameter ")
        << scopeAttr << " to implement ScopeAttrInterface";
    return {};
  }

  SmallVector<Type> structTypes;
  if (succeeded(parser.parseOptionalComma())) {
    // Parse {type, type, ...}.
    if (failed(parser.parseLBrace())) {
      return {};
    }
    if (failed(parser.parseTypeList(structTypes))) {
      return {};
    }
    if (failed(parser.parseRBrace())) {
      return {};
    }
  }

  if (parser.parseGreater()) {
    return {};
  }

  return ThreadGroupType::get(parser.getContext(), scope, structTypes);
}

void ThreadGroupType::print(AsmPrinter &printer) const {
  printer << "<";
  printer << getScope();
  if (hasStructElements()) {
    printer << ", {";
    llvm::interleaveComma(getStructTypes(), printer,
                          [&](Type type) { printer << type; });
    printer << "}";
  }
  printer << ">";
}

//===----------------------------------------------------------------------===//
// !pcf.cluster<...>
//===----------------------------------------------------------------------===//

/// Parses a simple affine expression token: an integer constant, a dim
/// variable (d0, d1, ...), or a symbol variable (s0, s1, ...). Updates
/// `numDims` and `numSyms` to track the maximum index seen.
static FailureOr<AffineExpr>
parseSimpleAffineExpr(AsmParser &parser, unsigned &numDims, unsigned &numSyms) {
  MLIRContext *ctx = parser.getContext();

  // Try parsing an integer constant.
  int64_t val;
  OptionalParseResult intResult = parser.parseOptionalInteger(val);
  if (intResult.has_value()) {
    if (failed(*intResult)) {
      return failure();
    }
    return getAffineConstantExpr(val, ctx);
  }

  // Try parsing a dim or symbol variable (d0, d1, s0, s1, ...).
  StringRef keyword;
  SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseKeyword(&keyword))) {
    parser.emitError(loc, "expected integer, dim variable, or symbol variable");
    return failure();
  }

  if (keyword.size() < 2) {
    parser.emitError(loc, "expected dim (dN) or symbol (sN) variable");
    return failure();
  }

  char prefix = keyword[0];
  unsigned idx;
  if (keyword.drop_front(1).getAsInteger(10, idx)) {
    parser.emitError(loc, "expected numeric index after '") << prefix << "'";
    return failure();
  }

  if (prefix == 'd') {
    numDims = std::max(numDims, idx + 1);
    return getAffineDimExpr(idx, ctx);
  }
  if (prefix == 's') {
    numSyms = std::max(numSyms, idx + 1);
    return getAffineSymbolExpr(idx, ctx);
  }

  parser.emitError(loc, "expected 'd' or 's' prefix for affine variable");
  return failure();
}

Type ClusterType::parse(AsmParser &parser) {
  if (parser.parseLess()) {
    return {};
  }

  // Parse scope attribute.
  Attribute scopeAttr;
  SMLoc scopeLoc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(scopeAttr))) {
    parser.emitError(scopeLoc, "failed to parse parameter 'scope'");
    return {};
  }

  ScopeAttrInterface scope = dyn_cast<ScopeAttrInterface>(scopeAttr);
  if (!scope) {
    parser.emitError(scopeLoc, "expected 'scope' parameter ")
        << scopeAttr << " to implement ScopeAttrInterface";
    return {};
  }

  if (parser.parseComma()) {
    return {};
  }

  // Parse range list: [expr -> expr) separated by ` x `.
  SmallVector<AffineExpr> results;
  unsigned numDims = 0;
  unsigned numSyms = 0;

  auto parseOneRange = [&]() -> LogicalResult {
    if (parser.parseLParen()) {
      return failure();
    }
    FailureOr<AffineExpr> lower =
        parseSimpleAffineExpr(parser, numDims, numSyms);
    if (failed(lower)) {
      return failure();
    }
    // Parse `->`.
    if (parser.parseArrow()) {
      return failure();
    }
    FailureOr<AffineExpr> upper =
        parseSimpleAffineExpr(parser, numDims, numSyms);
    if (failed(upper)) {
      return failure();
    }
    // Parse `)`.
    if (parser.parseRParen()) {
      return failure();
    }
    results.push_back(*lower);
    results.push_back(*upper);
    return success();
  };

  // Parse first range.
  if (failed(parseOneRange())) {
    return {};
  }

  // Parse additional ranges separated by ` x `.
  while (succeeded(parser.parseOptionalKeyword("x"))) {
    if (failed(parseOneRange())) {
      return {};
    }
  }

  // Parse struct groups and required ID.
  // Struct groups use "keyword : { types }" -- the colon disambiguates from
  // the trailing ID keyword (which has no colon).
  SmallVector<Type> privateTypes, sharedTypes, uniformTypes;
  llvm::SmallDenseSet<StringRef> seenKeywords;
  NamespacedSymbolAttr id;

  auto parseStructGroup = [&](SmallVector<Type> &types) -> LogicalResult {
    if (failed(parser.parseLBrace())) {
      return failure();
    }
    if (failed(parser.parseTypeList(types))) {
      return failure();
    }
    if (failed(parser.parseRBrace())) {
      return failure();
    }
    return success();
  };

  // Must have at least one comma (for the ID, which is required).
  if (parser.parseComma()) {
    return {};
  }

  // Parse keyword -- could be struct group label or ID.
  while (true) {
    StringRef keyword;
    SMLoc kwLoc = parser.getCurrentLocation();
    if (failed(parser.parseKeyword(&keyword))) {
      return {};
    }

    // Try colon -- if present, this is a struct group.
    if (succeeded(parser.parseOptionalColon())) {
      if (!seenKeywords.insert(keyword).second) {
        parser.emitError(kwLoc, "duplicate keyword '") << keyword << "'";
        return {};
      }
      if (keyword == "private") {
        if (failed(parseStructGroup(privateTypes))) {
          return {};
        }
      } else if (keyword == "shared") {
        if (failed(parseStructGroup(sharedTypes))) {
          return {};
        }
      } else if (keyword == "uniform") {
        if (failed(parseStructGroup(uniformTypes))) {
          return {};
        }
      } else {
        parser.emitError(kwLoc, "expected 'private', 'shared', or 'uniform'");
        return {};
      }
      // Expect comma before next group or ID.
      if (parser.parseComma()) {
        return {};
      }
      continue;
    }

    // No colon -- this keyword is the ID.
    SmallVector<StringAttr> idPath;
    SmallVector<StringRef> idSegments;
    keyword.split(idSegments, '.');
    for (StringRef seg : idSegments) {
      if (seg.empty()) {
        parser.emitError(kwLoc, "empty segment in cluster ID");
        return {};
      }
      idPath.push_back(StringAttr::get(parser.getContext(), seg));
    }
    id = NamespacedSymbolAttr::get(parser.getContext(), idPath);
    break;
  }

  if (parser.parseGreater()) {
    return {};
  }

  AffineMap boundsMap =
      AffineMap::get(numDims, numSyms, results, parser.getContext());
  return ClusterType::get(parser.getContext(), scope, boundsMap, privateTypes,
                          sharedTypes, uniformTypes, id);
}

void ClusterType::print(AsmPrinter &printer) const {
  printer << "<";
  printer << getScope() << ", ";

  AffineMap map = getBoundsMap();
  int64_t rank = getRank();
  for (int64_t i = 0, e = rank; i < e; ++i) {
    if (i > 0) {
      printer << " x ";
    }
    printer << "(";
    map.getResult(2 * i).print(printer.getStream());
    printer << " -> ";
    map.getResult(2 * i + 1).print(printer.getStream());
    printer << ")";
  }

  auto printGroup = [&](StringRef label, ArrayRef<Type> types) {
    if (types.empty()) {
      return;
    }
    printer << ", " << label << ": {";
    llvm::interleaveComma(types, printer, [&](Type type) { printer << type; });
    printer << "}";
  };

  printGroup("private", getPrivateTypes());
  printGroup("shared", getSharedTypes());
  printGroup("uniform", getUniformTypes());

  // Print cluster ID (always last, always present).
  printer << ", ";
  llvm::interleave(
      getId().getPath(), printer.getStream(),
      [&](StringAttr seg) { printer << seg.getValue(); }, ".");
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
