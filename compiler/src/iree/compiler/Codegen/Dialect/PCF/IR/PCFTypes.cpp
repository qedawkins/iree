// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.cpp.inc" // IWYU pragma: keep

// clang-format off
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFEnums.cpp.inc" // IWYU pragma: keep
// clang-format on

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
      // Could be accessor mode keyword (readwrite/readonly) or sync_scope
      // attribute. Try specific accessor mode keywords first -- these do
      // not consume the token when they don't match.
      std::optional<AccessorMode> earlyAccessorMode;
      if (succeeded(parser.parseOptionalKeyword("readwrite"))) {
        earlyAccessorMode = AccessorMode::ReadWrite;
      } else if (succeeded(parser.parseOptionalKeyword("readonly"))) {
        earlyAccessorMode = AccessorMode::ReadOnly;
      }

      if (earlyAccessorMode) {
        // Accessor mode without sync scope.
        if (parser.parseGreater()) {
          return {};
        }
        MLIRContext *context = parser.getContext();
        return ShapedRefType::get(context, shape, elementType,
                                  cast<ScopeAttrInterface>(scope), Attribute(),
                                  *earlyAccessorMode);
      }

      // Not an accessor mode. Parse as sync_scope attribute.
      if (failed(parser.parseAttribute(syncScope))) {
        return {};
      }
    }
  }

  // After scope and optional sync_scope, check for optional accessor mode.
  std::optional<AccessorMode> accessorMode;
  if (succeeded(parser.parseOptionalComma())) {
    StringRef modeStr;
    SMLoc modeLoc = parser.getCurrentLocation();
    if (failed(parser.parseKeyword(&modeStr))) {
      return {};
    }
    std::optional<AccessorMode> mode = symbolizeAccessorMode(modeStr);
    if (!mode) {
      parser.emitError(modeLoc, "invalid accessor mode '") << modeStr << "'";
      return {};
    }
    accessorMode = *mode;
  }

  if (parser.parseGreater()) {
    return {};
  }

  MLIRContext *context = parser.getContext();
  return ShapedRefType::get(context, shape, elementType,
                            cast<ScopeAttrInterface>(scope), syncScope,
                            accessorMode);
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
    // Printer case with no sync scope.
    printer << getScope();
  }
  // Print accessor mode if present.
  if (hasAccessorMode()) {
    printer << ", " << stringifyAccessorMode(*getAccessorMode());
  }
  printer << ">";
}

ShapedRefType ShapedRefType::get(MLIRContext *context, ArrayRef<int64_t> shape,
                                 Type elementType, ScopeAttrInterface scope) {
  return ShapedRefType::get(context, shape, elementType, scope, Attribute(),
                            std::nullopt);
}

ShapedRefType ShapedRefType::get(MLIRContext *context, ArrayRef<int64_t> shape,
                                 Type elementType, ScopeAttrInterface scope,
                                 Attribute syncScope) {
  return ShapedRefType::get(context, shape, elementType, scope, syncScope,
                            std::nullopt);
}

ShapedRefType ShapedRefType::get(MLIRContext *context, ArrayRef<int64_t> shape,
                                 Type elementType, ScopeAttrInterface scope,
                                 AccessorMode accessorMode) {
  return ShapedRefType::get(context, shape, elementType, scope, Attribute(),
                            accessorMode);
}

//===----------------------------------------------------------------------===//
// #pcf.bundle<...>
//===----------------------------------------------------------------------===//

LogicalResult BundleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ScopeAttrInterface scope, int64_t id) {
  if (id < 0) {
    return emitError() << "bundle ID must be non-negative, got " << id;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShapedRefType helpers
//===----------------------------------------------------------------------===//

bool ShapedRefType::isReturnOnlySync() const {
  return isa_and_present<SyncOnReturnAttr>(getSyncScope());
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
