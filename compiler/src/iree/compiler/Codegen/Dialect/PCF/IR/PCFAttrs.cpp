// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFEnums.cpp.inc" // IWYU pragma: keep
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.cpp.inc" // IWYU pragma: keep

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// SharedLayoutAttr
//===----------------------------------------------------------------------===//

Attribute SharedLayoutAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()) || failed(parser.parseLBrace())) {
    return {};
  }

  SmallVector<int64_t> strides;
  StringAttr swizzle;

  // Parse "stride = [...]".
  if (failed(parser.parseKeyword("stride")) || failed(parser.parseEqual())) {
    return {};
  }
  if (failed(parser.parseLSquare())) {
    return {};
  }
  if (failed(parser.parseCommaSeparatedList([&]() -> ParseResult {
        int64_t val;
        if (failed(parser.parseInteger(val))) {
          return failure();
        }
        strides.push_back(val);
        return success();
      }))) {
    return {};
  }
  if (failed(parser.parseRSquare())) {
    return {};
  }

  // Parse optional ", swizzle = <value>".
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseKeyword("swizzle")) || failed(parser.parseEqual())) {
      return {};
    }
    StringRef swizzleStr;
    if (failed(parser.parseKeyword(&swizzleStr))) {
      return {};
    }
    swizzle = StringAttr::get(parser.getContext(), swizzleStr);
  }

  if (failed(parser.parseRBrace()) || failed(parser.parseGreater())) {
    return {};
  }

  return SharedLayoutAttr::get(parser.getContext(), strides, swizzle);
}

void SharedLayoutAttr::print(AsmPrinter &printer) const {
  printer << "<{stride = [";
  llvm::interleaveComma(getStrides(), printer);
  printer << "]";
  if (getSwizzle()) {
    printer << ", swizzle = " << getSwizzle().getValue();
  }
  printer << "}>";
}

LogicalResult
SharedLayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<int64_t> strides, StringAttr swizzle) {
  if (strides.empty()) {
    return emitError() << "shared layout must have at least one stride";
  }
  // Validate swizzle value if present.
  if (swizzle) {
    StringRef val = swizzle.getValue();
    if (val != "none" && val != "xor_128" && val != "xor_64" &&
        val != "xor_32") {
      return emitError() << "unknown swizzle pattern '" << val
                         << "'; expected none, xor_128, xor_64, or xor_32";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void PCFDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::PCF
