// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler::IREE::Template {

Value createTypeCast(OpBuilder &builder, Location loc, Type targetType,
                     ValueRange inputs) {
  if (inputs.size() == 1 && inputs[0].getType() == targetType) {
    return inputs[0];
  }
  return UnrealizedConversionCastOp::create(builder, loc, targetType, inputs)
      .getResult(0);
}

SmallVector<Value> createTypeCasts(OpBuilder &builder, Location loc,
                                   TypeRange targetTypes, ValueRange inputs) {
  SmallVector<Value> results;
  results.reserve(targetTypes.size());

  size_t inputIdx = 0;
  for (Type targetType : targetTypes) {
    // For now, assume 1:1 mapping. The pass handles 1:N via segment tracking.
    if (inputIdx < inputs.size()) {
      results.push_back(
          createTypeCast(builder, loc, targetType, inputs[inputIdx]));
      ++inputIdx;
    }
  }

  return results;
}

} // namespace mlir::iree_compiler::IREE::Template

// Include generated interface definitions.
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateInterfaces.cpp.inc"
