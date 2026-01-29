// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateDialect.h"

#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateOps.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::iree_compiler::IREE::Template {

namespace {

// Used to control inlining behavior.
struct TemplateInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
};

} // namespace

TemplateDialect::TemplateDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TemplateDialect>()) {

  registerTypes();
  registerAttributes();
  registerOperations();

  addInterfaces<TemplateInlinerInterface>();
}

} // namespace mlir::iree_compiler::IREE::Template
