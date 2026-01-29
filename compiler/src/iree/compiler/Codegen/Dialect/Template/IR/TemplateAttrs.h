// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_IR_TEMPLATEATTRS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_IR_TEMPLATEATTRS_H_

#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateAttrs.h.inc" // IWYU pragma: keep

#endif // IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_IR_TEMPLATEATTRS_H_
