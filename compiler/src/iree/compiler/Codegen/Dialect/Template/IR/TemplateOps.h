// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_IR_TEMPLATEOPS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_IR_TEMPLATEOPS_H_

#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateDialect.h"
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateOps.h.inc" // IWYU pragma: keep

#endif // IREE_COMPILER_CODEGEN_DIALECT_TEMPLATE_IR_TEMPLATEOPS_H_
