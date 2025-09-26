// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#define DEBUG_TYPE "iree-pcf-convert-sref-to-memref"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_CONVERTSREFTOMEMREFPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

struct ConvertSRefToMemRefPass final
    : impl::ConvertSRefToMemRefPassBase<ConvertSRefToMemRefPass> {
  void runOnOperation() override;
};

void ConvertSRefToMemRefPass::runOnOperation() { return; }

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
