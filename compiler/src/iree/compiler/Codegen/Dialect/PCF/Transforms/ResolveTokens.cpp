// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"

#define DEBUG_TYPE "iree-pcf-resolve-tokens"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_RESOLVETOKENSPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

struct ResolveTokensPass final
    : impl::ResolveTokensPassBase<ResolveTokensPass> {
  void runOnOperation() override;
};

void ResolveTokensPass::runOnOperation() { return; }

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
