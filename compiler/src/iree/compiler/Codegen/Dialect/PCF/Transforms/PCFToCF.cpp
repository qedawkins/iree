// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-pcf-to-cf"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_PCFTOCFPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

// DO NOT SUBMIT
struct PCFToCFPass final : impl::PCFToCFPassBase<PCFToCFPass> {
  void runOnOperation() override;
};

void PCFToCFPass::runOnOperation() { return; }

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
