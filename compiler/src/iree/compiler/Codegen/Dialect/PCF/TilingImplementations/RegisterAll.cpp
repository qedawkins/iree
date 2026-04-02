// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/TilingImplementations/RegisterAll.h"

namespace mlir::iree_compiler::IREE::PCF {

// Defined in each TilingImplementations/*.cpp file.
void attachLinalgDistributedTilingModels(MLIRContext *ctx);
void attachLinalgExtDistributedTilingModels(MLIRContext *ctx);
void attachTensorDistributedTilingModels(MLIRContext *ctx);

void registerAllDistributedTilingModels(DialectRegistry &registry) {
  registerLinalgDistributedTilingModels(registry);
  registerLinalgExtDistributedTilingModels(registry);
  registerTensorDistributedTilingModels(registry);
}

void attachAllDistributedTilingModels(MLIRContext *ctx) {
  // Only attach to dialects that are loaded.
  if (ctx->getLoadedDialect("linalg")) {
    attachLinalgDistributedTilingModels(ctx);
  }
  if (ctx->getLoadedDialect("iree_linalg_ext")) {
    attachLinalgExtDistributedTilingModels(ctx);
  }
  if (ctx->getLoadedDialect("tensor")) {
    attachTensorDistributedTilingModels(ctx);
  }
}

} // namespace mlir::iree_compiler::IREE::PCF
