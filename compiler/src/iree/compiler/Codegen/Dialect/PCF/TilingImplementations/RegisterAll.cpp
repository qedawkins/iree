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
  attachLinalgDistributedTilingModels(ctx);
  attachLinalgExtDistributedTilingModels(ctx);
  attachTensorDistributedTilingModels(ctx);
}

} // namespace mlir::iree_compiler::IREE::PCF
