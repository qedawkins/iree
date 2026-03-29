// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_TILINGIMPLEMENTATIONS_REGISTERALL_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_TILINGIMPLEMENTATIONS_REGISTERALL_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler::IREE::PCF {

/// Registers PCFTilingOpInterface external models for LinalgOps.
void registerLinalgDistributedTilingModels(DialectRegistry &registry);

/// Registers PCFTilingOpInterface external models for LinalgExt ops
/// (ScatterOp, MapStoreOp, etc.).
void registerLinalgExtDistributedTilingModels(DialectRegistry &registry);

/// Registers PCFTilingOpInterface external models for tensor ops
/// (pad, pack, unpack).
void registerTensorDistributedTilingModels(DialectRegistry &registry);

/// Registers all PCFTilingOpInterface external models via DialectRegistry.
/// NOTE: This uses addExtension which may have ordering issues with
/// TilingInterface registration. Prefer the MLIRContext* overload for
/// direct attachment from passes.
void registerAllDistributedTilingModels(DialectRegistry &registry);

/// Directly attaches PCFTilingOpInterface external models to ops in the
/// given context. Call this from passes after all dialects are loaded to
/// avoid ordering issues with TilingInterface registration.
void attachAllDistributedTilingModels(MLIRContext *ctx);

} // namespace mlir::iree_compiler::IREE::PCF

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_TILINGIMPLEMENTATIONS_REGISTERALL_H_
