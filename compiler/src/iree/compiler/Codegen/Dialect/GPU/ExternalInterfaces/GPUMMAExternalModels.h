// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_GPUMMAEXTERNALMODELS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_GPUMMAEXTERNALMODELS_H_

#include "mlir/IR/DialectRegistry.h"

namespace mlir::iree_compiler::IREE::GPU {

/// Registers external model implementations for PCF::MMALayoutInterface on
/// GPU MMA attributes (MMAAttr).
void registerGPUMMAExternalModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_GPUMMAEXTERNALMODELS_H_
