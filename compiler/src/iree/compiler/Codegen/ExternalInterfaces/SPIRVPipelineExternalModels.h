// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_SPIRVPIPELINEEXTERNALMODELS_H_
#define IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_SPIRVPIPELINEEXTERNALMODELS_H_

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::iree_compiler {
void registerSPIRVPipelineExternalModels(DialectRegistry &registry);
} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_SPIRVPIPELINEEXTERNALMODELS_H_
