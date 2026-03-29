// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/ExternalInterfaces/Interfaces.h"

#include "iree/compiler/Codegen/Dialect/PCF/ExternalInterfaces/BufferizationExternalModels.h"

namespace mlir::iree_compiler {

void registerPCFExternalInterfaces(DialectRegistry &registry) {
  IREE::PCF::registerBufferizationExternalModels(registry);
  // NOTE: PCFTilingOpInterface external models (TilingImplementations) are NOT
  // registered here because they extend TilingInterface, which may not be
  // attached to ops yet at dialect-load time. Instead, passes that use
  // PCFTilingOpInterface call registerAllDistributedTilingModels() explicitly.
}

} // namespace mlir::iree_compiler
