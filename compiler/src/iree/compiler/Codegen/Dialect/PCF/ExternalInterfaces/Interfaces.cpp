// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/ExternalInterfaces/Interfaces.h"

#include "iree/compiler/Codegen/Dialect/PCF/ExternalInterfaces/BufferizationExternalModels.h"
#include "iree/compiler/Codegen/Dialect/PCF/TilingImplementations/RegisterAll.h"

namespace mlir::iree_compiler {

void registerPCFExternalInterfaces(DialectRegistry &registry) {
  IREE::PCF::registerBufferizationExternalModels(registry);
  IREE::PCF::registerAllDistributedTilingModels(registry);
}

} // namespace mlir::iree_compiler
