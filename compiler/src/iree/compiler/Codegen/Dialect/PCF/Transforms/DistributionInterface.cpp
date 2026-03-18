// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/Transforms/DistributionInterface.h"

namespace mlir::iree_compiler::IREE::PCF {

/// Global distribution factory. Set once during pass registration
/// (before any compilation threads start) and read-only thereafter.
/// Thread-safe because registration happens during static initialization.
static DistributionFactory globalFactory;

void registerDistributionFactory(DistributionFactory factory) {
  globalFactory = std::move(factory);
}

DistributionFactory getDistributionFactory() { return globalFactory; }

} // namespace mlir::iree_compiler::IREE::PCF
