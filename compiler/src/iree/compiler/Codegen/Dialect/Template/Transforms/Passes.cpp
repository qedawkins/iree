// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Template/Transforms/Passes.h"

namespace mlir::iree_compiler {

void registerTemplateTransformsPasses() {
  IREE::Template::registerConcretizeTemplateCallsPass();
  IREE::Template::registerInlineTemplateInstancesPass();
}

} // namespace mlir::iree_compiler
