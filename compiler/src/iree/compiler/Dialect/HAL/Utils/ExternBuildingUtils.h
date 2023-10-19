// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_UTILS_EXTERN_BUILDING_UTILS_H_
#define IREE_COMPILER_DIALECT_HAL_UTILS_EXTERN_BUILDING_UTILS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

void registerExternDispatchRewriteFunction(PDLPatternModule &pdlPatterns);

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_HAL_UTILS_EXTERN_BUILDING_UTILS_H_
