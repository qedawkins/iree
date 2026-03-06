// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_CODEGENPIPELINEOPTIONS_H_
#define IREE_COMPILER_CODEGEN_UTILS_CODEGENPIPELINEOPTIONS_H_

#include "mlir/Support/TypeID.h"

namespace mlir::iree_compiler {

/// Polymorphic base class for per-pipeline codegen options. Backends derive
/// from this to pass backend-specific options through
/// PipelineAttrInterface::buildPipeline.
///
/// Uses LLVM-style RTTI via TypeID for safe downcasting.
struct CodegenPipelineOptions {
  explicit CodegenPipelineOptions(TypeID typeID) : typeID(typeID) {}
  virtual ~CodegenPipelineOptions() = default;

  TypeID getTypeID() const { return typeID; }

  /// Downcast to a concrete options type. Returns nullptr on type mismatch.
  template <typename T>
  const T *getAs() const {
    if (typeID == TypeID::get<T>()) {
      return static_cast<const T *>(this);
    }
    return nullptr;
  }

private:
  TypeID typeID;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_CODEGENPIPELINEOPTIONS_H_
