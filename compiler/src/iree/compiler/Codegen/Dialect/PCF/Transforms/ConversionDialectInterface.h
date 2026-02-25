// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_CONVERSIONDIALECTINTERFACE_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_CONVERSIONDIALECTINTERFACE_H_

#include "mlir/IR/DialectInterface.h"

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace mlir::iree_compiler {

/// An interface for dialects to expose conversion functionality out of PCF.
class PCFConversionDialectInterface
    : public DialectInterface::Base<PCFConversionDialectInterface> {
public:
  PCFConversionDialectInterface(Dialect *dialect) : Base(dialect) {}
  virtual void loadSRefLoweringDependentDialects(MLIRContext *context) const {}
  virtual void loadTokenLoweringDependentDialects(MLIRContext *context) const {}
  virtual void
  loadStructuralLoweringDependentDialects(MLIRContext *context) const {}
  virtual void loadGroupLoweringDependentDialects(MLIRContext *context) const {}
  virtual void populateGroupLoweringPatterns(const TypeConverter &typeConverter,
                                             RewritePatternSet &patterns,
                                             ConversionTarget &target,
                                             int64_t regionId) const {}
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_CONVERSIONDIALECTINTERFACE_H_
