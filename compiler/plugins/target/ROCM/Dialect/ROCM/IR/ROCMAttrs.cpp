// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.h"
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.cpp.inc"

namespace mlir::iree_compiler::IREE::ROCM {

//===----------------------------------------------------------------------===//
// BuiltinTuningModuleAttr
//===----------------------------------------------------------------------===//

FailureOr<mlir::ModuleOp>
BuiltinTuningModuleAttr::getModule(Operation * /*annotationSite*/) const {
  auto &rocmDialect = cast<ROCMDialect>(getDialect());
  return rocmDialect.getOrLoadBuiltinModule(getBuiltinFilename());
}

//===----------------------------------------------------------------------===//
// BuiltinUKernelProviderAttr
//===----------------------------------------------------------------------===//

FailureOr<Operation *>
BuiltinUKernelProviderAttr::getMLIRUKernel(StringRef name,
                                           DictionaryAttr targetConfiguration,
                                           Operation *annotationSite) const {
  auto &rocmDialect = cast<ROCMDialect>(getDialect());

  auto targetAttr = dyn_cast_if_present<IREE::GPU::TargetAttr>(
      targetConfiguration.get("iree.gpu.target"));
  if (!targetAttr) {
    return annotationSite->emitError(
        "Invalid ROCM target config lacks a gpu target");
  }

  std::string filename = Twine("iree_mlir_ukernel_" + name + "_" +
                               targetAttr.getArch().str() + ".mlir")
                             .str();
  FailureOr<mlir::ModuleOp> moduleOp =
      rocmDialect.getOrLoadBuiltinModule(filename);
  if (failed(moduleOp)) {
    return annotationSite->emitError("Could not load ukernel module " +
                                     filename);
  }

  Operation *ukernel = moduleOp->lookupSymbol(name);
  if (!ukernel) {
    return annotationSite->emitError("Could not find ukernel " + name);
  }
  return ukernel;
}

//===----------------------------------------------------------------------===//
// Attribute Registration
//===----------------------------------------------------------------------===//

void ROCMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::ROCM
