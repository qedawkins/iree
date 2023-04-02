// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMLinkerUtils.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Transforms/IPO/Internalize.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

bool anyRequiredSymbols(const llvm::Module &module, StringRef prefix) {
  for (const auto &function : module.functions()) {
    if (!function.isIntrinsic() && function.isDeclaration() &&
        (function.getName().startswith(prefix))) {
      return true;
    }
  }
  return false;
}

LogicalResult linkBitcodeModule(
    Location loc, llvm::Linker &linker, unsigned linkerFlags,
    llvm::TargetMachine &targetMachine, StringRef name,
    llvm::Expected<std::unique_ptr<llvm::Module>> bitcodeModuleValue,
    ModuleSpecializationCallback specializationCallback) {
  // Ensure the bitcode loaded correctly. It may fail if the LLVM version is
  // incompatible.
  if (!bitcodeModuleValue) {
    return mlir::emitError(loc)
           << "failed to parse " << name
           << " bitcode: " << llvm::toString(bitcodeModuleValue.takeError())
           << " (possible LLVM bitcode incompatibility?)";
  }

  // Override the data layout and target triple with the final one we expect.
  // This is at the module level and if functions have their own specified
  // target attributes they won't be modified.
  auto bitcodeModule = std::move(bitcodeModuleValue.get());
  bitcodeModule->setDataLayout(targetMachine.createDataLayout());
  bitcodeModule->setTargetTriple(targetMachine.getTargetTriple().str());

  // Inject target-specific flags to specialize the bitcode prior to linking.
  if (specializationCallback) {
    specializationCallback(*bitcodeModule);
  }

  // Link the bitcode into the base module. This will merge in any required
  // symbols and override declarations that may exist.
  if (linker.linkInModule(
          std::move(bitcodeModule), linkerFlags,
          [&](llvm::Module &m, const StringSet<> &gvs) {
            if (linkerFlags & llvm::Linker::LinkOnlyNeeded) {
              llvm::internalizeModule(m, [&gvs](const llvm::GlobalValue &gv) {
                return !gv.hasName() || (gvs.count(gv.getName()) == 0);
              });
            }
          })) {
    return mlir::emitError(loc) << "failed to link " << name << " bitcode";
  }

  return success();
}

llvm::Expected<std::unique_ptr<llvm::Module>> loadBitcodeObject(
    IREE::HAL::ExecutableObjectAttr objectAttr, llvm::LLVMContext &context) {
  // Load the object data into memory.
  auto objectData = objectAttr.loadData();
  if (!objectData) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to load bitcode object file");
  }

  // Load the generic bitcode file contents.
  llvm::MemoryBufferRef bitcodeBufferRef(objectData.value(),
                                         objectAttr.getPath());
  auto bitcodeModuleValue = llvm::parseBitcodeFile(bitcodeBufferRef, context);
  if (!bitcodeModuleValue) return bitcodeModuleValue;
  auto bitcodeModule = std::move(bitcodeModuleValue.get());

  // NOTE: at this point the bitcode may not have the expected data layout!
  return std::move(bitcodeModule);
}

LogicalResult linkBitcodeObjects(
    Location loc, llvm::Linker &linker, unsigned linkerFlags,
    llvm::TargetMachine &targetMachine, ArrayAttr objectAttrs,
    llvm::LLVMContext &context,
    ModuleSpecializationCallback specializationCallback) {
  // Gather only the bitcode objects.
  SmallVector<IREE::HAL::ExecutableObjectAttr> bitcodeObjectAttrs;
  IREE::HAL::ExecutableObjectAttr::filterObjects(objectAttrs, {".bc"},
                                                 bitcodeObjectAttrs);

  // Load and link each object in the order declared.
  for (auto objectAttr : bitcodeObjectAttrs) {
    if (failed(linkBitcodeModule(
            loc, linker, linkerFlags, targetMachine, objectAttr.getPath(),
            loadBitcodeObject(objectAttr, context), specializationCallback))) {
      return mlir::emitError(loc)
             << "failed linking in user object bitcode `"
             << objectAttr.getPath() << "` for target triple '"
             << targetMachine.getTargetTriple().str() << "'";
    }
  }

  return success();
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
