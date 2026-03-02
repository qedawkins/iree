// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/Interfaces.h"

#include "iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.h"
#include "iree/compiler/Codegen/ExternalInterfaces/CodegenExternalModels.h"
#include "iree/compiler/Codegen/ExternalInterfaces/GPUEncodingExternalModels.h"
#include "iree/compiler/Codegen/ExternalInterfaces/LLVMCPUPipelineExternalModels.h"
#include "iree/compiler/Codegen/ExternalInterfaces/LLVMGPUPipelineExternalModels.h"
#include "iree/compiler/Codegen/ExternalInterfaces/SPIRVPipelineExternalModels.h"
#include "iree/compiler/Codegen/ExternalInterfaces/UtilExternalModels.h"
#include "iree/compiler/Codegen/ExternalInterfaces/VMVXPipelineExternalModels.h"

namespace mlir::iree_compiler {

void registerCodegenExternalInterfaces(DialectRegistry &registry) {
  IREE::Codegen::registerCodegenExternalModels(registry);
  IREE::Codegen::registerUtilExternalModels(registry);
  IREE::CPU::registerCPUEncodingExternalModels(registry);
  IREE::GPU::registerGPUEncodingExternalModels(registry);
  registerLLVMCPUPipelineExternalModels(registry);
  registerLLVMGPUPipelineExternalModels(registry);
  registerSPIRVPipelineExternalModels(registry);
  registerVMVXPipelineExternalModels(registry);
}

} // namespace mlir::iree_compiler
