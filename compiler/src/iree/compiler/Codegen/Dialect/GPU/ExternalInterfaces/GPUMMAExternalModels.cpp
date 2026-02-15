// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/ExternalInterfaces/GPUMMAExternalModels.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"

namespace mlir::iree_compiler::IREE::GPU {

namespace {

/// Converts a StringRef operand role ("lhs", "rhs", "acc") to the integer
/// operand index used by getSingleSubgroupLayout.
static int getOperandIndex(StringRef operandRole) {
  if (operandRole == "lhs") {
    return kMMAOperandLhs;
  }
  if (operandRole == "rhs") {
    return kMMAOperandRhs;
  }
  assert(operandRole == "acc" && "expected lhs, rhs, or acc");
  return kMMAOperandAcc;
}

/// External model for MMAAttr implementing PCF::MMALayoutInterface.
/// Converts MMA intrinsic layout information into NestedLayoutAttr instances
/// that describe how each operand is distributed across threads.
struct MMALayoutModel final
    : public PCF::MMALayoutInterface::ExternalModel<MMALayoutModel,
                                                    GPU::MMAAttr> {
  Attribute getOperandLayout(Attribute attr, StringRef operandRole) const {
    MMAAttr mma = cast<MMAAttr>(attr);
    MLIRContext *ctx = attr.getContext();
    int operandIndex = getOperandIndex(operandRole);

    // Get the single-subgroup layout for this operand.
    MMASingleSubgroupLayout layout = getSingleSubgroupLayout(
        mma.getIntrinsic(), operandIndex, mma.getColMajor());

    // Construct a NestedLayoutAttr with subgroup_tile = [1, 1] since this
    // describes a single-subgroup MMA. Batch tiles are [1, 1] as there is
    // no unrolling at this level.
    int64_t rank = layout.element.size();
    SmallVector<int64_t> subgroupTile(rank, 1);
    SmallVector<int64_t> batchTile(rank, 1);
    SmallVector<int64_t> subgroupStrides(rank, 0);

    return VectorExt::NestedLayoutAttr::get(
        ctx, subgroupTile, batchTile, layout.outer, layout.thread,
        layout.element, subgroupStrides, layout.tstrides);
  }

  SmallVector<int64_t> getOperandShape(Attribute attr,
                                       StringRef operandRole) const {
    MMAAttr mma = cast<MMAAttr>(attr);
    auto [m, n, k] = mma.getMNKShape();
    if (operandRole == "lhs") {
      return {m, k};
    }
    if (operandRole == "rhs") {
      return {k, n};
    }
    return {m, n}; // acc
  }

  Type getOperandElementType(Attribute attr, MLIRContext *context,
                             StringRef operandRole) const {
    MMAAttr mma = cast<MMAAttr>(attr);
    auto [aType, bType, cType] = mma.getABCElementTypes();
    if (operandRole == "lhs") {
      return aType;
    }
    if (operandRole == "rhs") {
      return bType;
    }
    return cType; // acc
  }
};

} // namespace

void registerGPUMMAExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, GPU::IREEGPUDialect *dialect) {
        GPU::MMAAttr::attachInterface<MMALayoutModel>(*context);
      });
}

} // namespace mlir::iree_compiler::IREE::GPU
