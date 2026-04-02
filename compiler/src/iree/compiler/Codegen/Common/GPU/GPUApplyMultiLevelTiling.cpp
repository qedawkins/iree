// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTilingInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/TilingImplementations/RegisterAll.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-multi-level-tiling"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYMULTILEVELTILINGPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct GPUApplyMultiLevelTilingPass final
    : impl::GPUApplyMultiLevelTilingPassBase<GPUApplyMultiLevelTilingPass> {
  using GPUApplyMultiLevelTilingPassBase::GPUApplyMultiLevelTilingPassBase;
  void runOnOperation() override;
};

void GPUApplyMultiLevelTilingPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();

  // Ensure PCFTilingOpInterface external models are attached.
  IREE::PCF::attachAllDistributedTilingModels(context);

  // Find ops with lowering configs that have thread/subgroup/reduction levels.
  IRRewriter rewriter(context);

  funcOp->walk([&](Operation *op) {
    auto pcfTilingOp = dyn_cast<IREE::PCF::PCFTilingOpInterface>(op);
    if (!pcfTilingOp) {
      return;
    }

    IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
        getLoweringConfig(op);
    if (!loweringConfig) {
      return;
    }

    auto gpuConfig = dyn_cast<IREE::GPU::LoweringConfigAttr>(loweringConfig);
    if (!gpuConfig) {
      return;
    }

    // Check if this op has subgroup or thread tiling.
    bool hasSubgroup =
        gpuConfig.hasTilingLevel(
            llvm::to_underlying(IREE::GPU::TilingLevel::Subgroup));
    bool hasThread =
        gpuConfig.hasTilingLevel(
            llvm::to_underlying(IREE::GPU::TilingLevel::Thread));
    bool hasReduction =
        gpuConfig.hasTilingLevel(
            llvm::to_underlying(IREE::GPU::TilingLevel::Reduction));

    if (!hasSubgroup && !hasThread) {
      return;
    }

    // Get tile sizes.
    SmallVector<OpFoldResult> sgTileSizes;
    SmallVector<OpFoldResult> laneTileSizes;
    SmallVector<OpFoldResult> reductionTileSizes;

    if (hasSubgroup) {
      sgTileSizes = gpuConfig.getTilingLevelSizes(
          rewriter,
          llvm::to_underlying(IREE::GPU::TilingLevel::Subgroup), op);
    }
    if (hasThread) {
      // Thread tiling — if subgroup is not set, thread = subgroup level.
      SmallVector<OpFoldResult> threadTiles = gpuConfig.getTilingLevelSizes(
          rewriter,
          llvm::to_underlying(IREE::GPU::TilingLevel::Thread), op);
      if (!hasSubgroup) {
        sgTileSizes = threadTiles;
      } else {
        laneTileSizes = threadTiles;
      }
    }
    if (hasReduction) {
      reductionTileSizes = gpuConfig.getTilingLevelSizes(
          rewriter,
          llvm::to_underlying(IREE::GPU::TilingLevel::Reduction), op);
    }

    // Get promotion info.
    SmallVector<unsigned> operandsToPromote;
    if (auto promotedList =
            IREE::GPU::getPromotedOperandList(gpuConfig)) {
      for (int64_t idx : *promotedList) {
        operandsToPromote.push_back(static_cast<unsigned>(idx));
      }
    }

    // Get per-operand promotion types if specified.
    SmallVector<Attribute> promotionTypes;
    if (auto typesList =
            IREE::GPU::getPromotionTypesList(gpuConfig)) {
      promotionTypes.assign(typesList->begin(), typesList->end());
    }

    // Build params.
    IREE::PCF::MultiLevelTilingParams params;
    params.subgroup.scope = IREE::GPU::SubgroupScopeAttr::get(context);
    params.subgroup.tileSizes = sgTileSizes;
    params.lane.scope = IREE::GPU::LaneScopeAttr::get(context);
    params.lane.tileSizes = laneTileSizes;
    params.reductionTileSizes = reductionTileSizes;
    params.operandsToPromote = operandsToPromote;
    params.promotionTypes = promotionTypes;

    // Check for MMA kind.
    if (auto mmaKind = IREE::GPU::getMmaKind(gpuConfig)) {
      params.mmaKind = mmaKind;
    }

    rewriter.setInsertionPoint(op);
    (void)IREE::PCF::applyMultiLevelTiling(rewriter, pcfTilingOp, params);
  });
}

} // namespace
} // namespace mlir::iree_compiler
