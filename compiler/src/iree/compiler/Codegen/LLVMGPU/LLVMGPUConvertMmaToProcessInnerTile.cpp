// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/Template/IR/Template.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-llvmgpu-convert-mma-to-process-inner-tile"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUCONVERTMMATOPROCESSINNERTILEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Check if a lowering config is eligible for conversion to process_inner_tile.
/// Requirements:
/// - Has mma_kind
/// - Has template_call
/// - Has promote_operands = [0, 1]
static bool
isEligibleForProcessInnerTile(IREE::GPU::LoweringConfigAttr config) {
  if (!config) {
    return false;
  }

  // Must have mma_kind.
  auto mmaKind = IREE::GPU::getMmaKind(config);
  if (!mmaKind) {
    return false;
  }

  // Must have template_call.
  if (!IREE::GPU::getTemplateCall(config)) {
    return false;
  }

  // Must have promote_operands = [0, 1].
  auto promoteOperands = IREE::GPU::getPromotedOperandList(config);
  if (!promoteOperands) {
    return false;
  }
  if (promoteOperands->size() != 2 || (*promoteOperands)[0] != 0 ||
      (*promoteOperands)[1] != 1) {
    return false;
  }

  return true;
}

/// Get the outer dimension distribution from the subgroup tile sizes.
/// This tells us how to distribute outer dimensions across subgroups.
static SmallVector<int64_t>
getOuterDimDistribution(IREE::GPU::LoweringConfigAttr config,
                        linalg::LinalgOp linalgOp,
                        ArrayRef<utils::IteratorType> iteratorTypes) {
  SmallVector<int64_t> subgroupTiles = config.getStaticTilingLevelSizes(
      llvm::to_underlying(IREE::GPU::TilingLevel::Subgroup), linalgOp);
  if (subgroupTiles.empty()) {
    // Default to [1, 1] if not specified.
    return SmallVector<int64_t>(2, 1);
  }

  // Extract only parallel dimensions (skip reduction dimensions).
  SmallVector<int64_t> distribution;
  for (auto [tile, iterType] : llvm::zip(subgroupTiles, iteratorTypes)) {
    if (iterType == utils::IteratorType::parallel && tile != 0) {
      distribution.push_back(tile);
    }
  }

  // Ensure we have at least 2 dimensions.
  while (distribution.size() < 2) {
    distribution.push_back(1);
  }

  return distribution;
}

class LLVMGPUConvertMmaToProcessInnerTilePass final
    : public impl::LLVMGPUConvertMmaToProcessInnerTilePassBase<
          LLVMGPUConvertMmaToProcessInnerTilePass> {
public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Collect eligible ops first to avoid modifying while iterating.
    SmallVector<std::pair<linalg::LinalgOp, IREE::GPU::LoweringConfigAttr>>
        eligibleOps;

    funcOp->walk([&](linalg::LinalgOp linalgOp) {
      auto config = dyn_cast_or_null<IREE::GPU::LoweringConfigAttr>(
          getLoweringConfig(linalgOp));
      if (isEligibleForProcessInnerTile(config)) {
        eligibleOps.push_back({linalgOp, config});
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "Found " << eligibleOps.size()
                            << " eligible ops for process_inner_tile\n");

    // Convert each eligible op.
    for (auto &[linalgOp, config] : eligibleOps) {
      Location loc = linalgOp.getLoc();
      OpBuilder builder(linalgOp);

      // Get MMA kind.
      auto mmaKind = IREE::GPU::getMmaKind(config);
      assert(mmaKind && "Expected mma_kind to be present");

      // Get template symbol.
      auto templateCall = IREE::GPU::getTemplateCall(config);
      assert(templateCall && "Expected template_call to be present");

      // Get iterator types.
      SmallVector<utils::IteratorType> iteratorTypes =
          linalgOp.getIteratorTypesArray();

      // Get indexing maps.
      SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();

      // Create bounds from workgroup + reduction tile sizes.
      // The bounds represent the iteration space dimensions.
      SmallVector<int64_t> workgroupTiles = config.getWorkgroupTileSizes();
      SmallVector<int64_t> reductionTiles = config.getStaticTilingLevelSizes(
          llvm::to_underlying(IREE::GPU::TilingLevel::Reduction), linalgOp);

      // Build bounds values (using constants for static sizes).
      SmallVector<Value> bounds;
      for (size_t i = 0; i < iteratorTypes.size(); ++i) {
        int64_t tileSize = 0;
        if (i < workgroupTiles.size() && workgroupTiles[i] != 0) {
          tileSize = workgroupTiles[i];
        } else if (i < reductionTiles.size() && reductionTiles[i] != 0) {
          tileSize = reductionTiles[i];
        }
        // For dynamic sizes or zero, use a placeholder.
        if (tileSize == 0) {
          tileSize = 1;
        }
        Value boundValue = arith::ConstantIndexOp::create(builder, loc, tileSize);
        bounds.push_back(boundValue);
      }

      // Get outer dimension distribution.
      SmallVector<int64_t> outerDimDistribution =
          getOuterDimDistribution(config, linalgOp, iteratorTypes);

      // Get inputs and outputs.
      SmallVector<Value> inputs(linalgOp.getDpsInputs());
      SmallVector<Value> outputs(linalgOp.getDpsInits());

      // Create iterator type attrs.
      SmallVector<Attribute> iteratorTypeAttrs;
      for (utils::IteratorType iterType : iteratorTypes) {
        iteratorTypeAttrs.push_back(
            linalg::IteratorTypeAttr::get(context, iterType));
      }

      // Create indexing map array attr.
      SmallVector<Attribute> mapAttrs;
      for (AffineMap map : indexingMaps) {
        mapAttrs.push_back(AffineMapAttr::get(map));
      }

      // Get result types.
      SmallVector<Type> resultTypes;
      for (Value output : outputs) {
        resultTypes.push_back(output.getType());
      }

      // Create process_inner_tile op.
      auto processInnerTileOp = IREE::GPU::ProcessInnerTileOp::create(
          builder, loc, resultTypes, bounds, mmaKind,
          builder.getArrayAttr(mapAttrs),
          builder.getArrayAttr(iteratorTypeAttrs),
          builder.getDenseI64ArrayAttr(outerDimDistribution), inputs, outputs,
          *templateCall);

      LLVM_DEBUG(llvm::dbgs() << "Created process_inner_tile op: "
                              << processInnerTileOp << "\n");

      // Replace uses of linalg op results.
      for (auto [oldResult, newResult] :
           llvm::zip(linalgOp->getResults(), processInnerTileOp.getResults())) {
        oldResult.replaceAllUsesWith(newResult);
      }

      // Erase the original linalg op.
      linalgOp->erase();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
