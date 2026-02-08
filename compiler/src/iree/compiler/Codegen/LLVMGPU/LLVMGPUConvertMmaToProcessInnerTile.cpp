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
/// - Has mma_kind.
/// - Has template_call.
/// - Has subgroup tiling level with non-zero values.
static bool
isEligibleForProcessInnerTile(IREE::GPU::LoweringConfigAttr config,
                              linalg::LinalgOp linalgOp) {
  if (!config)
    return false;

  if (!IREE::GPU::getMmaKind(config))
    return false;

  if (!IREE::GPU::getTemplateCall(config))
    return false;

  // Require subgroup tiles to be present (distinguishes pingpong from
  // standard MMA ops that use the template infrastructure differently).
  SmallVector<int64_t> subgroupTiles = config.getStaticTilingLevelSizes(
      llvm::to_underlying(IREE::GPU::TilingLevel::Subgroup), linalgOp);
  if (subgroupTiles.empty())
    return false;

  // Must have at least one non-zero subgroup tile.
  bool hasNonZero = llvm::any_of(subgroupTiles, [](int64_t v) { return v != 0; });
  return hasNonZero;
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
      if (isEligibleForProcessInnerTile(config, linalgOp)) {
        eligibleOps.push_back({linalgOp, config});
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "Found " << eligibleOps.size()
                            << " eligible ops for process_inner_tile\n");

    // Convert each eligible op.
    for (auto &[linalgOp, config] : eligibleOps) {
      Location loc = linalgOp.getLoc();
      OpBuilder builder(linalgOp);

      // --- Read only 3 fields from the config ---

      // 1. MMA kind.
      auto mmaKind = IREE::GPU::getMmaKind(config);
      assert(mmaKind && "Expected mma_kind to be present");

      // 2. Template symbol.
      auto templateCall = IREE::GPU::getTemplateCall(config);
      assert(templateCall && "Expected template_call to be present");

      // 3. Subgroup tile sizes.
      //    For parallel dims: distribution factors (number of subgroups).
      //    For reduction dims: tile size (e.g. K tile size).
      SmallVector<int64_t> subgroupTiles = config.getStaticTilingLevelSizes(
          llvm::to_underlying(IREE::GPU::TilingLevel::Subgroup), linalgOp);

      // --- Derive everything else from the linalg op ---

      SmallVector<utils::IteratorType> iteratorTypes =
          linalgOp.getIteratorTypesArray();
      SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();

      // Get static loop ranges from the linalg op shapes.
      // After workgroup tiling, parallel dims reflect workgroup tile sizes.
      SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();

      // Build bounds values.
      // Parallel dims: from linalg op shapes (workgroup tile sizes after
      //   tileAndDistributeToWorkgroup).
      // Reduction dims: from subgroup tiles (K tile size stored there).
      SmallVector<Value> bounds;
      for (size_t i = 0; i < iteratorTypes.size(); ++i) {
        int64_t tileSize = 0;
        if (iteratorTypes[i] == utils::IteratorType::parallel) {
          tileSize = loopRanges[i];
        } else {
          // Reduction dim: K tile size from subgroup tiles.
          tileSize = (i < subgroupTiles.size()) ? subgroupTiles[i] : 0;
        }
        if (tileSize == 0 || ShapedType::isDynamic(tileSize)) {
          tileSize = 1;
        }
        bounds.push_back(arith::ConstantIndexOp::create(builder, loc, tileSize));
      }

      // Outer dimension distribution: subgroup tiles for parallel dims.
      SmallVector<int64_t> outerDimDistribution;
      for (auto [sg, iterType] :
           llvm::zip(subgroupTiles, iteratorTypes)) {
        if (iterType == utils::IteratorType::parallel && sg != 0) {
          outerDimDistribution.push_back(sg);
        }
      }
      while (outerDimDistribution.size() < 2) {
        outerDimDistribution.push_back(1);
      }

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
