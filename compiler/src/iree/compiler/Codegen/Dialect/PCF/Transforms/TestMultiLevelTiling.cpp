// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTilingInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/TilingImplementations/RegisterAll.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_TESTMULTILEVELTILINGPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

static constexpr StringLiteral kLoweringConfigAttrName = "lowering_config";
static constexpr StringLiteral kSubgroupTilesKey = "subgroup";
static constexpr StringLiteral kLaneTilesKey = "lane";
static constexpr StringLiteral kReductionTilesKey = "reduction";

static FailureOr<SmallVector<OpFoldResult>>
parseTileSizes(IRRewriter &rewriter, Operation *op, DictionaryAttr config,
               StringRef key, int64_t rank) {
  SmallVector<OpFoldResult> parsedTiles(rank, rewriter.getIndexAttr(0));
  Attribute levelAttr = config.get(key);
  if (!levelAttr) {
    return parsedTiles;
  }

  SmallVector<int64_t> rawTiles;
  if (auto denseArray = dyn_cast<DenseI64ArrayAttr>(levelAttr)) {
    rawTiles.assign(denseArray.asArrayRef().begin(),
                    denseArray.asArrayRef().end());
  } else if (auto arrayAttr = dyn_cast<ArrayAttr>(levelAttr)) {
    rawTiles.reserve(arrayAttr.size());
    for (Attribute attr : arrayAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(attr);
      if (!intAttr) {
        op->emitOpError() << "'" << kLoweringConfigAttrName << "." << key
                          << "' expects integer elements";
        return failure();
      }
      rawTiles.push_back(intAttr.getInt());
    }
  } else {
    op->emitOpError() << "'" << kLoweringConfigAttrName << "." << key
                      << "' expects an integer array";
    return failure();
  }

  for (int64_t i = 0, e = std::min<int64_t>(rank, rawTiles.size()); i < e;
       ++i) {
    parsedTiles[i] = rewriter.getIndexAttr(rawTiles[i]);
  }
  return parsedTiles;
}

struct TestMultiLevelTilingPass final
    : impl::TestMultiLevelTilingPassBase<TestMultiLevelTilingPass> {
  using TestMultiLevelTilingPassBase::TestMultiLevelTilingPassBase;
  void runOnOperation() override {
    attachAllDistributedTilingModels(&getContext());

    IRRewriter rewriter(&getContext());
    SmallVector<PCFTilingOpInterface> targets;
    getOperation()->walk([&](Operation *op) {
      auto tilingOp = dyn_cast<PCFTilingOpInterface>(op);
      if (!tilingOp) {
        return;
      }
      if (op->hasAttr(kLoweringConfigAttrName)) {
        targets.push_back(tilingOp);
      }
    });

    for (PCFTilingOpInterface target : targets) {
      Operation *op = target.getOperation();
      auto config = dyn_cast_or_null<DictionaryAttr>(
          op->getAttr(kLoweringConfigAttrName));
      if (!config) {
        op->emitOpError() << "'" << kLoweringConfigAttrName
                          << "' must be a DictionaryAttr in this test pass";
        return signalPassFailure();
      }

      int64_t rank = cast<TilingInterface>(op).getLoopIteratorTypes().size();
      FailureOr<SmallVector<OpFoldResult>> subgroupTiles =
          parseTileSizes(rewriter, op, config, kSubgroupTilesKey, rank);
      FailureOr<SmallVector<OpFoldResult>> laneTiles =
          parseTileSizes(rewriter, op, config, kLaneTilesKey, rank);
      FailureOr<SmallVector<OpFoldResult>> reductionTiles =
          parseTileSizes(rewriter, op, config, kReductionTilesKey, rank);
      if (failed(subgroupTiles) || failed(laneTiles) ||
          failed(reductionTiles)) {
        return signalPassFailure();
      }

      MultiLevelTilingParams params;
      params.subgroup.scope = PCF::SequentialAttr::get(&getContext());
      params.subgroup.tileSizes = std::move(*subgroupTiles);
      params.lane.scope = PCF::SequentialAttr::get(&getContext());
      params.lane.tileSizes = std::move(*laneTiles);
      params.reductionTileSizes = std::move(*reductionTiles);

      rewriter.setInsertionPoint(op);
      if (failed(applyMultiLevelTiling(rewriter, target, params))) {
        op->emitOpError("failed to apply test multi-level tiling");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::PCF
