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
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_TESTMULTILEVELTILINGPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

struct TestMultiLevelTilingPass final
    : impl::TestMultiLevelTilingPassBase<TestMultiLevelTilingPass> {
  using TestMultiLevelTilingPassBase::TestMultiLevelTilingPassBase;
  void runOnOperation() override {
    attachAllDistributedTilingModels(&getContext());

    IRRewriter rewriter(&getContext());

    // Build tile size OFRs from options.
    SmallVector<OpFoldResult> sgTiles =
        llvm::map_to_vector(subgroupTileSizes, [&](int64_t s) -> OpFoldResult {
          return rewriter.getIndexAttr(s);
        });
    SmallVector<OpFoldResult> laneTiles =
        llvm::map_to_vector(laneTileSizes, [&](int64_t s) -> OpFoldResult {
          return rewriter.getIndexAttr(s);
        });
    SmallVector<OpFoldResult> redTiles =
        llvm::map_to_vector(reductionTileSizes, [&](int64_t s) -> OpFoldResult {
          return rewriter.getIndexAttr(s);
        });

    // Use sequential scopes for testing.
    ScopeAttrInterface sgScope = PCF::SequentialAttr::get(&getContext());
    ScopeAttrInterface laneScope = PCF::SequentialAttr::get(&getContext());

    MultiLevelTilingParams params;
    params.subgroup.scope = sgScope;
    params.subgroup.tileSizes = sgTiles;
    params.lane.scope = laneScope;
    params.lane.tileSizes = laneTiles;
    params.reductionTileSizes = redTiles;

    // Walk and tile all PCFTilingOpInterface ops.
    getOperation()->walk([&](Operation *op) {
      auto tilingOp = dyn_cast<PCFTilingOpInterface>(op);
      if (!tilingOp) {
        return;
      }
      rewriter.setInsertionPoint(op);
      (void)applyMultiLevelTiling(rewriter, tilingOp, params);
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::PCF
