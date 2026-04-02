// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/TilingImplementations/RegisterAll.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_TESTTILETOPCFPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

struct TestTileToPCFPass final
    : impl::TestTileToPCFPassBase<TestTileToPCFPass> {
  using TestTileToPCFPassBase::TestTileToPCFPassBase;
  void runOnOperation() override {
    attachAllDistributedTilingModels(&getContext());
    Operation *rootOp = getOperation();
    IRRewriter rewriter(&getContext());
    SmallVector<OpFoldResult> tileSizeOFRs =
        llvm::map_to_vector(tileSizes, [&](int64_t s) -> OpFoldResult {
          return rewriter.getIndexAttr(s);
        });
    ScopeAttrInterface scope = PCF::SequentialAttr::get(&getContext());
    // Collect ops first to avoid modifying the IR while walking.
    SmallVector<PCFTilingOpInterface> targets;
    rootOp->walk([&](Operation *op) {
      if (auto tilingOp = dyn_cast<PCFTilingOpInterface>(op)) {
        targets.push_back(tilingOp);
      }
    });
    for (PCFTilingOpInterface target : targets) {
      rewriter.setInsertionPoint(target);
      (void)tileToPCFLoop(rewriter, target, scope, tileSizeOFRs);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::PCF
