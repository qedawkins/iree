// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_UNROLLANNOTATEDLOOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Returns the trip count of `forOp` if its' low bound, high bound and step are
/// constants, or optional otherwise. Trip count is computed as
/// ceilDiv(highBound - lowBound, step).
static std::optional<int64_t> getConstantTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lbCstOp = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ubCstOp = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> stepCstOp = getConstantIntValue(forOp.getStep());
  if (!lbCstOp.has_value() || !ubCstOp.has_value() || !stepCstOp.has_value())
    return std::nullopt;

  // Constant loop bounds computation.
  int64_t lbCst = lbCstOp.value();
  int64_t ubCst = ubCstOp.value();
  int64_t stepCst = stepCstOp.value();
  if (lbCst < 0 || ubCst < 0 || stepCst <= 0) {
    return std::nullopt;
  }
  return llvm::divideCeil(ubCst - lbCst, stepCst);
}

struct UnrollAnnotatedLoopsPass final
    : impl::UnrollAnnotatedLoopsPassBase<UnrollAnnotatedLoopsPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();

    // Get the list of operations to unroll in pre-order so that the inner
    // most loops get unrolled before the outer most loops.
    SmallVector<scf::ForOp> unrollTargets;
    funcOp.walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
      if (getLoopUnrollMarker(forOp)) {
        unrollTargets.push_back(forOp);
      }
    });

    for (auto forOp : unrollTargets) {
      removeLoopUnrollMarker(forOp);

      std::optional<int64_t> maybeTripCount = getConstantTripCount(forOp);
      if (!maybeTripCount || maybeTripCount.value() <= 0) {
        continue;
      }

      (void)loopUnrollByFactor(forOp, maybeTripCount.value());
    }

    // Cleanup unrolled loops.
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);
      scf::ForOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        funcOp->emitError("Failed to apply post unroll cleanup");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
