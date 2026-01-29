// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTWORKGROUPFORALLTOPCFPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct ConvertWorkgroupForall : OpRewritePattern<scf::ForallOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override;
};

/// Folds an scf.forall with split-reduction mapping containing a pcf.loop
/// with workgroup scope into a single pcf.generic. This handles the "split-k"
/// pattern where the outer forall distributes work and the inner pcf.loop
/// represents the split-reduction dimension.
struct FoldSplitKWorkgroupLoop : OpRewritePattern<scf::ForallOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override;
};

struct ConvertWorkgroupForallToPCFPass final
    : public impl::ConvertWorkgroupForallToPCFPassBase<
          ConvertWorkgroupForallToPCFPass> {
  void runOnOperation() override;
  using Base::Base;
};

} // namespace

LogicalResult
ConvertWorkgroupForall::matchAndRewrite(scf::ForallOp op,
                                        PatternRewriter &rewriter) const {
  ArrayAttr mappingAttr = op.getMappingAttr();
  if (!mappingAttr || mappingAttr.empty() ||
      !llvm::all_of(mappingAttr,
                    llvm::IsaPred<IREE::Codegen::WorkgroupMappingAttr>)) {
    return failure();
  }
  // Linearize all ids down to 1 so that in cases when there are multiple
  // scf.foralls with incompatible delinearization bases. This technically
  // may be a small pessimization in very specific static cases, so if someone
  // ever finds they care they can try doing the analysis here to figure out
  // when it's ok not to linearize.
  //
  // Interface is implemented via external models hence the cast.
  auto scope = cast<IREE::PCF::ScopeAttrInterface>(
      IREE::Codegen::WorkgroupScopeAttr::get(rewriter.getContext(),
                                             /*linearize=*/true));
  FailureOr<IREE::PCF::LoopOp> res =
      convertForallToPCFLoop(rewriter, op, scope, 1);
  if (failed(res)) {
    return failure();
  }

  // Create a workgroup count hint to launch all workgroups along x.
  auto counts = llvm::to_vector_of<OpFoldResult>(res->getCount());
  rewriter.setInsertionPoint(*res);
  [[maybe_unused]] LogicalResult hintRes = createWorkgroupCountHint(
      rewriter, res->getLoc(), counts, /*maxWorkgroupParallelDims=*/1,
      /*reverse=*/false);
  assert(succeeded(hintRes) &&
         "Unexpected failure to construct workgroup count hint");
  return success();
}

LogicalResult
FoldSplitKWorkgroupLoop::matchAndRewrite(scf::ForallOp op,
                                         PatternRewriter &rewriter) const {
  // Match scf.forall with split-reduction mapping.
  if (!forallOpHasMappingType<IREE::LinalgExt::SplitReductionMappingAttr>(op)) {
    return failure();
  }

  // Find pcf.loop with workgroup scope in the forall body.
  IREE::PCF::LoopOp loopOp = nullptr;
  for (Operation &bodyOp : op.getBody()->without_terminator()) {
    if (auto loop = dyn_cast<IREE::PCF::LoopOp>(&bodyOp)) {
      // Check if the loop has workgroup scope.
      if (isa<IREE::Codegen::WorkgroupScopeAttr>(loop.getScope())) {
        loopOp = loop;
        break;
      }
    }
  }
  if (!loopOp) {
    return failure();
  }

  // Compute total workgroup count before folding (forall iterations * loop
  // count). We need this for the workgroup count hint since pcf.generic
  // doesn't store the count directly.
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);

  // Compute forall total iterations.
  SmallVector<OpFoldResult> upperBounds = op.getMixedUpperBound();
  Value forallCount = nullptr;
  for (OpFoldResult ub : upperBounds) {
    Value ubVal = getValueOrCreateConstantIndexOp(rewriter, loc, ub);
    if (!forallCount) {
      forallCount = ubVal;
    } else {
      forallCount = arith::MulIOp::create(rewriter, loc, forallCount, ubVal);
    }
  }

  // Get pcf.loop count.
  ValueRange loopCounts = loopOp.getCount();
  Value totalLoopCount = nullptr;
  for (Value count : loopCounts) {
    if (!totalLoopCount) {
      totalLoopCount = count;
    } else {
      totalLoopCount =
          arith::MulIOp::create(rewriter, loc, totalLoopCount, count);
    }
  }

  // Total workgroup count.
  Value totalCount =
      arith::MulIOp::create(rewriter, loc, forallCount, totalLoopCount);

  // Try to fold the forall into the pcf.loop.
  FailureOr<IREE::PCF::GenericOp> result =
      IREE::PCF::foldForallIntoPCFLoop(rewriter, op);
  if (failed(result)) {
    // On failure, we could remove the mapping attribute to let downstream
    // passes handle it differently, but for now just fail.
    return failure();
  }

  // Create workgroup count hint for the folded generic.
  SmallVector<OpFoldResult> counts = {OpFoldResult(totalCount)};
  rewriter.setInsertionPoint(*result);
  [[maybe_unused]] LogicalResult hintRes = createWorkgroupCountHint(
      rewriter, loc, counts, /*maxWorkgroupParallelDims=*/1,
      /*reverse=*/false);
  assert(succeeded(hintRes) &&
         "Unexpected failure to construct workgroup count hint");

  return success();
}

void ConvertWorkgroupForallToPCFPass::runOnOperation() {
  // First pass: Convert workgroup foralls to pcf.loop.
  RewritePatternSet patterns1(&getContext());
  patterns1.add<ConvertWorkgroupForall>(&getContext());
  walkAndApplyPatterns(getOperation(), std::move(patterns1));

  // Second pass: Fold split-k foralls containing pcf.loop into pcf.generic.
  RewritePatternSet patterns2(&getContext());
  patterns2.add<FoldSplitKWorkgroupLoop>(&getContext());
  walkAndApplyPatterns(getOperation(), std::move(patterns2));
}

} // namespace mlir::iree_compiler
