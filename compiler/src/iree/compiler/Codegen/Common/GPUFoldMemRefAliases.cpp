// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPUPatterns.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

static LogicalResult
resolveSourceIndicesSubView(Location loc, PatternRewriter &rewriter,
                            memref::SubViewOp subViewOp, ValueRange indices,
                            SmallVectorImpl<Value> &sourceIndices) {
  SmallVector<OpFoldResult> mixedOffsets = subViewOp.getMixedOffsets();
  SmallVector<OpFoldResult> mixedSizes = subViewOp.getMixedSizes();
  SmallVector<OpFoldResult> mixedStrides = subViewOp.getMixedStrides();

  SmallVector<Value> useIndices;
  // Check if this is rank-reducing case. Then for every unit-dim size add a
  // zero to the indices.
  unsigned resultDim = 0;
  llvm::SmallBitVector unusedDims = subViewOp.getDroppedDims();
  for (auto dim : llvm::seq<unsigned>(0, subViewOp.getSourceType().getRank())) {
    if (unusedDims.test(dim))
      useIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    else
      useIndices.push_back(indices[resultDim++]);
  }
  if (useIndices.size() != mixedOffsets.size())
    return failure();
  sourceIndices.resize(useIndices.size());
  for (auto index : llvm::seq<size_t>(0, mixedOffsets.size())) {
    SmallVector<Value> dynamicOperands;
    AffineExpr expr = rewriter.getAffineDimExpr(0);
    unsigned numSymbols = 0;
    dynamicOperands.push_back(useIndices[index]);

    // Multiply the stride;
    if (auto attr = mixedStrides[index].dyn_cast<Attribute>()) {
      expr = expr * attr.cast<IntegerAttr>().getInt();
    } else {
      dynamicOperands.push_back(mixedStrides[index].get<Value>());
      expr = expr * rewriter.getAffineSymbolExpr(numSymbols++);
    }

    // Add the offset.
    if (auto attr = mixedOffsets[index].dyn_cast<Attribute>()) {
      expr = expr + attr.cast<IntegerAttr>().getInt();
    } else {
      dynamicOperands.push_back(mixedOffsets[index].get<Value>());
      expr = expr + rewriter.getAffineSymbolExpr(numSymbols++);
    }
    Location loc = subViewOp.getLoc();
    sourceIndices[index] = rewriter.create<AffineApplyOp>(
        loc, AffineMap::get(1, numSymbols, expr), dynamicOperands);
  }
  return success();
}

static SmallVector<Value>
calculateExpandedAccessIndices(AffineMap affineMap,
                               const SmallVector<Value> &indices, Location loc,
                               PatternRewriter &rewriter) {
  SmallVector<Value> expandedIndices;
  for (unsigned i = 0, e = affineMap.getNumResults(); i < e; i++)
    expandedIndices.push_back(
        rewriter.create<AffineApplyOp>(loc, affineMap.getSubMap({i}), indices));
  return expandedIndices;
}

class SubgroupMmaLoadOpOfSubViewOpFolder final : public OpRewritePattern<gpu::SubgroupMmaLoadMatrixOp> {
public:
  using OpRewritePattern<gpu::SubgroupMmaLoadMatrixOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SubgroupMmaLoadMatrixOp loadOp,
                                PatternRewriter &rewriter) const {
    auto subViewOp =
        loadOp.getSrcMemref().getDefiningOp<memref::SubViewOp>();

    if (!subViewOp)
      return failure();

    SmallVector<Value> indices(loadOp.getIndices().begin(),
                               loadOp.getIndices().end());
    // For affine ops, we need to apply the map to get the operands to get the
    // "actual" indices.
    if (auto affineLoadOp = dyn_cast<AffineLoadOp>(loadOp.getOperation())) {
      AffineMap affineMap = affineLoadOp.getAffineMap();
      auto expandedIndices = calculateExpandedAccessIndices(
          affineMap, indices, loadOp.getLoc(), rewriter);
      indices.assign(expandedIndices.begin(), expandedIndices.end());
    }
    SmallVector<Value, 4> sourceIndices;
    if (failed(resolveSourceIndicesSubView(loadOp.getLoc(), rewriter, subViewOp,
                                           indices, sourceIndices)))
      return failure();

    //auto stride = loadOp.getLeadDimension().getSExtValue();
    //if (stride != 0)
    //  stride = cast<MemRefType>(subViewOp.getSource().getType()).getShape().back();

    //rewriter.replaceOpWithNewOp<gpu::SubgroupMmaLoadMatrixOp>(loadOp,
    //        loadOp.getType(), subViewOp.getSource(),
    //        sourceIndices, rewriter.getIndexAttr(stride), loadOp.getTransposeAttr());
    rewriter.replaceOpWithNewOp<gpu::SubgroupMmaLoadMatrixOp>(loadOp,
            loadOp.getType(), subViewOp.getSource(),
            sourceIndices, loadOp.getLeadDimension(), loadOp.getTransposeAttr());
    return success();
  }
};

struct GPUFoldMemRefAliasesPass final
    : public GPUFoldMemRefAliasesBase<GPUFoldMemRefAliasesPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, memref::MemRefDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    RewritePatternSet patterns(funcOp.getContext());
    memref::populateFoldMemRefAliasOpPatterns(patterns);
    patterns.add<SubgroupMmaLoadOpOfSubViewOpFolder>(patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createGPUFoldMemRefAliasesPass() {
  return std::make_unique<GPUFoldMemRefAliasesPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
