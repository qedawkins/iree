// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPCFDistribution.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/EquivalenceAnalysis.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/DistributionPatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// PCFVectorLayoutOptions
//===----------------------------------------------------------------------===//

IREE::VectorExt::VectorLayoutInterface
PCFVectorLayoutOptions::getDefaultLayout(VectorType type) const {
  // Only allow a default layout for 0-d vectors.
  if (type.getRank() > 0) {
    return IREE::VectorExt::VectorLayoutInterface();
  }
  ArrayRef<int64_t> empty = {};
  return IREE::VectorExt::NestedLayoutAttr::get(
      type.getContext(), empty, empty, empty, empty, empty, empty, empty);
}

//===----------------------------------------------------------------------===//
// VectorDistributionImpl
//===----------------------------------------------------------------------===//

LogicalResult VectorDistributionImpl::distributeRegions(
    ArrayRef<Region *> regions, ValueRange threadIDs, ValueRange threadCounts,
    const IREE::PCF::ClusterEquivalenceInfo &equivalenceInfo,
    const DenseSet<Operation *> &opsToSkip) {
  if (regions.empty()) {
    return success();
  }

  // Linearize multi-dimensional thread IDs into a single lane ID.
  // Row-major: lane = id0 * count1 * ... + id1 * count2 * ... + idN.
  Operation *parentOp = regions.front()->getParentOp();
  OpBuilder builder(parentOp);
  Location loc = parentOp->getLoc();
  Value laneId = threadIDs.front();
  for (int64_t i = 1, e = threadIDs.size(); i < e; ++i) {
    laneId = arith::MulIOp::create(builder, loc, laneId, threadCounts[i]);
    laneId = arith::AddIOp::create(builder, loc, laneId, threadIDs[i]);
  }

  // opsToSkip contains already-distributed run_thread ops. These should
  // not have to_layout anchors and thus won't receive distribution
  // signatures. Assert this invariant.
  for (Operation *skipOp : opsToSkip) {
    skipOp->walk([](Operation *op) {
      assert(!IREE::VectorExt::hasOpSignature(op) &&
             "skipped op should not have vector distribution signature");
    });
  }

  // Collect the parent ops of all regions as roots for distribution.
  SmallVector<Operation *> roots;
  for (Region *region : regions) {
    roots.push_back(region->getParentOp());
  }

  // Pre-compute an equivalence lookup map from the ClusterEquivalenceInfo. This
  // avoids iterating all cluster groups and run_cluster ops on every callback
  // invocation.
  DenseMap<Value, SmallVector<Value>> equivMap;
  SmallVector<Value> allResults;
  for (auto &[clusterId, ops] : equivalenceInfo.getClusterGroups()) {
    for (IREE::PCF::RunClusterOp rc : ops) {
      Value result = rc.getResult();
      if (result) {
        allResults.push_back(result);
      }
    }
  }
  for (Value val : allResults) {
    for (Value other : allResults) {
      if (other != val && equivalenceInfo.areEquivalent(val, other)) {
        equivMap[val].push_back(other);
      }
    }
  }

  LayoutEquivalenceCallback equivCallback =
      [&equivMap](OpOperand &operand) -> SmallVector<Value> {
    auto it = equivMap.find(operand.get());
    if (it != equivMap.end()) {
      return it->second;
    }
    return {};
  };

  // Run layout analysis across all roots with equivalence propagation.
  llvm::MapVector<Value, IREE::VectorExt::VectorLayoutInterface> layouts;
  propagateVectorLayoutInfo(roots, layouts, equivCallback);

  // Create layout options and distribution patterns.
  PCFVectorLayoutOptions options(roots.front());
  RewritePatternSet patterns(parentOp->getContext());
  populateGPUDistributionPatterns(patterns);
  IREE::VectorExt::populateNestedLayoutDistributionPatterns(
      patterns, laneId, subgroupSize, workgroupSize);

  // Run distribution with pre-computed layouts.
  return distributeVectorOps(roots, patterns, options, layouts);
}

void registerPCFVectorDistribution() {
  IREE::PCF::registerDistributionFactory(
      [](int64_t subgroupSize, ArrayRef<int64_t> workgroupSize)
          -> std::unique_ptr<IREE::PCF::DistributionInterface> {
        return std::make_unique<VectorDistributionImpl>(subgroupSize,
                                                        workgroupSize);
      });
}

} // namespace mlir::iree_compiler
