// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/Transforms/EquivalenceAnalysis.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"

namespace mlir::iree_compiler {

llvm::EquivalenceClasses<Value>
computeValueEquivalences(Operation *root, ArrayRef<Value> seeds,
                         function_ref<bool(Value)> isRelevant) {
  llvm::EquivalenceClasses<Value> equivalences;
  if (seeds.empty()) {
    return equivalences;
  }

  // Create an Explorer rooted at the parent op to handle control flow
  // traversal through RegionBranchOpInterface, block arguments, yields,
  // etc.
  Explorer explorer(root, TraversalAction::RECURSE);
  explorer.initialize();

  for (Value seed : seeds) {
    // Insert the seed so it always has an equivalence class entry.
    equivalences.insert(seed);

    // Walk all transitive uses of the seed through control flow. The
    // Explorer automatically follows values through region boundaries
    // (yields -> successor block args, operands -> entry block args),
    // branches, calls, and tied ops. In the callback, use.get() is the
    // current worklist value being processed.
    explorer.walkTransitiveUses(seed, [&](OpOperand &use) -> WalkResult {
      Value current = use.get();
      if (current != seed && isRelevant(current)) {
        equivalences.unionSets(seed, current);
      }
      return WalkResult::advance();
    });
  }

  return equivalences;
}

} // namespace mlir::iree_compiler

namespace mlir::iree_compiler::IREE::PCF {

/// Helper that checks whether a value has a cluster type.
static bool isClusterTyped(Value value) {
  return isa<ClusterType>(value.getType());
}

ClusterEquivalenceInfo ClusterEquivalenceInfo::build(TileGroupOp tileGroup) {
  ClusterEquivalenceInfo info;

  // Step 1: Collect all run_cluster ops and group by cluster ID.
  tileGroup.getBody().walk([&](RunClusterOp runCluster) {
    ClusterType sourceType =
        cast<ClusterType>(runCluster.getSources().front().getType());
    Attribute clusterId = sourceType.getId();
    info.clusterGroups[clusterId].push_back(runCluster);
  });

  // Step 2: Collect seed values (run_cluster results) for equivalence
  // tracing.
  SmallVector<Value> seeds;
  for (auto &[clusterId, ops] : info.clusterGroups) {
    for (RunClusterOp runCluster : ops) {
      Value result = runCluster.getResult();
      if (result) {
        seeds.push_back(result);
      }
    }
  }

  // Step 3: Build equivalence classes using the generic Explorer-based
  // traversal. The isClusterTyped predicate ensures only cluster-typed
  // values are included in the equivalence classes.
  info.equivalences =
      computeValueEquivalences(tileGroup, seeds, isClusterTyped);

  return info;
}

void ClusterEquivalenceInfo::registerTypeConversion(Value original,
                                                    Type newType) {
  Value leader = getLeader(original);
  typeConversions[leader] = newType;
}

std::optional<Type>
ClusterEquivalenceInfo::getConvertedType(Value value) const {
  Value leader = getLeader(value);
  auto it = typeConversions.find(leader);
  if (it != typeConversions.end()) {
    return it->second;
  }
  return std::nullopt;
}

Value ClusterEquivalenceInfo::getLeader(Value value) const {
  auto it = equivalences.findLeader(value);
  if (it == equivalences.member_end()) {
    return value;
  }
  return equivalences.getLeaderValue(value);
}

bool ClusterEquivalenceInfo::areEquivalent(Value a, Value b) const {
  return equivalences.isEquivalent(a, b);
}

} // namespace mlir::iree_compiler::IREE::PCF
