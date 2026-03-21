// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_EQUIVALENCE_ANALYSIS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_EQUIVALENCE_ANALYSIS_H_

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler {

/// Build equivalence classes for a set of seed values by tracing them
/// through control flow (RegionBranchOpInterface, block arguments,
/// yields). Uses the Explorer infrastructure for traversal. Not
/// PCF-specific.
///
/// The |isRelevant| predicate filters which intermediate values are
/// included in the equivalence classes. Only values passing the
/// predicate are unioned with their seed.
llvm::EquivalenceClasses<Value>
computeValueEquivalences(Operation *root, ArrayRef<Value> seeds,
                         function_ref<bool(Value)> isRelevant);

} // namespace mlir::iree_compiler

namespace mlir::iree_compiler::IREE::PCF {

/// Tracks equivalence of cluster-typed values across run_cluster ops
/// within a tile_group.
///
/// Two values are equivalent when one run_cluster yields a result that
/// flows (through SSA def-use or RegionBranchOpInterface) into a
/// subsequent run_cluster's source operand. The struct elements bound
/// to equivalent cluster values must be distributed consistently.
class ClusterEquivalenceInfo {
public:
  /// Default constructor for empty equivalence info (no clusters).
  ClusterEquivalenceInfo() = default;

  /// Build equivalence info for all run_cluster ops in a tile_group.
  static ClusterEquivalenceInfo build(TileGroupOp tileGroup);

  /// Returns all run_cluster ops grouped by cluster ID.
  const llvm::MapVector<Attribute, SmallVector<RunClusterOp>> &
  getClusterGroups() const {
    return clusterGroups;
  }

  /// Register a type conversion for a value.
  void registerTypeConversion(Value original, Type newType);

  /// Returns the new type registered for a value, or nullopt.
  std::optional<Type> getConvertedType(Value value) const;

  /// Returns the equivalence class leader for a value.
  Value getLeader(Value value) const;

  /// Returns true if two values are in the same equivalence class.
  bool areEquivalent(Value a, Value b) const;

private:
  llvm::MapVector<Attribute, SmallVector<RunClusterOp>> clusterGroups;
  llvm::EquivalenceClasses<Value> equivalences;
  llvm::DenseMap<Value, Type> typeConversions;
};

} // namespace mlir::iree_compiler::IREE::PCF

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_EQUIVALENCE_ANALYSIS_H_
