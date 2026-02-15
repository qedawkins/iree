// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- DistributionAnalysis.h - Layout constraint analysis for PCF --------===//
//
// Iterative layout analysis for DistributeSharedExecutor. Determines how
// tensor values within a pcf.shared_executor region should be distributed
// across parallel workers using NestedLayoutAttr constraints.
//
// The analysis follows a seed-propagate-fill-resolve pattern modeled after
// the existing VectorLayoutAnalysis in VectorExt.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_DISTRIBUTIONANALYSIS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_DISTRIBUTIONANALYSIS_H_

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// LayoutConstraintInfo
//===----------------------------------------------------------------------===//

/// Stores layout constraints for values within a shared_executor region.
///
/// Follows the same seed-propagate-resolve pattern as the existing
/// VectorLayoutAnalysis but operates on pcf-level collective ops rather
/// than vector-level ops.
///
/// Each value in the shared_executor region may have an associated
/// LayoutAttrInterface attribute (concretely, a NestedLayoutAttr from
/// VectorExt) that describes how it should be distributed across workers.
struct LayoutConstraintInfo {
  /// Set the layout for a value if not already set. Returns true if the
  /// layout was newly set (and the value should be added to worklists).
  /// If the value already has a different layout, records a conflict.
  bool setLayout(Value value, Attribute layout);

  /// Get the layout for a value, or nullptr if not constrained.
  Attribute getLayout(Value value) const;

  /// Check whether a value has a layout constraint.
  bool hasLayout(Value value) const;

  /// Map from SSA values to their resolved layout constraints.
  /// Uses MapVector for deterministic iteration order.
  llvm::MapVector<Value, Attribute> layouts;

  /// Forward propagation worklist. Values whose layouts have been set
  /// and whose users need to be checked.
  SmallVector<Value> forwardWorklist;

  /// Backward propagation worklist. Values whose layouts have been set
  /// and whose operands need to be checked.
  SmallVector<Value> backwardWorklist;

  /// Recorded conflicts: values where two incompatible layouts met.
  /// Pairs of (value, conflicting layout). The existing layout in the map
  /// takes precedence; the second element is the one that was rejected.
  SmallVector<std::pair<Value, Attribute>> conflicts;
};

//===----------------------------------------------------------------------===//
// Analysis phases
//===----------------------------------------------------------------------===//

/// Phase 1: SEED. Collect explicit layout constraints from
/// pcf.constrain_layout and pcf.constrain_mma ops within the region.
/// Seeds are added to both forward and backward worklists.
///
/// Requires Pair B ops: ConstrainLayoutOp, ConstrainMmaOp.
void seedConstraints(Region &region, LayoutConstraintInfo &info);

/// Phase 2: PROPAGATE. Run iterative forward/backward constraint propagation
/// until both worklists are empty (fixed point). Uses per-op propagation
/// rules for linalg, pcf, scf, and elementwise operations.
void propagateToFixedPoint(LayoutConstraintInfo &info);

/// Phase 3: FILL UNCONSTRAINED. Assign default coalesced layouts to any
/// values that remain unconstrained after propagation. After assigning
/// defaults, runs another round of propagation.
///
/// |numThreads|: number of threads per subgroup for the innermost scope.
/// |numSubgroups|: number of subgroups for the outermost scope.
void fillUnconstrained(Region &region, LayoutConstraintInfo &info,
                       int64_t numThreads, int64_t numSubgroups);

/// Phase 4: RESOLVE CONFLICTS. Check for layout conflicts at operation
/// boundaries. If |strictLayoutChecking| is true, emits errors on conflicts.
/// Otherwise, inserts pcf.redistribute ops to convert between layouts.
///
/// Requires Pair B ops: RedistributeOp.
LogicalResult resolveConflicts(Region &region, LayoutConstraintInfo &info,
                               bool strictLayoutChecking, OpBuilder &builder);

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Create a default coalesced NestedLayoutAttr for the given shape.
/// Distributes the innermost dimensions across threads with unit stride
/// (coalesced access pattern).
///
/// |shape|: the tensor dimensions.
/// |numThreads|: threads per subgroup.
/// |numSubgroups|: number of subgroups.
/// |elementType|: element type for computing vector width.
/// |context|: MLIR context for attribute creation.
Attribute createCoalescedLayout(ArrayRef<int64_t> shape, int64_t numThreads,
                                int64_t numSubgroups, Type elementType,
                                MLIRContext *context);

/// Determine the redistribution method based on the relationship between
/// two layouts. Returns "registers", "shuffle", or "shared_memory".
StringRef determineRedistributionMethod(Attribute sourceLayout,
                                        Attribute targetLayout);

} // namespace mlir::iree_compiler::IREE::PCF

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_DISTRIBUTIONANALYSIS_H_
