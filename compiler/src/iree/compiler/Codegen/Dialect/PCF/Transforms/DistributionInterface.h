// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_DISTRIBUTION_INTERFACE_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_DISTRIBUTION_INTERFACE_H_

#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::PCF {

class ClusterEquivalenceInfo;

/// Abstract interface for distributing ops within run_cluster regions.
///
/// Implementations convert collective (run_cluster) semantics to per-thread
/// (run_thread) semantics. The interface receives thread IDs and counts for
/// the cluster being distributed, along with equivalence information for
/// cross-region value tracking.
class DistributionInterface {
public:
  virtual ~DistributionInterface() = default;

  /// Distribute a set of regions sharing the same cluster ID.
  ///
  /// \param regions      run_cluster bodies to distribute (same cluster ID).
  ///                     A run_cluster's cluster ID is determined by the
  ///                     NamespacedSymbolAttr ID on its source cluster types.
  /// \param threadIDs    Per-dimension worker IDs for this cluster.
  /// \param threadCounts Per-dimension worker counts for this cluster.
  /// \param equivalenceInfo Maps block args/results across regions to
  ///                        their equivalent counterparts.
  /// \param opsToSkip   Already-distributed ops (run_thread bodies) to
  ///                    leave untouched.
  virtual LogicalResult
  distributeRegions(ArrayRef<Region *> regions, ValueRange threadIDs,
                    ValueRange threadCounts,
                    const ClusterEquivalenceInfo &equivalenceInfo,
                    const DenseSet<Operation *> &opsToSkip) = 0;
};

} // namespace mlir::iree_compiler::IREE::PCF

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_DISTRIBUTION_INTERFACE_H_
