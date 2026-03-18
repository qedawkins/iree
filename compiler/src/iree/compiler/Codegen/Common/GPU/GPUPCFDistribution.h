// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_GPU_PCF_DISTRIBUTION_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_GPU_PCF_DISTRIBUTION_H_

#include "iree/compiler/Codegen/Dialect/PCF/Transforms/DistributionInterface.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/DistributionPatterns.h"

namespace mlir::iree_compiler {

/// Layout options for PCF vector distribution. Uses fullConversion=false
/// since not all vector ops may have anchors.
class PCFVectorLayoutOptions : public IREE::VectorExt::VectorLayoutOptions {
public:
  explicit PCFVectorLayoutOptions(Operation *root)
      : VectorLayoutOptions(root, /*fullConversion=*/false) {}

  IREE::VectorExt::VectorLayoutInterface
  getDefaultLayout(VectorType type) const override;
};

/// Implementation of DistributionInterface that wraps the existing
/// VectorDistribute infrastructure. Runs layout analysis across all
/// run_cluster regions (with equivalence propagation), then distributes
/// vector ops within those regions.
///
/// This class lives in Common/GPU (not PCF/Transforms) to avoid circular
/// dependencies: PCF/Transforms cannot depend on Common/GPU.
class VectorDistributionImpl final : public IREE::PCF::DistributionInterface {
public:
  /// \param subgroupSize  Subgroup (warp/wave) size for the target.
  /// \param workgroupSize Workgroup dimensions for linearization.
  VectorDistributionImpl(int64_t subgroupSize, ArrayRef<int64_t> workgroupSize)
      : subgroupSize(subgroupSize),
        workgroupSize(workgroupSize.begin(), workgroupSize.end()) {}

  LogicalResult
  distributeRegions(ArrayRef<Region *> regions, ValueRange threadIDs,
                    ValueRange threadCounts,
                    const IREE::PCF::ClusterEquivalenceInfo &equivalenceInfo,
                    const DenseSet<Operation *> &opsToSkip) override;

private:
  int64_t subgroupSize;
  SmallVector<int64_t> workgroupSize;
};

/// Register the VectorDistributionImpl factory with the PCF distribution
/// interface. Call this during initialization so that the
/// --vector-distribution pass option can create VectorDistributionImpl.
void registerPCFVectorDistribution();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_GPU_PCF_DISTRIBUTION_H_
