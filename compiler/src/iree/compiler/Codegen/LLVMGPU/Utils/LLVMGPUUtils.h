// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_LLVMGPUUTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_LLVMGPUUTILS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

/// Helper to convert copy to shared memory to async copy. This creates groups
/// of consecutive copies and emit wait operation right after.
void createAsyncGroups(RewriterBase &rewriter, mlir::FunctionOpInterface funcOp,
                       bool useMMASync);

/// Function to do layout analysis and distribution.
void doLayoutAnalysisAndDistribution(RewriterBase &rewriter,
                                     mlir::FunctionOpInterface funcOp);

/// Function to reorder transposes and elementwise ops.
void reorderTranspose(RewriterBase &rewriter, mlir::FunctionOpInterface funcOp);

/// Look for allocs in shared memory space with overlapping liveness,
/// group them, and then pack all the allocations in each group into one i8
/// alloc.
///
/// Also adds barriers to make sure we are done writing/reading
/// from the previous alias group before starting a new one.
void packSharedMemoryAlloc(mlir::FunctionOpInterface funcOp);

// Add patterns to distribute contractions to MFMA ops.
void populateAMDGPUDistributionPatterns(RewritePatternSet &patterns);

// Prefetches data written to shared memory for the next iteration. Returns the
// new loop on success or failure when the `forOp` is not supported.
FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
                                               scf::ForOp forOp);

/// Insert barriers and wait operations if there are allocs of a different alias
/// group before the given alloc.
void addBarrier(mlir::FunctionOpInterface funcOp, Operation *alloc,
                ArrayRef<Operation *> aliasGroup, bool hasAsyncCopies = true);

} // namespace mlir::iree_compiler

#endif
