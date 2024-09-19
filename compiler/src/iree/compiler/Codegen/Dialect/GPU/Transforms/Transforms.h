// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Transforms.h - Transformations for the IREE GPU dialect ------------===//
//
// Defines transformations that apply to IREE GPU ops for use in multiple
// places.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::linalg {
class LinalgOp;
}

namespace mlir::scf {
class ForallOp;
}

namespace mlir::vector {
struct UnrollVectorOptions;
}

namespace mlir::iree_compiler::IREE::GPU {

/// Function to fuse the given producer-consumer pair of forall loops into
/// the single consumer loop. This is managed by inserting an
/// `iree_gpu.barrier_region` at the boundary to synchronize the workers at
/// the fusion point.
///
/// The mapping attributes of both the producer and consumer `scf.forall` ops
/// must be in a relative descending order, for example:
///  [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>]
/// or
///  [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]
LogicalResult fuseForallIntoConsumer(RewriterBase &rewriter,
                                     scf::ForallOp producer,
                                     scf::ForallOp consumer,
                                     SmallVector<Operation *> consumerChain);

// Helper to convert a contraction-like linalg op to an iree_gpu.multi_mma.
FailureOr<IREE::GPU::MultiMmaOp>
convertContractionToMultiMma(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                             IREE::GPU::MmaInterfaceAttr mmaKind);

// Helper to distribute a multi_mma op to lanes.
FailureOr<Operation *> distributeMultiMmaOp(RewriterBase &rewriter,
                                            IREE::GPU::MultiMmaOp mmaOp);

// Helper to map all scf.forall ops on lanes.
void mapLaneForalls(RewriterBase &rewriter, Operation *funcOp,
                    bool insertBarrier);

// Various populate pattern methods.
void populateIREEGPUDropUnitDimsPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerMultiMmaPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerBarrierRegionPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerValueBarrierPatterns(RewritePatternSet &patterns);
void populateIREEGPUVectorUnrollPatterns(
    RewritePatternSet &patterns, const vector::UnrollVectorOptions &options);
// Version of unrolling with a preset configuration.
void populateIREEGPUVectorUnrollPatterns(RewritePatternSet &patterns);
void populateIREEGPUVectorizationPatterns(RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_
