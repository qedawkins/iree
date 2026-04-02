// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCONVERTTHREADFORALLTOSUBGROUPLANEPCFPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Returns true if the forall op has gpu::GPUThreadMappingAttr mapping
/// attributes.
static bool hasThreadMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  if (!mapping || mapping->empty()) {
    return false;
  }
  return llvm::all_of(mapping.value(),
                      llvm::IsaPred<gpu::GPUThreadMappingAttr>);
}

/// Create a pcf.sref type from a ranked tensor type and scope.
static IREE::PCF::ShapedRefType
getSrefTypeFromTensor(RankedTensorType tensorType,
                      IREE::PCF::ScopeAttrInterface scope) {
  return IREE::PCF::ShapedRefType::get(tensorType.getContext(),
                                       tensorType.getShape(),
                                       tensorType.getElementType(), scope);
}

/// Create a pcf.sref type matching a shaped type's shape and element type.
static IREE::PCF::ShapedRefType getSrefType(ShapedType shapedType,
                                            IREE::PCF::ScopeAttrInterface scope,
                                            Attribute syncScope = {}) {
  return IREE::PCF::ShapedRefType::get(
      shapedType.getContext(), shapedType.getShape(),
      shapedType.getElementType(), scope, syncScope);
}

/// Walk a single-use def chain from an sref value, converting tensor ops to
/// their PCF equivalents. Handles expand_shape, extract_slice, and
/// transfer_read dynamically — any ordering, any number of intermediate ops.
static void convertSrefConsumerChain(IRRewriter &rewriter, Value srefVal,
                                     IREE::PCF::ShapedRefType srefType,
                                     IREE::PCF::ScopeAttrInterface scope) {
  // Worklist of (current sref value, current sref type) pairs.
  SmallVector<std::pair<Value, IREE::PCF::ShapedRefType>> worklist;
  worklist.push_back({srefVal, srefType});

  while (!worklist.empty()) {
    auto [currentVal, currentSrefType] = worklist.pop_back_val();

    for (OpOperand &use : llvm::make_early_inc_range(currentVal.getUses())) {
      Operation *user = use.getOwner();

      if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(user)) {
        rewriter.setInsertionPoint(expandOp);
        RankedTensorType resultTensorType = expandOp.getResultType();
        IREE::PCF::ShapedRefType expandedSrefType = getSrefType(
            resultTensorType, scope, currentSrefType.getSyncScope());

        auto pcfExpand = IREE::PCF::ExpandShapeOp::create(
            rewriter, expandOp.getLoc(), expandedSrefType, currentVal,
            expandOp.getReassociation(), expandOp.getOutputShape());
        worklist.push_back({pcfExpand.getResult(), expandedSrefType});
        rewriter.replaceOp(expandOp, pcfExpand.getResult());
        continue;
      }

      if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
        rewriter.setInsertionPoint(extractOp);
        RankedTensorType resultTensorType = extractOp.getResultType();
        IREE::PCF::ShapedRefType subviewSrefType = getSrefType(
            resultTensorType, scope, currentSrefType.getSyncScope());

        auto pcfSubview = IREE::PCF::SubviewOp::create(
            rewriter, extractOp.getLoc(), subviewSrefType, currentVal,
            extractOp.getMixedOffsets(), extractOp.getMixedSizes(),
            extractOp.getMixedStrides());
        worklist.push_back({pcfSubview.getResult(), subviewSrefType});
        rewriter.replaceOp(extractOp, pcfSubview.getResult());
        continue;
      }

      if (auto readOp = dyn_cast<vector::TransferReadOp>(user)) {
        rewriter.setInsertionPoint(readOp);
        VectorType vecType = readOp.getVectorType();

        // Create offsets from the transfer_read indices.
        SmallVector<OpFoldResult> offsets = llvm::map_to_vector(
            readOp.getIndices(), [](Value v) -> OpFoldResult { return v; });
        SmallVector<OpFoldResult> sizes;
        for (int64_t dim : vecType.getShape()) {
          sizes.push_back(rewriter.getIndexAttr(dim));
        }
        SmallVector<OpFoldResult> strides(vecType.getRank(),
                                          rewriter.getIndexAttr(1));

        auto pcfRead =
            IREE::PCF::ReadSliceOp::create(rewriter, readOp.getLoc(), vecType,
                                           currentVal, offsets, sizes, strides);
        rewriter.replaceOp(readOp, pcfRead.getResult());
        continue;
      }

      // Unknown user — leave it alone. This handles cases like DMA ops
      // whose types already accept sref.
    }
  }
}

/// Find the outermost ancestor pcf.generic and return its scope.
/// For workgroup shared memory, we want the outermost scope (e.g.
/// subgroup_scope) rather than the innermost (e.g. lane_scope).
static IREE::PCF::ScopeAttrInterface findOutermostScope(Operation *op) {
  IREE::PCF::ScopeAttrInterface result;
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto genericOp = dyn_cast<IREE::PCF::GenericOp>(parent)) {
      result = genericOp.getScope();
    }
    parent = parent->getParentOp();
  }
  return result;
}

/// Convert barrier_region chains from tensor to sref semantics.
/// Pattern: bufferization.alloc_tensor → barrier_region → consumer chain.
/// Converts to: pcf.alloc → barrier_region(sref) → pcf ops.
/// The alloc_tensor may live outside the pcf.generic (e.g. at workgroup
/// scope) while the barrier_region consumers live inside it.
static void convertBarrierChainToSref(IRRewriter &rewriter, Operation *funcOp) {
  // Find all workgroup-scoped alloc_tensor ops in the function.
  SmallVector<bufferization::AllocTensorOp> allocOps;
  funcOp->walk([&](bufferization::AllocTensorOp allocOp) {
    std::optional<Attribute> memSpace = allocOp.getMemorySpace();
    if (!memSpace) {
      return;
    }
    auto addrSpace = dyn_cast<gpu::AddressSpaceAttr>(*memSpace);
    if (!addrSpace || addrSpace.getValue() != gpu::AddressSpace::Workgroup) {
      return;
    }
    allocOps.push_back(allocOp);
  });

  for (bufferization::AllocTensorOp allocOp : allocOps) {
    RankedTensorType tensorType = allocOp.getType();

    // Determine the scope from the first barrier_region user's enclosing
    // pcf.generic. The alloc may be outside the generic (e.g. at workgroup
    // level) while barriers are inside it.
    IREE::PCF::ScopeAttrInterface scope;
    for (Operation *user : allocOp->getUsers()) {
      if (auto barrierOp = dyn_cast<IREE::GPU::BarrierRegionOp>(user)) {
        scope = findOutermostScope(barrierOp);
        if (scope) {
          break;
        }
      }
    }
    if (!scope) {
      continue;
    }

    IREE::PCF::ShapedRefType srefType =
        getSrefTypeFromTensor(tensorType, scope);

    // Replace alloc_tensor with pcf.alloc.
    rewriter.setInsertionPoint(allocOp);
    Value srefAlloc = IREE::PCF::AllocOp::create(
        rewriter, allocOp.getLoc(), srefType, allocOp.getDynamicSizes());

    // Find barrier_region users and convert them.
    for (OpOperand &use : llvm::make_early_inc_range(allocOp->getUses())) {
      Operation *user = use.getOwner();
      auto barrierOp = dyn_cast<IREE::GPU::BarrierRegionOp>(user);
      if (!barrierOp) {
        continue;
      }

      int64_t inputIdx = use.getOperandNumber();

      Block &body = barrierOp.getRegion().front();
      BlockArgument blockArg = body.getArgument(inputIdx);

      // Phase A: Fix inner foralls BEFORE changing the block arg type.
      // The scf.forall verifier requires tensor shared_outs, so we must
      // decouple the forall from the block arg while it's still tensor-typed.
      SmallVector<scf::ForallOp> innerForalls;
      body.walk([&](scf::ForallOp forallOp) {
        for (auto [idx, output] : llvm::enumerate(forallOp.getOutputs())) {
          if (output == blockArg) {
            innerForalls.push_back(forallOp);
            break;
          }
        }
      });

      for (scf::ForallOp innerForall : innerForalls) {
        for (auto [outputIdx, output] :
             llvm::enumerate(innerForall.getOutputs())) {
          if (output != blockArg) {
            continue;
          }

          BlockArgument sharedOut = innerForall.getRegionIterArgs()[outputIdx];

          // Convert parallel_insert_slice ops to pcf.write_slice, writing
          // directly to the sref alloc (bypassing the forall's DPS).
          // Place write_slice in the forall body (before the in_parallel
          // terminator), not inside in_parallel which only allows
          // ParallelCombiningOpInterface ops.
          scf::InParallelOp terminator = innerForall.getTerminator();
          for (Operation &op :
               llvm::make_early_inc_range(*terminator.getBody())) {
            auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
            if (!insertOp || insertOp.getDest() != sharedOut) {
              continue;
            }

            // Insert the write_slice before the in_parallel terminator.
            rewriter.setInsertionPoint(terminator);
            IREE::PCF::WriteSliceOp::create(
                rewriter, insertOp.getLoc(), insertOp.getSource(), srefAlloc,
                insertOp.getMixedOffsets(), insertOp.getMixedSizes(),
                insertOp.getMixedStrides());
            rewriter.eraseOp(insertOp);
          }

          // Replace the forall's shared_out operand with a tensor.empty
          // so the forall stays valid in tensor land.
          rewriter.setInsertionPoint(innerForall);
          Value emptyTensor = tensor::EmptyOp::create(
              rewriter, innerForall.getLoc(), tensorType.getShape(),
              tensorType.getElementType());
          innerForall.getOutputsMutable()[outputIdx].assign(emptyTensor);
        }
      }

      // Phase B: Now safe to change the block arg type and barrier operand.
      barrierOp.setOperand(inputIdx, srefAlloc);
      blockArg.setType(srefType);

      // Update the yield op to yield the sref block arg.
      auto yieldOp = cast<IREE::GPU::YieldOp>(body.getTerminator());
      for (auto [idx, yieldOperand] : llvm::enumerate(yieldOp.getOperands())) {
        if (yieldOperand.getType() == tensorType) {
          yieldOp.setOperand(idx, blockArg);
        }
      }

      // Update barrier_region result types to sref and convert consumers.
      for (auto [idx, result] : llvm::enumerate(barrierOp.getResults())) {
        if (result.getType() == tensorType) {
          result.setType(srefType);
          convertSrefConsumerChain(rewriter, result, srefType, scope);
        }
      }
    }

    // Replace remaining uses of alloc_tensor with pcf.alloc.
    rewriter.replaceOp(allocOp, srefAlloc);
  }
}

struct GPUConvertThreadForallToSubgroupLanePCFPass final
    : impl::GPUConvertThreadForallToSubgroupLanePCFPassBase<
          GPUConvertThreadForallToSubgroupLanePCFPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Create nested scopes: subgroup (outer) + lane (inner).
    auto subgroupScope = cast<IREE::PCF::ScopeAttrInterface>(
        IREE::GPU::SubgroupScopeAttr::get(ctx));
    auto laneScope =
        cast<IREE::PCF::ScopeAttrInterface>(IREE::GPU::LaneScopeAttr::get(ctx));

    SmallVector<IREE::PCF::ScopeAttrInterface> scopes = {subgroupScope,
                                                         laneScope};

    IRRewriter rewriter(ctx);

    // Phase 1: Convert thread-mapped foralls to pcf.generic nests.
    SmallVector<scf::ForallOp> forallOps;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      if (hasThreadMapping(forallOp)) {
        forallOps.push_back(forallOp);
      }
    });

    SmallVector<IREE::PCF::GenericOp> createdGenerics;
    for (scf::ForallOp forallOp : forallOps) {
      rewriter.setInsertionPoint(forallOp);
      FailureOr<IREE::PCF::GenericOp> result =
          IREE::PCF::convertForallToGenericNest(rewriter, forallOp, scopes);
      if (failed(result)) {
        forallOp.emitError(
            "failed to convert thread forall to pcf.generic nest");
        return signalPassFailure();
      }
      createdGenerics.push_back(*result);
      rewriter.replaceOp(forallOp, result->getResults());
    }

    // Phase 2: Convert barrier_region chains to PCF ops.
    // Walk the whole function since alloc_tensors may live outside the
    // pcf.generic nests (e.g. at workgroup scope).
    convertBarrierChainToSref(rewriter, getOperation());
  }
};

} // namespace
} // namespace mlir::iree_compiler
