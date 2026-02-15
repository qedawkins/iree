// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- DistributeSharedExecutor.cpp - Distribute collective to per-thread -===//
//
// Converts the collective execution model of pcf.shared_executor into the
// per-thread execution model of pcf.generic. Runs iterative layout analysis
// using NestedLayoutAttr constraints, computes per-thread slices, and
// generates distributed code with thread IDs.
//
// Algorithm phases:
// 1. SEED: Collect explicit constraints (pcf.constrain_layout,
//    pcf.constrain_mma).
// 2. PROPAGATE: Iterative forward/backward constraint propagation to fixed
//    point.
// 3. FILL UNCONSTRAINED: Assign greedy coalesced defaults.
// 4. RESOLVE CONFLICTS: Insert pcf.redistribute where layouts conflict.
// 5. Generate distributed code: replace shared_executor with pcf.generic,
//    distribute collective ops to per-thread slices.
// 6. Handle multi-scope distribution via nested pcf.generic ops.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/DistributionAnalysis.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-pcf-distribute-shared-executor"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_DISTRIBUTESHAREDEXECUTORPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Phase 5: Generate Distributed Code
//===----------------------------------------------------------------------===//

/// Distribute a collective pcf.read_slice to a per-thread read with offsets
/// computed from the resolved layout and thread/subgroup IDs.
///
/// Input (collective):
///   %tile = pcf.read_slice %sref [0, %k] [128, 16] [1, 1]
///
/// Output (distributed):
///   %offset = <computed from layout + thread ID>
///   %my_tile = pcf.read_slice %sref [%offset, %k] [32, 16] [1, 1]
static LogicalResult distributeReadSlice(ReadSliceOp readOp, Attribute layout,
                                         Value threadId, Value subgroupId,
                                         int64_t subgroupSize,
                                         IRMapping &mapping,
                                         OpBuilder &builder) {
  auto layoutIface = dyn_cast<LayoutAttrInterface>(layout);
  if (!layoutIface) {
    return readOp.emitError("layout does not implement LayoutAttrInterface");
  }

  Location loc = readOp.getLoc();

  // Compute per-thread offsets from the layout.
  SmallVector<Value> threadOffsets = layoutIface.computeThreadSliceOffsets(
      builder, loc, threadId, subgroupId, subgroupSize);

  // FIXME: Compute distributed offsets and sizes from the layout.
  //
  // Original offsets: readOp.getMixedOffsets()
  // Distributed offsets: original + threadOffset per dimension.
  // Distributed sizes: derived from layout.getDistributedShape().
  //
  // Requires LayoutAttrInterface::getDistributedShape() implementation
  // to compute per-thread tile sizes from the layout decomposition.
  (void)threadOffsets;
  (void)mapping;
  return success();
}

/// Distribute a collective pcf.write_slice to a per-thread write with offsets
/// computed from the resolved layout and thread/subgroup IDs.
static LogicalResult
distributeWriteSlice(WriteSliceOp writeOp, Attribute layout, Value threadId,
                     Value subgroupId, int64_t subgroupSize, IRMapping &mapping,
                     OpBuilder &builder) {
  auto layoutIface = dyn_cast<LayoutAttrInterface>(layout);
  if (!layoutIface) {
    return writeOp.emitError("layout does not implement LayoutAttrInterface");
  }

  Location loc = writeOp.getLoc();

  SmallVector<Value> threadOffsets = layoutIface.computeThreadSliceOffsets(
      builder, loc, threadId, subgroupId, subgroupSize);

  // FIXME: Compute distributed offsets and sizes analogously to read_slice.
  // Requires LayoutAttrInterface::getDistributedShape() implementation.
  (void)threadOffsets;
  (void)mapping;
  return success();
}

/// Distribute operations within a form_bundles region by generating
/// conditional execution (scf.if / scf.switch) based on thread IDs.
///
/// For 2 bundles: scf.if %is_in_bundle_0 { ... } else { ... }
/// For >2 bundles: scf.switch %bundle_idx { case 0 { ... } ... }
static LogicalResult distributeBundles(FormBundlesOp formBundlesOp,
                                       LayoutConstraintInfo &info,
                                       Value threadId, Value numWorkers,
                                       OpBuilder &builder) {
  // FIXME: Bundle distribution is not yet implemented. The form_bundles op
  // should be lowered to scf.if/scf.switch conditional execution based on
  // thread IDs. Implementation requires:
  // 1. Compute bundle membership predicates from sizes and thread ID.
  //    For 2 bundles with sizes [s0, s1]:
  //      %is_bundle_0 = arith.cmpi ult, %threadId, %s0
  //    For >2 bundles: compute %bundle_idx = <piecewise from sizes>.
  // 2. For each execute_as, generate conditional code:
  //    2a. scf.if for 2 bundles.
  //    2b. scf.switch for >2 bundles.
  // 3. Compute local IDs within each bundle:
  //    %local_id = arith.subi %threadId, %bundle_start_offset
  //    %local_count = %bundle_size
  // 4. Distribute ops within each execute_as using local IDs and counts.
  (void)info;
  (void)threadId;
  (void)numWorkers;
  (void)builder;
  return formBundlesOp.emitError(
      "form_bundles distribution is not yet implemented");
}

/// Replace a shared_executor with a pcf.generic, distributing its contents.
///
/// Key transformations:
/// 1. shared_executor -> pcf.generic with scope and thread ID args.
/// 2. All sref args (captures and inits) become pcf.generic tied ref args.
///    NOTE: The design doc describes captures as values referenced from the
///    enclosing scope (since pcf.generic is NOT IsolatedFromAbove). However,
///    we pass them as tied refs to preserve sref typing for inner ops like
///    read_slice/write_slice. The generic produces dead results for capture
///    positions, which are not used by any consumer. This trade-off avoids
///    needing sref-to-tensor conversions at the generic boundary.
/// 3. Thread IDs introduced as new block arguments.
/// 4. Collective ops cloned into generic body with remapped operands.
///    Per-thread distribution of offsets/sizes is handled by
///    distributeReadSlice/distributeWriteSlice.
/// 5. Layout constraint ops consumed and removed.
/// 6. shared_executor replaced with generic results (non-capture only).
///
/// Returns the created GenericOp on success so callers can use it for
/// multi-scope distribution.
static FailureOr<GenericOp>
generateDistributedCode(RewriterBase &rewriter, SharedExecutorOp sharedExecOp,
                        LayoutConstraintInfo &info) {
  Location loc = sharedExecOp.getLoc();

  // Extract scope and count dimensions.
  ArrayAttr scopes = sharedExecOp.getScopesAttr();
  if (scopes.empty()) {
    return rewriter.notifyMatchFailure(sharedExecOp, "no scopes");
  }
  ScopeAttrInterface scope = cast<ScopeAttrInterface>(scopes[0]);
  ArrayRef<int64_t> countDimsPerScope = sharedExecOp.getCountDimsPerScope();
  int64_t numIterators = countDimsPerScope.empty() ? 1 : countDimsPerScope[0];

  // Build generic inits and result types. All shared_executor refs (both
  // captures and readwrite inits) become tied refs in the generic. This
  // preserves sref typing so cloned ops remain valid.
  ArrayRef<bool> isCapture = sharedExecOp.getIsCapture();
  int64_t numRefs = sharedExecOp.getNumRefArgs();
  SmallVector<Value> genericInits;
  SmallVector<Type> genericResultTypes;
  genericInits.reserve(numRefs);
  genericResultTypes.reserve(numRefs);

  int64_t captureIdx = 0;
  int64_t initIdx = 0;
  for (bool isCap : isCapture) {
    Value initVal;
    if (isCap) {
      initVal = sharedExecOp.getCaptures()[captureIdx++];
    } else {
      initVal = sharedExecOp.getInits()[initIdx++];
    }
    genericInits.push_back(initVal);
    genericResultTypes.push_back(initVal.getType());
  }

  // All refs are tied (each has a corresponding init value).
  SmallVector<bool> isTied(numRefs, true);

  // Create the generic op.
  rewriter.setInsertionPoint(sharedExecOp);
  GenericOp genericOp = rewriter.create<GenericOp>(
      loc, genericResultTypes, scope, genericInits,
      /*dynamicSizes=*/ValueRange{}, isTied, numIterators,
      // No implicit barrier at generic return. The shared_executor's
      // barrier semantics are explicit via pcf.barrier ops in the body.
      /*syncOnReturn=*/false);

  // The build method already created the block with sref args + index args.
  Block &genericBlock = genericOp.getRegion().front();

  // Set up block argument mapping from shared_executor to generic.
  //
  // shared_executor block args:
  //   [leading_args | ref_args | count_scope0 | count_scope1 | ...]
  // generic block args:
  //   [ref_args | id_args | count_args]
  //
  // FIXME: Leading args (from the initializer region) are not mapped.
  // After replaceOp, any uses of leading args will become dangling
  // references, causing crashes or verification failures. This is safe
  // only while the initializer region is unimplemented. Once initializer
  // support lands, leading args must be mapped to appropriate values
  // (e.g., hoisted above the generic or passed as additional inits).
  IRMapping mapping;

  Region &seBody = sharedExecOp.getBody();
  Block &seBlock = seBody.front();
  int64_t numLeading = sharedExecOp.getNumLeadingArgs();

  // Map ref block args (captures + inits) 1:1.
  for (int64_t i = 0; i < numRefs; ++i) {
    BlockArgument seRefArg = seBlock.getArgument(numLeading + i);
    BlockArgument genRefArg = genericBlock.getArgument(i);
    mapping.map(seRefArg, genRefArg);
  }

  // Map count block args. In shared_executor, count args follow ref args.
  // In generic, count args follow id args. We only map the outermost
  // scope's count dims (inner scopes handled by Phase 6).
  int64_t seCountStart = numLeading + numRefs;
  int64_t genIdStart = numRefs;
  int64_t genCountStart = numRefs + numIterators;
  for (int64_t i = 0; i < numIterators; ++i) {
    if (seCountStart + i < static_cast<int64_t>(seBlock.getNumArguments())) {
      BlockArgument seCountArg = seBlock.getArgument(seCountStart + i);
      BlockArgument genCountArg = genericBlock.getArgument(genCountStart + i);
      mapping.map(seCountArg, genCountArg);
    }
  }

  // The generic's ID args are new (no shared_executor equivalent).
  // They are available at indices [genIdStart, genIdStart + numIterators).

  LLVM_DEBUG({
    llvm::dbgs() << "  Block arg mapping:\n";
    llvm::dbgs() << "    " << numRefs << " ref args mapped\n";
    llvm::dbgs() << "    " << numIterators << " count args mapped\n";
    llvm::dbgs() << "    " << numIterators << " new ID args at indices ["
                 << genIdStart << ", " << genIdStart + numIterators << ")\n";
  });

  // Clone shared_executor body ops into generic body.
  // Skip constraint ops (consumed by analysis) and the terminator.
  rewriter.setInsertionPointToEnd(&genericBlock);

  for (Operation &op : seBlock.without_terminator()) {
    // Skip layout constraint ops — they were consumed by the analysis.
    if (isa<ConstrainLayoutOp>(&op)) {
      LLVM_DEBUG(llvm::dbgs() << "  Skipping constrain_layout: " << op << "\n");
      continue;
    }
    if (isa<ConstrainMmaOp>(&op)) {
      LLVM_DEBUG(llvm::dbgs() << "  Skipping constrain_mma: " << op << "\n");
      continue;
    }

    // Handle form_bundles via distributeBundles().
    if (auto formBundles = dyn_cast<FormBundlesOp>(&op)) {
      // Get the outermost scope's ID and count for bundle distribution.
      Value threadId = genericBlock.getArgument(genIdStart);
      Value numWorkers = genericBlock.getArgument(genCountStart);
      if (failed(distributeBundles(formBundles, info, threadId, numWorkers,
                                   rewriter))) {
        return failure();
      }
      continue;
    }

    // FIXME: read_slice and write_slice should be distributed to per-thread
    // slices using the resolved layout and thread IDs. Currently they are
    // cloned as-is (collective semantics preserved). This must be updated
    // for correctness before the pass is usable end-to-end.
    // See distributeReadSlice() and distributeWriteSlice().

    // Clone the op with remapped operands.
    rewriter.clone(op, mapping);
  }

  // Add the terminator.
  rewriter.create<ReturnOp>(loc);

  // Replace shared_executor results with the corresponding generic results.
  // shared_executor produces results only for non-capture (readwrite) refs.
  // In our generic, captures are at the front and inits follow, matching
  // the isCapture order.
  SmallVector<Value> replacements;
  for (auto [i, isCap] : llvm::enumerate(isCapture)) {
    if (!isCap) {
      replacements.push_back(genericOp.getResult(i));
    }
  }

  if (replacements.size() != sharedExecOp.getNumResults()) {
    return sharedExecOp.emitError("result count mismatch: expected ")
           << sharedExecOp.getNumResults() << " but got "
           << replacements.size();
  }

  rewriter.replaceOp(sharedExecOp, replacements);

  LLVM_DEBUG(llvm::dbgs() << "  Replaced shared_executor with generic ("
                          << genericResultTypes.size() << " refs, "
                          << numIterators << " iterators).\n");

  return genericOp;
}

//===----------------------------------------------------------------------===//
// Phase 6: Multi-Scope Distribution
//===----------------------------------------------------------------------===//

/// Handle multi-scope distribution by creating nested pcf.generic ops.
///
/// Strategy A (preferred): Nested pcf.generic, one per scope level.
///   The outer generic handles subgroup-level distribution, and an inner
///   generic handles lane-level distribution within each subgroup.
/// Strategy B: Single pcf.generic with manual sub-scope ID computation.
///   A single generic handles all scopes, computing inner-scope IDs as
///   functions of the outer-scope ID.
///
/// This function is called AFTER generateDistributedCode has already created
/// the outer pcf.generic for the outermost scope. It adds inner generics
/// for any additional scopes.
static LogicalResult handleMultiScopeDistribution(
    RewriterBase &rewriter, GenericOp outerGenericOp, ArrayAttr allScopes,
    ArrayRef<int64_t> countDimsPerScope, bool preferNestedGeneric) {
  // TODO: Walk the outer generic's body and create inner pcf.generic ops
  // for each additional scope level. For each inner scope:
  //   1. Get the scope attribute and count dimensions.
  //   2. If preferNestedGeneric: create a nested pcf.generic with the inner
  //      scope, wrapping ops that need lane-level distribution.
  //   3. Otherwise: compute inner-scope IDs from outer-scope IDs using
  //      delinearization and use them for distribution directly.
  //
  // For now, only single-scope distribution is fully supported. Multi-scope
  // shared_executors will be distributed at the outermost scope only.
  if (allScopes.size() <= 1) {
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "  Multi-scope distribution: " << allScopes.size()
                          << " scopes. "
                          << (preferNestedGeneric ? "Nested" : "Single")
                          << " generic strategy.\n");

  // FIXME: Implement multi-scope distribution. Currently only the
  // outermost scope is distributed. Inner scopes need either nested
  // pcf.generic ops (Strategy A) or manual ID computation (Strategy B).
  (void)rewriter;
  (void)outerGenericOp;
  (void)countDimsPerScope;
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct DistributeSharedExecutorPass final
    : impl::DistributeSharedExecutorPassBase<DistributeSharedExecutorPass> {
  using Base::Base;

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "=== DistributeSharedExecutor ===\n");
    LLVM_DEBUG(llvm::dbgs()
               << "  preferNestedGeneric: " << preferNestedGeneric << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "  strictLayoutChecking: " << strictLayoutChecking << "\n");

    IRRewriter rewriter(getOperation()->getContext());
    bool hadFailure = false;

    getOperation()->walk([&](SharedExecutorOp sharedExecOp) {
      if (hadFailure) {
        return WalkResult::interrupt();
      }

      LLVM_DEBUG(llvm::dbgs() << "Processing shared_executor at "
                              << sharedExecOp.getLoc() << "\n");

      Region &body = sharedExecOp.getBody();
      if (body.empty()) {
        return WalkResult::advance();
      }

      // Extract scope info for worker count estimation.
      ArrayAttr scopes = sharedExecOp.getScopesAttr();
      if (scopes.empty()) {
        return WalkResult::advance();
      }

      // Estimate worker counts for layout analysis. If we have multiple
      // scopes, the outer scope determines subgroup count and the inner
      // scope determines thread count.
      // FIXME: ScopeAttrInterface only provides runtime worker counts
      // via getWorkerCounts(). We need a getStaticWorkerCounts() method
      // or pass options to provide static estimates. Using hardcoded
      // defaults until that infrastructure is available.
      int64_t numScopes = scopes.size();
      int64_t numThreads = 64;  // Typical subgroup size.
      int64_t numSubgroups = 4; // Typical subgroup count.

      LLVM_DEBUG(llvm::dbgs() << "  numThreads: " << numThreads
                              << ", numSubgroups: " << numSubgroups << "\n");

      // 1. SEED: Collect explicit constraints from pcf.constrain_layout
      //    and pcf.constrain_mma ops.
      LayoutConstraintInfo info;
      seedConstraints(body, info);

      LLVM_DEBUG(llvm::dbgs()
                 << "  Seeded " << info.layouts.size() << " constraints.\n");

      // 2. PROPAGATE: Iterative forward/backward propagation to fixed point.
      propagateToFixedPoint(info);

      LLVM_DEBUG(llvm::dbgs() << "  After propagation: " << info.layouts.size()
                              << " values constrained.\n");

      // 3. FILL UNCONSTRAINED: Assign greedy coalesced defaults.
      fillUnconstrained(body, info, numThreads, numSubgroups);

      // 4. RESOLVE CONFLICTS: Insert pcf.redistribute where needed.
      OpBuilder builder(getOperation()->getContext());
      builder.setInsertionPointToStart(&body.front());
      if (failed(resolveConflicts(body, info, strictLayoutChecking, builder))) {
        hadFailure = true;
        return WalkResult::interrupt();
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "  Layout analysis complete: " << info.layouts.size()
                 << " values, " << info.conflicts.size()
                 << " resolved conflicts.\n");

      // Save scope info before generateDistributedCode erases sharedExecOp.
      ArrayRef<int64_t> countDimsPerScope = sharedExecOp.getCountDimsPerScope();
      SmallVector<int64_t> savedCountDims(countDimsPerScope);

      // 5. Generate distributed code: replace shared_executor with
      //    pcf.generic with per-thread slices. This erases sharedExecOp.
      FailureOr<GenericOp> genericOpOrFailure =
          generateDistributedCode(rewriter, sharedExecOp, info);
      if (failed(genericOpOrFailure)) {
        hadFailure = true;
        return WalkResult::interrupt();
      }

      // 6. Handle multi-scope distribution if needed. sharedExecOp is
      //    erased; we use the returned GenericOp for nested distribution.
      if (numScopes > 1) {
        if (failed(handleMultiScopeDistribution(rewriter, *genericOpOrFailure,
                                                scopes, savedCountDims,
                                                preferNestedGeneric))) {
          hadFailure = true;
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    if (hadFailure) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
