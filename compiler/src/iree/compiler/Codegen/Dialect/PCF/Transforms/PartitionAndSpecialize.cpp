// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- PartitionAndSpecialize.cpp - Partition workers and specialize ------===//
//
// Transforms a pcf.shared_executor region containing collective operations
// into one containing pcf.form_bundles with pcf.execute_as regions.
//
// Fuses the original Stages 4 (PartitionIntoBundles) and 5
// (SpecializeSharedExecutor) because MLIR attributes cannot reference SSA
// values, making a separate annotation intermediate form impossible.
//
// Algorithm:
// 1. Phase identification: partition ops by barrier boundaries.
// 2. Operation classification: classify each op as load/compute/shared.
// 3. Partitioning decision: assign ops to bundles based on strategy.
// 4. Dependency verification: check no cross-bundle SSA deps.
// 5. Transformation: create form_bundles + execute_as structure.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-pcf-partition-and-specialize"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_PARTITIONANDSPECIALIZEPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Phase Analysis
//===----------------------------------------------------------------------===//

/// Classification of an operation within a shared executor region.
enum class OpClassification {
  /// Load operations: reads from global/shared memory (pcf.read_slice from
  /// readonly or shared srefs, pcf.get_memref).
  Load,
  /// Store operations: writes to global/shared memory (pcf.write_slice into
  /// shared srefs).
  Store,
  /// Compute operations: arithmetic, contractions, elementwise (linalg ops,
  /// arith ops, etc.).
  Compute,
  /// Shared operations executed by all bundles (barriers, fences, allocs).
  Shared,
  /// Control flow (scf.for, scf.if, scf.yield).
  ControlFlow,
};

/// Represents a phase within a shared executor region. Phases are separated
/// by barriers. All operations within a phase execute concurrently.
struct PhaseInfo {
  /// Index of this phase in the sequence of phases.
  unsigned index = 0;

  /// Operations belonging to this phase, in program order.
  SmallVector<Operation *> ops;

  /// Cached result of classifyOp() for each op in this phase.
  DenseMap<Operation *, OpClassification> classifications;
};

/// Represents a bundle of workers that execute a subset of operations.
struct BundleInfo {
  /// Unique identifier for this bundle (0-indexed).
  unsigned id = 0;

  /// Number of workers in this bundle.
  int64_t size = 0;

  /// Operations assigned to this bundle, per phase.
  /// Maps phase index -> list of operations for this bundle.
  DenseMap<unsigned, SmallVector<Operation *>> opsByPhase;
};

//===----------------------------------------------------------------------===//
// Phase Identification
//===----------------------------------------------------------------------===//

/// Partition operations in a region by barrier boundaries into phases.
/// Each phase contains all operations between consecutive barriers.
/// Barriers themselves are NOT included in any phase; they form the
/// boundaries.
static SmallVector<PhaseInfo> identifyPhases(Region &region) {
  SmallVector<PhaseInfo> phases;
  if (region.empty()) {
    return phases;
  }

  PhaseInfo currentPhase;
  currentPhase.index = 0;

  for (Operation &op : region.front().getOperations()) {
    // Barriers end the current phase and start a new one.
    if (isa<BarrierOp>(&op)) {
      if (!currentPhase.ops.empty()) {
        phases.push_back(std::move(currentPhase));
      }
      currentPhase = PhaseInfo{};
      currentPhase.index = phases.size();
      continue;
    }

    // Skip terminators (pcf.return, pcf.yield).
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }

    currentPhase.ops.push_back(&op);
  }

  // Push the last phase if non-empty.
  if (!currentPhase.ops.empty()) {
    phases.push_back(std::move(currentPhase));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Identified " << phases.size() << " phases:\n";
    for (const PhaseInfo &phase : phases) {
      llvm::dbgs() << "  Phase " << phase.index << ": " << phase.ops.size()
                   << " ops\n";
      for (Operation *op : phase.ops) {
        llvm::dbgs() << "    " << op->getName() << "\n";
      }
    }
  });

  return phases;
}

//===----------------------------------------------------------------------===//
// Operation Classification
//===----------------------------------------------------------------------===//

/// Classify an operation as load, store, compute, shared, or control flow.
static OpClassification classifyOp(Operation *op) {
  // PCF-specific ops.
  if (isa<ReadSliceOp>(op)) {
    return OpClassification::Load;
  }
  if (isa<WriteSliceOp>(op)) {
    return OpClassification::Store;
  }
  if (isa<GetMemrefOp>(op)) {
    return OpClassification::Load;
  }
  if (isa<AllocOp>(op)) {
    return OpClassification::Shared;
  }
  if (isa<BarrierOp, FenceOp>(op)) {
    return OpClassification::Shared;
  }

  // Control flow.
  if (isa<scf::ForOp, scf::IfOp, scf::YieldOp>(op)) {
    return OpClassification::ControlFlow;
  }

  // Linalg ops are compute.
  if (isa<linalg::LinalgOp>(op)) {
    return OpClassification::Compute;
  }

  // Arith ops are compute.
  if (op->getDialect() && op->getDialect()->getNamespace() == "arith") {
    return OpClassification::Compute;
  }

  // Default: treat as compute.
  return OpClassification::Compute;
}

/// Classify all ops in each phase and cache the results.
static void classifyAllOps(SmallVectorImpl<PhaseInfo> &phases) {
  for (PhaseInfo &phase : phases) {
    for (Operation *op : phase.ops) {
      phase.classifications[op] = classifyOp(op);
    }
  }

  LLVM_DEBUG({
    for (const PhaseInfo &phase : phases) {
      unsigned loads = 0, stores = 0, computes = 0, shared = 0, cf = 0;
      for (Operation *op : phase.ops) {
        auto it = phase.classifications.find(op);
        if (it == phase.classifications.end()) {
          continue;
        }
        switch (it->second) {
        case OpClassification::Load:
          ++loads;
          break;
        case OpClassification::Store:
          ++stores;
          break;
        case OpClassification::Compute:
          ++computes;
          break;
        case OpClassification::Shared:
          ++shared;
          break;
        case OpClassification::ControlFlow:
          ++cf;
          break;
        }
      }
      llvm::dbgs() << "Phase " << phase.index << " classification: " << loads
                   << " loads, " << stores << " stores, " << computes
                   << " computes, " << shared << " shared, " << cf
                   << " control flow\n";
    }
  });
}

//===----------------------------------------------------------------------===//
// Partitioning Strategies
//===----------------------------------------------------------------------===//

/// Parse the producer-consumer ratio string (e.g., "1:3").
/// Returns failure if the string is not "auto" or a valid "N:M" format.
static FailureOr<std::pair<int64_t, int64_t>> parseRatio(StringRef ratioStr) {
  if (ratioStr == "auto") {
    return std::pair<int64_t, int64_t>{0, 0};
  }
  auto [lhsStr, rhsStr] = ratioStr.split(':');
  int64_t lhs = 0, rhs = 0;
  if (lhsStr.getAsInteger(10, lhs) || rhsStr.getAsInteger(10, rhs)) {
    return failure();
  }
  if (lhs <= 0 || rhs <= 0) {
    return failure();
  }
  return std::pair<int64_t, int64_t>{lhs, rhs};
}

/// Partition operations into bundles using the producer-consumer strategy.
///
/// Producer bundle: handles load and store-to-shared ops (global->shared
/// memory transfers). Consumer bundle: handles compute and store-to-global
/// ops (computation and result writeback).
///
/// Within each phase:
///   - Loads from readonly srefs + writes to shared srefs -> producer.
///   - Reads from shared srefs + compute + writes to readwrite srefs ->
///     consumer.
///   - Shared ops (allocs, barriers) stay at the form_bundles level.
///   - Control flow is replicated or lifted depending on nesting.
static SmallVector<BundleInfo>
partitionProducerConsumer(ArrayRef<PhaseInfo> phases,
                          std::pair<int64_t, int64_t> ratio,
                          int64_t totalWorkers) {
  SmallVector<BundleInfo> bundles(2);

  // Determine producer/consumer split.
  int64_t prodSize = ratio.first;
  int64_t consSize = ratio.second;
  if (prodSize == 0 && consSize == 0) {
    // Auto: split evenly.
    prodSize = totalWorkers / 2;
    consSize = totalWorkers - prodSize;
  }

  bundles[0].id = 0;
  bundles[0].size = prodSize;
  bundles[1].id = 1;
  bundles[1].size = consSize;

  for (const PhaseInfo &phase : phases) {
    // Iterate phase.ops (deterministic program order) rather than
    // phase.classifications (DenseMap, non-deterministic).
    for (Operation *op : phase.ops) {
      auto it = phase.classifications.find(op);
      if (it == phase.classifications.end()) {
        continue;
      }
      OpClassification cls = it->second;
      switch (cls) {
      case OpClassification::Load:
      case OpClassification::Store:
        // Classify stores more precisely: writes to shared srefs go to
        // producer, writes to readwrite global srefs go to consumer.
        // For now, all loads/stores go to producer.
        // TODO: Distinguish shared vs global sref targets using the
        // sref accessor mode from Pair A.
        bundles[0].opsByPhase[phase.index].push_back(op);
        break;
      case OpClassification::Compute:
        bundles[1].opsByPhase[phase.index].push_back(op);
        break;
      case OpClassification::Shared:
      case OpClassification::ControlFlow:
        // Shared ops and control flow stay at the form_bundles level.
        // They are NOT assigned to any bundle.
        break;
      }
    }
  }

  LLVM_DEBUG({
    for (const BundleInfo &bundle : bundles) {
      llvm::dbgs() << "Bundle " << bundle.id << " (size=" << bundle.size
                   << "): ";
      unsigned total = 0;
      for (auto &[phaseIdx, ops] : bundle.opsByPhase) {
        total += ops.size();
      }
      llvm::dbgs() << total << " ops\n";
    }
  });

  return bundles;
}

/// Partition operations using the pingpong strategy for double-buffered
/// pipeline-parallel execution.
///
/// Two bundles (A and B) alternate roles between pipeline stages:
///   - Prologue: A loads.
///   - Main loop, half 1: A computes + B loads.
///   - Main loop, half 2: B computes + A loads.
///   - Epilogue: B computes.
///
/// Requires a pre-pipelined loop structure (prologue, main, epilogue).
static SmallVector<BundleInfo>
partitionPingpong(ArrayRef<PhaseInfo> phases, std::pair<int64_t, int64_t> ratio,
                  int64_t totalWorkers) {
  SmallVector<BundleInfo> bundles(2);

  int64_t halfWorkers = totalWorkers / 2;
  bundles[0].id = 0;
  bundles[0].size = halfWorkers;
  bundles[1].id = 1;
  bundles[1].size = totalWorkers - halfWorkers;

  // TODO: Detect prologue/main/epilogue structure from phases.
  // The scheduling pass (Stage 3, future) produces this structure.
  //
  // For now, fall back to producer-consumer assignment.
  for (const PhaseInfo &phase : phases) {
    // Iterate phase.ops for deterministic order.
    for (Operation *op : phase.ops) {
      auto it = phase.classifications.find(op);
      if (it == phase.classifications.end()) {
        continue;
      }
      switch (it->second) {
      case OpClassification::Load:
      case OpClassification::Store:
        bundles[0].opsByPhase[phase.index].push_back(op);
        break;
      case OpClassification::Compute:
        bundles[1].opsByPhase[phase.index].push_back(op);
        break;
      default:
        break;
      }
    }
  }

  return bundles;
}

//===----------------------------------------------------------------------===//
// Dependency Analysis
//===----------------------------------------------------------------------===//

/// Verify that there are no cross-bundle SSA dependencies within a phase.
/// Cross-bundle communication must happen through shared memory (srefs).
///
/// Within a single phase (between two barriers), ops assigned to different
/// bundles must not have def-use chains between them. Such chains indicate
/// that the partitioning is invalid and the ops cannot be independently
/// parallelized.
static LogicalResult verifyNoCrossBundleDeps(ArrayRef<BundleInfo> bundles,
                                             ArrayRef<PhaseInfo> phases) {
  // Build a map: operation -> bundle ID.
  DenseMap<Operation *, unsigned> opToBundle;
  for (const BundleInfo &bundle : bundles) {
    for (auto &[phaseIdx, ops] : bundle.opsByPhase) {
      for (Operation *op : ops) {
        opToBundle[op] = bundle.id;
      }
    }
  }

  // Check each defined value: if it is defined in bundle X, all users
  // within the same phase must also be in bundle X (or unassigned).
  for (auto &[op, bundleId] : opToBundle) {
    for (Value result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();
        auto userIt = opToBundle.find(user);
        if (userIt == opToBundle.end()) {
          continue; // User not assigned to a bundle (shared/control flow).
        }
        if (userIt->second != bundleId) {
          return op->emitError("cross-bundle SSA dependency: value defined in "
                               "bundle ")
                 << bundleId << " used by op in bundle " << userIt->second
                 << ". Cross-bundle communication must use shared memory "
                    "(pcf.sref).";
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "No cross-bundle SSA dependencies found.\n");
  return success();
}

//===----------------------------------------------------------------------===//
// Transformation
//===----------------------------------------------------------------------===//

/// Create the form_bundles and execute_as IR structure from analysis results.
///
/// This is the core transformation that builds the specialized IR:
/// 1. Create pcf.form_bundles with scope and sizes from bundle analysis.
/// 2. For each phase, create pcf.execute_as regions per bundle.
/// 3. Move ops into appropriate execute_as regions.
/// 4. Keep barriers, allocs, and loop structure at the form_bundles level.
static LogicalResult applyPartitioning(RewriterBase &rewriter,
                                       SharedExecutorOp sharedExecOp,
                                       ArrayRef<BundleInfo> bundles,
                                       ArrayRef<PhaseInfo> phases) {
  Location loc = sharedExecOp.getLoc();

  // Get the scope for form_bundles. Use the outermost scope.
  ArrayAttr scopes = sharedExecOp.getScopesAttr();
  if (scopes.empty()) {
    return sharedExecOp.emitError("shared_executor has no scopes");
  }
  Attribute scope = scopes[0];

  // Build sizes array from bundles.
  SmallVector<int64_t> sizes;
  sizes.reserve(bundles.size());
  for (const BundleInfo &bundle : bundles) {
    sizes.push_back(bundle.size);
  }

  // Set insertion point inside the shared_executor body, before the
  // terminator. The form_bundles will be inserted into the body.
  Region &body = sharedExecOp.getBody();
  Block &entryBlock = body.front();
  Operation *terminator = entryBlock.getTerminator();
  rewriter.setInsertionPoint(terminator);

  // Create form_bundles op with the computed scope and sizes.
  auto formBundlesOp = rewriter.create<FormBundlesOp>(
      loc, scope, rewriter.getDenseI64ArrayAttr(sizes));

  // Create the entry block with bundle-typed arguments.
  Block *fbBlock = rewriter.createBlock(&formBundlesOp.getBody());
  for (int64_t i = 0, e = bundles.size(); i < e; ++i) {
    BundleType bundleType = BundleType::get(rewriter.getContext(),
                                            cast<ScopeAttrInterface>(scope), i);
    fbBlock->addArgument(bundleType, loc);
  }

  // For each phase, create execute_as regions for bundles that have ops
  // in that phase.
  rewriter.setInsertionPointToEnd(fbBlock);

  for (const PhaseInfo &phase : phases) {
    for (const BundleInfo &bundle : bundles) {
      auto it = bundle.opsByPhase.find(phase.index);
      if (it == bundle.opsByPhase.end() || it->second.empty()) {
        continue;
      }

      // Create execute_as with this bundle's block argument.
      Value bundleArg = fbBlock->getArgument(bundle.id);
      auto executeAsOp =
          rewriter.create<ExecuteAsOp>(loc, ValueRange{bundleArg});

      // Create the body block.
      Block *eaBlock = rewriter.createBlock(&executeAsOp.getBody());
      rewriter.setInsertionPointToEnd(eaBlock);

      // Clone ops into the execute_as body. The ops still reference
      // the original SSA values from the shared_executor region, which
      // is fine since execute_as is NOT IsolatedFromAbove.
      IRMapping mapping;
      for (Operation *op : it->second) {
        rewriter.clone(*op, mapping);
      }

      // Add terminator.
      rewriter.create<ReturnOp>(loc);

      // Reset insertion point to form_bundles body for next execute_as.
      rewriter.setInsertionPointToEnd(fbBlock);
    }

    // Insert barrier between phases (except after the last phase).
    if (phase.index + 1 < phases.size()) {
      auto scopeIface = dyn_cast<ScopeAttrInterface>(scope);
      if (scopeIface) {
        if (failed(scopeIface.addBarrier(rewriter))) {
          return rewriter.notifyMatchFailure(
              sharedExecOp, "failed to create inter-phase barrier");
        }
      }
    }
  }

  // Add yield terminator to form_bundles.
  rewriter.create<YieldOp>(loc);

  // FIXME: Erase original ops that were cloned into execute_as regions.
  // Side-effecting ops (read_slice, write_slice, barrier) will NOT be
  // removed by DCE, so they will execute twice — once from the originals
  // in the shared_executor body and once from clones inside execute_as.
  // Erasing requires distinguishing ops that were fully moved into
  // bundles from ops that remain shared (e.g., allocs used across
  // multiple bundles).
  //
  // Until erasure is implemented, the transformation produces incorrect
  // IR (duplicated side effects). Guard against silent miscompilation.
  return sharedExecOp.emitError(
      "partitioning transformation is not yet complete: original ops are "
      "not erased after cloning into execute_as regions, causing duplicated "
      "side effects. Implement op erasure to enable this pass.");
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct PartitionAndSpecializePass final
    : impl::PartitionAndSpecializePassBase<PartitionAndSpecializePass> {
  using Base::Base;

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "=== PartitionAndSpecialize ===\n");
    LLVM_DEBUG(llvm::dbgs()
               << "  partitioningStrategy: " << partitioningStrategy << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "  producerConsumerRatio: " << producerConsumerRatio << "\n");

    // Early exit for no-op strategy.
    if (partitioningStrategy == "none") {
      return;
    }

    // Parse options.
    FailureOr<std::pair<int64_t, int64_t>> ratio =
        parseRatio(producerConsumerRatio);
    if (failed(ratio)) {
      emitError(getOperation()->getLoc())
          << "invalid producer-consumer-ratio: '" << producerConsumerRatio
          << "', expected 'auto' or 'N:M'";
      return signalPassFailure();
    }

    IRRewriter rewriter(getOperation()->getContext());
    bool hadFailure = false;

    getOperation()->walk([&](SharedExecutorOp sharedExecOp) {
      if (hadFailure) {
        return WalkResult::interrupt();
      }

      LLVM_DEBUG(llvm::dbgs() << "Processing shared_executor at "
                              << sharedExecOp.getLoc() << "\n");

      // 1. Extract the region and scope information.
      Region &body = sharedExecOp.getBody();
      if (body.empty()) {
        return WalkResult::advance();
      }

      // Estimate total worker count for partitioning decisions.
      // FIXME: ScopeAttrInterface only provides runtime worker counts
      // via getWorkerCounts(). We need a getStaticWorkerCounts() method
      // or pass options to provide static estimates. Using hardcoded
      // defaults until that infrastructure is available.
      ArrayAttr scopes = sharedExecOp.getScopesAttr();
      if (scopes.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "  No scopes, skipping.\n");
        return WalkResult::advance();
      }
      int64_t totalWorkers = 4;

      LLVM_DEBUG(llvm::dbgs() << "  totalWorkers: " << totalWorkers << "\n");

      // 2. Phase identification.
      SmallVector<PhaseInfo> phases = identifyPhases(body);
      classifyAllOps(phases);

      if (phases.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "  No phases found, skipping.\n");
        return WalkResult::advance();
      }

      // 3. Partitioning decision.
      SmallVector<BundleInfo> bundles;
      if (partitioningStrategy == "producer-consumer" ||
          partitioningStrategy == "auto") {
        bundles = partitionProducerConsumer(phases, *ratio, totalWorkers);
      } else if (partitioningStrategy == "pingpong") {
        bundles = partitionPingpong(phases, *ratio, totalWorkers);
      } else {
        sharedExecOp.emitError("unknown partitioning strategy: '")
            << partitioningStrategy << "'";
        hadFailure = true;
        return WalkResult::interrupt();
      }

      // 4. Dependency verification.
      if (failed(verifyNoCrossBundleDeps(bundles, phases))) {
        hadFailure = true;
        return WalkResult::interrupt();
      }

      // 5. Transformation: create form_bundles + execute_as structure.
      if (failed(applyPartitioning(rewriter, sharedExecOp, bundles, phases))) {
        hadFailure = true;
        return WalkResult::interrupt();
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
