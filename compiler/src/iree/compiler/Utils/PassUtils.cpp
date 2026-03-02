// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/PassUtils.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/Threading.h"

#define DEBUG_TYPE "iree-pass-utils"

namespace mlir::iree_compiler {

void signalFixedPointModified(Operation *rootOp) {
  MLIRContext *context = rootOp->getContext();
  if (!rootOp->hasAttr("iree.fixedpoint.iteration")) {
    LLVM_DEBUG(llvm::dbgs() << "Not signaling fixed-point modification: not "
                               "running under fixed point iterator");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Signalling fixed-point iterator modification");
  rootOp->setAttr("iree.fixedpoint.modified", UnitAttr::get(context));
}

//===----------------------------------------------------------------------===//
// OpPipelineAdaptorPass
//===----------------------------------------------------------------------===//

namespace detail {

OpPipelineAdaptorPass::OpPipelineAdaptorPass(SmallVector<Entry> entries)
    : entries(std::move(entries)) {}

OpPipelineAdaptorPass::OpPipelineAdaptorPass(
    const OpPipelineAdaptorPass &other)
    : PassWrapper(other) {
  entries.reserve(other.entries.size());
  for (const Entry &e : other.entries) {
    entries.emplace_back(e.condition, OpPassManager(e.pipeline));
  }
}

void OpPipelineAdaptorPass::getDependentDialects(
    DialectRegistry &registry) const {
  for (const Entry &e : entries) {
    e.pipeline.getDependentDialects(registry);
  }
}

void OpPipelineAdaptorPass::runOnOperation() {
  if (getContext().isMultithreadingEnabled()) {
    runOnOperationAsync();
  } else {
    runOnOperationSync();
  }
}

void OpPipelineAdaptorPass::runOnOperationSync() {
  Operation *parentOp = getOperation();
  LDBG() << "Running op pipeline adaptor synchronously on '"
         << parentOp->getName() << "'";

  for (Region &region : parentOp->getRegions()) {
    for (Operation &op : region.getOps()) {
      for (Entry &e : entries) {
        if (e.condition(&op)) {
          LDBG() << "  Dispatching '" << op.getName() << "' to pipeline";
          if (failed(runPipeline(e.pipeline, &op))) {
            return signalPassFailure();
          }
          break;
        }
      }
    }
  }
}

void OpPipelineAdaptorPass::runOnOperationAsync() {
  Operation *parentOp = getOperation();
  MLIRContext *context = &getContext();

  LDBG() << "Running op pipeline adaptor asynchronously on '"
         << parentOp->getName() << "'";

  // --- Phase 1: Collect matched operations (sequential). ---

  // Information for a single matched operation.
  struct OpDispatchInfo {
    unsigned entryIdx;
    Operation *op;
  };

  SmallVector<OpDispatchInfo> opInfos;
  for (Region &region : parentOp->getRegions()) {
    for (Operation &op : region.getOps()) {
      for (unsigned i = 0, e = entries.size(); i < e; ++i) {
        if (entries[i].condition(&op)) {
          opInfos.push_back({i, &op});
          break;
        }
      }
    }
  }

  if (opInfos.empty()) {
    return;
  }

  // Pre-create nested analysis managers so that the dynamic pipeline
  // callback (invoked from runPipeline) only performs lookups — not
  // insertions — on the parent AnalysisManager's child map during the
  // parallel section. This is required because DenseMap insertion is not
  // thread-safe.
  AnalysisManager am = getAnalysisManager();
  for (OpDispatchInfo &info : opInfos) {
    (void)am.nest(info.op);
  }

  // --- Phase 2: Ensure per-thread entry clones exist. ---

  unsigned numThreads = context->getThreadPool().getMaxConcurrency();
  // Entries are immutable after construction (moved in from MultiPipelineNest
  // destructor), so size comparison suffices for staleness detection.
  if (asyncExecutors.empty() ||
      asyncExecutors.front().size() != entries.size()) {
    LDBG() << "  Creating " << numThreads << " async executors with "
           << entries.size() << " entries each";
    asyncExecutors.clear();
    asyncExecutors.resize(numThreads);
    for (SmallVector<Entry> &executor : asyncExecutors) {
      executor.reserve(entries.size());
      for (const Entry &e : entries) {
        executor.emplace_back(e.condition, OpPassManager(e.pipeline));
      }
    }
  }

  // --- Phase 3: Parallel dispatch. ---

  // Track which executors are in use (one per thread).
  std::vector<std::atomic<bool>> activeExecutors(asyncExecutors.size());
  for (std::atomic<bool> &active : activeExecutors) {
    active.store(false);
  }
  std::atomic<bool> hasFailure(false);

  // NOTE: Using the dynamic pipeline API (Pass::runPipeline) rather than
  // the internal OpToOpPassAdaptor::runPipeline means instrumentation events
  // will report the parent thread ID rather than the actual worker thread.
  // This is acceptable since OpPipelineAdaptorPass is an internal
  // implementation detail.
  parallelForEach(context, opInfos, [&](OpDispatchInfo &info) {
    // Claim an inactive executor via atomic compare-and-swap.
    auto it = llvm::find_if(activeExecutors, [](std::atomic<bool> &isActive) {
      bool expected = false;
      return isActive.compare_exchange_strong(expected, true);
    });
    assert(it != activeExecutors.end() &&
           "more concurrent tasks than available executor slots");
    unsigned executorIdx = it - activeExecutors.begin();

    // Run the matched pipeline from this executor's clone.
    OpPassManager &pm = asyncExecutors[executorIdx][info.entryIdx].pipeline;
    if (failed(runPipeline(pm, info.op))) {
      hasFailure.store(true);
    }

    // Release the executor.
    activeExecutors[executorIdx].store(false);
  });

  if (hasFailure) {
    signalPassFailure();
  }
}

} // namespace detail

//===----------------------------------------------------------------------===//
// MultiPipelineNest
//===----------------------------------------------------------------------===//

void MultiPipelineNest::addTo(OpPassManager &pm) {
  if (entries.empty()) {
    return;
  }
  // Move entries from the deque (used for stable references during building)
  // into a SmallVector for the adaptor pass.
  SmallVector<detail::OpPipelineAdaptorPass::Entry> vec;
  vec.reserve(entries.size());
  for (detail::OpPipelineAdaptorPass::Entry &e : entries) {
    vec.emplace_back(std::move(e.condition), std::move(e.pipeline));
  }
  pm.addPass(
      std::make_unique<detail::OpPipelineAdaptorPass>(std::move(vec)));
}

OpPassManager &MultiPipelineNest::nestIf(ConditionFn condition) {
  entries.emplace_back(std::move(condition), OpPassManager());
  return entries.back().pipeline;
}

} // namespace mlir::iree_compiler
