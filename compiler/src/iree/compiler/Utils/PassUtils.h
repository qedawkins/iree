// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_PASSUTILS_H_
#define IREE_COMPILER_UTILS_PASSUTILS_H_

#include <deque>
#include <functional>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

// If running under a FixedPointIterator pass, annotate that a modification
// has been made which requires another iteration. No-op otherwise.
void signalFixedPointModified(Operation *rootOp);

//===----------------------------------------------------------------------===//
// OpPipelineAdaptorPass
//===----------------------------------------------------------------------===//

namespace detail {

/// An unregistered pass that adapts a parent pass manager to run sub-pipelines
/// on child operations. The pass walks top-level operations in all regions of
/// the parent operation, evaluates conditions in order for each child, and runs
/// the first matching sub-pipeline. Children that match no condition are
/// skipped.
///
/// When MLIR multithreading is enabled, child operations are dispatched in
/// parallel using the same pattern as OpToOpPassAdaptor: sub-pipelines are
/// cloned per thread, and operations are processed via parallelForEach.
///
/// This pass is used internally by MultiPipelineNest and is not intended for
/// direct use.
class OpPipelineAdaptorPass final
    : public PassWrapper<OpPipelineAdaptorPass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpPipelineAdaptorPass)

  using ConditionFn = std::function<bool(Operation *)>;

  struct Entry {
    ConditionFn condition;
    OpPassManager pipeline;

    Entry(ConditionFn condition, OpPassManager pipeline)
        : condition(std::move(condition)), pipeline(std::move(pipeline)) {}
  };

  explicit OpPipelineAdaptorPass(SmallVector<Entry> entries);

  /// Copy for thread-safe cloning. OpPassManagers are deep-copied;
  /// ConditionFn objects are value-copied (safe because they are read-only).
  OpPipelineAdaptorPass(const OpPipelineAdaptorPass &other);

  StringRef getArgument() const override { return "iree-op-pipeline-adaptor"; }

  StringRef getDescription() const override {
    return "Adapts a parent pass manager to run sub-pipelines on child "
           "operations based on conditions.";
  }

  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;

private:
  /// Run dispatch synchronously (single-threaded).
  void runOnOperationSync();

  /// Run dispatch in parallel using parallelForEach.
  void runOnOperationAsync();

  /// The condition+pipeline entries. First match wins.
  SmallVector<Entry> entries;

  /// Per-thread copies of entries for parallel execution. Lazily initialized
  /// when multithreading is enabled. Each element is a complete copy of
  /// `entries` for one thread.
  SmallVector<SmallVector<Entry>> asyncExecutors;
};

} // namespace detail

//===----------------------------------------------------------------------===//
// MultiPipelineNest
//===----------------------------------------------------------------------===//

/// Builder for conditional pipeline dispatch over child operations. Each
/// sub-pipeline is guarded by a condition predicate; the first matching
/// condition wins at runtime. When MLIR multithreading is enabled, matched
/// operations are processed in parallel.
///
/// Usage:
///   MultiPipelineNest nest;
///
///   // Static pipeline: pre-built at construction time.
///   addMyPipeline(nest.nestIf(myCondition));
///
///   // Dynamic fallback: builds pipeline per-operation at runtime.
///   nest.nestIf(fallbackCondition)
///       .addPass(createDynamicDispatchPass());
///
///   // Common passes appended to ALL sub-pipelines.
///   nest.addPass(createVerifyPass);
///
///   // Finalize: create the adaptor pass and add it to the parent PM.
///   nest.addTo(modulePassManager);
///
/// Internally, addTo() creates an OpPipelineAdaptorPass containing all
/// condition+pipeline pairs and adds it to the parent pass manager. The
/// adaptor walks child operations and dispatches each to the first matching
/// sub-pipeline, with parallel execution when available.
class MultiPipelineNest {
public:
  using ConditionFn = std::function<bool(Operation *)>;

  MultiPipelineNest() = default;

  // Movable but not copyable.
  MultiPipelineNest(MultiPipelineNest &&) = default;
  MultiPipelineNest &operator=(MultiPipelineNest &&) = default;
  MultiPipelineNest(const MultiPipelineNest &) = delete;
  MultiPipelineNest &operator=(const MultiPipelineNest &) = delete;

  /// Finalize the nest by creating an OpPipelineAdaptorPass and adding it
  /// to the given pass manager. The nest is consumed (moved-from) after this.
  void addTo(OpPassManager &pm);

  /// Add a sub-pipeline that runs when the condition returns true.
  /// Returns the OpPassManager for the caller to populate with passes.
  /// The first matching condition wins at runtime.
  OpPassManager &nestIf(ConditionFn condition);

  /// Convenience: add a sub-pipeline for a specific op type.
  template <typename OpT>
  OpPassManager &nest() {
    return nestIf([](Operation *op) { return isa<OpT>(op); });
  }

  /// Add a pass to ALL existing sub-pipelines.
  template <typename F = std::unique_ptr<Pass> (*)()>
  MultiPipelineNest &addPass(F constructor) {
    for (detail::OpPipelineAdaptorPass::Entry &entry : entries) {
      entry.pipeline.addPass(constructor());
    }
    return *this;
  }

  /// Add a pass to ALL existing sub-pipelines if the predicate is true.
  template <typename F = std::unique_ptr<Pass> (*)()>
  MultiPipelineNest &addPredicatedPass(bool enable, F constructor) {
    if (enable) {
      addPass(constructor);
    }
    return *this;
  }

private:
  // Uses std::deque so that nestIf() can return stable OpPassManager&
  // references that are not invalidated by subsequent nestIf() calls.
  std::deque<detail::OpPipelineAdaptorPass::Entry> entries;
};

//===----------------------------------------------------------------------===//
// MultiOpNest
//===----------------------------------------------------------------------===//

/// Constructs a pipeline of passes across multiple nested op types.
/// Uses MultiPipelineNest internally, enabling parallel dispatch across
/// all op types when MLIR multithreading is enabled.
///
/// Usage:
///   using FunctionLikeNest = MultiOpNest<IREE::Util::InitializerOp,
///                                        IREE::Util::FuncOp>;
///
///   FunctionLikeNest(passManager)
///     .addPass(createMyPass)
///     .addPredicatedPass(enable, createMyOtherPass);
template <typename... OpTys>
struct MultiOpNest {
public:
  MultiOpNest(OpPassManager &parentPm) : parentPm(&parentPm) {
    initNests<OpTys...>();
  }

  ~MultiOpNest() {
    if (parentPm) {
      nest.addTo(*parentPm);
    }
  }

  // Movable but not copyable. Moved-from objects will not finalize.
  MultiOpNest(MultiOpNest &&other)
      : parentPm(other.parentPm), nest(std::move(other.nest)) {
    other.parentPm = nullptr;
  }
  MultiOpNest &operator=(MultiOpNest &&other) {
    if (this != &other) {
      parentPm = other.parentPm;
      nest = std::move(other.nest);
      other.parentPm = nullptr;
    }
    return *this;
  }
  MultiOpNest(const MultiOpNest &) = delete;
  MultiOpNest &operator=(const MultiOpNest &) = delete;

  // We give the template param a default to support passing overload
  // constructors (i.e. createCanonicalizerPass).
  template <typename F = std::unique_ptr<Pass> (*)()>
  MultiOpNest &addPass(F constructor) {
    nest.addPass(constructor);
    return *this;
  }

  template <typename F = std::unique_ptr<Pass> (*)()>
  MultiOpNest &addPredicatedPass(bool enable, F constructor) {
    nest.addPredicatedPass(enable, constructor);
    return *this;
  }

private:
  template <typename T, typename... Rest>
  void initNests() {
    nest.template nest<T>();
    if constexpr (sizeof...(Rest) > 0) {
      initNests<Rest...>();
    }
  }

  OpPassManager *parentPm;
  MultiPipelineNest nest;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_PASSUTILS_H_
