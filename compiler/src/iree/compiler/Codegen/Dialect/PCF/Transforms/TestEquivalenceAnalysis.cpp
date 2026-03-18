// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/Transforms/EquivalenceAnalysis.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_TESTVALUEEQUIVALENCESPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {
struct TestValueEquivalencesPass final
    : impl::TestValueEquivalencesPassBase<TestValueEquivalencesPass> {
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    // Collect seed values: results of ops with test.seed attribute.
    SmallVector<Value> seeds;
    rootOp->walk([&](Operation *op) {
      if (op->hasAttr("test.seed")) {
        for (Value result : op->getResults()) {
          seeds.push_back(result);
        }
      }
    });

    if (seeds.empty()) {
      return;
    }

    // Run the generic equivalence analysis with a predicate that accepts
    // all values.
    llvm::EquivalenceClasses<Value> equivalences =
        computeValueEquivalences(rootOp, seeds, [](Value) { return true; });

    // Group seeds by their equivalence class leader.
    llvm::DenseMap<Value, SmallVector<Value>> leaderToSeeds;
    for (Value seed : seeds) {
      Value leader = equivalences.getLeaderValue(seed);
      leaderToSeeds[leader].push_back(seed);
    }

    // Emit remarks for each equivalence class.
    for (Value seed : seeds) {
      Value leader = equivalences.getLeaderValue(seed);
      // Only emit from the leader to avoid duplicate remarks.
      if (seed != leader) {
        continue;
      }
      SmallVector<Value> &members = leaderToSeeds[leader];
      if (members.size() <= 1) {
        // Singleton class -- emit remark on the seed's defining op.
        if (Operation *defOp = seed.getDefiningOp()) {
          defOp->emitRemark("equivalence class: {self}");
        }
        continue;
      }
      // Multi-member class -- emit on the leader's defining op.
      if (Operation *defOp = leader.getDefiningOp()) {
        std::string msg = "equivalence class with " +
                          std::to_string(members.size()) + " seeds";
        defOp->emitRemark(msg);
      }
    }

    // Also emit per-seed which leader it maps to, for detailed checking.
    for (Value seed : seeds) {
      Value leader = equivalences.getLeaderValue(seed);
      if (Operation *defOp = seed.getDefiningOp()) {
        if (seed == leader) {
          defOp->emitRemark("seed is leader of its class");
        } else {
          defOp->emitRemark("seed is equivalent to leader");
        }
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::IREE::PCF
