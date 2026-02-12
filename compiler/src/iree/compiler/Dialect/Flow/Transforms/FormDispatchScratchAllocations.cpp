// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_FORMDISPATCHSCRATCHALLOCATIONSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

/// Returns true if the function body contains a matmul-like operation.
/// A matmul-like operation is a contraction with at least 2 parallel loops.
static bool containsMatmulLikeOp(Region &region) {
  bool found = false;
  region.walk([&](linalg::LinalgOp linalgOp) {
    if (found) {
      return WalkResult::interrupt();
    }
    if (linalg::isaContractionOpInterface(linalgOp) &&
        linalgOp.getNumParallelLoops() >= 2) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Populates the scratch_size region of an export op with a placeholder
/// computation. The region takes workload arguments and returns a constant
/// scratch size.
static void populateScratchSizeRegion(ExecutableExportOp exportOp,
                                      unsigned numWorkloadArgs) {
  Region &scratchRegion = exportOp.getScratchSize();
  assert(scratchRegion.empty() && "scratch_size region already populated");

  // Create a single block with the same workload arguments as the workgroup
  // count region.
  Block *block = new Block();
  scratchRegion.push_back(block);
  Location loc = exportOp.getLoc();
  for (unsigned i = 0; i < numWorkloadArgs; ++i) {
    block->addArgument(IndexType::get(exportOp.getContext()), loc);
  }

  // Populate with a placeholder constant. The actual scratch size computation
  // will be refined later when the codegen strategy is known.
  OpBuilder builder(exportOp.getContext());
  builder.setInsertionPointToStart(block);
  Value scratchSize =
      arith::ConstantIndexOp::create(builder, loc, 4096).getResult();
  Flow::ReturnOp::create(builder, loc, scratchSize);
}

/// Collects the number of workload arguments from the workgroup_count region
/// or from the dispatch's workload operands.
static unsigned getNumWorkloadArgs(ExecutableExportOp exportOp) {
  if (!exportOp.getWorkgroupCount().empty()) {
    return exportOp.getWorkgroupCount().front().getNumArguments();
  }
  return 0;
}

struct FormDispatchScratchAllocationsPass
    : public IREE::Flow::impl::FormDispatchScratchAllocationsPassBase<
          FormDispatchScratchAllocationsPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // Step 1: Identify matmul-like executables and populate scratch_size
    // regions.
    DenseSet<StringAttr> scratchExecutables;
    for (auto executableOp :
         moduleOp.getBody()->getOps<IREE::Flow::ExecutableOp>()) {
      auto innerModuleOp = executableOp.getInnerModule();
      if (!innerModuleOp) {
        continue;
      }
      for (auto exportOp :
           executableOp.getBlock().getOps<ExecutableExportOp>()) {
        // Skip if already has a scratch_size region.
        if (!exportOp.getScratchSize().empty()) {
          continue;
        }

        auto funcOp = innerModuleOp.lookupSymbol<mlir::FunctionOpInterface>(
            exportOp.getFunctionRef());
        if (!funcOp) {
          continue;
        }

        // Check if the function contains a matmul-like operation.
        if (!containsMatmulLikeOp(funcOp.getFunctionBody())) {
          continue;
        }

        // Populate the scratch_size region.
        unsigned numWorkloadArgs = getNumWorkloadArgs(exportOp);
        populateScratchSizeRegion(exportOp, numWorkloadArgs);

        // Track this executable for dispatch modification.
        scratchExecutables.insert(executableOp.getNameAttr());
      }
    }

    if (scratchExecutables.empty()) {
      return;
    }

    // Step 2: Walk all dispatch ops and add scratch buffer arguments for
    // matching executables.
    moduleOp.walk([&](IREE::Flow::DispatchOp dispatchOp) {
      // Check if any entry point references a scratch executable.
      bool needsScratch = false;
      for (SymbolRefAttr entryRef : dispatchOp.getEntryPointRefs()) {
        auto rootRef = entryRef.getRootReference();
        if (scratchExecutables.contains(rootRef)) {
          needsScratch = true;
          break;
        }
      }
      if (!needsScratch) {
        return;
      }

      // Insert scratch size calculation before the dispatch.
      OpBuilder builder(dispatchOp);
      Location loc = dispatchOp.getLoc();

      // Use the first entry point for the scratch size call.
      SymbolRefAttr entryPointRef = *dispatchOp.getEntryPointRefs().begin();
      Value scratchSize = IREE::Flow::ExecutableScratchSizeOp::create(
                              builder, loc, builder.getIndexType(),
                              entryPointRef, dispatchOp.getWorkload())
                              .getResult();

      // Create a dynamic scratch tensor.
      auto scratchTensorType =
          RankedTensorType::get({ShapedType::kDynamic}, builder.getI8Type());
      Value scratchTensor =
          tensor::EmptyOp::create(builder, loc, scratchTensorType,
                                  ValueRange{scratchSize})
              .getResult();

      // Rebuild the dispatch op with the scratch tensor as an additional
      // argument. We need to reconstruct the op because AttrSizedOperandSegments
      // requires careful handling.
      SmallVector<Value> newArguments =
          llvm::to_vector(dispatchOp.getArguments());
      newArguments.push_back(scratchTensor);

      SmallVector<Value> newArgumentDims =
          llvm::to_vector(dispatchOp.getArgumentDims());
      // The scratch tensor has one dynamic dim (the size).
      newArgumentDims.push_back(scratchSize);

      auto newDispatchOp = IREE::Flow::DispatchOp::create(
          builder, loc, dispatchOp.getResultTypes(),
          dispatchOp.getWorkload(), dispatchOp.getEntryPointsAttr(),
          newArguments, newArgumentDims, dispatchOp.getResultDims(),
          dispatchOp.getTiedOperandsAttr());
      newDispatchOp->setDialectAttrs(dispatchOp->getDialectAttrs());

      // Replace the old dispatch with the new one.
      dispatchOp.replaceAllUsesWith(newDispatchOp.getResults());
      dispatchOp.erase();
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
