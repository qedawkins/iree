// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_FORMDISPATCHSCRATCHALLOCATIONSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

/// Returns true if the function body contains a matmul-like operation.
/// A matmul-like operation is a contraction with at least 2 parallel loops.
bool containsMatmulLikeOp(Region &region) {
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

/// Populates the scratch_size region of an export op with a
/// `dispatch.scratch_size_from_slice` placeholder. The region takes workload
/// arguments and returns the result of the from_slice op, which will be
/// resolved later during codegen by materializing the backwards program slice.
void populateScratchSizeRegion(ExecutableExportOp exportOp,
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

  // Populate with scratch_size_from_slice placeholder. The actual scratch size
  // computation will be materialized later by cloning the backwards program
  // slice from the dispatch body (through workload ordinals).
  OpBuilder builder(exportOp.getContext());
  builder.setInsertionPointToStart(block);
  Value scratchSize =
      IREE::TensorExt::DispatchScratchSizeFromSliceOp::create(
          builder, loc, builder.getIndexType(), block->getArguments())
          .getResult();
  Flow::ReturnOp::create(builder, loc, scratchSize);
}

/// Collects the number of workload arguments from the workgroup_count region
/// or from the dispatch's workload operands.
unsigned getNumWorkloadArgs(ExecutableExportOp exportOp) {
  if (!exportOp.getWorkgroupCount().empty()) {
    return exportOp.getWorkgroupCount().front().getNumArguments();
  }
  return 0;
}

struct FormDispatchScratchAllocationsPass final
    : public IREE::Flow::impl::FormDispatchScratchAllocationsPassBase<
          FormDispatchScratchAllocationsPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // Step 1: Identify matmul-like executables and populate scratch_size
    // regions. Also track the inner functions for binding updates.
    DenseSet<StringAttr> scratchExecutables;
    DenseMap<StringAttr, mlir::FunctionOpInterface> executableFuncs;
    for (auto executableOp :
         moduleOp.getBody()->getOps<IREE::Flow::ExecutableOp>()) {
      mlir::ModuleOp innerModuleOp = executableOp.getInnerModule();
      if (!innerModuleOp) {
        continue;
      }
      for (auto exportOp :
           executableOp.getBlock().getOps<ExecutableExportOp>()) {
        // Skip if already has a scratch_size region.
        if (!exportOp.getScratchSize().empty()) {
          continue;
        }

        mlir::FunctionOpInterface funcOp =
            innerModuleOp.lookupSymbol<mlir::FunctionOpInterface>(
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
        executableFuncs[executableOp.getNameAttr()] = funcOp;
      }
    }

    if (scratchExecutables.empty()) {
      return;
    }

    // Step 2: Collect dispatch ops that need scratch buffers.
    SmallVector<IREE::Flow::DispatchOp> dispatchOpsToModify;
    moduleOp.walk([&](IREE::Flow::DispatchOp dispatchOp) {
      for (SymbolRefAttr entryRef : dispatchOp.getEntryPointRefs()) {
        StringAttr rootRef = entryRef.getRootReference();
        if (scratchExecutables.contains(rootRef)) {
          dispatchOpsToModify.push_back(dispatchOp);
          return;
        }
      }
    });

    // Step 3: Add scratch buffer arguments to collected dispatch ops and
    // their corresponding inner workgroup functions.
    DenseSet<StringAttr> updatedFuncs;
    for (IREE::Flow::DispatchOp dispatchOp : dispatchOpsToModify) {
      OpBuilder builder(dispatchOp);
      Location loc = dispatchOp.getLoc();

      // Use the first entry point for the scratch size call.
      SymbolRefAttr entryPointRef = *dispatchOp.getEntryPointRefs().begin();
      Value scratchSize = IREE::Flow::ExecutableScratchSizeOp::create(
                              builder, loc, builder.getIndexType(),
                              entryPointRef, dispatchOp.getWorkload())
                              .getResult();

      // Create a dynamic scratch tensor using flow.tensor.empty so that the
      // Flow→Stream conversion can handle it (tensor.empty would be illegal).
      RankedTensorType scratchTensorType =
          RankedTensorType::get({ShapedType::kDynamic}, builder.getI8Type());
      Value scratchTensor =
          IREE::Flow::TensorEmptyOp::create(builder, loc, scratchTensorType,
                                            ValueRange{scratchSize})
              .getResult();

      // The number of original dispatch arguments determines where to insert
      // the scratch binding in the inner function (before result bindings).
      unsigned numOriginalArgs = dispatchOp.getArguments().size();

      // Rebuild the dispatch op with the scratch tensor as an additional
      // argument. We need to reconstruct the op because
      // AttrSizedOperandSegments requires careful handling.
      SmallVector<Value> newArguments =
          llvm::to_vector(dispatchOp.getArguments());
      newArguments.push_back(scratchTensor);

      SmallVector<Value> newArgumentDims =
          llvm::to_vector(dispatchOp.getArgumentDims());
      // The scratch tensor has one dynamic dim (the size).
      newArgumentDims.push_back(scratchSize);

      IREE::Flow::DispatchOp newDispatchOp = IREE::Flow::DispatchOp::create(
          builder, loc, dispatchOp.getResultTypes(), dispatchOp.getWorkload(),
          dispatchOp.getEntryPointsAttr(), newArguments, newArgumentDims,
          dispatchOp.getResultDims(), dispatchOp.getTiedOperandsAttr());
      newDispatchOp->setDialectAttrs(dispatchOp->getDialectAttrs());

      // Replace the old dispatch with the new one.
      dispatchOp.replaceAllUsesWith(newDispatchOp.getResults());
      dispatchOp.erase();

      // Add the scratch binding to the inner workgroup function. Insert it
      // after the existing argument bindings and before the result bindings.
      // Each dispatch may share the same executable, so only update once.
      StringAttr execName = entryPointRef.getRootReference();
      if (updatedFuncs.contains(execName)) {
        continue;
      }
      updatedFuncs.insert(execName);

      auto funcIt = executableFuncs.find(execName);
      if (funcIt == executableFuncs.end()) {
        continue;
      }
      mlir::FunctionOpInterface funcOp = funcIt->second;
      Block &entryBlock = funcOp.getFunctionBody().front();
      auto scratchBindingType = IREE::TensorExt::DispatchTensorType::get(
          IREE::TensorExt::TensorAccess::ReadWrite, scratchTensorType);
      BlockArgument scratchArg =
          entryBlock.insertArgument(numOriginalArgs, scratchBindingType, loc);

      // Collect existing workload ordinal ops from the function body, sorted
      // by ordinal index. These are created by
      // MaterializeDefaultWorkgroupCountRegion in DispatchCreation (before the
      // Flow pipeline), so they should already be present.
      SmallVector<std::pair<int64_t, Value>> ordinalPairs;
      Operation *lastOrdinal = nullptr;
      for (auto &op : entryBlock) {
        if (auto ordinalOp =
                dyn_cast<IREE::TensorExt::DispatchWorkloadOrdinalOp>(op)) {
          ordinalPairs.push_back(
              {ordinalOp.getOrdinal().getSExtValue(), ordinalOp.getResult()});
          lastOrdinal = &op;
        }
      }
      llvm::sort(ordinalPairs, llvm::less_first());
      SmallVector<Value> workloadOrdinals;
      for (auto &[idx, val] : ordinalPairs) {
        workloadOrdinals.push_back(val);
      }

      // Insert a dispatch.tensor.scratch op to anchor the scratch binding.
      // This ties the scratch argument to the workload ordinals, providing
      // the root for the backwards program slice that will later populate
      // the scratch_size region. Must be inserted after ordinal ops to
      // maintain SSA dominance.
      OpBuilder funcBuilder(funcOp.getContext());
      if (lastOrdinal) {
        funcBuilder.setInsertionPointAfter(lastOrdinal);
      } else {
        funcBuilder.setInsertionPointToStart(&entryBlock);
      }

      IREE::TensorExt::DispatchTensorScratchOp::create(
          funcBuilder, loc, scratchBindingType, scratchArg,
          /*source_dims=*/ValueRange{}, workloadOrdinals);

      // Update the function type to include the new argument.
      SmallVector<Type> argTypes;
      for (BlockArgument arg : entryBlock.getArguments()) {
        argTypes.push_back(arg.getType());
      }
      funcOp.setType(
          FunctionType::get(funcOp.getContext(), argTypes, /*results=*/{}));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
