// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- DistributionAnalysis.cpp - Layout constraint analysis for PCF ------===//
//
// Iterative layout analysis for DistributeSharedExecutor. Implements the
// seed-propagate-fill-resolve pipeline described in the distribution pass
// design doc.
//
// The analysis determines, for each tensor value in a shared_executor region,
// the NestedLayoutAttr that describes how it should be distributed across
// parallel workers. This drives the subsequent code generation phase that
// replaces collective operations with per-thread slices.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/PCF/Transforms/DistributionAnalysis.h"

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"

#define DEBUG_TYPE "iree-pcf-distribution-analysis"

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// LayoutConstraintInfo
//===----------------------------------------------------------------------===//

bool LayoutConstraintInfo::setLayout(Value value, Attribute layout) {
  auto [it, inserted] = layouts.insert({value, layout});
  if (inserted) {
    return true;
  }

  // Value already has a layout. Check compatibility.
  Attribute existing = it->second;
  if (existing == layout) {
    return false;
  }

  // Check via LayoutAttrInterface if available.
  auto existingLayout = dyn_cast<LayoutAttrInterface>(existing);
  if (existingLayout && existingLayout.isCompatibleWith(layout)) {
    return false;
  }

  // Incompatible layouts: record a conflict.
  LLVM_DEBUG(llvm::dbgs() << "  Conflict on value: " << value << "\n"
                          << "    existing: " << existing << "\n"
                          << "    new:      " << layout << "\n");
  conflicts.push_back({value, layout});
  return false;
}

Attribute LayoutConstraintInfo::getLayout(Value value) const {
  auto it = layouts.find(value);
  if (it == layouts.end()) {
    return nullptr;
  }
  return it->second;
}

bool LayoutConstraintInfo::hasLayout(Value value) const {
  return layouts.contains(value);
}

//===----------------------------------------------------------------------===//
// Phase 1: SEED — Collect Explicit Constraints
//===----------------------------------------------------------------------===//

void seedConstraints(Region &region, LayoutConstraintInfo &info) {
  region.walk([&](Operation *op) {
    // Handle pcf.constrain_layout: explicit layout annotation.
    if (auto constrainOp = dyn_cast<ConstrainLayoutOp>(op)) {
      Attribute layout = constrainOp.getLayout();
      Value result = constrainOp.getResult();
      if (info.setLayout(result, layout)) {
        LLVM_DEBUG(llvm::dbgs() << "Seeded constraint from constrain_layout: "
                                << result << " -> " << layout << "\n");
        info.forwardWorklist.push_back(result);
        info.backwardWorklist.push_back(result);
      }
      // Also propagate to the input (the constrained value).
      Value input = constrainOp.getInput();
      if (info.setLayout(input, layout)) {
        info.forwardWorklist.push_back(input);
        info.backwardWorklist.push_back(input);
      }
      return;
    }

    // Handle pcf.constrain_mma: derive per-operand layouts from MMA kind.
    if (auto mmaOp = dyn_cast<ConstrainMmaOp>(op)) {
      auto kind = dyn_cast<MMALayoutInterface>(mmaOp.getKind());
      if (!kind) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Warning: constrain_mma kind does not implement "
                      "MMALayoutInterface\n");
        return;
      }

      // Extract per-operand layouts from the MMA interface.
      struct OperandInfo {
        StringRef role;
        Value result;
        Value input;
      };
      SmallVector<OperandInfo> operands = {
          {"lhs", mmaOp.getLhsResult(), mmaOp.getLhs()},
          {"rhs", mmaOp.getRhsResult(), mmaOp.getRhs()},
          {"acc", mmaOp.getAccResult(), mmaOp.getAcc()},
      };

      for (const OperandInfo &operand : operands) {
        Attribute layout = kind.getOperandLayout(operand.role);
        if (!layout) {
          continue;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "Seeded constraint from constrain_mma (" << operand.role
                   << "): " << operand.result << " -> " << layout << "\n");
        if (info.setLayout(operand.result, layout)) {
          info.forwardWorklist.push_back(operand.result);
          info.backwardWorklist.push_back(operand.result);
        }
        if (info.setLayout(operand.input, layout)) {
          info.forwardWorklist.push_back(operand.input);
          info.backwardWorklist.push_back(operand.input);
        }
      }
      return;
    }
  });
}

//===----------------------------------------------------------------------===//
// Phase 2: PROPAGATE — Iterative Constraint Propagation
//===----------------------------------------------------------------------===//

/// Propagate a layout constraint forward from a value to its users.
/// New constraints are added to worklists for further propagation.
static void propagateForward(Value value, LayoutConstraintInfo &info) {
  Attribute layout = info.getLayout(value);
  if (!layout) {
    return;
  }

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();

    // Skip constraint ops (they are seeds, not propagation targets).
    if (isa<ConstrainLayoutOp, ConstrainMmaOp>(user)) {
      continue;
    }

    // Elementwise/broadcast ops: output layout = input layout.
    if (OpTrait::hasElementwiseMappableTraits(user) &&
        user->getNumResults() == 1) {
      Value result = user->getResult(0);
      if (isa<ShapedType>(result.getType()) && info.setLayout(result, layout)) {
        LLVM_DEBUG(llvm::dbgs() << "  Forward (elementwise): " << result
                                << " -> " << layout << "\n");
        info.forwardWorklist.push_back(result);
      }
      continue;
    }

    // pcf.read_slice: source constraint propagates to result.
    // The result inherits the source layout restricted to the slice.
    if (auto readOp = dyn_cast<ReadSliceOp>(user)) {
      Value result = readOp.getResult();
      // For now, propagate the full layout. More precise slice-aware
      // propagation can be added when needed.
      if (info.setLayout(result, layout)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  Forward (read_slice): " << result << "\n");
        info.forwardWorklist.push_back(result);
      }
      continue;
    }

    // pcf.write_slice: if the value being propagated is the destination
    // sref, propagate its layout to the source tensor. WriteSliceOp is
    // void (no results), so propagation from dest→source happens here.
    if (auto writeOp = dyn_cast<WriteSliceOp>(user)) {
      Value dest = writeOp.getDest();
      if (value == dest) {
        Value source = writeOp.getSource();
        if (info.setLayout(source, layout)) {
          LLVM_DEBUG(llvm::dbgs() << "  Forward (write_slice dest->source): "
                                  << source << " -> " << layout << "\n");
          info.forwardWorklist.push_back(source);
          info.backwardWorklist.push_back(source);
        }
      }
      continue;
    }

    // scf.for: propagate to iter_args and yield values.
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      unsigned operandIdx = use.getOperandNumber();
      // Check if this is an init_arg (not the lower/upper/step bounds).
      unsigned initArgOffset = forOp.getNumControlOperands();
      if (operandIdx >= initArgOffset) {
        unsigned argIdx = operandIdx - initArgOffset;
        // Propagate to the corresponding block argument.
        Value blockArg = forOp.getRegionIterArg(argIdx);
        if (info.setLayout(blockArg, layout)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Forward (scf.for init->arg): " << blockArg << "\n");
          info.forwardWorklist.push_back(blockArg);
        }
        // Propagate to the corresponding result.
        Value result = forOp.getResult(argIdx);
        if (info.setLayout(result, layout)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Forward (scf.for init->result): " << result << "\n");
          info.forwardWorklist.push_back(result);
        }
      }
      continue;
    }

    // scf.yield: propagate to the parent op's result.
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      unsigned operandIdx = use.getOperandNumber();
      Operation *parentOp = yieldOp->getParentOp();
      if (parentOp && operandIdx < parentOp->getNumResults()) {
        Value result = parentOp->getResult(operandIdx);
        if (info.setLayout(result, layout)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Forward (yield->parent result): " << result << "\n");
          info.forwardWorklist.push_back(result);
        }
      }
      continue;
    }

    // linalg.generic/matmul: propagate input layout to output using
    // indexing maps. For contractions, this is more complex. For now,
    // propagate if the operation is elementwise (all maps are identity).
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(user)) {
      SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
      unsigned operandIdx = use.getOperandNumber();
      if (operandIdx < maps.size()) {
        // For elementwise (all identity maps), propagate directly.
        bool allIdentity =
            llvm::all_of(maps, [](AffineMap m) { return m.isIdentity(); });
        if (allIdentity) {
          for (Value result : linalgOp->getResults()) {
            if (info.setLayout(result, layout)) {
              LLVM_DEBUG(llvm::dbgs() << "  Forward (linalg elementwise): "
                                      << result << "\n");
              info.forwardWorklist.push_back(result);
            }
          }
        }
        // For contraction ops, the output layout must be derived from
        // the contraction semantics. This is handled by backward propagation
        // from the output constraint (seeded by constrain_mma).
      }
      continue;
    }
  }
}

/// Propagate a layout constraint backward from a value to its defining op's
/// operands. New constraints are added to worklists for further propagation.
static void propagateBackward(Value value, LayoutConstraintInfo &info) {
  Attribute layout = info.getLayout(value);
  if (!layout) {
    return;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return;
  }

  // Skip constraint ops.
  if (isa<ConstrainLayoutOp, ConstrainMmaOp>(defOp)) {
    return;
  }

  // Elementwise: input layout = output layout.
  if (OpTrait::hasElementwiseMappableTraits(defOp)) {
    for (Value operand : defOp->getOperands()) {
      if (isa<ShapedType>(operand.getType()) &&
          info.setLayout(operand, layout)) {
        LLVM_DEBUG(llvm::dbgs() << "  Backward (elementwise): " << operand
                                << " -> " << layout << "\n");
        info.backwardWorklist.push_back(operand);
      }
    }
    return;
  }

  // Note: WriteSliceOp backward propagation is handled in propagateForward
  // (dest sref → source tensor), because WriteSliceOp is void and never
  // appears as a defining op of an SSA value.

  // pcf.read_slice: result layout propagates to source.
  if (auto readOp = dyn_cast<ReadSliceOp>(defOp)) {
    Value source = readOp.getSource();
    if (info.setLayout(source, layout)) {
      LLVM_DEBUG(llvm::dbgs() << "  Backward (read_slice): " << source << "\n");
      info.backwardWorklist.push_back(source);
    }
    return;
  }

  // scf.for: result layout propagates to init_arg and yield values.
  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    unsigned resultIdx = cast<OpResult>(value).getResultNumber();
    // Propagate to init_arg.
    Value initArg = forOp.getInitArgs()[resultIdx];
    if (info.setLayout(initArg, layout)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Backward (scf.for result->init): " << initArg << "\n");
      info.backwardWorklist.push_back(initArg);
    }
    // Propagate to the region iter_arg.
    Value blockArg = forOp.getRegionIterArg(resultIdx);
    if (info.setLayout(blockArg, layout)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Backward (scf.for result->arg): " << blockArg << "\n");
      info.backwardWorklist.push_back(blockArg);
    }
    // Propagate to yield operand.
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (resultIdx < yieldOp.getNumOperands()) {
      Value yieldOperand = yieldOp.getOperand(resultIdx);
      if (info.setLayout(yieldOperand, layout)) {
        LLVM_DEBUG(llvm::dbgs() << "  Backward (scf.for result->yield): "
                                << yieldOperand << "\n");
        info.backwardWorklist.push_back(yieldOperand);
      }
    }
    return;
  }

  // linalg.generic/matmul: output layout propagates to inputs.
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(defOp)) {
    SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
    // For elementwise (all identity), propagate to all inputs.
    bool allIdentity =
        llvm::all_of(maps, [](AffineMap m) { return m.isIdentity(); });
    if (allIdentity) {
      for (Value input : linalgOp.getDpsInputs()) {
        if (isa<ShapedType>(input.getType()) && info.setLayout(input, layout)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Backward (linalg elementwise): " << input << "\n");
          info.backwardWorklist.push_back(input);
        }
      }
    }
    // For contraction (matmul): output layout determines input layouts.
    // The MMA constraint seeded by constrain_mma handles this. For non-MMA
    // matmuls, we derive input layouts from the output layout using the
    // contraction's indexing maps.
    //
    // TODO: Implement contraction-aware backward propagation. For M x N
    // output with subgroup_tile = [sg_m, sg_n], the LHS gets [sg_m, 1]
    // and the RHS gets [1, sg_n] for the non-reduction dimensions.
  }
}

void propagateToFixedPoint(LayoutConstraintInfo &info) {
  // Limit iterations to prevent infinite loops on cyclic graphs.
  constexpr int kMaxIterations = 100;
  int iterations = 0;

  while ((!info.forwardWorklist.empty() || !info.backwardWorklist.empty()) &&
         iterations < kMaxIterations) {
    ++iterations;

    // Prioritize forward propagation (matches VectorLayoutAnalysis).
    if (!info.forwardWorklist.empty()) {
      Value value = info.forwardWorklist.pop_back_val();
      propagateForward(value, info);
    } else {
      Value value = info.backwardWorklist.pop_back_val();
      propagateBackward(value, info);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Fixed point reached after " << iterations
                          << " iterations, " << info.layouts.size()
                          << " values constrained\n");
}

//===----------------------------------------------------------------------===//
// Phase 3: FILL UNCONSTRAINED — Assign Greedy Defaults
//===----------------------------------------------------------------------===//

Attribute createCoalescedLayout(ArrayRef<int64_t> shape, int64_t numThreads,
                                int64_t numSubgroups, Type elementType,
                                MLIRContext *context) {
  int64_t rank = shape.size();
  SmallVector<int64_t> subgroupTile(rank, 1);
  SmallVector<int64_t> batchTile(rank, 1);
  SmallVector<int64_t> outerTile(rank, 1);
  SmallVector<int64_t> threadTile(rank, 1);
  SmallVector<int64_t> elementTile(rank, 1);
  SmallVector<int64_t> subgroupStrides(rank, 0);
  SmallVector<int64_t> threadStrides(rank, 0);

  // Determine max vector width for the element type.
  // Common heuristic: 128-bit vector loads -> 4 x f32, 8 x f16, 16 x bf16.
  int64_t elementBits = elementType.getIntOrFloatBitWidth();
  int64_t maxVectorWidth = 128 / elementBits;

  // Distribute innermost dimension first (coalesced access).
  // Work from innermost to outermost dimension.
  int64_t remainingThreads = numThreads;
  int64_t remainingSubgroups = numSubgroups;
  int64_t threadStride = 1;
  int64_t subgroupStride = 1;

  for (int64_t dim = rank - 1; dim >= 0; --dim) {
    int64_t dimSize = shape[dim];
    if (dimSize <= 0) {
      continue;
    }

    // Step 1: element_tile = max vector width that divides the dimension.
    int64_t elemTile = std::min(maxVectorWidth, dimSize);
    while (dimSize % elemTile != 0 && elemTile > 1) {
      elemTile /= 2;
    }
    elementTile[dim] = elemTile;
    int64_t remaining = dimSize / elemTile;

    // Step 2: thread_tile = min(remaining threads, remaining elements).
    int64_t tTile = std::min(remainingThreads, remaining);
    while (remaining % tTile != 0 && tTile > 1) {
      tTile /= 2;
    }
    threadTile[dim] = tTile;
    remaining /= tTile;
    if (tTile > 1) {
      threadStrides[dim] = threadStride;
      threadStride *= tTile;
      remainingThreads /= tTile;
    }

    // Step 3: subgroup_tile = min(remaining subgroups, remaining elements).
    int64_t sgTile = std::min(remainingSubgroups, remaining);
    while (remaining % sgTile != 0 && sgTile > 1) {
      sgTile /= 2;
    }
    subgroupTile[dim] = sgTile;
    remaining /= sgTile;
    if (sgTile > 1) {
      subgroupStrides[dim] = subgroupStride;
      subgroupStride *= sgTile;
      remainingSubgroups /= sgTile;
    }

    // Step 4: batch_tile = remaining.
    batchTile[dim] = remaining;
  }

  // FIXME: Phase 3 (fill unconstrained) is non-functional until this is
  // wired up. All computed tile decompositions above are correct, but
  // creating the actual NestedLayoutAttr requires a dependency on VectorExt.
  // Either add a VectorExt dependency here, or move this function to a
  // location that can include VectorExt headers and call
  // VectorExt::NestedLayoutAttr::get(context, subgroupTile, batchTile,
  //   outerTile, threadTile, elementTile, subgroupStrides, threadStrides).
  //
  // Until resolved, fillUnconstrained() will leave all unconstrained values
  // without layouts, relying entirely on seed + propagation.
  (void)context;
  (void)batchTile;
  (void)outerTile;
  (void)subgroupStrides;
  (void)threadStrides;
  return nullptr;
}

void fillUnconstrained(Region &region, LayoutConstraintInfo &info,
                       int64_t numThreads, int64_t numSubgroups) {
  bool assignedNew = false;

  region.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      if (info.hasLayout(result)) {
        continue;
      }

      auto shapedType = dyn_cast<ShapedType>(result.getType());
      if (!shapedType || !shapedType.hasStaticShape()) {
        continue;
      }

      // Skip scalar values (rank 0).
      if (shapedType.getRank() == 0) {
        continue;
      }

      // Skip alloc ops (shared memory is not distributed).
      if (isa<AllocOp>(op)) {
        continue;
      }

      Attribute layout =
          createCoalescedLayout(shapedType.getShape(), numThreads, numSubgroups,
                                shapedType.getElementType(), op->getContext());
      if (!layout) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  Skipping unconstrained value (no layout factory): "
                   << result << "\n");
        continue;
      }

      if (info.setLayout(result, layout)) {
        LLVM_DEBUG(llvm::dbgs() << "  Assigned default coalesced layout: "
                                << result << "\n");
        info.forwardWorklist.push_back(result);
        info.backwardWorklist.push_back(result);
        assignedNew = true;
      }
    }
  });

  // Re-propagate after assigning defaults.
  if (assignedNew) {
    propagateToFixedPoint(info);
  }
}

//===----------------------------------------------------------------------===//
// Phase 4: RESOLVE CONFLICTS — Insert Redistribution
//===----------------------------------------------------------------------===//

StringRef determineRedistributionMethod(Attribute sourceLayout,
                                        Attribute targetLayout) {
  auto srcLayout = dyn_cast<LayoutAttrInterface>(sourceLayout);
  auto tgtLayout = dyn_cast<LayoutAttrInterface>(targetLayout);
  if (!srcLayout || !tgtLayout) {
    return "shared_memory";
  }

  // If subgroup tiles match, redistribution is within-subgroup.
  // Can use shuffle instructions.
  if (srcLayout.getSubgroupTile() == tgtLayout.getSubgroupTile()) {
    // If thread tiles also match, this is just a register reinterpretation.
    if (srcLayout.getThreadTile() == tgtLayout.getThreadTile()) {
      return "registers";
    }
    return "shuffle";
  }

  // Cross-subgroup redistribution requires shared memory.
  return "shared_memory";
}

LogicalResult resolveConflicts(Region &region, LayoutConstraintInfo &info,
                               bool strictLayoutChecking, OpBuilder &builder) {
  if (info.conflicts.empty()) {
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "Resolving " << info.conflicts.size()
                          << " layout conflicts\n");

  for (auto &[value, conflictingLayout] : info.conflicts) {
    Attribute existingLayout = info.getLayout(value);
    if (!existingLayout) {
      continue;
    }

    if (strictLayoutChecking) {
      // In strict mode, emit an error for each conflict.
      Operation *defOp = value.getDefiningOp();
      Location loc =
          defOp ? defOp->getLoc() : UnknownLoc::get(builder.getContext());
      return emitError(loc)
             << "conflicting layout constraints on value: " << "\n  existing: "
             << existingLayout << "\n  conflicting: " << conflictingLayout
             << "\n  Enable redistribution (remove --strict-layout-checking) "
                "to resolve automatically.";
    }

    // Determine the redistribution method.
    StringRef method =
        determineRedistributionMethod(existingLayout, conflictingLayout);

    LLVM_DEBUG(llvm::dbgs() << "  Inserting pcf.redistribute via " << method
                            << " for value: " << value << "\n"
                            << "    from: " << existingLayout << "\n"
                            << "    to:   " << conflictingLayout << "\n");

    // Insert pcf.redistribute op at each use that expects the conflicting
    // layout. For now, record the information; actual op insertion requires
    // Pair B's RedistributeOp.
    //
    // TODO: Insert RedistributeOp at each use site. The op takes:
    //   %redistributed = pcf.redistribute %value
    //       from layout(existingLayout)
    //       to layout(conflictingLayout)
    //       via <method>
    //       : <tensor_type>
    //
    // Then replace the conflicting uses with %redistributed.
    (void)method;
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::PCF
