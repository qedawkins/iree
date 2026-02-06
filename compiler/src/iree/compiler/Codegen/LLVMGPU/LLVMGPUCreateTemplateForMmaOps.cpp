// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/Template/IR/Template.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "iree-llvmgpu-create-template-for-mma-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUCREATETEMPLATEFORMMAOPSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Check if a lowering config is eligible for template creation.
/// Requirements:
/// - Has mma_kind
/// - Has promote_operands = [0, 1]
/// - Does NOT have template_call already set
static bool isEligibleForTemplateCreation(IREE::GPU::LoweringConfigAttr config) {
  if (!config) {
    return false;
  }

  // Must have mma_kind.
  auto mmaKind = IREE::GPU::getMmaKind(config);
  if (!mmaKind) {
    return false;
  }

  // Must have promote_operands = [0, 1].
  auto promoteOperands = IREE::GPU::getPromotedOperandList(config);
  if (!promoteOperands) {
    return false;
  }
  if (promoteOperands->size() != 2 || (*promoteOperands)[0] != 0 ||
      (*promoteOperands)[1] != 1) {
    return false;
  }

  // Must NOT already have template_call.
  if (IREE::GPU::getTemplateCall(config)) {
    return false;
  }

  return true;
}

/// Create a template.func for the inner-tile pattern.
///
/// Default mode (splitSubgroups=false):
///   All subgroups execute both copy and compute in a single scf.for loop:
///     pcf.generic lane_scope {
///       %acc = template.branch 1(%sg_id, %lane_id, %dest) -> type<2>
///       scf.for %i iter_args(%acc) {
///         template.branch 2(%sg_id, %lane_id, %i, %allocs...)
///         %r = template.branch 3(%sg_id, %lane_id, %acc, %allocs...)
///         scf.yield %r
///       }
///       template.branch 4(%sg_id, %lane_id, %result, %dest)
///     }
///
/// Split-subgroups mode (splitSubgroups=true):
///   Even subgroups do shared memory copies; odd subgroups do compute:
///     pcf.generic lane_scope {
///       %is_even = (sg_id % 2 == 0)
///       scf.if %is_even {
///         scf.for %i { template.branch 2(...) }
///       } else {
///         %acc = template.branch 1(...)
///         scf.for %i iter_args(%acc) {
///           %r = template.branch 3(...)
///           scf.yield %r
///         }
///         template.branch 4(...)
///       }
///     }
///
/// Type bindings:
///   type<0>: Output tensor type.
///   type<1>: pcf.sref for outputs (ref arg in pcf.generic execute region).
///   type<2>: Per-thread accumulator.
///   type<3+i>: pcf.sref for input i's shared memory (one binding per input).
///
/// - 5 implementation blocks for each phase
static IREE::Template::FuncOp
createTemplateFuncForMma(OpBuilder &builder, Location loc, StringRef name,
                         Type outputTensorType, int64_t numInputs,
                         bool splitSubgroups) {
  MLIRContext *context = builder.getContext();

  // Template types:
  // type<0>: Output tensor type.
  // type<1>: pcf.sref for outputs (ref arg in pcf.generic execute region).
  // type<2>: Per-thread accumulator.
  // type<3+i>: pcf.sref for input i's shared memory.
  IREE::Template::TypeType type0 =
      IREE::Template::TypeType::get(context, 0);
  IREE::Template::TypeType type1 =
      IREE::Template::TypeType::get(context, 1);
  IREE::Template::TypeType type2 =
      IREE::Template::TypeType::get(context, 2);
  // Create one template type per input for shared memory allocations.
  SmallVector<IREE::Template::TypeType> allocTypes;
  for (int64_t i = 0; i < numInputs; ++i) {
    allocTypes.push_back(IREE::Template::TypeType::get(context, 3 + i));
  }
  Type indexType = builder.getIndexType();

  // template.func @name(%inits: type<0>, %k: index) -> type<0>
  FunctionType funcType =
      FunctionType::get(context, {type0, indexType}, {type0});
  auto funcOp = IREE::Template::FuncOp::create(builder, loc, name, funcType);

  // Create entry block in main region.
  Region &mainRegion = funcOp.getMain();
  Block *mainBlock = builder.createBlock(&mainRegion);
  mainBlock->addArguments({type0, indexType}, {loc, loc});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(mainBlock);

  Value initsArg = mainBlock->getArgument(0);
  Value kArg = mainBlock->getArgument(1);

  // Constants for the reduction loop bounds.
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

  // ===== Outer pcf.generic: subgroup scope =====
  // Creates subgroup-level parallelism with an initializer for shared memory
  // allocation.
  IREE::PCF::ScopeAttrInterface subgroupScope =
      cast<IREE::PCF::ScopeAttrInterface>(
          IREE::GPU::SubgroupScopeAttr::get(context));
  auto outerGeneric = IREE::PCF::GenericOp::create(
      builder, loc,
      /*resultTypes=*/TypeRange{type0},
      /*scope=*/subgroupScope,
      /*inits=*/ValueRange{initsArg},
      /*dynamicSizes=*/ValueRange{},
      /*isTied=*/ArrayRef<bool>{true},
      /*numIterators=*/1,
      /*syncOnReturn=*/false);

  // The builder created the execute region with block args:
  //   [dest(type<0>), sg_id(index), sg_n(index)]
  // We need:
  //   [alloc0(type<3>), alloc1(type<4>), ..., dest(type<1>), sg_id, sg_n]
  Block &outerExecBlock = outerGeneric.getRegion().front();
  // Fix ref arg type: type<0> -> type<1> (sref for outputs).
  outerExecBlock.getArgument(0).setType(type1);
  // Insert leading args for allocs (yielded from initializer), one per input.
  for (int64_t i = numInputs - 1; i >= 0; --i) {
    outerExecBlock.insertArgument(/*index=*/0u, allocTypes[i], loc);
  }
  outerGeneric.setNumLeadingArgs(numInputs);

  // Populate the initializer region.
  {
    Region &initRegion = outerGeneric.getInitializer();
    Block *initBlock = builder.createBlock(&initRegion);
    builder.setInsertionPointToStart(initBlock);

    // template.branch 0() -> (type<3>, type<4>, ...)  (allocate shared memory).
    SmallVector<Type> allocResultTypes(allocTypes.begin(), allocTypes.end());
    auto allocBranch = IREE::Template::BranchOp::create(
        builder, loc, TypeRange{allocResultTypes},
        builder.getI64IntegerAttr(0), ValueRange{});

    // pcf.yield %alloc0, %alloc1, ... : type<3>, type<4>, ...
    IREE::PCF::YieldOp::create(builder, loc, allocBranch.getResults());
  }

  // Populate the outer execute block.
  {
    builder.setInsertionPointToStart(&outerExecBlock);

    // Collect per-input alloc values from leading args.
    SmallVector<Value> allocValues;
    for (int64_t i = 0; i < numInputs; ++i) {
      allocValues.push_back(outerExecBlock.getArgument(i));
    }
    Value dest = outerExecBlock.getArgument(numInputs);          // type<1>
    Value subgroupId = outerExecBlock.getArgument(numInputs + 1); // index

    // ===== Inner pcf.generic: lane scope =====
    // Creates lane-level parallelism within each subgroup.
    IREE::PCF::ScopeAttrInterface laneScope =
        cast<IREE::PCF::ScopeAttrInterface>(
            IREE::GPU::LaneScopeAttr::get(context));
    auto innerGeneric = IREE::PCF::GenericOp::create(
        builder, loc, /*scope=*/laneScope, /*numIterators=*/1,
        /*syncOnReturn=*/false);

    // Populate the inner execute block.
    Block &innerExecBlock = innerGeneric.getRegion().front();
    {
      builder.setInsertionPointToStart(&innerExecBlock);

      Value laneId = innerExecBlock.getArgument(0);  // index

      if (splitSubgroups) {
        // ===== Split-subgroups mode =====
        // Even subgroups copy inputs to shared; odd subgroups compute.
        Value c2 = arith::ConstantIndexOp::create(builder, loc, 2);
        Value rem =
            arith::RemUIOp::create(builder, loc, subgroupId, c2);
        Value isEven = arith::CmpIOp::create(builder, loc,
                                              arith::CmpIPredicate::eq,
                                              rem, c0);

        auto ifOp = scf::IfOp::create(builder, loc, /*resultTypes=*/
                                       TypeRange{}, isEven,
                                       /*withElseRegion=*/true);

        // Then branch: even subgroups do shared memory copies.
        {
          // scf.if already created a yield terminator; insert before it.
          Block &thenBlock = ifOp.getThenRegion().front();
          builder.setInsertionPoint(thenBlock.getTerminator());

          auto copyFor = scf::ForOp::create(builder, loc, c0, kArg, c1,
                                             ValueRange{});
          {
            Block *forBody = copyFor.getBody();
            if (!forBody->empty() &&
                forBody->back().hasTrait<OpTrait::IsTerminator>())
              forBody->back().erase();
            builder.setInsertionPointToEnd(forBody);

            Value iv = copyFor.getInductionVar();

            // template.branch 2(%sg_id, %lane_id, %i, %allocs...)
            SmallVector<Value> branch2Args = {subgroupId, laneId, iv};
            branch2Args.append(allocValues);
            IREE::Template::BranchOp::create(
                builder, loc, TypeRange{}, builder.getI64IntegerAttr(2),
                branch2Args);

            scf::YieldOp::create(builder, loc, ValueRange{});
          }
        }

        // Else branch: odd subgroups do init, compute loop, writeback.
        {
          // scf.if already created a yield terminator; insert before it.
          Block &elseBlock = ifOp.getElseRegion().front();
          builder.setInsertionPoint(elseBlock.getTerminator());

          // template.branch 1(%sg_id, %lane_id, %dest) -> type<2>
          auto initAccBranch = IREE::Template::BranchOp::create(
              builder, loc, TypeRange{type2}, builder.getI64IntegerAttr(1),
              ValueRange{subgroupId, laneId, dest});
          Value localInits = initAccBranch.getResults()[0];

          // scf.for %i iter_args(%acc = %localInits) -> type<2>
          auto computeFor = scf::ForOp::create(
              builder, loc, c0, kArg, c1, ValueRange{localInits});
          {
            Block *forBody = computeFor.getBody();
            if (!forBody->empty() &&
                forBody->back().hasTrait<OpTrait::IsTerminator>())
              forBody->back().erase();
            builder.setInsertionPointToEnd(forBody);

            Value acc = computeFor.getRegionIterArg(0);

            // template.branch 3(%sg_id, %lane_id, %acc, %allocs...)
            SmallVector<Value> branch3Args = {subgroupId, laneId, acc};
            branch3Args.append(allocValues);
            auto computeBranch = IREE::Template::BranchOp::create(
                builder, loc, TypeRange{type2},
                builder.getI64IntegerAttr(3), branch3Args);
            Value localRes = computeBranch.getResults()[0];

            scf::YieldOp::create(builder, loc, ValueRange{localRes});
          }

          Value localResult = computeFor.getResult(0);

          // template.branch 4(%sg_id, %lane_id, %result, %dest)
          builder.setInsertionPointAfter(computeFor);
          IREE::Template::BranchOp::create(
              builder, loc, TypeRange{}, builder.getI64IntegerAttr(4),
              ValueRange{subgroupId, laneId, localResult, dest});
        }

        // pcf.return after the scf.if.
        builder.setInsertionPointAfter(ifOp);
        IREE::PCF::ReturnOp::create(builder, loc);

      } else {
        // ===== Default mode: all subgroups do both copy and compute =====

        // template.branch 1(%sg_id, %lane_id, %dest) -> type<2>
        // Initialize per-thread accumulators.
        auto initAccBranch = IREE::Template::BranchOp::create(
            builder, loc, TypeRange{type2}, builder.getI64IntegerAttr(1),
            ValueRange{subgroupId, laneId, dest});
        Value localInits = initAccBranch.getResults()[0];

        // scf.for %i = %c0 to %k step %c1 iter_args(%acc = %localInits)
        auto forOp = scf::ForOp::create(builder, loc, c0, kArg, c1,
                                         ValueRange{localInits});

        // Populate the for loop body.
        {
          Block *forBody = forOp.getBody();
          // Remove default terminator if present.
          if (!forBody->empty() &&
              forBody->back().hasTrait<OpTrait::IsTerminator>())
            forBody->back().erase();
          builder.setInsertionPointToEnd(forBody);

          Value iv = forOp.getInductionVar();
          Value acc = forOp.getRegionIterArg(0);

          // template.branch 2(%sg_id, %lane_id, %i, %alloc0, %alloc1, ...)
          // Copy inputs to shared memory.
          SmallVector<Value> branch2Args = {subgroupId, laneId, iv};
          branch2Args.append(allocValues);
          IREE::Template::BranchOp::create(
              builder, loc, TypeRange{}, builder.getI64IntegerAttr(2),
              branch2Args);

          // template.branch 3(%sg_id, %lane_id, %acc, %alloc0, %alloc1, ...)
          // Perform inner-tiled computation.
          SmallVector<Value> branch3Args = {subgroupId, laneId, acc};
          branch3Args.append(allocValues);
          auto computeBranch = IREE::Template::BranchOp::create(
              builder, loc, TypeRange{type2}, builder.getI64IntegerAttr(3),
              branch3Args);
          Value localRes = computeBranch.getResults()[0];

          // scf.yield %local_res : type<2>
          scf::YieldOp::create(builder, loc, ValueRange{localRes});
        }

        Value localResult = forOp.getResult(0);

        // template.branch 4(%sg_id, %lane_id, %local_result, %dest)
        // Write results back to destination.
        builder.setInsertionPointAfter(forOp);
        IREE::Template::BranchOp::create(
            builder, loc, TypeRange{}, builder.getI64IntegerAttr(4),
            ValueRange{subgroupId, laneId, localResult, dest});

        // pcf.return (lane scope terminator).
        IREE::PCF::ReturnOp::create(builder, loc);
      }
    }

    // pcf.return (subgroup scope terminator).
    builder.setInsertionPointAfter(innerGeneric);
    IREE::PCF::ReturnOp::create(builder, loc);
  }

  // template.return %res : type<0>
  builder.setInsertionPointAfter(outerGeneric);
  IREE::Template::ReturnOp::create(builder, loc,
                                   ValueRange{outerGeneric.getResult(0)});

  // ===== Create implementation blocks =====
  Region &implRegion = funcOp.getImplementations();

  // Block 0: () -> (type<3>, type<4>, ...)
  {
    Block *block = builder.createBlock(&implRegion);
    builder.setInsertionPointToStart(block);
    SmallVector<Type> allocResultTypes(allocTypes.begin(), allocTypes.end());
    IREE::Template::UnimplementedOp::create(builder, loc,
                                            TypeRange{allocResultTypes},
                                            ValueRange{});
  }

  // Block 1: (sg_id: index, lane_id: index, dest: type<1>) -> type<2>
  {
    Block *block = builder.createBlock(&implRegion);
    block->addArguments({indexType, indexType, type1}, {loc, loc, loc});
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{type2},
                                            ValueRange{});
  }

  // Block 2: (sg_id, lane_id, k_idx, alloc0: type<3>, alloc1: type<4>, ...)
  {
    Block *block = builder.createBlock(&implRegion);
    SmallVector<Type> argTypes = {indexType, indexType, indexType};
    SmallVector<Location> argLocs(3, loc);
    for (int64_t i = 0; i < numInputs; ++i) {
      argTypes.push_back(allocTypes[i]);
      argLocs.push_back(loc);
    }
    block->addArguments(argTypes, argLocs);
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{},
                                            ValueRange{});
  }

  // Block 3: (sg_id, lane_id, acc: type<2>, alloc0: type<3>, alloc1: type<4>,
  //           ...) -> type<2>
  {
    Block *block = builder.createBlock(&implRegion);
    SmallVector<Type> argTypes = {indexType, indexType, type2};
    SmallVector<Location> argLocs(3, loc);
    for (int64_t i = 0; i < numInputs; ++i) {
      argTypes.push_back(allocTypes[i]);
      argLocs.push_back(loc);
    }
    block->addArguments(argTypes, argLocs);
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{type2},
                                            ValueRange{});
  }

  // Block 4: (sg_id: index, lane_id: index, result: type<2>, dest: type<1>)
  {
    Block *block = builder.createBlock(&implRegion);
    block->addArguments({indexType, indexType, type2, type1},
                        {loc, loc, loc, loc});
    builder.setInsertionPointToStart(block);
    IREE::Template::UnimplementedOp::create(builder, loc, TypeRange{},
                                            ValueRange{});
  }

  return funcOp;
}

class LLVMGPUCreateTemplateForMmaOpsPass final
    : public impl::LLVMGPUCreateTemplateForMmaOpsPassBase<
          LLVMGPUCreateTemplateForMmaOpsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    SymbolTable symbolTable(moduleOp);

    // Collect eligible ops first to avoid modifying while iterating.
    SmallVector<std::pair<linalg::LinalgOp, IREE::GPU::LoweringConfigAttr>>
        eligibleOps;

    moduleOp.walk([&](linalg::LinalgOp linalgOp) {
      auto config =
          dyn_cast_or_null<IREE::GPU::LoweringConfigAttr>(getLoweringConfig(linalgOp));
      if (isEligibleForTemplateCreation(config)) {
        eligibleOps.push_back({linalgOp, config});
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "Found " << eligibleOps.size()
                            << " eligible ops for template creation\n");

    // Create templates and update lowering configs.
    unsigned templateCounter = 0;
    for (auto &[linalgOp, config] : eligibleOps) {
      Location loc = linalgOp.getLoc();

      // Generate unique name for the template.
      std::string templateName =
          ("__mma_template_" + Twine(templateCounter++)).str();

      // Get output tensor type and number of inputs.
      Type outputType = linalgOp.getDpsInits()[0].getType();
      int64_t numInputs = linalgOp.getDpsInputs().size();

      // Create the template.func at module level.
      OpBuilder moduleBuilder(context);
      moduleBuilder.setInsertionPointToStart(moduleOp.getBody());
      auto templateFunc = createTemplateFuncForMma(
          moduleBuilder, loc, templateName, outputType, numInputs,
          splitSubgroups);
      symbolTable.insert(templateFunc);

      LLVM_DEBUG(llvm::dbgs() << "Created template function: " << templateName
                              << "\n");

      // Update lowering config with template_call.
      FlatSymbolRefAttr templateSymbol =
          FlatSymbolRefAttr::get(context, templateName);
      auto newConfig =
          IREE::GPU::setTemplateCall(context, config, templateSymbol);
      setLoweringConfig(linalgOp, newConfig);

      LLVM_DEBUG(llvm::dbgs() << "Updated lowering config for op\n");
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
