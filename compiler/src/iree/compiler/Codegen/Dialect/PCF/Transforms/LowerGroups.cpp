// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/ConversionDialectInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-pcf-lower-groups"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_LOWERGROUPSPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

/// Returns true if `regionId` is contained in the group type's IDs.
static bool isMatchingGroup(GroupType groupType, int64_t regionId) {
  return llvm::is_contained(groupType.getIds(), regionId);
}

/// Computes the group index from worker ID and cumulative sizes using a
/// select chain. Returns a null Value for single group (no switch needed).
///
/// For N groups with sizes [s0, s1, ...]:
///   idx = 0
///   if (worker_id >= s0) idx = 1
///   if (worker_id >= s0+s1) idx = 2
///   ...
static Value computeGroupIndex(OpBuilder &builder, Location loc, Value workerID,
                               ArrayRef<int64_t> sizes) {
  int64_t numGroups = sizes.size();
  if (numGroups <= 1) {
    return nullptr;
  }

  Value idx = arith::ConstantIndexOp::create(builder, loc, 0);
  int64_t cumsum = 0;
  for (int64_t i = 1; i < numGroups; ++i) {
    cumsum += sizes[i - 1];
    Value boundary = arith::ConstantIndexOp::create(builder, loc, cumsum);
    Value ge = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::uge,
                                     workerID, boundary);
    Value iVal = arith::ConstantIndexOp::create(builder, loc, i);
    idx = arith::SelectOp::create(builder, loc, ge, iVal, idx);
  }
  return idx;
}

/// Computes workspace offset/size values for a DefineWorkspaceOp.
/// Emits arithmetic at the current insertion point of `builder`.
/// Returns input struct values + [all offsets] + [all sizes].
static LogicalResult
computeWorkspaceValues(DefineWorkspaceOp defineWs, Value workerID,
                       ArrayRef<int64_t> formGroupSizes,
                       ValueRange inputStructValues,
                       ValueRange iterationSpaceValues, OpBuilder &builder,
                       SmallVectorImpl<Value> &resultStructValues) {
  Location loc = defineWs.getLoc();
  GroupType inputType = cast<GroupType>(defineWs.getGroup().getType());
  int64_t groupID = inputType.getId();

  if (groupID < 0 || groupID >= static_cast<int64_t>(formGroupSizes.size())) {
    return defineWs.emitOpError("group ID out of range for form_groups sizes");
  }

  // Compute base offset and worker count from form_groups sizes.
  int64_t baseOffset = 0;
  for (int64_t i = 0; i < groupID; ++i) {
    baseOffset += formGroupSizes[i];
  }
  int64_t workerCount = formGroupSizes[groupID];
  if (workerCount <= 0) {
    return defineWs.emitOpError("worker count must be positive");
  }

  // Validate all indexing maps before creating any IR. Each distributed dim
  // must appear as a direct AffineDimExpr in the map results.
  for (Attribute mapAttr : defineWs.getIndexingMapsAttr()) {
    AffineMap map = cast<AffineMapAttr>(mapAttr).getValue();
    for (int64_t d = 0, numDims = map.getNumDims(); d < numDims; ++d) {
      bool found = false;
      for (int64_t r = 0, re = map.getNumResults(); r < re; ++r) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(r))) {
          if (dimExpr.getPosition() == d) {
            found = true;
            break;
          }
        }
      }
      if (!found) {
        return defineWs.emitOpError(
            "distributed dim not found as direct map result");
      }
    }
  }

  // local_id = workerID - baseOffset.
  Value baseOffsetVal =
      arith::ConstantIndexOp::create(builder, loc, baseOffset);
  Value localID = arith::SubIOp::create(builder, loc, workerID, baseOffsetVal,
                                        arith::IntegerOverflowFlags::nsw);
  Value workerCountVal =
      arith::ConstantIndexOp::create(builder, loc, workerCount);

  // Materialize iteration space values, using the converted operands.
  ArrayRef<int64_t> staticIterSpace = defineWs.getStaticIterationSpace();
  int64_t dynamicIdx = 0;
  SmallVector<Value> iterSpaceVals;
  for (int64_t staticVal : staticIterSpace) {
    if (ShapedType::isDynamic(staticVal)) {
      iterSpaceVals.push_back(iterationSpaceValues[dynamicIdx++]);
    } else {
      iterSpaceVals.push_back(
          arith::ConstantIndexOp::create(builder, loc, staticVal));
    }
  }

  // For each distributed dim, compute offset and size.
  // Ordering: [all offsets..., all sizes...] to match the design doc.
  SmallVector<Value> offsets;
  SmallVector<Value> sizes;
  for (Attribute mapAttr : defineWs.getIndexingMapsAttr()) {
    AffineMap map = cast<AffineMapAttr>(mapAttr).getValue();
    for (int64_t d = 0, numDims = map.getNumDims(); d < numDims; ++d) {
      // Find which iteration space dimension this distributed dim maps to.
      int64_t iterDimIdx = -1;
      for (int64_t r = 0, re = map.getNumResults(); r < re; ++r) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(r))) {
          if (dimExpr.getPosition() == d) {
            iterDimIdx = r;
            break;
          }
        }
      }
      assert(iterDimIdx >= 0 && "upfront validation should have caught this");

      // Even split: size = iter_extent / worker_count.
      // NOTE: Assumes iter_extent is evenly divisible by worker_count.
      Value iterExtent = iterSpaceVals[iterDimIdx];
      Value sizeVal =
          arith::DivUIOp::create(builder, loc, iterExtent, workerCountVal);
      // offset = local_id * size.
      Value offsetVal = arith::MulIOp::create(builder, loc, localID, sizeVal,
                                              arith::IntegerOverflowFlags::nsw);
      offsets.push_back(offsetVal);
      sizes.push_back(sizeVal);
    }
  }

  // Result: input struct + offsets + sizes.
  resultStructValues.clear();
  llvm::append_range(resultStructValues, inputStructValues);
  llvm::append_range(resultStructValues, offsets);
  llvm::append_range(resultStructValues, sizes);
  return success();
}

//===----------------------------------------------------------------------===//
// TypeConverter and ConversionTarget construction.
//===----------------------------------------------------------------------===//

/// Constructs a TypeConverter for a specific case region.
/// Matching groups (containing regionId) expand to struct element types.
/// Non-matching groups convert to 0 types (erasure).
static TypeConverter buildGroupTypeConverter(MLIRContext *context,
                                             int64_t regionId) {
  TypeConverter typeConverter;
  // Pass through all non-group types.
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (isa<GroupType>(type)) {
      return std::nullopt;
    }
    return type;
  });
  // Group type conversion: matching -> expand, non-matching -> erase.
  typeConverter.addConversion(
      [regionId](GroupType groupType,
                 SmallVectorImpl<Type> &results) -> LogicalResult {
        if (isMatchingGroup(groupType, regionId)) {
          llvm::append_range(results, groupType.getStructTypes());
        }
        // Non-matching: results stay empty -> 1:0 erasure.
        return success();
      });
  return typeConverter;
}

/// Sets up the ConversionTarget for group lowering.
/// All PCF group ops are illegal. Unknown ops are dynamically legal
/// if they have no group-typed operands, results, or block arguments.
static ConversionTarget
buildGroupConversionTarget(MLIRContext &context,
                           const TypeConverter &typeConverter) {
  ConversionTarget target(context);
  target
      .addIllegalOp<ExecuteAsOp, BindOp, JoinOp, SplitOp, DefineWorkspaceOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) -> bool { return typeConverter.isLegal(op); });
  // Register SCF structural type conversion legality.
  scf::populateSCFStructuralTypeConversionTarget(typeConverter, target);
  return target;
}

//===----------------------------------------------------------------------===//
// DialectExtensionBase for loading external dialect dependencies.
//===----------------------------------------------------------------------===//

class LoadGroupLoweringDialectExtension : public DialectExtensionBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LoadGroupLoweringDialectExtension)

  LoadGroupLoweringDialectExtension()
      : DialectExtensionBase(/*dialectNames=*/{}) {}

  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    for (Dialect *dialect : dialects) {
      auto *iface = dyn_cast<PCFConversionDialectInterface>(dialect);
      if (!iface) {
        continue;
      }
      iface->loadGroupLoweringDependentDialects(context);
    }
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<LoadGroupLoweringDialectExtension>(*this);
  }
};

//===----------------------------------------------------------------------===//
// ConversionPatterns for PCF group ops.
//===----------------------------------------------------------------------===//

/// Conversion pattern for pcf.execute_as.
/// Matching groups: inline the body. Non-matching: erase entirely.
struct LowerExecuteAsOp final : OpConversionPattern<ExecuteAsOp> {
  int64_t regionId;

  LowerExecuteAsOp(const TypeConverter &converter, MLIRContext *ctx,
                   int64_t regionId)
      : OpConversionPattern(converter, ctx), regionId(regionId) {}

  LogicalResult
  matchAndRewrite(ExecuteAsOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    GroupType gt = op.getGroupType();
    if (!isMatchingGroup(gt, regionId)) {
      // Non-matching: erase. Result (if any) converts to 0 types.
      if (op.hasResult()) {
        rewriter.replaceOpWithMultiple(op, {ValueRange{}});
      } else {
        rewriter.eraseOp(op);
      }
      return success();
    }

    // Matching: inline the execute_as body.
    // The adaptor's group operand has been expanded to struct element values.
    SmallVector<Value> bodyArgs(adaptor.getGroup());

    Block &eaBody = op.getBody().front();

    // Capture yielded values before inlining (terminator will be erased).
    Operation *terminator = eaBody.getTerminator();
    SmallVector<Value> yieldedValues;
    if (auto yieldOp = dyn_cast<YieldOp>(terminator)) {
      llvm::append_range(yieldedValues, yieldOp.getOperands());
    }

    // Inline the body block before the execute_as op.
    rewriter.inlineBlockBefore(&eaBody, op, bodyArgs);

    // Erase the terminator (pcf.return or pcf.yield).
    rewriter.eraseOp(terminator);

    // Replace the execute_as with its yielded values (1:N replacement).
    if (op.hasResult()) {
      rewriter.replaceOpWithMultiple(op, {yieldedValues});
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

/// Conversion pattern for pcf.bind.
/// Matching: result = input struct values + newly bound values.
/// Non-matching: erase (result converts to 0 types).
struct LowerBindOp final : OpConversionPattern<BindOp> {
  int64_t regionId;

  LowerBindOp(const TypeConverter &converter, MLIRContext *ctx,
              int64_t regionId)
      : OpConversionPattern(converter, ctx), regionId(regionId) {}

  LogicalResult
  matchAndRewrite(BindOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    GroupType resultType = cast<GroupType>(op.getResult().getType());
    if (!isMatchingGroup(resultType, regionId)) {
      rewriter.replaceOpWithMultiple(op, {ValueRange{}});
      return success();
    }
    // Matching: concatenate existing group struct values + bound values.
    SmallVector<Value> resultVals(adaptor.getGroup());
    for (ValueRange vals : adaptor.getValues()) {
      llvm::append_range(resultVals, vals);
    }
    rewriter.replaceOpWithMultiple(op, {resultVals});
    return success();
  }
};

/// Conversion pattern for pcf.join.
/// In a matching region, exactly one input group is matching. Its converted
/// values become the result. Non-matching: erase.
struct LowerJoinOp final : OpConversionPattern<JoinOp> {
  int64_t regionId;

  LowerJoinOp(const TypeConverter &converter, MLIRContext *ctx,
              int64_t regionId)
      : OpConversionPattern(converter, ctx), regionId(regionId) {}

  LogicalResult
  matchAndRewrite(JoinOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    GroupType resultType = cast<GroupType>(op.getResult().getType());
    if (!isMatchingGroup(resultType, regionId)) {
      rewriter.replaceOpWithMultiple(op, {ValueRange{}});
      return success();
    }
    // Find the matching input group's converted values.
    for (auto [origGroup, convertedVals] :
         llvm::zip(op.getGroups(), adaptor.getGroups())) {
      GroupType gt = cast<GroupType>(origGroup.getType());
      if (isMatchingGroup(gt, regionId)) {
        SmallVector<Value> vals(convertedVals);
        rewriter.replaceOpWithMultiple(op, {vals});
        return success();
      }
    }
    return rewriter.notifyMatchFailure(op, "no matching group in join");
  }
};

/// Conversion pattern for pcf.split.
/// Matching sub-groups get the input's converted struct values.
/// Non-matching sub-groups are erased (0 types).
struct LowerSplitOp final : OpConversionPattern<SplitOp> {
  int64_t regionId;

  LowerSplitOp(const TypeConverter &converter, MLIRContext *ctx,
               int64_t regionId)
      : OpConversionPattern(converter, ctx), regionId(regionId) {}

  LogicalResult
  matchAndRewrite(SplitOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Build replacement value ranges for each result.
    SmallVector<ValueRange> replacements;
    SmallVector<Value> matchingVals(adaptor.getGroup());
    for (OpResult result : op.getResults()) {
      GroupType resultType = cast<GroupType>(result.getType());
      if (isMatchingGroup(resultType, regionId)) {
        replacements.push_back(matchingVals);
      } else {
        replacements.push_back(ValueRange{});
      }
    }
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

/// Conversion pattern for pcf.define_workspace.
/// Computes workspace offsets and sizes from form_groups partition info.
struct LowerDefineWorkspaceOp final : OpConversionPattern<DefineWorkspaceOp> {
  int64_t regionId;
  // NOTE: Storing a Value in a pattern is safe here because the
  // FrozenRewritePatternSet is local to lowerCaseRegion and destroyed at
  // function exit. The workerID is defined outside the conversion scope
  // (not in opsToConvert) so it remains valid throughout conversion.
  Value workerID;
  SmallVector<int64_t> formGroupSizes;

  LowerDefineWorkspaceOp(const TypeConverter &converter, MLIRContext *ctx,
                         int64_t regionId, Value workerID,
                         ArrayRef<int64_t> formGroupSizes)
      : OpConversionPattern(converter, ctx), regionId(regionId),
        workerID(workerID),
        formGroupSizes(formGroupSizes.begin(), formGroupSizes.end()) {}

  LogicalResult
  matchAndRewrite(DefineWorkspaceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    GroupType resultType = cast<GroupType>(op.getResult().getType());
    if (!isMatchingGroup(resultType, regionId)) {
      rewriter.replaceOpWithMultiple(op, {ValueRange{}});
      return success();
    }

    // Compute workspace values. The adaptor provides converted group
    // operands and iteration space values (index-typed, pass through).
    SmallVector<Value> inputStructValues(adaptor.getGroup());
    // Iteration space values are index-typed, so they pass through
    // the TypeConverter unchanged. Flatten from the 1:N adaptor.
    SmallVector<Value> iterSpaceValues;
    for (ValueRange vals : adaptor.getIterationSpace()) {
      llvm::append_range(iterSpaceValues, vals);
    }
    SmallVector<Value> resultStruct;
    if (failed(computeWorkspaceValues(op, workerID, formGroupSizes,
                                      inputStructValues, iterSpaceValues,
                                      rewriter, resultStruct))) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to compute workspace values");
    }
    rewriter.replaceOpWithMultiple(op, {resultStruct});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerFormGroupsOp and pass.
//===----------------------------------------------------------------------===//

/// Populates all group lowering patterns for a given case region.
static void populateGroupLoweringPatterns(const TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          ConversionTarget &target,
                                          int64_t regionId, Value workerID,
                                          ArrayRef<int64_t> sizes,
                                          MLIRContext *context) {
  // PCF group op patterns.
  patterns.add<LowerExecuteAsOp, LowerBindOp, LowerJoinOp, LowerSplitOp>(
      typeConverter, context, regionId);
  patterns.add<LowerDefineWorkspaceOp>(typeConverter, context, regionId,
                                       workerID, sizes);
  // SCF structural type conversions (handles scf.for, scf.yield, etc.).
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
  // External dialect patterns via interface.
  for (Dialect *dialect : context->getLoadedDialects()) {
    auto *iface = dyn_cast<PCFConversionDialectInterface>(dialect);
    if (!iface) {
      continue;
    }
    iface->populateGroupLoweringPatterns(typeConverter, patterns, target,
                                         regionId);
  }
}

/// Lowers a single case region by cloning the form_groups body,
/// inserting unrealized_conversion_cast ops for group-typed block args,
/// and running applyPartialConversion.
static LogicalResult lowerCaseRegion(FormGroupsOp formGroups, int64_t regionId,
                                     Value workerID, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  MLIRContext *context = formGroups.getContext();
  Block &originalBody = formGroups.getBody().front();
  ArrayRef<int64_t> sizes = formGroups.getSizes();
  Location loc = formGroups.getLoc();

  // Step 1: Insert unrealized_conversion_cast ops for each group-typed
  // block arg. These produce a group-typed SSA value from nothing, which
  // serves as a placeholder for the dialect conversion framework.
  IRMapping mapping;
  SmallVector<Operation *> castOps;
  for (BlockArgument arg : originalBody.getArguments()) {
    GroupType gt = cast<GroupType>(arg.getType());
    // Create a cast from nothing -> group type.
    UnrealizedConversionCastOp castOp =
        UnrealizedConversionCastOp::create(builder, loc, gt, ValueRange{});
    mapping.map(arg, castOp.getResult(0));
    castOps.push_back(castOp);
  }

  // Step 2: Clone body ops (except terminator) and track them for conversion.
  SmallVector<Operation *> opsToConvert(castOps.begin(), castOps.end());
  for (Operation &bodyOp : originalBody.without_terminator()) {
    opsToConvert.push_back(builder.clone(bodyOp, mapping));
  }

  // Step 4: Build per-region TypeConverter and ConversionTarget.
  TypeConverter typeConverter = buildGroupTypeConverter(context, regionId);
  ConversionTarget target = buildGroupConversionTarget(*context, typeConverter);

  // Step 5: Populate patterns.
  RewritePatternSet patterns(context);
  populateGroupLoweringPatterns(typeConverter, patterns, target, regionId,
                                workerID, sizes, context);

  // Step 6: Run dialect conversion on the case region ops.
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(opsToConvert, target, frozenPatterns))) {
    return failure();
  }

  // Step 7: Clean up unrealized_conversion_cast ops.
  // After conversion, all their uses must have been replaced.
  for (Operation *castOp : llvm::reverse(castOps)) {
    if (!castOp->use_empty()) {
      return castOp->emitOpError(
          "unrealized_conversion_cast was not resolved during group lowering");
    }
    castOp->erase();
  }

  return success();
}

/// Lowers a FormGroupsOp by creating an scf.index_switch and running
/// per-region dialect conversion in each case.
static LogicalResult lowerFormGroupsOp(FormGroupsOp op) {
  Location loc = op.getLoc();
  ScopeAttrInterface scope = op.getScope();
  ArrayRef<int64_t> sizes = op.getSizes();
  int64_t numGroups = op.getNumGroups();

  OpBuilder builder(op);

  // Get the current worker ID.
  Value workerID = scope.getWorkerIDs(builder, loc, 1)[0];

  if (numGroups == 1) {
    // Single group: no switch needed. Emit directly.
    if (failed(lowerCaseRegion(op, /*regionId=*/0, workerID, builder))) {
      return failure();
    }
    op->erase();
    return success();
  }

  // Multi-group: create scf.index_switch.
  Value groupIdx = computeGroupIndex(builder, loc, workerID, sizes);
  SmallVector<int64_t> caseIndices =
      llvm::to_vector(llvm::seq<int64_t>(0, numGroups));
  scf::IndexSwitchOp switchOp =
      scf::IndexSwitchOp::create(builder, loc, /*resultTypes=*/TypeRange{},
                                 groupIdx, caseIndices, numGroups);

  // Specialize each case region.
  for (int64_t i = 0; i < numGroups; ++i) {
    Region &caseRegion = switchOp.getCaseRegions()[i];
    Block *caseBlock = builder.createBlock(&caseRegion);
    OpBuilder caseBuilder = OpBuilder::atBlockEnd(caseBlock);
    scf::YieldOp::create(caseBuilder, loc);
    // Set insertion point before the yield for body ops.
    caseBuilder.setInsertionPointToStart(caseBlock);
    if (failed(lowerCaseRegion(op, /*regionId=*/i, workerID, caseBuilder))) {
      return failure();
    }
  }

  // Fill default region (unreachable at runtime).
  Block *defaultBlock = builder.createBlock(&switchOp.getDefaultRegion());
  OpBuilder defaultBuilder = OpBuilder::atBlockEnd(defaultBlock);
  scf::YieldOp::create(defaultBuilder, loc);

  op->erase();
  return success();
}

/// Verifies that no group ops remain after lowering.
static WalkResult verifyGroupLegality(Operation *op) {
  if (isa<FormGroupsOp, ExecuteAsOp, BindOp, JoinOp, SplitOp,
          DefineWorkspaceOp>(op)) {
    op->emitOpError("expected group ops to be lowered by LowerGroupsPass");
    return WalkResult::interrupt();
  }
  return WalkResult::advance();
}

//===----------------------------------------------------------------------===//
// Pass definition.
//===----------------------------------------------------------------------===//

struct LowerGroupsPass final : impl::LowerGroupsPassBase<LowerGroupsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<PCFDialect, arith::ArithDialect, scf::SCFDialect>();
    registry.addExtensions<LoadGroupLoweringDialectExtension>();
  }

  void runOnOperation() override {
    // Walk collects in post-order (inner before outer), which is the
    // correct lowering order for nested form_groups.
    SmallVector<FormGroupsOp> formGroupsOps;
    getOperation()->walk([&](FormGroupsOp op) { formGroupsOps.push_back(op); });

    for (FormGroupsOp op : formGroupsOps) {
      if (failed(lowerFormGroupsOp(op))) {
        return signalPassFailure();
      }
    }

    // Verify no group ops remain.
    if (getOperation()->walk(verifyGroupLegality).wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
