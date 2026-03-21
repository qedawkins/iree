// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/DistributionInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/EquivalenceAnalysis.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-pcf-distribute-and-lower"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_DISTRIBUTEANDLOWERPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// TestDistributionImpl
//===----------------------------------------------------------------------===//

/// Test implementation of DistributionInterface that succeeds without
/// modifying IR. This exercises the distribution dispatch plumbing without
/// requiring VectorDistribute. Run_cluster ops remain unchanged and are
/// handled by structural lowering.
struct TestDistributionImpl final : public DistributionInterface {
  LogicalResult
  distributeRegions(ArrayRef<Region *> regions, ValueRange threadIDs,
                    ValueRange threadCounts,
                    const ClusterEquivalenceInfo &equivalenceInfo,
                    const DenseSet<Operation *> &opsToSkip) override {
    // No-op: run_cluster ops are preserved for structural lowering.
    return success();
  }
};

//===----------------------------------------------------------------------===//
// VectorDistributionImpl
//===----------------------------------------------------------------------===//

// Note: The VectorDistributionImpl is defined in
// Codegen/Common/GPU/GPUPCFDistribution.cpp to avoid circular dependencies
// (PCF/Transforms cannot depend on Common/GPU which already depends on
// PCF/Transforms). The implementation is registered via the
// DistributionInterface pointer on the pass, set by pipeline construction
// code.
//
// For standalone testing, use --test-distribution which uses the no-op
// TestDistributionImpl above.

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns true if the given cluster type's ID matches the specified cluster
/// argument's ID (i.e., they refer to the same partition).
static bool isMatchingCluster(ClusterType clusterType,
                              BlockArgument matchingClusterArg) {
  ClusterType matchingType = cast<ClusterType>(matchingClusterArg.getType());
  return clusterType.getId() == matchingType.getId();
}

/// Compute the cluster index for a 1D tile_group given a thread ID and
/// split points. The thread belongs to cluster i if
/// split_points[i-1] <= tid < split_points[i] (with implicit 0 and total_size
/// boundaries).
///
/// This builds a chain of comparisons:
///   idx = 0
///   if (tid >= split[0]) idx = 1
///   if (tid >= split[1]) idx = 2
///   ...
static Value computeClusterIndex(OpBuilder &builder, Location loc,
                                 Value workerID, ArrayRef<Value> splitPoints) {
  if (splitPoints.empty()) {
    return arith::ConstantIndexOp::create(builder, loc, 0);
  }

  Value idx = arith::ConstantIndexOp::create(builder, loc, 0);
  for (int64_t i = 0, e = splitPoints.size(); i < e; ++i) {
    Value ge = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::uge,
                                     workerID, splitPoints[i]);
    Value iVal = arith::ConstantIndexOp::create(builder, loc, i + 1);
    idx = arith::SelectOp::create(builder, loc, ge, iVal, idx);
  }
  return idx;
}

/// Compute local thread IDs for a run_thread op given global worker IDs.
/// Returns local_tid[dim] = global_tid[dim] - lower_bound[dim], where the
/// lower bound comes from the cluster's boundsMap.
static SmallVector<Value>
computeLocalThreadIDs(OpBuilder &builder, Location loc, ClusterType clusterType,
                      ValueRange rangeValues, ValueRange globalThreadIDs) {
  AffineMap boundsMap = clusterType.getBoundsMap();
  int64_t rank = clusterType.getRank();
  ScopeAttrInterface scope = clusterType.getScope();
  SmallVector<Value> workerCounts = scope.getWorkerCounts(builder, loc, rank);

  // Build the shared operand list: dims from rangeValues, symbols from counts.
  SmallVector<OpFoldResult> mapOperands;
  llvm::append_range(mapOperands, llvm::map_range(rangeValues, [](Value v) {
                       return OpFoldResult(v);
                     }));
  llvm::append_range(mapOperands, llvm::map_range(workerCounts, [](Value v) {
                       return OpFoldResult(v);
                     }));

  SmallVector<Value> localThreadIDs;
  for (int64_t i = 0; i < rank; ++i) {
    AffineExpr lowerExpr = boundsMap.getResult(2 * i);

    // Build a single-result AffineMap for the lower bound.
    AffineMap lowerMap =
        AffineMap::get(boundsMap.getNumDims(), boundsMap.getNumSymbols(),
                       lowerExpr, builder.getContext());

    // Materialize the lower bound as a Value.
    affine::AffineApplyOp lowerBoundOp =
        affine::makeComposedAffineApply(builder, loc, lowerMap, mapOperands);
    Value lowerBound = lowerBoundOp.getResult();

    Value localID =
        arith::SubIOp::create(builder, loc, globalThreadIDs[i], lowerBound);
    localThreadIDs.push_back(localID);
  }
  return localThreadIDs;
}

//===----------------------------------------------------------------------===//
// TypeConverter and ConversionTarget for tile_group lowering.
//===----------------------------------------------------------------------===//

/// Constructs a TypeConverter for a specific case region of a tile_group.
/// All ClusterType values convert to 0 types (erasure), since tile_group
/// cluster block args have no struct elements and serve only as partition
/// identifiers. ClusterTypes appearing as run_cluster/run_thread results
/// (which may have struct elements) are also erased; the conversion patterns
/// handle their semantics.
static TypeConverter buildClusterTypeConverter(MLIRContext *context) {
  TypeConverter typeConverter;
  // Pass through all non-cluster, non-threadgroup types.
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (isa<ClusterType, ThreadGroupType>(type)) {
      return std::nullopt;
    }
    return type;
  });
  // ClusterType conversion: all erase to 0 types. The tile_group constraint
  // guarantees cluster block args have no struct elements.
  typeConverter.addConversion(
      [](ClusterType clusterType,
         SmallVectorImpl<Type> &results) -> LogicalResult {
        // Results stay empty -> 1:0 erasure.
        return success();
      });
  // ThreadGroupType conversion: also erase (shouldn't appear in tile_group
  // body, but handle defensively).
  typeConverter.addConversion(
      [](ThreadGroupType tgType,
         SmallVectorImpl<Type> &results) -> LogicalResult {
        return success();
      });
  return typeConverter;
}

/// Sets up the ConversionTarget for tile_group lowering.
/// RunThreadOp is illegal (handled by conversion patterns).
/// RunClusterOp is legal (preserved for later lowering phases; non-matching
/// instances are erased in a post-conversion cleanup step).
/// Unknown ops are dynamically legal if they have no cluster-typed operands
/// or results.
static ConversionTarget
buildClusterConversionTarget(MLIRContext &context,
                             const TypeConverter &typeConverter) {
  ConversionTarget target(context);
  target.addIllegalOp<RunThreadOp>();
  target.addLegalOp<UnrealizedConversionCastOp, RunClusterOp>();
  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) -> bool { return typeConverter.isLegal(op); });
  return target;
}

//===----------------------------------------------------------------------===//
// Conversion patterns for tile_group lowering.
//===----------------------------------------------------------------------===//

/// Conversion pattern for pcf.shared_executor.run_thread.
/// Matching cluster: inline body with computed local thread IDs.
/// Non-matching cluster: erase entirely.
struct LowerRunThreadOp final : OpConversionPattern<RunThreadOp> {
  BlockArgument matchingClusterArg;
  ValueRange globalThreadIDs;

  LowerRunThreadOp(const TypeConverter &converter, MLIRContext *ctx,
                   BlockArgument matchingClusterArg, ValueRange globalThreadIDs)
      : OpConversionPattern(converter, ctx),
        matchingClusterArg(matchingClusterArg),
        globalThreadIDs(globalThreadIDs) {}

  LogicalResult
  matchAndRewrite(RunThreadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ClusterType sourceClusterType =
        cast<ClusterType>(op.getSources().front().getType());

    if (!isMatchingCluster(sourceClusterType, matchingClusterArg)) {
      // Non-matching: erase. Result (if any) converts to 0 types.
      if (op.getResult()) {
        rewriter.replaceOpWithMultiple(op, {ValueRange{}});
      } else {
        rewriter.eraseOp(op);
      }
      return success();
    }

    // Matching: inline the run_thread body with local thread IDs.
    Location loc = op.getLoc();

    // Compute local thread IDs from global IDs and cluster bounds.
    SmallVector<Value> localIDs = computeLocalThreadIDs(
        rewriter, loc, sourceClusterType, op.getRangeValues(), globalThreadIDs);

    // Build body arguments: struct args (as unrealized_conversion_casts for
    // now, since cluster struct types are abstract tokens) + local thread IDs.
    SmallVector<Value> bodyArgs;
    MutableArrayRef<BlockArgument> structArgs = op.getStructArgs();
    for (BlockArgument arg : structArgs) {
      UnrealizedConversionCastOp castOp = UnrealizedConversionCastOp::create(
          rewriter, loc, arg.getType(), ValueRange{});
      bodyArgs.push_back(castOp.getResult(0));
    }
    llvm::append_range(bodyArgs, localIDs);

    // Capture yielded values before inlining (terminator will be erased).
    Block &body = op.getBody().front();
    Operation *terminator = body.getTerminator();

    // Inline the body block before the run_thread op.
    rewriter.inlineBlockBefore(&body, op, bodyArgs);

    // Erase the terminator (cluster_yield).
    rewriter.eraseOp(terminator);

    // Replace the run_thread with nothing (result converts to 0 types).
    if (op.getResult()) {
      rewriter.replaceOpWithMultiple(op, {ValueRange{}});
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

/// Erase non-matching RunClusterOps from the given list.
/// Matching RunClusterOps are preserved for later lowering phases.
/// Non-matching ones are erased (their results should have no remaining uses
/// in a correctly-structured tile_group).
static LogicalResult
eraseNonMatchingRunClusters(ArrayRef<RunClusterOp> runClusters,
                            BlockArgument matchingClusterArg) {
  for (RunClusterOp runCluster : runClusters) {
    // Check if any source cluster matches the current case.
    bool matches = false;
    for (Value source : runCluster.getSources()) {
      ClusterType sourceClusterType = cast<ClusterType>(source.getType());
      if (isMatchingCluster(sourceClusterType, matchingClusterArg)) {
        matches = true;
        break;
      }
    }
    if (matches) {
      continue;
    }
    // Non-matching: verify result has no uses, then erase.
    if (Value result = runCluster.getResult()) {
      if (!result.use_empty()) {
        return runCluster.emitOpError(
            "non-matching run_cluster result still has uses after tile_group "
            "lowering");
      }
    }
    runCluster.erase();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Case region lowering.
//===----------------------------------------------------------------------===//

/// Lowers a single case region by cloning the tile_group body, inserting
/// unrealized_conversion_cast ops for cluster-typed block args, and running
/// applyPartialConversion to selectively inline matching ops and erase
/// non-matching ops.
static LogicalResult lowerCaseRegion(TileGroupOp tileGroup, int64_t caseIdx,
                                     ValueRange workerIDs, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  MLIRContext *context = tileGroup.getContext();
  Block &originalBody = tileGroup.getBody().front();
  ArrayRef<BlockArgument> clusterArgs = tileGroup.getClusterArgs();
  BlockArgument matchingClusterArg = clusterArgs[caseIdx];
  Location loc = tileGroup.getLoc();

  // Step 1: Insert unrealized_conversion_cast ops for each cluster-typed
  // block arg. These produce a cluster-typed SSA value from nothing, serving
  // as placeholders for the dialect conversion framework.
  IRMapping mapping;
  SmallVector<Operation *> castOps;
  for (BlockArgument arg : clusterArgs) {
    UnrealizedConversionCastOp castOp = UnrealizedConversionCastOp::create(
        builder, loc, arg.getType(), ValueRange{});
    mapping.map(arg, castOp.getResult(0));
    castOps.push_back(castOp);
  }

  // Step 2: Clone body ops (except terminator) and track them for conversion.
  // Also collect cloned RunClusterOps separately for post-conversion cleanup
  // (they are not part of the conversion target and need manual handling).
  SmallVector<Operation *> opsToConvert(castOps.begin(), castOps.end());
  SmallVector<RunClusterOp> clonedRunClusters;
  for (Operation &bodyOp : originalBody.without_terminator()) {
    Operation *cloned = builder.clone(bodyOp, mapping);
    opsToConvert.push_back(cloned);
    if (RunClusterOp rc = dyn_cast<RunClusterOp>(cloned)) {
      clonedRunClusters.push_back(rc);
    }
  }

  // Step 3: Build per-case TypeConverter and ConversionTarget.
  TypeConverter typeConverter = buildClusterTypeConverter(context);
  ConversionTarget target =
      buildClusterConversionTarget(*context, typeConverter);

  // Step 4: Populate patterns (only RunThreadOp; RunClusterOp is handled
  // in a post-conversion cleanup step).
  RewritePatternSet patterns(context);
  patterns.add<LowerRunThreadOp>(typeConverter, context, matchingClusterArg,
                                 workerIDs);

  // Step 5: Run dialect conversion on the cloned ops.
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(opsToConvert, target, frozenPatterns))) {
    return failure();
  }

  // Step 6: Erase non-matching RunClusterOps. Matching ones are preserved
  // for later lowering phases.
  if (failed(
          eraseNonMatchingRunClusters(clonedRunClusters, matchingClusterArg))) {
    return failure();
  }

  // Step 7: Clean up unrealized_conversion_cast ops for cluster block args.
  // After conversion and run_cluster cleanup, matching run_cluster ops still
  // reference the cast for the matching cluster. Those uses are expected and
  // the cast must be kept alive for them.
  for (Operation *castOp : llvm::reverse(castOps)) {
    if (castOp->use_empty()) {
      castOp->erase();
    }
  }

  return success();
}

/// Lower a tile_group op to scf.index_switch. Gets thread IDs from the
/// source scope, computes which cluster the current thread belongs to,
/// and emits an scf.index_switch with one case per cluster. Each case
/// uses dialect conversion to selectively include only the matching
/// cluster's run_thread/run_cluster ops.
static LogicalResult lowerTileGroup(TileGroupOp tileGroup) {
  Location loc = tileGroup.getLoc();
  OpBuilder builder(tileGroup);

  // Get the scope from the source type to obtain thread IDs.
  ScopeAttrInterface scope = tileGroup.getScope();
  SmallVector<SmallVector<Value>> splitsPerDim =
      tileGroup.getSplitPointsPerDim();
  int64_t rank = splitsPerDim.size();
  SmallVector<Value> workerIDs = scope.getWorkerIDs(builder, loc, rank);

  // Compute a per-dimension partition index for each dimension.
  SmallVector<Value> dimIndices;
  SmallVector<int64_t> numPartitionsPerDim;
  for (int64_t d = 0; d < rank; ++d) {
    Value dimIdx =
        computeClusterIndex(builder, loc, workerIDs[d], splitsPerDim[d]);
    dimIndices.push_back(dimIdx);
    numPartitionsPerDim.push_back(splitsPerDim[d].size() + 1);
  }

  // Linearize the per-dimension partition indices into a single cluster
  // index (row-major). For dimensions with partition counts [n0, n1, ...],
  // the linearized index is:
  //   idx = idx_dim0 * n1 * n2 * ... + idx_dim1 * n2 * ... + ... + idx_dimN
  Value clusterIdx;
  if (rank == 1) {
    clusterIdx = dimIndices[0];
  } else {
    clusterIdx = dimIndices[0];
    for (int64_t d = 1; d < rank; ++d) {
      Value stride =
          arith::ConstantIndexOp::create(builder, loc, numPartitionsPerDim[d]);
      clusterIdx = arith::MulIOp::create(builder, loc, clusterIdx, stride);
      clusterIdx =
          arith::AddIOp::create(builder, loc, clusterIdx, dimIndices[d]);
    }
  }

  // Build the scf.index_switch. Tile_group has no results (terminates with
  // pcf.return which has no operands), so the switch yields nothing.
  ArrayRef<BlockArgument> clusterArgs = tileGroup.getClusterArgs();
  int64_t numClusters = clusterArgs.size();

  // Build case values: 0, 1, ..., numClusters-2. The last cluster is the
  // default case.
  SmallVector<int64_t> caseValues;
  for (int64_t i = 0; i < numClusters - 1; ++i) {
    caseValues.push_back(i);
  }

  scf::IndexSwitchOp switchOp = scf::IndexSwitchOp::create(
      builder, loc, TypeRange{}, clusterIdx, caseValues, caseValues.size());

  // Specialize each case region using dialect conversion.
  for (int64_t i = 0; i < numClusters; ++i) {
    Region &targetRegion = (i < numClusters - 1) ? switchOp.getCaseRegions()[i]
                                                 : switchOp.getDefaultRegion();
    Block *caseBlock = new Block();
    targetRegion.push_back(caseBlock);
    OpBuilder caseBuilder = OpBuilder::atBlockEnd(caseBlock);

    // Set insertion point before the yield for body ops.
    scf::YieldOp::create(caseBuilder, loc);
    caseBuilder.setInsertionPointToStart(caseBlock);

    if (failed(lowerCaseRegion(tileGroup, i, workerIDs, caseBuilder))) {
      return failure();
    }
  }

  // Erase the tile_group op.
  tileGroup.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Distribution dispatch
//===----------------------------------------------------------------------===//

/// Distribute threadgroup-level ops within a shared_executor body by calling
/// the distribution interface. This handles the case where the shared_executor
/// body contains undistributed ops (e.g., vectorized code with to_layout
/// anchors) without any tile_group/run_cluster structure.
static LogicalResult
distributeSharedExecutor(SharedExecutorOp sharedExec,
                         DistributionInterface &distInterface) {
  ScopeAttrInterface scope = sharedExec.getScope();
  Location loc = sharedExec.getLoc();
  OpBuilder builder(sharedExec);

  // Get thread IDs and counts from the scope.
  int64_t numIds = scope.getNativeNumProcessorIds();
  SmallVector<Value> threadIDs = scope.getWorkerIDs(builder, loc, numIds);
  SmallVector<Value> threadCounts =
      scope.getWorkerCounts(builder, loc, numIds);

  // The shared_executor body is the region to distribute.
  SmallVector<Region *> regions;
  regions.push_back(&sharedExec.getRegion());

  // Collect ops to skip: any run_thread ops already inside.
  DenseSet<Operation *> opsToSkip;
  sharedExec.getRegion().walk([&](RunThreadOp rt) {
    opsToSkip.insert(rt);
  });

  // Create an empty equivalence info (no cluster equivalences for
  // direct shared_executor distribution).
  ClusterEquivalenceInfo emptyEquivInfo;

  if (failed(distInterface.distributeRegions(
          regions, threadIDs, threadCounts, emptyEquivInfo, opsToSkip))) {
    return sharedExec.emitOpError("distribution failed for shared_executor");
  }

  return success();
}

/// Distribute run_cluster ops within a tile_group by calling the distribution
/// interface for each cluster ID group.
static LogicalResult
distributeWithinTileGroup(TileGroupOp tileGroup,
                          DistributionInterface &distInterface) {
  ClusterEquivalenceInfo equivInfo = ClusterEquivalenceInfo::build(tileGroup);

  for (auto &[clusterId, runClusterOps] : equivInfo.getClusterGroups()) {
    if (runClusterOps.empty()) {
      continue;
    }

    // Get scope from first run_cluster's source type.
    RunClusterOp firstOp = runClusterOps.front();
    ClusterType clusterType =
        cast<ClusterType>(firstOp.getSources().front().getType());
    ScopeAttrInterface scope = clusterType.getScope();
    int64_t rank = clusterType.getRank();
    Location loc = firstOp.getLoc();

    // Build thread IDs and counts before the tile_group op.
    OpBuilder builder(tileGroup);

    // Get global thread IDs and counts from the scope.
    SmallVector<Value> globalIDs = scope.getWorkerIDs(builder, loc, rank);
    SmallVector<Value> threadCounts = scope.getWorkerCounts(builder, loc, rank);

    // Compute cluster-local IDs: local_tid = global_tid - lower_bound.
    // The boundsMap dims are dependent variables (from rangeValues) and
    // symbols are scope grid sizes.
    AffineMap boundsMap = clusterType.getBoundsMap();
    SmallVector<Value> threadIDs;
    for (int64_t i = 0; i < rank; ++i) {
      AffineExpr lowerExpr = boundsMap.getResult(2 * i);

      // Build a single-result AffineMap for the lower bound expression.
      AffineMap lowerMap =
          AffineMap::get(boundsMap.getNumDims(), boundsMap.getNumSymbols(),
                         lowerExpr, builder.getContext());

      // Collect operands: dims are rangeValues, symbols are worker counts.
      SmallVector<OpFoldResult> mapOperands;
      llvm::append_range(mapOperands,
                         llvm::map_range(firstOp.getRangeValues(), [](Value v) {
                           return OpFoldResult(v);
                         }));
      llvm::append_range(mapOperands,
                         llvm::map_range(threadCounts, [](Value v) {
                           return OpFoldResult(v);
                         }));

      // Materialize the lower bound as a Value.
      affine::AffineApplyOp lowerBoundOp =
          affine::makeComposedAffineApply(builder, loc, lowerMap, mapOperands);
      Value lowerBound = lowerBoundOp.getResult();

      Value localID =
          arith::SubIOp::create(builder, loc, globalIDs[i], lowerBound);
      threadIDs.push_back(localID);
    }

    // Collect regions and ops to skip.
    SmallVector<Region *> regions;
    DenseSet<Operation *> opsToSkip;
    for (RunClusterOp op : runClusterOps) {
      regions.push_back(&op.getBody());
      // Skip any run_thread ops already inside.
      op.getBody().walk([&](RunThreadOp rt) { opsToSkip.insert(rt); });
    }

    if (failed(distInterface.distributeRegions(regions, threadIDs, threadCounts,
                                               equivInfo, opsToSkip))) {
      return tileGroup.emitOpError("distribution failed for cluster ID '")
             << clusterId << "'";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SharedExecutor → pcf.generic lowering
//===----------------------------------------------------------------------===//

/// Auto-detect a child scope from telescope ops in the shared_executor body.
/// Returns the child scope if all telescope ops agree on the same scope,
/// or nullptr if no telescope ops exist.
static ScopeAttrInterface detectChildScope(SharedExecutorOp sharedExec) {
  ScopeAttrInterface childScope;
  sharedExec.getRegion().walk([&](TelescopeOp telescopeOp) {
    // The first result of a telescope op is a ThreadGroupType whose scope
    // is the child scope.
    ThreadGroupType resultTgType =
        cast<ThreadGroupType>(telescopeOp.getResults().front().getType());
    ScopeAttrInterface scope = resultTgType.getScope();
    if (!childScope) {
      childScope = scope;
    }
    // If multiple telescope ops disagree, we still use the first one found.
    // The verifier on telescope ops ensures consistency within a single op.
  });
  return childScope;
}

/// Lower a SharedExecutorOp to a pcf.generic op. This handles the basic case
/// where tile_groups have already been lowered (Phase 3). The execute region's
/// body is moved into a new pcf.generic, with readwrite sref args mapped
/// directly and the threadgroup arg replaced with a placeholder.
///
/// Readonly refs are lowered to pcf.to_sref ops inside the generic body. The
/// original tensor is captured from outside the generic (legal since generic is
/// not IsolatedFromAbove).
///
/// If the shared_executor has an initializer region, it is moved into the
/// pcf.generic's initializer region. Leading args (yielded from the
/// initializer) are mapped from the shared_executor's execute region to the
/// pcf.generic's execute region.
///
/// When a non-null childScope is provided, the body is wrapped in a child
/// SharedExecutorOp inside the pcf.generic. InitSubscopeOp bodies become the
/// child's initializer, and TelescopeOp results are replaced with the child's
/// threadgroup and leading args.
static LogicalResult
lowerSharedExecutor(SharedExecutorOp sharedExec,
                    ScopeAttrInterface childScope = nullptr) {
  Location loc = sharedExec.getLoc();

  ScopeAttrInterface scope = sharedExec.getScope();
  ValueRange readwriteInits = sharedExec.getReadwriteInits();

  // Use the scope's native number of IDs for the iterator count.
  int64_t numIterators = scope.getNativeNumProcessorIds();
  OpBuilder builder(sharedExec);
  GenericOp genericOp =
      GenericOp::create(builder, loc, scope, readwriteInits, numIterators);

  // If the shared_executor has an initializer region, move it into the
  // pcf.generic's initializer region and add corresponding leading args
  // to the generic's execute region.
  int64_t numLeadingArgs = sharedExec.getNumLeadingArgs();
  if (!sharedExec.getInitializer().empty()) {
    // Move the initializer region contents into the generic's initializer.
    Region &srcInitializer = sharedExec.getInitializer();
    Region &dstInitializer = genericOp.getInitializer();
    dstInitializer.takeBody(srcInitializer);

    // Add leading block arguments to the generic's execute region. These
    // correspond to the values yielded by the initializer.
    Block &genericBlock = genericOp.getRegion().front();
    ArrayRef<BlockArgument> sharedExecLeadingArgs = sharedExec.getLeadingArgs();
    for (int64_t i = 0; i < numLeadingArgs; ++i) {
      // Insert leading args at the front of the block, before sref args.
      Type argType = sharedExecLeadingArgs[i].getType();
      genericBlock.insertArgument(i, argType, loc);
    }

    // Update the generic's num_leading_args property.
    genericOp.setNumLeadingArgs(numLeadingArgs);
  }

  // The generic's region has block args:
  //   [leading_args] [sref_args (one per result)] [id_args] [count_args]
  // The shared_executor's region has block args:
  //   [leading_args] [readwrite_refs] [threadgroup]
  // Map leading args and readwrite refs to generic block args.
  Block &sharedExecBlock = sharedExec.getRegion().front();
  Block &genericBlock = genericOp.getRegion().front();

  // Map leading args from shared_executor to generic.
  MutableArrayRef<BlockArgument> sharedExecLeadingArgs =
      sharedExec.getLeadingArgsMutable();
  MutableArrayRef<BlockArgument> genericLeadingArgs =
      genericOp.getLeadingArgsMutable();
  for (int64_t i = 0; i < numLeadingArgs; ++i) {
    sharedExecLeadingArgs[i].replaceAllUsesWith(genericLeadingArgs[i]);
  }

  // Map readwrite refs to generic sref args.
  MutableArrayRef<BlockArgument> readwriteRefArgs =
      sharedExec.getReadwriteRefArgsMutable();
  ArrayRef<BlockArgument> genericRefArgs = genericOp.getRegionRefArgs();
  assert(readwriteRefArgs.size() == genericRefArgs.size() &&
         "readwrite ref count must match generic sref arg count");

  for (int64_t i = 0, e = readwriteRefArgs.size(); i < e; ++i) {
    readwriteRefArgs[i].replaceAllUsesWith(genericRefArgs[i]);
  }

  // Emit pcf.to_sref for each readonly ref. The original tensor is captured
  // from outside the generic (legal since generic is not IsolatedFromAbove).
  MutableArrayRef<BlockArgument> readonlyRefArgs =
      sharedExec.getReadonlyRefArgsMutable();
  if (!readonlyRefArgs.empty()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&genericBlock);
    ValueRange readonlyInits = sharedExec.getReadonlyInits();
    for (int64_t i = 0, e = readonlyRefArgs.size(); i < e; ++i) {
      // The readonly block arg already has the correct sref type with scope.
      ShapedRefType srefType =
          cast<ShapedRefType>(readonlyRefArgs[i].getType());
      ToSrefOp toSrefOp =
          ToSrefOp::create(builder, loc, srefType, readonlyInits[i]);
      readonlyRefArgs[i].replaceAllUsesWith(toSrefOp.getResult());
    }
  }

  // Replace the threadgroup arg with an unrealized_conversion_cast placeholder.
  // After tile_group lowering the threadgroup should have no remaining uses,
  // but we still need to provide a replacement to keep the IR valid during
  // body migration.
  BlockArgument threadGroupArg = sharedExec.getThreadGroup();
  if (!threadGroupArg.use_empty()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&genericBlock);
    UnrealizedConversionCastOp castOp = UnrealizedConversionCastOp::create(
        builder, loc, threadGroupArg.getType(), ValueRange{});
    threadGroupArg.replaceAllUsesWith(castOp.getResult(0));
  }

  // Move all operations from the shared_executor body into the generic body.
  // The GenericOp builder creates an empty block (no terminator), so we splice
  // everything including the pcf.return terminator.
  genericBlock.getOperations().splice(genericBlock.end(),
                                      sharedExecBlock.getOperations());

  // If a child scope is specified, wrap the body in a child SharedExecutorOp.
  if (childScope) {
    // Collect init_subscope and telescope ops from the generic body.
    SmallVector<InitSubscopeOp> initSubscopeOps;
    SmallVector<TelescopeOp> telescopeOps;
    for (Operation &op : genericBlock.getOperations()) {
      if (InitSubscopeOp initOp = dyn_cast<InitSubscopeOp>(&op)) {
        initSubscopeOps.push_back(initOp);
      } else if (TelescopeOp telOp = dyn_cast<TelescopeOp>(&op)) {
        telescopeOps.push_back(telOp);
      }
    }

    // Create the child SharedExecutorOp inside the generic body, just before
    // the pcf.return terminator.
    Operation *returnOp = genericBlock.getTerminator();
    builder.setInsertionPoint(returnOp);
    SharedExecutorOp childExec =
        SharedExecutorOp::create(builder, loc, childScope,
                                 /*readwriteInits=*/ValueRange());

    // If there are init_subscope ops, move their body ops into the child's
    // initializer region and set up leading args.
    if (!initSubscopeOps.empty()) {
      // For now, support a single init_subscope op.
      InitSubscopeOp initOp = initSubscopeOps.front();
      Block &initBody = initOp.getBody().front();

      // Get the yield op from the init_subscope body.
      YieldOp initYield = cast<YieldOp>(initBody.getTerminator());
      SmallVector<Type> yieldedTypes;
      for (Value v : initYield.getOperands()) {
        yieldedTypes.push_back(v.getType());
      }

      // Create the initializer region in the child shared_executor.
      Region &childInitializer = childExec.getInitializer();
      Block *childInitBlock = new Block();
      childInitializer.push_back(childInitBlock);

      // Move init_subscope body ops into the child initializer.
      childInitBlock->getOperations().splice(childInitBlock->end(),
                                             initBody.getOperations());

      // Add leading args to the child execute region for each yielded value.
      Block &childExecBlock = childExec.getRegion().front();
      int64_t numChildLeadingArgs = static_cast<int64_t>(yieldedTypes.size());
      for (int64_t i = 0; i < numChildLeadingArgs; ++i) {
        // Insert leading args at the front, before the threadgroup arg.
        childExecBlock.insertArgument(i, yieldedTypes[i], loc);
      }
      childExec.setNumLeadingArgs(numChildLeadingArgs);
    }

    // Move all remaining ops (except init_subscope ops and the pcf.return
    // terminator) from the generic body into the child execute region.
    // First, collect the ops to move (excluding init_subscope ops, telescope
    // ops will be handled after moving, and the return/child exec stay).
    Block &childExecBlock = childExec.getRegion().front();
    // The child's block currently has [leading_args] [threadgroup].
    // We need to insert ops before the child's terminator. Since the child
    // was just created, its block has no ops yet (no terminator either).
    // The builder created a block but no terminator.

    // Collect ops to move: everything in genericBlock that is not the child
    // shared_executor, not an init_subscope op, and not the pcf.return.
    SmallVector<Operation *> opsToMove;
    for (Operation &op : genericBlock.getOperations()) {
      if (&op == childExec.getOperation() || &op == returnOp) {
        continue;
      }
      if (isa<InitSubscopeOp>(&op)) {
        continue;
      }
      opsToMove.push_back(&op);
    }

    // Move ops into the child execute block.
    for (Operation *op : opsToMove) {
      op->moveBefore(&childExecBlock, childExecBlock.end());
    }

    // Now handle telescope ops: replace their results with the child
    // shared_executor's threadgroup and leading args.
    BlockArgument childTgArg = childExec.getThreadGroup();
    ArrayRef<BlockArgument> childLeadingArgs = childExec.getLeadingArgs();

    for (TelescopeOp telOp : telescopeOps) {
      // First result is the child threadgroup.
      telOp.getResults().front().replaceAllUsesWith(childTgArg);

      // Remaining results are struct fields, which correspond to leading
      // args from the initializer.
      for (int64_t i = 1, e = telOp.getNumResults(); i < e; ++i) {
        telOp.getResults()[i].replaceAllUsesWith(childLeadingArgs[i - 1]);
      }

      // Erase the telescope op.
      telOp.erase();
    }

    // Erase init_subscope ops (their results should have no remaining uses
    // since telescope ops consumed them and have been erased).
    for (InitSubscopeOp initOp : initSubscopeOps) {
      // The init_subscope result was used by telescope ops, which are now
      // erased. But we need to verify the result is unused.
      if (!initOp.getResult().use_empty()) {
        return initOp.emitOpError(
            "init_subscope result still has uses after telescope lowering");
      }
      initOp.erase();
    }

    // Add a pcf.return terminator to the child execute block if there isn't
    // one already (the moved ops should include the original pcf.return from
    // the generic block, which we need to repurpose).
    // Check if the child block has a terminator.
    if (childExecBlock.empty() ||
        !childExecBlock.back().hasTrait<OpTrait::IsTerminator>()) {
      OpBuilder childBuilder = OpBuilder::atBlockEnd(&childExecBlock);
      ReturnOp::create(childBuilder, loc);
    }

    // The generic block needs a new pcf.return since the old one was moved
    // to the child. Check if genericBlock still has its terminator.
    if (genericBlock.empty() ||
        !genericBlock.back().hasTrait<OpTrait::IsTerminator>()) {
      OpBuilder genericBuilder = OpBuilder::atBlockEnd(&genericBlock);
      ReturnOp::create(genericBuilder, loc);
    }

    // Clean up dead unrealized_conversion_cast ops in the child block.
    // These were created for the parent threadgroup replacement and are
    // now unused since telescope ops consumed the threadgroup.
    SmallVector<Operation *> deadOps;
    for (Operation &childOp : childExecBlock) {
      if (isa<UnrealizedConversionCastOp>(&childOp) && childOp.use_empty()) {
        deadOps.push_back(&childOp);
      }
    }
    for (Operation *deadOp : deadOps) {
      deadOp->erase();
    }
  }

  // Replace the shared_executor results with the generic results.
  sharedExec.replaceAllUsesWith(genericOp.getResults());

  // Erase the shared_executor op.
  sharedExec.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct DistributeAndLowerPass final
    : impl::DistributeAndLowerPassBase<DistributeAndLowerPass> {
  using DistributeAndLowerPassBase::DistributeAndLowerPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Select the distribution interface based on pass options.
    std::unique_ptr<DistributionInterface> distInterface;
    if (useTestDistribution) {
      distInterface = std::make_unique<TestDistributionImpl>();
    } else if (useVectorDistribution) {
      // Create VectorDistributionImpl via the factory callback if registered.
      DistributionFactory factory = getDistributionFactory();
      if (factory) {
        distInterface = factory(subgroupSize, workgroupSize);
      } else {
        getOperation()->emitOpError(
            "vector-distribution requested but no factory registered; "
            "ensure the GPU dialect extension is loaded");
        return signalPassFailure();
      }
    }

    // Phase 1: Distribute children within tile_groups.
    if (distInterface) {
      SmallVector<TileGroupOp> tileGroups;
      op->walk([&](TileGroupOp tileGroup) { tileGroups.push_back(tileGroup); });

      for (TileGroupOp tileGroup : tileGroups) {
        if (failed(distributeWithinTileGroup(tileGroup, *distInterface))) {
          return signalPassFailure();
        }
      }

      // Phase 2: Distribute threadgroup-level ops in shared_executor.
      // For shared_executors without tile_groups, the body contains
      // undistributed ops (e.g., vectorized code with to_layout anchors).
      // Distribute them using the DistributionInterface.
      SmallVector<SharedExecutorOp> sharedExecs;
      op->walk([&](SharedExecutorOp se) { sharedExecs.push_back(se); });

      for (SharedExecutorOp se : sharedExecs) {
        if (failed(distributeSharedExecutor(se, *distInterface))) {
          return signalPassFailure();
        }
      }
    }

    // Phase 3: Structural lowering.
    // Re-collect tile_group ops since distribution may have modified IR.
    // Use post-order walk to process inner tile_groups before outer ones.
    SmallVector<TileGroupOp> tileGroupsForLowering;
    op->walk<WalkOrder::PostOrder>([&](TileGroupOp tileGroup) {
      tileGroupsForLowering.push_back(tileGroup);
    });

    for (TileGroupOp tileGroup : tileGroupsForLowering) {
      if (failed(lowerTileGroup(tileGroup))) {
        signalPassFailure();
        return;
      }
    }

    // Phase 4: Lower shared_executor → pcf.generic.
    // Collect after Phase 3 since tile_groups inside shared_executors are now
    // lowered. Only lower top-level shared_executors (not those nested inside
    // a pcf.generic, which are child shared_executors created during this
    // phase for telescoping distribution).
    SmallVector<SharedExecutorOp> sharedExecs;
    op->walk([&](SharedExecutorOp sharedExec) {
      sharedExecs.push_back(sharedExec);
    });

    for (SharedExecutorOp sharedExec : sharedExecs) {
      // Auto-detect child scope from telescope ops in the body.
      ScopeAttrInterface detectedChildScope = detectChildScope(sharedExec);
      if (failed(lowerSharedExecutor(sharedExec, detectedChildScope))) {
        signalPassFailure();
        return;
      }
    }

    // Phase 5: Verify all tile_group ops were lowered. SharedExecutorOps
    // may remain as children inside pcf.generic when telescoping was used.
    WalkResult result = op->walk([](Operation *innerOp) -> WalkResult {
      if (isa<TileGroupOp>(innerOp)) {
        innerOp->emitOpError(
            "expected to be lowered by DistributeAndLowerPass");
        return WalkResult::interrupt();
      }
      // SharedExecutorOps nested inside pcf.generic are expected (child
      // scopes from telescoping). Only flag top-level ones.
      if (isa<SharedExecutorOp>(innerOp) &&
          !innerOp->getParentOfType<GenericOp>()) {
        innerOp->emitOpError(
            "expected to be lowered by DistributeAndLowerPass");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
