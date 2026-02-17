// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-pcf-fuse-consumers"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_FUSECONSUMERSPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

struct FuseConsumersPass final
    : impl::FuseConsumersPassBase<FuseConsumersPass> {
  void runOnOperation() override;
};

/// Returns which region of |parentOp| contains |childOp|.
static unsigned getRegionIndexOf(Operation *childOp, Operation *parentOp) {
  Region *childRegion = childOp->getParentRegion();
  for (auto [i, region] : llvm::enumerate(parentOp->getRegions())) {
    if (&region == childRegion || region.isAncestor(childRegion))
      return i;
  }
  llvm_unreachable("childOp not in any region of parentOp");
}

/// Walk from |consumer| up through structured control flow ops, collecting
/// each one until reaching the block that contains |producerDefiningOp|.
/// If the consumer's operand is a pcf.guarantee_value, only control flow
/// between the consumer and the guarantee_value is collected (the rest is
/// already mirrored in the producer).
///
/// Returns the chain innermost-first. Only scf.if is supported for V1;
/// other structured CF ops cause an empty return (caller should reject).
static SmallVector<ControlFlowEntry>
computeControlFlowContext(Operation *consumer, Operation *producerDefiningOp) {
  SmallVector<ControlFlowEntry> context;

  // Walk up from the consumer's parent.
  Block *producerBlock = producerDefiningOp->getBlock();
  Operation *current = consumer;
  while (current->getBlock() != producerBlock) {
    Operation *parent = current->getParentOp();
    if (!parent)
      break;

    if (isa<scf::IfOp>(parent)) {
      unsigned regionIdx = getRegionIndexOf(current, parent);
      context.push_back({parent, regionIdx});
      current = parent;
      continue;
    }

    // Unsupported structured CF (scf.for, scf.while, etc.) — bail out.
    // The caller will reject fusion.
    if (isa<scf::ForOp, scf::WhileOp>(parent)) {
      context.clear();
      return context;
    }

    // Non-CF parent (e.g., func.func) — keep walking.
    current = parent;
  }

  return context;
}

/// Find the first fusable consumer among the users of |producerOp|. Looks
/// through pcf.guarantee_value ops to find indirect consumers.
template <typename OpTy>
static std::pair<TilingInterface, ConsumerFusionParams>
findFirstFusableConsumer(RewriterBase &rewriter, OpTy producerOp) {
  auto tryMatch = [&](Operation *user)
      -> std::optional<std::pair<TilingInterface, ConsumerFusionParams>> {
    auto target = dyn_cast<TilingInterface>(user);
    if (!target)
      return std::nullopt;
    ConsumerFusionParams tempParams;
    if (succeeded(matchTilableConsumer(rewriter, producerOp, target,
                                       tempParams))) {
      return std::make_pair(target, std::move(tempParams));
    }
    return std::nullopt;
  };

  for (Operation *user : producerOp->getUsers()) {
    // Direct consumer.
    if (auto result = tryMatch(user))
      return *result;
    // Indirect consumer through guarantee_value.
    if (auto gv = dyn_cast<PCF::GuaranteeValueOp>(user)) {
      for (Operation *gvUser : gv->getUsers()) {
        if (auto result = tryMatch(gvUser))
          return *result;
      }
    }
  }
  return {TilingInterface(), ConsumerFusionParams()};
}

struct FuseIntoGenericOp : public OpRewritePattern<IREE::PCF::GenericOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto [fusionTarget, params] =
        findFirstFusableConsumer(rewriter, genericOp);
    if (!fusionTarget)
      return failure();
    fuseTilableConsumer(rewriter, genericOp, fusionTarget, params);
    return success();
  }
};

struct FuseIntoLoopOp : public OpRewritePattern<IREE::PCF::LoopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    auto [fusionTarget, params] = findFirstFusableConsumer(rewriter, loopOp);
    if (!fusionTarget)
      return failure();
    fuseTilableConsumer(rewriter, loopOp, fusionTarget, params);
    return success();
  }
};

struct FuseExtractSliceIntoLoopOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    auto loopOp = extractSliceOp.getSource().getDefiningOp<IREE::PCF::LoopOp>();
    if (!loopOp) {
      return rewriter.notifyMatchFailure(extractSliceOp, "No loop op producer");
    }

    if (failed(fuseExtractSliceIntoProducerLoop(rewriter, loopOp,
                                                extractSliceOp))) {
      return failure();
    }
    return success();
  }
};

struct FuseExtractSliceIntoGenericOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    auto genericOp =
        extractSliceOp.getSource().getDefiningOp<IREE::PCF::GenericOp>();
    if (!genericOp) {
      return rewriter.notifyMatchFailure(extractSliceOp,
                                         "No generic op producer");
    }

    if (failed(fuseExtractSliceIntoProducerGeneric(rewriter, genericOp,
                                                   extractSliceOp))) {
      return failure();
    }
    return success();
  }
};

struct FuseCollapseShapeIntoGenericOp final
    : public OpRewritePattern<tensor::CollapseShapeOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    auto genericOp =
        collapseOp.getSrc().getDefiningOp<IREE::PCF::GenericOp>();
    if (!genericOp) {
      return rewriter.notifyMatchFailure(collapseOp,
                                         "No generic op producer");
    }
    return fuseCollapseShapeIntoProducerGeneric(rewriter, genericOp,
                                                collapseOp);
  }
};

struct FuseCollapseShapeIntoLoopOp final
    : public OpRewritePattern<tensor::CollapseShapeOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    auto loopOp = collapseOp.getSrc().getDefiningOp<IREE::PCF::LoopOp>();
    if (!loopOp) {
      return rewriter.notifyMatchFailure(collapseOp, "No loop op producer");
    }
    return fuseCollapseShapeIntoProducerLoop(rewriter, loopOp, collapseOp);
  }
};

/// Creates a nested pcf.generic scope nest with subgroup (outer) + lane
/// (inner) scopes.  Returns the linearized thread ID and total worker count.
/// The builder's insertion point is left inside the innermost scope body.
/// The caller is responsible for creating pcf.return terminators.
///
/// Structure:
///   pcf.generic scope(#iree_gpu.subgroup_scope) execute[%sg_id, %sg_count] {
///     pcf.generic scope(#iree_gpu.lane_scope) execute[%lane_id, %lane_count] {
///       // builder positioned here
///     }
///   }
struct DistributedScopeNest {
  Value tid;
  Value totalWorkers;
  PCF::GenericOp outerGeneric;
  PCF::GenericOp innerGeneric;
};

static DistributedScopeNest buildDistributedScopeNest(OpBuilder &b,
                                                      Location loc) {
  MLIRContext *ctx = b.getContext();
  auto subgroupScope =
      cast<PCF::ScopeAttrInterface>(GPU::SubgroupScopeAttr::get(ctx));
  auto laneScope = cast<PCF::ScopeAttrInterface>(GPU::LaneScopeAttr::get(ctx));

  // Outer: subgroup scope with 1 iterator.
  // Builder creates block args: [sg_id, sg_count] (0 ref args + 2 index args).
  auto outerGeneric =
      PCF::GenericOp::create(b, loc, subgroupScope, /*numIterators=*/1);
  Value sgId = outerGeneric.getIdArgs()[0];
  Value sgCount = outerGeneric.getCountArgs()[0];

  // Inner: lane scope with 1 iterator, nested inside the outer body.
  // Builder creates block args: [lane_id, lane_count].
  OpBuilder innerB(ctx);
  innerB.setInsertionPointToStart(&outerGeneric.getRegion().front());
  auto innerGeneric =
      PCF::GenericOp::create(innerB, loc, laneScope, /*numIterators=*/1);
  Value laneId = innerGeneric.getIdArgs()[0];
  Value laneCount = innerGeneric.getCountArgs()[0];

  // Linearize: tid = sg_id * lane_count + lane_id.
  b.setInsertionPointToStart(&innerGeneric.getRegion().front());
  Value sgTimesLane = arith::MulIOp::create(b, loc, sgId, laneCount);
  Value tid = arith::AddIOp::create(b, loc, sgTimesLane, laneId);

  // total = sg_count * lane_count.
  Value totalWorkers = arith::MulIOp::create(b, loc, sgCount, laneCount);

  return {tid, totalWorkers, outerGeneric, innerGeneric};
}

/// Creates an scf.for loop that distributes |numElements| elements across
/// threads.  Each thread processes a chunk of elements.  Inside the loop
/// body, delinearized (row, col) indices are available.
///
/// Returns the loop op.  The builder's insertion point is left inside the
/// loop body after the row/col computation.
struct PerThreadChunkLoop {
  scf::ForOp loop;
  Value elemIdx;
  Value row;
  Value col;
};

static PerThreadChunkLoop buildPerThreadChunkLoop(OpBuilder &b, Location loc,
                                                  Value tid, Value totalWorkers,
                                                  ArrayRef<int64_t> tileShape) {
  int64_t numElements = 1;
  for (int64_t dim : tileShape) {
    numElements *= dim;
  }

  Value cNumElements = arith::ConstantIndexOp::create(b, loc, numElements);

  // chunkSize = ceildiv(numElements, totalWorkers).
  Value chunkSize =
      arith::CeilDivUIOp::create(b, loc, cNumElements, totalWorkers);

  // lb = tid * chunkSize.
  Value lb = arith::MulIOp::create(b, loc, tid, chunkSize);

  // ub = min(lb + chunkSize, numElements).
  Value lbPlusChunk = arith::AddIOp::create(b, loc, lb, chunkSize);
  Value ub = arith::MinUIOp::create(b, loc, lbPlusChunk, cNumElements);

  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto forOp = scf::ForOp::create(b, loc, lb, ub, c1);

  // Delinearize indices inside the loop body.
  b.setInsertionPointToStart(forOp.getBody());
  Value elemIdx = forOp.getInductionVar();
  Value cCols = arith::ConstantIndexOp::create(b, loc, tileShape[1]);
  Value row = arith::DivUIOp::create(b, loc, elemIdx, cCols);
  Value col = arith::RemUIOp::create(b, loc, elemIdx, cCols);

  return {forOp, elemIdx, row, col};
}

/// Decomposes `pcf.stream_k_recombine` into fully distributed control flow:
///
/// 1. Conditional scratch write (if tile is split among workgroups) —
///    distributed across all threads via pcf.generic scope nests.
/// 2. Global memory fence + thread-0 atomic increment + broadcast.
/// 3. If last contributor: distributed recombine from scratch + writeback.
/// 4. Else (non-split): distributed writeback of producer result.

/// Recursively hoist |op| (and its transitive operand-defining ops) before
/// |before| if they do not already dominate it.  Block arguments always
/// dominate everything in the block, so they are safe leaves.  Returns
/// failure if any leaf operation cannot be hoisted (e.g. it has side effects
/// that prevent movement, or its own operands do not dominate |before|).
static LogicalResult hoistBeforeIfNeeded(Operation *op, Operation *before,
                                         DominanceInfo &domInfo) {
  if (domInfo.properlyDominates(op, before)) {
    return success();
  }
  // Recursively hoist all operand-defining ops first.
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      // Block argument -- always dominates everything in the block.
      continue;
    }
    if (failed(hoistBeforeIfNeeded(defOp, before, domInfo))) {
      return failure();
    }
  }
  op->moveBefore(before);
  return success();
}

WalkResult verifyOperationLegality(Operation *op) {
  if (isa<UnrealizedConversionCastOp>(op)) {
    return WalkResult::interrupt();
  }
  return WalkResult::advance();
}

/// Decomposes a pcf.stream_k_recombine into fully distributed control flow:
///
/// 1. Sets sync_on_return=true on the direct producer pcf.generic.
/// 2. Inside the producer body, adds conditional scratch writes (if split).
/// 3. Outside the producer, creates 4-phase distributed decomposition:
///    Phase 1: All threads write partialTile to scratch via scope nests.
///    Phase 2: Barrier + thread-0 atomic + broadcast dword.
///    Phase 3: If last contributor, all threads recombine from scratch and
///             write to output dest via scope nests.
///    Phase 4 (else): All threads write partialTile directly to output
///             dest via scope nests.
/// 4. Final workgroup barrier.
struct FuseStreamKRecombineIntoGeneric final
    : public OpRewritePattern<StreamKRecombineOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(StreamKRecombineOp recombineOp,
                                PatternRewriter &rewriter) const override {
    Value partialTile = recombineOp.getPartialTile();

    // If you touch this line you die. If the exact producer of the
    // recombine op is not pcf.generic, you die.
    PCF::GenericOp producerOp = partialTile.getDefiningOp<PCF::GenericOp>();
    if (!producerOp) {
      // if you touch this line, you die.
      return failure();
    }

    // Gather recombine operands.
    Value scratch = recombineOp.getScratch();
    Value counter = recombineOp.getCounter();
    Value counterIndex = recombineOp.getCounterIndex();
    Value numInGroup = recombineOp.getNumInGroup();
    Value contributorOrdinal = recombineOp.getContributorOrdinal();
    Value dest = recombineOp.getDest();
    SmallVector<OpFoldResult> destOffsets = recombineOp.getMixedOffsets();

    // Hoist recombine operands before the producer so they are visible
    // inside the producer body (pcf.generic is not IsolatedFromAbove).
    Operation *hoistTarget = producerOp.getOperation();
    DominanceInfo dominanceInfo(recombineOp->getParentOp());
    for (Value operand :
         {scratch, counterIndex, numInGroup, contributorOrdinal, dest}) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        // Block argument -- always dominates.
        continue;
      }
      if (failed(hoistBeforeIfNeeded(defOp, hoistTarget, dominanceInfo))) {
        return rewriter.notifyMatchFailure(
            recombineOp, "recombine operand cannot be hoisted");
      }
    }
    // Hoist dynamic dest offset values.
    for (OpFoldResult offset : destOffsets) {
      if (auto val = dyn_cast<Value>(offset)) {
        if (Operation *defOp = val.getDefiningOp()) {
          if (failed(hoistBeforeIfNeeded(defOp, hoistTarget, dominanceInfo))) {
            return rewriter.notifyMatchFailure(recombineOp,
                                               "dest offset cannot be hoisted");
          }
        }
      }
    }

    Location loc = recombineOp.getLoc();

    RankedTensorType partialTileType =
        cast<RankedTensorType>(partialTile.getType());
    ArrayRef<int64_t> tileShape = partialTileType.getShape();

    PCF::ShapedRefType scratchType =
        cast<PCF::ShapedRefType>(scratch.getType());
    PCF::ShapedRefType counterType = recombineOp.getCounterType();

    // Step 1: Set sync_on_return=true on the direct producer generic.
    // For the indirect case the inner producer already synchronizes via
    // its sref sync-scope attributes.
    if (producerOp) {
      rewriter.modifyOpInPlace(producerOp,
                               [&]() { producerOp.setSyncOnReturn(true); });
    }

    // Step 2: Compute constants and conditions before the producer.
    // Since pcf.generic does not have IsolatedFromAbove, these values
    // are captured inside the producer body for the scratch write.
    OpBuilder b(hoistTarget);

    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);

    // Split condition: tile is split across multiple workgroups.
    Value isSplit = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ne, numInGroup, c1);

    // Per-tile base offset in scratch.
    int64_t scratchTotalRows = scratchType.getShape()[0];
    int64_t numTiles = counterType.getShape()[0];
    int64_t rowsPerTile = scratchTotalRows / numTiles;
    Value cRowsPerTile =
        arith::ConstantIndexOp::create(b, loc, rowsPerTile);
    Value tileBaseOffset =
        arith::MulIOp::create(b, loc, counterIndex, cRowsPerTile);

    // Scratch write offset for this contributor.
    Value tileHeight =
        arith::ConstantIndexOp::create(b, loc, tileShape[0]);
    Value localOffset =
        arith::MulIOp::create(b, loc, contributorOrdinal, tileHeight);
    Value scratchWriteOffset =
        arith::AddIOp::create(b, loc, tileBaseOffset, localOffset);

    // Non-split condition (complement of isSplit).
    Value isNotSplit =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, numInGroup, c1);

    // Step 2b: Inject scf.if guards around each write_slice to the result
    // sref inside the producer body.  This gives us:
    //   scf.if (isSplit)    { pcf.write_slice %local to %scratch }
    //   scf.if (!isSplit)   { pcf.write_slice %local to %sref   }
    // The scratch write happens inside the producer so each thread writes
    // its sub-tile directly from registers — no round-trip through the
    // tensor result.
    {
      unsigned resultIdx = cast<OpResult>(partialTile).getResultNumber();
      Value refArg = producerOp.getRegionRefArgs()[resultIdx];

      SmallVector<PCF::WriteSliceOp> writeSlices;
      for (Operation *user : refArg.getUsers()) {
        if (auto ws = dyn_cast<PCF::WriteSliceOp>(user)) {
          writeSlices.push_back(ws);
        }
      }

      for (PCF::WriteSliceOp ws : writeSlices) {
        OpBuilder wsB(ws);
        Location wsLoc = ws.getLoc();

        SmallVector<OpFoldResult> wsOffsets = ws.getMixedOffsets();
        SmallVector<OpFoldResult> wsSizes = ws.getMixedSizes();
        SmallVector<OpFoldResult> wsStrides = ws.getMixedStrides();
        Value wsSource = ws.getSource();

        // scf.if(isSplit) { write to scratch }.
        auto scratchIf = scf::IfOp::create(wsB, wsLoc, TypeRange{}, isSplit,
                                           /*withElseRegion=*/false);
        {
          OpBuilder scratchB(wsB.getContext());
          scratchB.setInsertionPointToStart(&scratchIf.getThenRegion().front());

          // scratch row = scratchWriteOffset + row_offset_in_tile.
          Value wsRow =
              getValueOrCreateConstantIndexOp(scratchB, wsLoc, wsOffsets[0]);
          Value scratchRow =
              arith::AddIOp::create(scratchB, wsLoc, scratchWriteOffset, wsRow);

          SmallVector<OpFoldResult> scratchWriteOffsets = {scratchRow,
                                                           wsOffsets[1]};
          PCF::WriteSliceOp::create(scratchB, wsLoc, wsSource, scratch,
                                    scratchWriteOffsets, wsSizes, wsStrides);
        }

        // scf.if(!isSplit) { original write to sref }.
        auto srefIf = scf::IfOp::create(wsB, wsLoc, TypeRange{}, isNotSplit,
                                        /*withElseRegion=*/false);
        // Move the original write_slice inside the then body.
        ws->moveBefore(srefIf.getThenRegion().front().getTerminator());
      }
    }

    // Step 3: Create fully distributed post-producer control flow.
    // Use plain OpBuilder for all newly-created regions to avoid
    // greedy driver listener issues.
    b.setInsertionPoint(recombineOp);

    Type elemType = partialTileType.getElementType();

    // Allocate a broadcast dword in workgroup shared memory (LDS).
    // Thread-0 stores the atomic result here; all threads load it after
    // the barrier to determine if this workgroup is the last contributor.
    // Use SubgroupScopeAttr so ConvertSRefToMemRef lowers to
    // memref.alloc with gpu.address_space<workgroup>.
    auto broadcastScope = cast<PCF::ScopeAttrInterface>(
        GPU::SubgroupScopeAttr::get(b.getContext()));
    auto broadcastSrefType = PCF::ShapedRefType::get(
        b.getContext(), {1}, b.getI32Type(), broadcastScope);
    Value broadcastSref =
        PCF::AllocOp::create(b, loc, broadcastSrefType, ValueRange{});

    auto outerIf = scf::IfOp::create(
        b, loc, TypeRange{}, isSplit, /*withElseRegion=*/true);

    // ── THEN BRANCH: split path ──
    {
      OpBuilder thenB(b.getContext());
      thenB.setInsertionPointToStart(
          &outerIf.getThenRegion().front());

      IndexType indexType = thenB.getIndexType();

      // Phase 1 (scratch write) is now inside the producer body via scf.if
      // guards injected in Step 2b.  Phase 2 starts here.

      // Position before the scf.yield terminator of the then block.
      thenB.setInsertionPoint(outerIf.getThenRegion().front().getTerminator());

      // ── Phase 2: Barrier + atomic + broadcast ──
      // Global memory fence (release): ensure scratch writes are
      // visible before the atomic counter increment.
      gpu::BarrierOp::create(thenB, loc, gpu::AddressSpace::Global);

      // Get memref view of broadcast dword for store/load.
      auto broadcastMemrefType = MemRefType::get(
          {1}, thenB.getI32Type(),
          StridedLayoutAttr::get(thenB.getContext(),
                                 ShapedType::kDynamic,
                                 {ShapedType::kDynamic}));
      Value broadcastMemref = PCF::GetMemrefOp::create(
          thenB, loc, broadcastMemrefType, broadcastSref,
          SmallVector<OpFoldResult>{thenB.getIndexAttr(0)},
          SmallVector<OpFoldResult>{thenB.getIndexAttr(1)},
          SmallVector<OpFoldResult>{thenB.getIndexAttr(1)});

      // Thread-0 does the atomic and stores result to broadcast dword.
      Value threadId = gpu::ThreadIdOp::create(
          thenB, loc, indexType, gpu::Dimension::x);
      Value isThread0 = arith::CmpIOp::create(
          thenB, loc, arith::CmpIPredicate::eq, threadId, c0);

      auto thread0If = scf::IfOp::create(
          thenB, loc, TypeRange{}, isThread0,
          /*withElseRegion=*/false);

      {
        OpBuilder::InsertionGuard thread0Guard(thenB);
        thenB.setInsertionPointToStart(
            &thread0If.getThenRegion().front());

        // Get memref view of counter for atomic operation.
        int64_t counterSize = counterType.getShape()[0];
        auto counterMemrefType = MemRefType::get(
            {counterSize}, thenB.getI32Type(),
            StridedLayoutAttr::get(thenB.getContext(),
                                   ShapedType::kDynamic,
                                   {ShapedType::kDynamic}));

        SmallVector<OpFoldResult> counterOffsets = {
            thenB.getIndexAttr(0)};
        SmallVector<OpFoldResult> counterSizes = {
            thenB.getIndexAttr(counterSize)};
        SmallVector<OpFoldResult> counterStrides = {
            thenB.getIndexAttr(1)};

        Value counterMemref = PCF::GetMemrefOp::create(
            thenB, loc, counterMemrefType, counter, counterOffsets,
            counterSizes, counterStrides);

        // Atomic increment using generic_atomic_rmw (cmpxchg loop).
        // We must NOT use memref.atomic_rmw here because the LLVM
        // AMDGPU backend's AtomicOptimizer pass incorrectly converts
        // it to a wavefront-level collective atomic.
        Value c1I32 = arith::ConstantOp::create(
            thenB, loc, thenB.getI32IntegerAttr(1));
        auto genericAtomic = memref::GenericAtomicRMWOp::create(
            thenB, loc, counterMemref, ValueRange{counterIndex});
        {
          OpBuilder::InsertionGuard atomicGuard(thenB);
          Block *atomicBody = genericAtomic.getBody();
          thenB.setInsertionPointToStart(atomicBody);
          Value currentVal = atomicBody->getArgument(0);
          Value newVal = arith::AddIOp::create(thenB, loc, currentVal, c1I32);
          memref::AtomicYieldOp::create(thenB, loc, newVal);
        }
        Value oldVal = genericAtomic.getResult();

        // Apply modulo for counter reuse across dispatch iterations.
        Value numInGroupI32 = arith::IndexCastOp::create(
            thenB, loc, thenB.getI32Type(), numInGroup);
        Value oldValMod =
            arith::RemUIOp::create(thenB, loc, oldVal, numInGroupI32);

        // Store modular result to broadcast dword.
        memref::StoreOp::create(thenB, loc, oldValMod, broadcastMemref,
                                ValueRange{c0});
      }

      // Barrier: sync all threads after thread-0 writes broadcast dword
      // AND invalidate global caches for scratch visibility (acquire).
      {
        SmallVector<Attribute> fenceSpaces = {
            gpu::AddressSpaceAttr::get(thenB.getContext(),
                                       gpu::AddressSpace::Workgroup),
            gpu::AddressSpaceAttr::get(thenB.getContext(),
                                       gpu::AddressSpace::Global)};
        gpu::BarrierOp::create(thenB, loc, thenB.getArrayAttr(fenceSpaces));
      }

      // ALL threads load the broadcast result and check last-contributor.
      Value broadcastedOld =
          memref::LoadOp::create(thenB, loc, broadcastMemref, ValueRange{c0});
      Value oldIdx = arith::IndexCastOp::create(
          thenB, loc, indexType, broadcastedOld);
      Value expectedLast =
          arith::SubIOp::create(thenB, loc, numInGroup, c1);
      Value isLast = arith::CmpIOp::create(
          thenB, loc, arith::CmpIPredicate::eq, oldIdx, expectedLast);

      // ── Phase 3: Distributed recombine (if last contributor) ──
      auto lastContribIf = scf::IfOp::create(
          thenB, loc, TypeRange{}, isLast,
          /*withElseRegion=*/false);

      {
        thenB.setInsertionPointToStart(
            &lastContribIf.getThenRegion().front());

        DistributedScopeNest recombineNest =
            buildDistributedScopeNest(thenB, loc);
        PerThreadChunkLoop recombineLoop =
            buildPerThreadChunkLoop(thenB, loc, recombineNest.tid,
                                    recombineNest.totalWorkers, tileShape);

        // Read from scratch at contributor 0.
        Value initRow = arith::AddIOp::create(thenB, loc, tileBaseOffset,
                                              recombineLoop.row);
        SmallVector<OpFoldResult> elemReadOffsets = {initRow,
                                                     recombineLoop.col};
        SmallVector<OpFoldResult> elemReadSizes = {thenB.getIndexAttr(1),
                                                   thenB.getIndexAttr(1)};
        SmallVector<OpFoldResult> elemReadStrides = {thenB.getIndexAttr(1),
                                                     thenB.getIndexAttr(1)};
        auto elemTensorType = RankedTensorType::get({1, 1}, elemType);
        Value accInit = PCF::ReadSliceOp::create(
            thenB, loc, elemTensorType, scratch, elemReadOffsets, elemReadSizes,
            elemReadStrides);

        // Accumulation loop: for each contributor i in [1, numInGroup).
        auto accLoop = scf::ForOp::create(thenB, loc, /*lb=*/c1,
                                          /*ub=*/numInGroup, /*step=*/c1,
                                          /*iterArgs=*/ValueRange{accInit});

        {
          OpBuilder::InsertionGuard accGuard(thenB);
          thenB.setInsertionPointToStart(accLoop.getBody());
          Value loopIdx = accLoop.getInductionVar();
          Value loopAcc = accLoop.getRegionIterArg(0);

          // Compute scratch offset for contributor i.
          Value contribOffset =
              arith::MulIOp::create(thenB, loc, loopIdx, tileHeight);
          Value contribBase =
              arith::AddIOp::create(thenB, loc, tileBaseOffset, contribOffset);
          Value contribRow =
              arith::AddIOp::create(thenB, loc, contribBase, recombineLoop.row);
          SmallVector<OpFoldResult> contribReadOffsets = {contribRow,
                                                          recombineLoop.col};

          Value partialElem = PCF::ReadSliceOp::create(
              thenB, loc, elemTensorType, scratch, contribReadOffsets,
              elemReadSizes, elemReadStrides);

          // Apply combiner element-wise.
          // Extract scalar elements from the 1x1 tensors.
          Value cIdx0 = arith::ConstantIndexOp::create(thenB, loc, 0);
          Value accScalar = tensor::ExtractOp::create(thenB, loc, loopAcc,
                                                      ValueRange{cIdx0, cIdx0});
          Value partialScalar = tensor::ExtractOp::create(
              thenB, loc, partialElem, ValueRange{cIdx0, cIdx0});

          // Clone the combiner region inline.
          Block &combinerBlock = recombineOp.getCombiner().front();
          IRMapping combinerMapping;
          combinerMapping.map(combinerBlock.getArgument(0), accScalar);
          combinerMapping.map(combinerBlock.getArgument(1), partialScalar);
          for (Operation &op : combinerBlock.without_terminator()) {
            thenB.clone(op, combinerMapping);
          }
          auto combinerYield =
              cast<PCF::YieldOp>(combinerBlock.getTerminator());
          Value combined =
              combinerMapping.lookupOrDefault(combinerYield.getOperand(0));

          // Re-insert into the iter arg tensor (not a fresh empty) for
          // bufferization equivalence.
          Value result = tensor::InsertOp::create(thenB, loc, combined, loopAcc,
                                                  ValueRange{cIdx0, cIdx0});
          scf::YieldOp::create(thenB, loc, ValueRange{result});
        }

        Value accumulated = accLoop.getResult(0);

        // Write accumulated element to output dest.
        Value destRow = arith::AddIOp::create(
            thenB, loc,
            getValueOrCreateConstantIndexOp(thenB, loc, destOffsets[0]),
            recombineLoop.row);
        Value destCol = arith::AddIOp::create(
            thenB, loc,
            getValueOrCreateConstantIndexOp(thenB, loc, destOffsets[1]),
            recombineLoop.col);
        SmallVector<OpFoldResult> destWriteOffsets = {destRow, destCol};
        SmallVector<OpFoldResult> destWriteSizes = {thenB.getIndexAttr(1),
                                                    thenB.getIndexAttr(1)};
        SmallVector<OpFoldResult> destWriteStrides = {thenB.getIndexAttr(1),
                                                      thenB.getIndexAttr(1)};
        PCF::WriteSliceOp::create(thenB, loc, accumulated, dest,
                                  destWriteOffsets, destWriteSizes,
                                  destWriteStrides);

        // Terminate scope nests.
        thenB.setInsertionPointAfter(recombineLoop.loop);
        PCF::ReturnOp::create(thenB, loc);
        thenB.setInsertionPointAfter(recombineNest.innerGeneric);
        PCF::ReturnOp::create(thenB, loc);
      }
    }

    // ── ELSE BRANCH: non-split direct writeback ──
    {
      OpBuilder elseB(b.getContext());
      elseB.setInsertionPointToStart(
          &outerIf.getElseRegion().front());

      DistributedScopeNest writebackNest =
          buildDistributedScopeNest(elseB, loc);
      PerThreadChunkLoop writebackLoop = buildPerThreadChunkLoop(
          elseB, loc, writebackNest.tid, writebackNest.totalWorkers, tileShape);

      // Extract element from partialTile.
      auto wbElemType = RankedTensorType::get({1, 1}, elemType);
      SmallVector<OpFoldResult> wbExtractOffsets = {writebackLoop.row,
                                                    writebackLoop.col};
      SmallVector<OpFoldResult> wbSizes = {elseB.getIndexAttr(1),
                                           elseB.getIndexAttr(1)};
      SmallVector<OpFoldResult> wbStrides = {elseB.getIndexAttr(1),
                                             elseB.getIndexAttr(1)};
      Value wbElem =
          tensor::ExtractSliceOp::create(elseB, loc, wbElemType, partialTile,
                                         wbExtractOffsets, wbSizes, wbStrides);

      // Write to output dest at [destOffset + row, destOffset + col].
      Value wbDestRow = arith::AddIOp::create(
          elseB, loc,
          getValueOrCreateConstantIndexOp(elseB, loc, destOffsets[0]),
          writebackLoop.row);
      Value wbDestCol = arith::AddIOp::create(
          elseB, loc,
          getValueOrCreateConstantIndexOp(elseB, loc, destOffsets[1]),
          writebackLoop.col);
      SmallVector<OpFoldResult> wbDestOffsets = {wbDestRow, wbDestCol};
      PCF::WriteSliceOp::create(elseB, loc, wbElem, dest, wbDestOffsets,
                                wbSizes, wbStrides);

      // Terminate scope nests.
      elseB.setInsertionPointAfter(writebackLoop.loop);
      PCF::ReturnOp::create(elseB, loc);
      elseB.setInsertionPointAfter(writebackNest.innerGeneric);
      PCF::ReturnOp::create(elseB, loc);
    }

    // Step 4: Final workgroup barrier.
    b.setInsertionPointAfter(outerIf);
    gpu::BarrierOp::create(b, loc, gpu::AddressSpace::Workgroup);

    // Erase the original recombine op.
    rewriter.eraseOp(recombineOp);

    return success();
  }
};

void FuseConsumersPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateConsumerFusionPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }

  // Verify that no unrealized conversion casts remain.
  if (getOperation()->walk(verifyOperationLegality).wasInterrupted()) {
    return signalPassFailure();
  }
}

//===---------------------------------------------------------------------===//
// Consumer fusion impls
//===---------------------------------------------------------------------===//

template <typename OpTy>
static LogicalResult
lookupProducerSlices(OpResult result,
                     SmallVectorImpl<PCF::WriteSliceOp> &slices) {
  OpTy owner = cast<OpTy>(result.getOwner());
  Value tiedArg = owner.getRegionRefArgs()[result.getResultNumber()];
  auto srefType = cast<PCF::ShapedRefType>(tiedArg.getType());
  if (!srefType.isReturnOnlySync()) {
    return failure();
  }
  for (auto user : tiedArg.getUsers()) {
    // We can ignore memory reads.
    if (isa<PCF::ReadSliceOp>(user)) {
      continue;
    }
    auto sliceOp = dyn_cast<PCF::WriteSliceOp>(user);
    // TODO: Support vector operands.
    if (!sliceOp || !isa<RankedTensorType>(sliceOp.getSourceType()) ||
        !sliceOp.hasUnitStride()) {
      return failure();
    }
    slices.push_back(sliceOp);
  }
  return success();
}

// Two cases: one operand with multiple producer slices or multiple operands
// with a single producer slice per operand.
// Currently multiple operands <-> multiple producers is unsupported.
template <typename OpTy>
static LogicalResult
matchTilableConsumerImpl(RewriterBase &rewriter, OpTy producerOp,
                         TilingInterface target, ConsumerFusionParams &params) {
  // To create a loop result we need either an initializer or a shape. This
  // can come from either ReifyRankedShapedTypeOpInterface or DPS. If the
  // operand being fused along is itself a destination we get the shape via
  // passthrough with the producer's init/shape.
  if (!isa<ReifyRankedShapedTypeOpInterface, DestinationStyleOpInterface>(
          *target)) {
    return rewriter.notifyMatchFailure(
        target, "unsupported non-reify result shapes or dps op");
  }

  SmallVector<unsigned> &targetOperands = params.operands;
  SetVector<unsigned> &targetResults = params.results;
  SmallVector<PCF::WriteSliceOp> &slices = params.slices;
  assert(targetOperands.empty() && "unexpected non-empty operand list");
  assert(targetResults.empty() && "unexpected non-empty result set");
  assert(slices.empty() && "unexpected non-empty slice list");
  mlir::DominanceInfo dominanceInfo(producerOp->getParentOp());
  // First collect the set of operands/results fused along. Additionally verify
  // dominance for other operands. Look through guarantee_value to find the
  // actual producer result.
  for (OpOperand &operand : target->getOpOperands()) {
    Value val = operand.get();
    // Look through guarantee_value to find the actual producer result.
    if (auto gv = val.getDefiningOp<PCF::GuaranteeValueOp>())
      val = gv.getSource();
    auto opResult = dyn_cast<OpResult>(val);
    if (opResult && opResult.getOwner() == producerOp) {
      targetOperands.push_back(operand.getOperandNumber());
      targetResults.insert(opResult.getResultNumber());
    } else {
      if (!dominanceInfo.dominates(operand.get(), producerOp)) {
        return rewriter.notifyMatchFailure(
            target, "unable to fuse due to operand dominance");
      }
    }
  }

  if (targetOperands.empty()) {
    return rewriter.notifyMatchFailure(target, "no operands to fuse along");
  }

  // This dominance check can be expensive in the most general case, however
  // the majority of tilable ops have no or small regions so in practice this
  // isn't so bad.
  WalkResult res = target->walk([&](Operation *containedOp) -> WalkResult {
    if (containedOp == target) {
      return WalkResult::advance();
    }
    bool dominates = llvm::all_of(containedOp->getOperands(), [&](Value v) {
      auto bbArg = dyn_cast<BlockArgument>(v);
      // Check if the tilable op owns the producer of this operand or if the
      // producer dominates the loop we're fusing into.
      Operation *owner = bbArg ? bbArg.getParentRegion()->getParentOp()
                               : cast<OpResult>(v).getOwner();
      return target->isAncestor(owner) ||
             dominanceInfo.dominates(v, producerOp);
    });
    return dominates ? WalkResult::advance() : WalkResult::interrupt();
  });
  if (res.wasInterrupted()) {
    return rewriter.notifyMatchFailure(
        target, "target region users don't dominate producer");
  }

  // Compute control flow context between the consumer and the producer.
  // If the consumer uses a guarantee_value, only the control flow between
  // the consumer and the guarantee_value matters (already-mirrored CF is
  // skipped).
  Operation *scopeOp = producerOp;
  for (unsigned operandIdx : targetOperands) {
    Value val = target->getOperand(operandIdx);
    if (auto gv = val.getDefiningOp<PCF::GuaranteeValueOp>()) {
      scopeOp = gv;
      break;
    }
  }
  params.controlFlowContext = computeControlFlowContext(target, scopeOp);

  // Verify that the conditions/bounds of enclosing control flow dominate the
  // producer so they can be referenced inside the producer's region.
  for (const ControlFlowEntry &entry : params.controlFlowContext) {
    if (auto ifOp = dyn_cast<scf::IfOp>(entry.op)) {
      if (!dominanceInfo.dominates(ifOp.getCondition(), producerOp)) {
        return rewriter.notifyMatchFailure(
            target,
            "control flow condition does not dominate producer");
      }
    }
  }

  // Reject fusion through unsupported CF (computeControlFlowContext returns
  // empty for scf.for/while).
  if (target->getBlock() != scopeOp->getBlock() &&
      params.controlFlowContext.empty()) {
    return rewriter.notifyMatchFailure(
        target, "unsupported control flow between consumer and producer");
  }

  // Case 1: Single result to fuse along.
  if (targetResults.size() == 1) {
    unsigned resultIndex = *targetResults.begin();
    if (failed(lookupProducerSlices<OpTy>(producerOp->getOpResult(resultIndex),
                                          slices))) {
      return rewriter.notifyMatchFailure(producerOp,
                                         "non write slice producer");
    }

    for (PCF::WriteSliceOp writeSlice : slices) {
      SmallVector<SmallVector<OpFoldResult>> allOffsets(
          targetOperands.size(), writeSlice.getMixedOffsets());
      SmallVector<SmallVector<OpFoldResult>> allSizes(
          targetOperands.size(), writeSlice.getMixedSizes());
      if (!target.isOpFusableWithProducerSlices(targetOperands, allOffsets,
                                                allSizes)) {
        return rewriter.notifyMatchFailure(
            target, "unsupported fusion for single producer");
      }
    }
  } else {
    // Case 2: Multiple results. We must find the most dominated slice to use
    // as the insertion point in this case.
    int64_t leader = -1;
    for (auto operandIndex : targetOperands) {
      int64_t currNumSlices = slices.size();
      auto opResult = cast<OpResult>(target->getOperand(operandIndex));
      if (failed(lookupProducerSlices<OpTy>(opResult, slices))) {
        return rewriter.notifyMatchFailure(producerOp,
                                           "non write slice producer");
      }
      if (slices.size() - currNumSlices != 1) {
        return rewriter.notifyMatchFailure(
            target,
            "multiple operand fusion with multiple producers unsupported");
      }

      if (leader < 0) {
        leader = currNumSlices;
      } else {
        PCF::WriteSliceOp currLeader = slices[leader];
        PCF::WriteSliceOp next = slices.back();
        if (next != currLeader) {
          // Match for all writes within the same block to guarantee all
          // required result values are produced "together" (i.e. all required
          // operand slices are written at the same time by the same thread per
          // the *control flow* rather than just the slice offsets/shape).
          if (next->getBlock() != currLeader->getBlock()) {
            return rewriter.notifyMatchFailure(
                target, "unsupported different block insertion points");
          }
          if (dominanceInfo.dominates(currLeader, next)) {
            leader = currNumSlices;
          } else if (!dominanceInfo.dominates(next, currLeader)) {
            return rewriter.notifyMatchFailure(
                target, "could not find single insertion point for multiple "
                        "producer slices");
          }
        }
      }
    }
    SmallVector<SmallVector<OpFoldResult>> allOffsets = llvm::map_to_vector(
        slices, [](PCF::WriteSliceOp op) { return op.getMixedOffsets(); });
    SmallVector<SmallVector<OpFoldResult>> allSizes = llvm::map_to_vector(
        slices, [](PCF::WriteSliceOp op) { return op.getMixedSizes(); });
    if (!target.isOpFusableWithProducerSlices(targetOperands, allOffsets,
                                              allSizes)) {
      return rewriter.notifyMatchFailure(
          target, "unsupported fusion for multiple producer slices");
    }
    if (leader != 0) {
      // Swap the most dominant slice to the beginning.
      std::swap(slices[leader], slices[0]);
      std::swap(targetOperands[leader], targetOperands[0]);
    }
  }
  return success();
}

/// Fuses tilable op |target| into the list of |slices|, one per operand.
/// For example, if |operands| was [0, 2, 4], then the 3 entries in |slices|
/// correspond to the inputs for operands 0, 2, and 4 respectively.
/// The most dominant slice (i.e. the insertion point for the tiled + fused op)
/// is always assumed to be slices[0] unless |insertionOverride| is set.
/// |newResultDests| is the list of new DPS destinations for the tiled op to
/// write to.
static void fuseIntoWriteSlices(
    RewriterBase &rewriter, TilingInterface target,
    ArrayRef<unsigned> operands, MutableArrayRef<PCF::WriteSliceOp> slices,
    ValueRange newResultDests,
    std::optional<OpBuilder::InsertPoint> insertionOverride = std::nullopt) {
  assert(operands.size() == slices.size() &&
         "expected same number of operands and slices to fuse into");
  OpBuilder::InsertionGuard g(rewriter);
  Location loc = target.getLoc();

  // The contract with the matcher is that the first slice in the list is the
  // most dominant and thus the insertion point for the fused op, unless an
  // override is provided (e.g., inside mirrored control flow).
  if (insertionOverride) {
    rewriter.restoreInsertionPoint(*insertionOverride);
  } else {
    rewriter.setInsertionPoint(slices.front());
  }

  // Clone the op and replace all operands being fused along with unrealized
  // conversion casts from the distributed producer tile to the undistributed
  // tile. We will forward the input to the unrealized conversion cast directly
  // to the tiled op once finished.
  auto clonedOp = cast<TilingInterface>(rewriter.clone(*target));
  SmallVector<UnrealizedConversionCastOp> unrealizedConversions;
  for (auto [operand, slice] : llvm::zip_equal(operands, slices)) {
    OpOperand &currOperand = clonedOp->getOpOperand(operand);
    Type undistributedType = currOperand.get().getType();
    auto conversion = UnrealizedConversionCastOp::create(
        rewriter, loc, undistributedType, slice.getSource());
    currOperand.assign(conversion.getResult(0));
    unrealizedConversions.push_back(conversion);
  }

  // Get the iteration domain in terms of the operand tiles. This is required
  // to fetch the result tile positions. This and all subsequent interface
  // queries must succeed per the matcher check.
  SmallVector<SmallVector<OpFoldResult>> allOffsets = llvm::map_to_vector(
      slices, [](PCF::WriteSliceOp op) { return op.getMixedOffsets(); });
  SmallVector<SmallVector<OpFoldResult>> allSizes = llvm::map_to_vector(
      slices, [](PCF::WriteSliceOp op) { return op.getMixedSizes(); });
  SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;
  [[maybe_unused]] LogicalResult res =
      clonedOp.getIterationDomainTileFromOperandTiles(
          rewriter, operands, allOffsets, allSizes, iterDomainOffsets,
          iterDomainSizes);
  assert(succeeded(res) && "unexpected iteration domain fetch failed");

  unsigned numResults = clonedOp->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsets(numResults);
  SmallVector<SmallVector<OpFoldResult>> resultSizes(numResults);
  for (auto [idx, v] : llvm::enumerate(clonedOp->getResults())) {
    [[maybe_unused]] LogicalResult res = clonedOp.getResultTilePosition(
        rewriter, idx, iterDomainOffsets, iterDomainSizes, resultOffsets[idx],
        resultSizes[idx]);
    assert(succeeded(res) && "Unexpected failure to get result tile position");
  }

  // Tile the cloned op based on the slice shapes.
  FailureOr<TilingResult> tiledResult =
      clonedOp.getTiledImplementationFromOperandTiles(rewriter, operands,
                                                      allOffsets, allSizes);
  assert(succeeded(tiledResult) && "unexpected tiling failure");

  // Create write_slice ops updating the destination for each result.
  OpFoldResult one = rewriter.getIndexAttr(1);
  for (auto [offsets, sizes, replacement, dest] :
       llvm::zip_equal(resultOffsets, resultSizes, tiledResult->tiledValues,
                       newResultDests)) {
    SmallVector<OpFoldResult> strides(offsets.size(), one);
    PCF::WriteSliceOp::create(rewriter, loc, replacement, dest, offsets, sizes,
                              strides);
  }

  // Finally forward the sources of the unrealized conversion casts past each
  // `tensor.extract_slice` consumer. If this fails for any reason we leave the
  // unrealized cast in and fail later for better diagnostics as it is
  // unrecoverable.
  for (auto unrealizedCast : unrealizedConversions) {
    SmallVector<Operation *> users(unrealizedCast->getUsers());
    for (Operation *user : users) {
      if (auto extract = dyn_cast<tensor::ExtractSliceOp>(user)) {
        if (extract.getResultType() ==
            unrealizedCast->getOperandTypes().front()) {
          rewriter.replaceOp(extract, unrealizedCast.getOperand(0));
        }
      }
    }
    if (unrealizedCast->use_empty()) {
      rewriter.eraseOp(unrealizedCast);
    }
  }
}

static void addSrefArguments(MLIRContext *context, Location loc,
                             int64_t newArgIndex, Block *entryBlock,
                             TypeRange resultTypes,
                             PCF::ScopeAttrInterface scope) {
  // Add the new region arguments with parent sync scope.
  Attribute syncScope = PCF::SyncOnReturnAttr::get(context);
  for (auto resultType : resultTypes) {
    auto tensorType = cast<RankedTensorType>(resultType);
    auto newSrefType =
        PCF::ShapedRefType::get(context, tensorType.getShape(),
                                tensorType.getElementType(), scope, syncScope);
    entryBlock->insertArgument(newArgIndex, newSrefType, loc);
    ++newArgIndex;
  }
}

static PCF::LoopOp addResults(RewriterBase &rewriter, PCF::LoopOp loopOp,
                              ArrayRef<bool> isTied, ArrayRef<Value> tiedArgs,
                              ArrayRef<Value> dynamicSizes,
                              TypeRange resultTypes) {

  // Append the parameters for the new results to the existing lists.
  SmallVector<Type> newResultTypes(loopOp->getResultTypes());
  newResultTypes.append(resultTypes.begin(), resultTypes.end());
  SmallVector<bool> newIsTied(loopOp.getIsTied());
  newIsTied.append(isTied.begin(), isTied.end());
  SmallVector<Value> newDynamicSizes(loopOp.getDynamicSizes());
  newDynamicSizes.append(dynamicSizes.begin(), dynamicSizes.end());
  SmallVector<Value> newTiedArgs(loopOp.getInits());
  newTiedArgs.append(tiedArgs.begin(), tiedArgs.end());

  int64_t numOriginalResults = loopOp->getNumResults();

  // Get the index of the last region ref arg before moving the body over.
  // + 1 because we want the new args to go at the end.
  int64_t newArgIndex = loopOp.getRegionRefArgs().back().getArgNumber() + 1;

  auto newLoopOp =
      PCF::LoopOp::create(rewriter, loopOp.getLoc(), newResultTypes,
                          loopOp.getScope(), loopOp.getCount(), newTiedArgs,
                          newDynamicSizes, newIsTied, loopOp.getSyncOnReturn());
  newLoopOp.getRegion().takeBody(loopOp.getRegion());

  // Add the new region arguments with parent sync scope.
  addSrefArguments(rewriter.getContext(), loopOp.getLoc(), newArgIndex,
                   newLoopOp.getBody(), resultTypes, loopOp.getScope());

  rewriter.replaceOp(loopOp,
                     newLoopOp->getResults().take_front(numOriginalResults));
  return newLoopOp;
}

static PCF::GenericOp
addResults(RewriterBase &rewriter, PCF::GenericOp genericOp,
           ArrayRef<bool> isTied, ArrayRef<Value> tiedArgs,
           ArrayRef<Value> dynamicSizes, TypeRange resultTypes) {

  // Append the parameters for the new results to the existing lists.
  SmallVector<Type> newResultTypes(genericOp->getResultTypes());
  newResultTypes.append(resultTypes.begin(), resultTypes.end());
  SmallVector<bool> newIsTied(genericOp.getIsTied());
  newIsTied.append(isTied.begin(), isTied.end());
  SmallVector<Value> newDynamicSizes(genericOp.getDynamicSizes());
  newDynamicSizes.append(dynamicSizes.begin(), dynamicSizes.end());
  SmallVector<Value> newTiedArgs(genericOp.getInits());
  newTiedArgs.append(tiedArgs.begin(), tiedArgs.end());

  int64_t numOriginalResults = genericOp->getNumResults();

  // Get the index of the last region ref arg before moving the body over.
  // + 1 because we want the new args to go at the end.
  int64_t newArgIndex = genericOp.getRegionRefArgs().back().getArgNumber() + 1;

  auto newGenericOp = PCF::GenericOp::create(
      rewriter, genericOp.getLoc(), newResultTypes, genericOp.getScope(),
      newTiedArgs, newDynamicSizes, newIsTied, genericOp.getNumIterators(),
      genericOp.getSyncOnReturn());
  newGenericOp.getRegion().takeBody(genericOp.getRegion());
  newGenericOp.getInitializer().takeBody(genericOp.getInitializer());
  newGenericOp.setNumLeadingArgs(genericOp.getNumLeadingArgs());

  // Add the new region arguments with parent sync scope.
  addSrefArguments(rewriter.getContext(), genericOp.getLoc(), newArgIndex,
                   &newGenericOp.getRegion().front(), resultTypes,
                   genericOp.getScope());

  rewriter.replaceOp(genericOp,
                     newGenericOp->getResults().take_front(numOriginalResults));
  return newGenericOp;
}

template <typename OpTy>
static void fuseTilableConsumerImpl(RewriterBase &rewriter, OpTy producerOp,
                                    TilingInterface target,
                                    const ConsumerFusionParams &params) {
  assert(!params.results.empty() && "unexpected empty number of results");

  Location loc = target.getLoc();

  // Step 1. To compute the set of new result shapes, we need to either reify
  // result shapes or get it from a destination a la DPS.
  SmallVector<bool> isTied;
  SmallVector<Value> tiedArgs;
  SmallVector<Value> dynamicSizes;
  SmallVector<Type> resultTypes(target->getResultTypes());

  auto getInitOrCreateEmpty = [&](int64_t resultNumber) -> Value {
    if (OpOperand *tiedInit = producerOp.getTiedInit(resultNumber)) {
      return tiedInit->get();
    }
    return mlir::tensor::EmptyOp::create(
        rewriter, loc, producerOp.getResultType(resultNumber),
        producerOp.getResultDims(resultNumber));
  };

  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(*target)) {
    for (Value init : dpsOp.getDpsInits()) {
      if (init.getDefiningOp() == producerOp) {
        // There are two options if we are fusing along an init operand:
        //  1. Create a new empty init with the same shape.
        //  2. Use the init of the producer op.
        //
        // This opts for the latter because when fusing we'll replace the actual
        // operand with the thread-local version, so the code will still be
        // correct and it's the closest to the original intent of the
        // ops destination.
        auto result = cast<OpResult>(init);
        if (OpOperand *operand =
                producerOp.getTiedInit(result.getResultNumber())) {
          isTied.push_back(true);
          tiedArgs.push_back(operand->get());
        } else {
          // If there is no tied init, copy the dimensions over.
          isTied.push_back(false);
          ValueRange resultDims =
              producerOp.getResultDims(result.getResultNumber());
          dynamicSizes.append(resultDims.begin(), resultDims.end());
        }
      } else {
        // Otherwise we just use the init as the tied operand directly.
        isTied.push_back(true);
        tiedArgs.push_back(init);
      }
    }
  } else {
    // For reification ops, we need to construct the result dims in terms of
    // the producer's operands. To do this we replace all operands of |target|
    // coming from the producer with equivalently shaped inits/tensor.empty ops
    // and call reification on that. This is ostensibly a hack as there is no
    // formal guarantee that swapping out operands like this is valid per the
    // interface, however in practice this is valid for all known operations
    // that implement the interface today.

    SmallVector<Value> originalOperands;
    rewriter.setInsertionPoint(producerOp);
    for (unsigned operandIndex : params.operands) {
      Value operand = target->getOperand(operandIndex);
      originalOperands.push_back(operand);
      target->getOpOperand(operandIndex)
          .assign(
              getInitOrCreateEmpty(cast<OpResult>(operand).getResultNumber()));
    }

    SmallVector<SmallVector<OpFoldResult>> outputShapes;
    Operation *nextNode = target->getNextNode();
    Block *currBlock = target->getBlock();

    // Move the op immediately before the producer to get the SSA dominance
    // needed for the result shape dims.
    rewriter.moveOpBefore(target, producerOp);

    // If a fusable operation cannot reify its result shapes under any
    // circumstance, then it was not fusable and should not have been marked as
    // such.
    [[maybe_unused]] auto reifyOp =
        cast<ReifyRankedShapedTypeOpInterface>(*target);
    assert(succeeded(reifyOp.reifyResultShapes(rewriter, outputShapes)) &&
           "unexpected reify result shapes failed");
    if (nextNode) {
      rewriter.moveOpBefore(target, nextNode);
    } else {
      rewriter.moveOpAfter(target, &currBlock->back());
    }

    for (ArrayRef<OpFoldResult> outputShape : outputShapes) {
      dynamicSizes.append(
          llvm::map_to_vector(outputShape, [&](OpFoldResult ofr) {
            return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
          }));
    }
    isTied.append(outputShapes.size(), false);
  }

  OpTy newRegionOp = addResults(rewriter, producerOp, isTied, tiedArgs,
                                dynamicSizes, resultTypes);
  ValueRange newResultDests =
      newRegionOp.getRegionRefArgs().take_back(resultTypes.size());
  ValueRange replacements =
      newRegionOp.getResults().take_back(resultTypes.size());

  // Helper to create mirrored control flow before |anchorOp| and return
  // an insertion point inside the innermost mirrored branch.
  auto createMirroredControlFlow =
      [&](Operation *anchorOp) -> OpBuilder::InsertPoint {
    rewriter.setInsertionPoint(anchorOp);
    // Mirror outermost-first (context is innermost-first, so reverse).
    for (const ControlFlowEntry &entry :
         llvm::reverse(params.controlFlowContext)) {
      if (auto ifOp = dyn_cast<scf::IfOp>(entry.op)) {
        auto mirroredIf = scf::IfOp::create(
            rewriter, ifOp.getLoc(), /*resultTypes=*/TypeRange{},
            ifOp.getCondition(), /*withElseRegion=*/entry.regionIndex == 1);
        Block *targetBlock =
            &mirroredIf->getRegion(entry.regionIndex).front();
        rewriter.setInsertionPointToStart(targetBlock);
      }
    }
    return rewriter.saveInsertionPoint();
  };

  if (params.results.size() == 1) {
    // Single fusion vector means each slice is a different insertion point.
    for (PCF::WriteSliceOp slice : params.slices) {
      SmallVector<PCF::WriteSliceOp> slices(params.operands.size(), slice);
      if (!params.controlFlowContext.empty()) {
        // Create mirrored CF before the write_slice and fuse inside it.
        OpBuilder::InsertPoint innerPt =
            createMirroredControlFlow(slice);
        fuseIntoWriteSlices(rewriter, target, params.operands, slices,
                            newResultDests, innerPt);
      } else {
        fuseIntoWriteSlices(rewriter, target, params.operands, slices,
                            newResultDests);
      }
    }
  } else {
    if (!params.controlFlowContext.empty()) {
      PCF::WriteSliceOp leaderSlice = params.slices.front();
      OpBuilder::InsertPoint innerPt =
          createMirroredControlFlow(leaderSlice);
      fuseIntoWriteSlices(rewriter, target, params.operands, params.slices,
                          newResultDests, innerPt);
    } else {
      fuseIntoWriteSlices(rewriter, target, params.operands, params.slices,
                          newResultDests);
    }
  }

  // Replace the original fusion target. When control flow was mirrored,
  // emit guarantee_value to mark that the fused result is valid at this
  // control flow point. Otherwise, replace directly.
  if (!params.controlFlowContext.empty()) {
    rewriter.setInsertionPoint(target);
    SmallVector<Value> guaranteedValues;
    for (Value replacement : replacements) {
      Value gv = PCF::GuaranteeValueOp::create(rewriter, loc, replacement);
      guaranteedValues.push_back(gv);
    }
    rewriter.replaceOp(target, guaranteedValues);
  } else {
    rewriter.replaceOp(target, replacements);
  }
}

//===---------------------------------------------------------------------===//
// Extract slice consumer fusion
//===---------------------------------------------------------------------===//

// Compute clamped offsets and sizes of a write_slice to fit within extract
// bounds. This creates affine.min ops to clamp the sizes.
static void computeClampedOffsetsAndSizes(
    RewriterBase &rewriter, Location loc, ArrayRef<OpFoldResult> sliceOffsets,
    ArrayRef<OpFoldResult> sliceSizes, ArrayRef<OpFoldResult> extractSizes,
    SmallVectorImpl<OpFoldResult> &clampedOffsets,
    SmallVectorImpl<OpFoldResult> &clampedSizes) {
  // Clamp sizes to fit within extract bounds.
  for (auto [sliceOffset, sliceSize, extractSize] :
       llvm::zip_equal(sliceOffsets, sliceSizes, extractSizes)) {
    // Compute min(sliceOffset + sliceSize, extractSize) - sliceOffset
    // = min(sliceSize, extractSize - sliceOffset)
    AffineExpr d0, d1, d2;
    bindDims(rewriter.getContext(), d0, d1, d2);
    // d0 = sliceSize, d1 = extractSize, d2 = sliceOffset
    // clampedSize = min(d0, d1 - d2)
    AffineMap minMap =
        AffineMap::get(3, 0, {d0, d1 - d2}, rewriter.getContext());
    OpFoldResult clampedSize = affine::makeComposedFoldedAffineMin(
        rewriter, loc, minMap, {sliceSize, extractSize, sliceOffset});
    clampedOffsets.push_back(sliceOffset);
    clampedSizes.push_back(clampedSize);
  }
}

template <typename OpTy>
static LogicalResult
fuseExtractSliceIntoProducerImpl(RewriterBase &rewriter, OpTy producerOp,
                                 tensor::ExtractSliceOp extractSliceOp) {
  OpResult producerResult = cast<OpResult>(extractSliceOp.getSource());
  if (!producerResult.hasOneUse()) {
    return rewriter.notifyMatchFailure(producerOp,
                                       "producer result has multiple uses");
  }

  // Only zero offset extract_slice ops are supported.
  if (!llvm::all_of(extractSliceOp.getMixedOffsets(), [](OpFoldResult ofr) {
        return isConstantIntValue(ofr, 0);
      })) {
    return rewriter.notifyMatchFailure(extractSliceOp,
                                       "extract_slice has non-zero offsets");
  }

  // Rank-reducing extract_slice is not yet supported.
  auto extractedType = extractSliceOp.getType();
  auto producerResultType = cast<RankedTensorType>(producerResult.getType());
  if (extractedType.getRank() != producerResultType.getRank()) {
    return rewriter.notifyMatchFailure(
        extractSliceOp, "rank-reducing extract_slice not yet supported");
  }

  // Get the write_slice ops for this result.
  unsigned resultIdx = producerResult.getResultNumber();
  SmallVector<PCF::WriteSliceOp> slices;
  if (failed(lookupProducerSlices<OpTy>(producerResult, slices))) {
    return rewriter.notifyMatchFailure(producerOp,
                                       "failed to lookup producer slices");
  }

  if (slices.empty()) {
    return rewriter.notifyMatchFailure(producerOp, "no write_slice producers");
  }

  // Verify all write_slices have unit stride.
  // Only zero-offset extract_slice is supported (already checked above).
  SmallVector<OpFoldResult> extractSizes = extractSliceOp.getMixedSizes();
  for (PCF::WriteSliceOp slice : slices) {
    if (!slice.hasUnitStride()) {
      return rewriter.notifyMatchFailure(slice,
                                         "write_slice has non-unit stride");
    }
  }

  // Get the tied init for this result if it exists.
  OpOperand *tiedInit = producerOp.getTiedInit(resultIdx);
  Value initValue;
  if (tiedInit) {
    // Extract from the tied init.
    rewriter.setInsertionPoint(producerOp);
    initValue = tensor::ExtractSliceOp::create(
        rewriter, producerOp.getLoc(), tiedInit->get(),
        extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides());
  }

  // Compute new dynamic sizes for the result.
  SmallVector<Value> newDynamicSizes;
  int64_t dynamicDimIdx = 0;

  // First, copy dynamic sizes for results before this one.
  for (unsigned i = 0; i < resultIdx; ++i) {
    auto prevResultType =
        cast<RankedTensorType>(producerOp->getResult(i).getType());
    for (int64_t j = 0; j < prevResultType.getRank(); ++j) {
      if (prevResultType.isDynamicDim(j)) {
        newDynamicSizes.push_back(
            producerOp.getDynamicSizes()[dynamicDimIdx++]);
      }
    }
  }

  // Skip dynamic sizes for the current result (we'll add new ones).
  for (int64_t j = 0; j < producerResultType.getRank(); ++j) {
    if (producerResultType.isDynamicDim(j)) {
      dynamicDimIdx++;
    }
  }

  // Add new dynamic sizes from the extract_slice.
  rewriter.setInsertionPoint(producerOp);
  for (int64_t j = 0; j < extractedType.getRank(); ++j) {
    if (extractedType.isDynamicDim(j)) {
      OpFoldResult size = extractSliceOp.getMixedSizes()[j];
      newDynamicSizes.push_back(
          getValueOrCreateConstantIndexOp(rewriter, producerOp.getLoc(), size));
    }
  }

  // Copy remaining dynamic sizes.
  while (dynamicDimIdx <
         static_cast<int64_t>(producerOp.getDynamicSizes().size())) {
    newDynamicSizes.push_back(producerOp.getDynamicSizes()[dynamicDimIdx++]);
  }

  // Update tied init if present.
  SmallVector<Value> newInits(producerOp.getInits());
  if (tiedInit) {
    // Find the init index using the same logic as getTiedInit.
    int64_t initIdx =
        llvm::count(producerOp.getIsTied().take_front(resultIdx), true);
    newInits[initIdx] = initValue;
  }

  // Create the new result types.
  SmallVector<Type> newResultTypes;
  for (unsigned i = 0; i < producerOp->getNumResults(); ++i) {
    if (i == resultIdx) {
      newResultTypes.push_back(extractedType);
    } else {
      newResultTypes.push_back(producerOp->getResult(i).getType());
    }
  }

  // Clone the producer op with updated result types and dynamic sizes.
  OpTy newOp;
  if constexpr (std::is_same_v<OpTy, PCF::LoopOp>) {
    newOp = PCF::LoopOp::create(
        rewriter, producerOp.getLoc(), newResultTypes, producerOp.getScope(),
        producerOp.getCount(), newInits, newDynamicSizes,
        producerOp.getIsTied(), producerOp.getSyncOnReturn());
    newOp.getRegion().takeBody(producerOp.getRegion());
  } else {
    newOp = PCF::GenericOp::create(
        rewriter, producerOp.getLoc(), newResultTypes, producerOp.getScope(),
        newInits, newDynamicSizes, producerOp.getIsTied(),
        producerOp.getNumIterators(), producerOp.getSyncOnReturn());
    newOp.getRegion().takeBody(producerOp.getRegion());
    newOp.getInitializer().takeBody(producerOp.getInitializer());
    newOp.setNumLeadingArgs(producerOp.getNumLeadingArgs());
  }

  // Update the region ref arg type to match the new result size.
  Value newRefArg = newOp.getRegionRefArgs()[resultIdx];
  auto oldSrefType = cast<PCF::ShapedRefType>(newRefArg.getType());
  auto newSrefType = PCF::ShapedRefType::get(
      rewriter.getContext(), extractedType.getShape(),
      extractedType.getElementType(), producerOp.getScope(),
      oldSrefType.getSyncScope());
  newRefArg.setType(newSrefType);

  // Get the write_slices in the new op's region (they were moved with the
  // body).
  SmallVector<PCF::WriteSliceOp> newSlices;
  for (auto user : newRefArg.getUsers()) {
    if (auto writeSlice = dyn_cast<PCF::WriteSliceOp>(user)) {
      newSlices.push_back(writeSlice);
    }
  }

  // For each write_slice, clamp it to fit within the extracted bounds.
  for (PCF::WriteSliceOp slice : newSlices) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(slice);
    Location loc = slice.getLoc();

    // Compute clamped offsets and sizes.
    SmallVector<OpFoldResult> clampedOffsets, clampedSizes;
    computeClampedOffsetsAndSizes(rewriter, loc, slice.getMixedOffsets(),
                                  slice.getMixedSizes(), extractSizes,
                                  clampedOffsets, clampedSizes);

    Value source = slice.getSource();
    auto sourceType = cast<RankedTensorType>(source.getType());

    // Create extract_slice of source to get the clamped portion.
    SmallVector<OpFoldResult> sourceOffsets(sourceType.getRank(),
                                            rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sourceStrides(sourceType.getRank(),
                                            rewriter.getIndexAttr(1));
    auto clampedSource = tensor::ExtractSliceOp::create(
        rewriter, loc, source, sourceOffsets, clampedSizes, sourceStrides);

    // Create the new write_slice with clamped offsets/sizes.
    SmallVector<OpFoldResult> strides(clampedOffsets.size(),
                                      rewriter.getIndexAttr(1));
    PCF::WriteSliceOp::create(rewriter, loc, clampedSource, slice.getDest(),
                              clampedOffsets, clampedSizes, strides);

    rewriter.eraseOp(slice);
  }

  // Replace the producer and extract_slice.
  SmallVector<Value> replacements(newOp->getResults());
  rewriter.replaceOp(producerOp, replacements);
  rewriter.replaceOp(extractSliceOp, newOp->getResult(resultIdx));

  return success();
}

//===---------------------------------------------------------------------===//
// Collapse shape consumer fusion
//===---------------------------------------------------------------------===//

/// Computes collapsed offsets and sizes for a write_slice given reassociation
/// indices. For each reassociation group, inner dimensions must be fully
/// covered (offset=0, size=dim_size). The outer dimension of each group
/// provides the collapsed offset and size via linearization.
static void computeCollapsedOffsetsAndSizes(
    RewriterBase &rewriter, Location loc,
    ArrayRef<OpFoldResult> sliceOffsets, ArrayRef<OpFoldResult> sliceSizes,
    ArrayRef<int64_t> producerShape,
    ArrayRef<ReassociationIndices> reassociation,
    SmallVectorImpl<OpFoldResult> &collapsedOffsets,
    SmallVectorImpl<OpFoldResult> &collapsedSizes) {
  for (const ReassociationIndices &group : reassociation) {
    if (group.size() == 1) {
      // Singleton group: offset and size pass through.
      collapsedOffsets.push_back(sliceOffsets[group[0]]);
      collapsedSizes.push_back(sliceSizes[group[0]]);
      continue;
    }
    // Multi-dim group. Compute the product of inner dimension sizes.
    int64_t innerProduct = 1;
    for (size_t i = 1, e = group.size(); i < e; ++i) {
      innerProduct *= producerShape[group[i]];
    }

    // Collapsed offset = outer_offset * innerProduct.
    AffineExpr d0;
    bindDims(rewriter.getContext(), d0);
    AffineMap mulMap =
        AffineMap::get(1, 0, d0 * innerProduct, rewriter.getContext());
    OpFoldResult collapsedOffset = affine::makeComposedFoldedAffineApply(
        rewriter, loc, mulMap, {sliceOffsets[group[0]]});
    collapsedOffsets.push_back(collapsedOffset);

    // Collapsed size = outer_size * innerProduct.
    OpFoldResult collapsedSize = affine::makeComposedFoldedAffineApply(
        rewriter, loc, mulMap, {sliceSizes[group[0]]});
    collapsedSizes.push_back(collapsedSize);
  }
}

template <typename OpTy>
static LogicalResult
fuseCollapseShapeIntoProducerImpl(RewriterBase &rewriter, OpTy producerOp,
                                   tensor::CollapseShapeOp collapseOp) {
  OpResult producerResult = cast<OpResult>(collapseOp.getSrc());
  if (!producerResult.hasOneUse()) {
    return rewriter.notifyMatchFailure(producerOp,
                                       "producer result has multiple uses");
  }

  unsigned resultIdx = producerResult.getResultNumber();
  auto producerResultType = cast<RankedTensorType>(producerResult.getType());
  RankedTensorType collapsedType = collapseOp.getResultType();

  // Only static shapes are supported for now.
  if (!producerResultType.hasStaticShape()) {
    return rewriter.notifyMatchFailure(
        collapseOp, "dynamic producer result shape not supported");
  }

  SmallVector<ReassociationIndices> reassociation =
      collapseOp.getReassociationIndices();
  ArrayRef<int64_t> producerShape = producerResultType.getShape();

  // Get the write_slice ops for this result.
  SmallVector<PCF::WriteSliceOp> slices;
  if (failed(lookupProducerSlices<OpTy>(producerResult, slices))) {
    return rewriter.notifyMatchFailure(producerOp,
                                       "failed to lookup producer slices");
  }

  if (slices.empty()) {
    return rewriter.notifyMatchFailure(producerOp, "no write_slice producers");
  }

  // Verify that each write_slice can be collapsed.  Two cases are supported:
  // 1. Groups where all inner dims are fully covered (contiguous collapse).
  // 2. Two-dim groups with partial inner coverage (decomposed via loops).
  for (PCF::WriteSliceOp slice : slices) {
    if (!slice.hasUnitStride()) {
      return rewriter.notifyMatchFailure(slice,
                                         "write_slice has non-unit stride");
    }
    for (const ReassociationIndices &group : reassociation) {
      if (group.size() > 2) {
        return rewriter.notifyMatchFailure(
            slice, "groups with >2 dims not supported for collapse");
      }
    }
  }

  // Get the tied init for this result if it exists.
  OpOperand *tiedInit = producerOp.getTiedInit(resultIdx);
  Value initValue;
  if (tiedInit) {
    rewriter.setInsertionPoint(producerOp);
    initValue = tensor::CollapseShapeOp::create(
        rewriter, producerOp.getLoc(), collapsedType, tiedInit->get(),
        reassociation);
  }

  // Compute new dynamic sizes. Since we require static shapes, no dynamic
  // sizes are needed for the collapsed result dimension.
  SmallVector<Value> newDynamicSizes;
  int64_t dynamicDimIdx = 0;

  // Copy dynamic sizes for results before this one.
  for (unsigned i = 0; i < resultIdx; ++i) {
    auto prevResultType =
        cast<RankedTensorType>(producerOp->getResult(i).getType());
    for (int64_t j = 0; j < prevResultType.getRank(); ++j) {
      if (prevResultType.isDynamicDim(j)) {
        newDynamicSizes.push_back(
            producerOp.getDynamicSizes()[dynamicDimIdx++]);
      }
    }
  }

  // Skip dynamic sizes for the current result.
  for (int64_t j = 0; j < producerResultType.getRank(); ++j) {
    if (producerResultType.isDynamicDim(j)) {
      dynamicDimIdx++;
    }
  }

  // Add dynamic sizes for the collapsed type (none for static shapes).
  rewriter.setInsertionPoint(producerOp);
  for (int64_t j = 0; j < collapsedType.getRank(); ++j) {
    if (collapsedType.isDynamicDim(j)) {
      Value dimSize = arith::ConstantIndexOp::create(
          rewriter, producerOp.getLoc(), collapsedType.getDimSize(j));
      newDynamicSizes.push_back(dimSize);
    }
  }

  // Copy remaining dynamic sizes.
  while (dynamicDimIdx <
         static_cast<int64_t>(producerOp.getDynamicSizes().size())) {
    newDynamicSizes.push_back(producerOp.getDynamicSizes()[dynamicDimIdx++]);
  }

  // Update tied init if present.
  SmallVector<Value> newInits(producerOp.getInits());
  if (tiedInit) {
    int64_t initIdx =
        llvm::count(producerOp.getIsTied().take_front(resultIdx), true);
    newInits[initIdx] = initValue;
  }

  // Create the new result types with collapsed type for this result.
  SmallVector<Type> newResultTypes;
  for (unsigned i = 0, e = producerOp->getNumResults(); i < e; ++i) {
    if (i == resultIdx) {
      newResultTypes.push_back(collapsedType);
    } else {
      newResultTypes.push_back(producerOp->getResult(i).getType());
    }
  }

  // Clone the producer op with updated result type.
  OpTy newOp;
  if constexpr (std::is_same_v<OpTy, PCF::LoopOp>) {
    newOp = PCF::LoopOp::create(
        rewriter, producerOp.getLoc(), newResultTypes, producerOp.getScope(),
        producerOp.getCount(), newInits, newDynamicSizes,
        producerOp.getIsTied(), producerOp.getSyncOnReturn());
    newOp.getRegion().takeBody(producerOp.getRegion());
  } else {
    newOp = PCF::GenericOp::create(
        rewriter, producerOp.getLoc(), newResultTypes, producerOp.getScope(),
        newInits, newDynamicSizes, producerOp.getIsTied(),
        producerOp.getNumIterators(), producerOp.getSyncOnReturn());
    newOp.getRegion().takeBody(producerOp.getRegion());
    newOp.getInitializer().takeBody(producerOp.getInitializer());
    newOp.setNumLeadingArgs(producerOp.getNumLeadingArgs());
  }

  // Update the region ref arg type to match the collapsed shape.
  Value newRefArg = newOp.getRegionRefArgs()[resultIdx];
  auto oldSrefType = cast<PCF::ShapedRefType>(newRefArg.getType());
  auto newSrefType = PCF::ShapedRefType::get(
      rewriter.getContext(), collapsedType.getShape(),
      collapsedType.getElementType(), producerOp.getScope(),
      oldSrefType.getSyncScope());
  newRefArg.setType(newSrefType);

  // Get the write_slices in the new op's region.
  SmallVector<PCF::WriteSliceOp> newSlices;
  for (Operation *user : newRefArg.getUsers()) {
    if (auto writeSlice = dyn_cast<PCF::WriteSliceOp>(user)) {
      newSlices.push_back(writeSlice);
    }
  }

  // Transform each write_slice to use collapsed offsets/sizes.
  for (PCF::WriteSliceOp slice : newSlices) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(slice);
    Location loc = slice.getLoc();

    SmallVector<OpFoldResult> offsets = slice.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = slice.getMixedSizes();
    Value source = slice.getSource();

    // Check if any group needs loop decomposition (partial inner dims).
    bool needsDecomposition = false;
    for (const ReassociationIndices &group : reassociation) {
      if (group.size() <= 1) {
        continue;
      }
      for (size_t i = 1, e = group.size(); i < e; ++i) {
        int64_t dim = group[i];
        if (!isConstantIntValue(offsets[dim], 0) ||
            !isConstantIntValue(sizes[dim], producerShape[dim])) {
          needsDecomposition = true;
          break;
        }
      }
      if (needsDecomposition) {
        break;
      }
    }

    if (!needsDecomposition) {
      // Fast path: all inner dims fully covered, contiguous in collapsed
      // layout.  Use simple linearization.
      SmallVector<OpFoldResult> collapsedOffsets, collapsedSizes;
      computeCollapsedOffsetsAndSizes(rewriter, loc, offsets, sizes,
                                      producerShape, reassociation,
                                      collapsedOffsets, collapsedSizes);
      Value collapsedSource =
          tensor::CollapseShapeOp::create(rewriter, loc, source, reassociation);
      SmallVector<OpFoldResult> strides(collapsedOffsets.size(),
                                        rewriter.getIndexAttr(1));
      PCF::WriteSliceOp::create(rewriter, loc, collapsedSource, slice.getDest(),
                                collapsedOffsets, collapsedSizes, strides);
      rewriter.eraseOp(slice);
      continue;
    }

    // Decomposition path: for groups with partial inner coverage, create
    // scf.for loops over the outer dim range.  Each iteration writes a
    // contiguous inner slice at a linearized offset.
    //
    // Example: write tensor<2x4x4x1> into sref<4x16x8x16> at [o0,o1,o2,o3]
    // with reassociation [[0,1],[2,3]].  Inner dims 1,3 have partial coverage
    // (offsets != 0 or sizes != dimSize).  Decomposition:
    //   for d0 in [0, 2):
    //     for d2 in [0, 4):
    //       %inner = extract_slice source[d0, :, d2, :] → tensor<1x4x1x1>
    //       %col   = collapse_shape %inner [[0,1],[2,3]] → tensor<4x1>
    //       write %col into sref_2d[(o0+d0)*16+o1, (o2+d2)*16+o3] [4, 1]
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Per-group decomposition metadata.
    struct GroupInfo {
      int64_t outerDim = -1;
      int64_t innerDim = -1;
      int64_t innerDimSize = 0;
      bool needsLoop = false;
    };
    SmallVector<GroupInfo> groupInfos;
    for (const ReassociationIndices &group : reassociation) {
      GroupInfo info;
      if (group.size() == 1) {
        info.outerDim = group[0];
        groupInfos.push_back(info);
        continue;
      }
      info.outerDim = group[0];
      info.innerDim = group[1];
      info.innerDimSize = producerShape[group[1]];
      info.needsLoop =
          !isConstantIntValue(offsets[info.innerDim], 0) ||
          !isConstantIntValue(sizes[info.innerDim], info.innerDimSize);
      groupInfos.push_back(info);
    }

    // Create nested scf.for loops for groups needing decomposition.
    SmallVector<Value> loopIVs(groupInfos.size(), Value());
    for (auto [gi, info] : llvm::enumerate(groupInfos)) {
      if (!info.needsLoop) {
        continue;
      }
      Value ub =
          getValueOrCreateConstantIndexOp(rewriter, loc, sizes[info.outerDim]);
      scf::ForOp forOp = scf::ForOp::create(rewriter, loc, c0, ub, c1);
      loopIVs[gi] = forOp.getInductionVar();
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    // Build extract_slice from the source tensor.  For decomposed groups,
    // the outer dim is indexed by the loop IV with size 1; other dims keep
    // their original extent.
    auto sourceType = cast<RankedTensorType>(source.getType());
    int64_t sourceRank = sourceType.getRank();
    SmallVector<OpFoldResult> extractOffsets(sourceRank,
                                             rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> extractSizes;
    for (int64_t i = 0; i < sourceRank; ++i) {
      extractSizes.push_back(rewriter.getIndexAttr(sourceType.getDimSize(i)));
    }
    SmallVector<OpFoldResult> extractStrides(sourceRank,
                                             rewriter.getIndexAttr(1));
    for (auto [gi, info] : llvm::enumerate(groupInfos)) {
      if (!info.needsLoop) {
        continue;
      }
      extractOffsets[info.outerDim] = loopIVs[gi];
      extractSizes[info.outerDim] = rewriter.getIndexAttr(1);
    }

    SmallVector<int64_t> extractShape;
    for (OpFoldResult size : extractSizes) {
      std::optional<int64_t> val = getConstantIntValue(size);
      extractShape.push_back(val ? *val : ShapedType::kDynamic);
    }
    RankedTensorType extractType =
        RankedTensorType::get(extractShape, sourceType.getElementType());

    Value extracted = tensor::ExtractSliceOp::create(
        rewriter, loc, extractType, source, extractOffsets, extractSizes,
        extractStrides);

    // Collapse the extracted inner slice (e.g. tensor<1x4x1x1> → tensor<4x1>
    // for reassociation [[0,1],[2,3]]).
    Value collapsedExtracted = tensor::CollapseShapeOp::create(
        rewriter, loc, extracted, reassociation);

    // Compute collapsed write offsets and sizes.
    SmallVector<OpFoldResult> collapsedWriteOffsets;
    SmallVector<OpFoldResult> collapsedWriteSizes;
    for (auto [gi, info] : llvm::enumerate(groupInfos)) {
      if (info.innerDim < 0) {
        // Singleton group: pass through.
        collapsedWriteOffsets.push_back(offsets[info.outerDim]);
        collapsedWriteSizes.push_back(sizes[info.outerDim]);
        continue;
      }
      if (!info.needsLoop) {
        // Full inner coverage: collapsed = outer * innerDimSize.
        AffineExpr d0;
        bindDims(rewriter.getContext(), d0);
        AffineMap mulMap =
            AffineMap::get(1, 0, d0 * info.innerDimSize, rewriter.getContext());
        collapsedWriteOffsets.push_back(affine::makeComposedFoldedAffineApply(
            rewriter, loc, mulMap, {offsets[info.outerDim]}));
        collapsedWriteSizes.push_back(affine::makeComposedFoldedAffineApply(
            rewriter, loc, mulMap, {sizes[info.outerDim]}));
      } else {
        // Partial inner coverage:
        //   offset = (origOuter + loopIV) * innerDimSize + innerOffset.
        //   size = innerSize.
        AffineExpr d0, d1, d2;
        bindDims(rewriter.getContext(), d0, d1, d2);
        AffineMap offsetMap = AffineMap::get(
            3, 0, (d0 + d1) * info.innerDimSize + d2, rewriter.getContext());
        collapsedWriteOffsets.push_back(affine::makeComposedFoldedAffineApply(
            rewriter, loc, offsetMap,
            {offsets[info.outerDim], loopIVs[gi], offsets[info.innerDim]}));
        collapsedWriteSizes.push_back(sizes[info.innerDim]);
      }
    }

    SmallVector<OpFoldResult> collapsedStrides(collapsedWriteOffsets.size(),
                                               rewriter.getIndexAttr(1));
    PCF::WriteSliceOp::create(rewriter, loc, collapsedExtracted,
                              slice.getDest(), collapsedWriteOffsets,
                              collapsedWriteSizes, collapsedStrides);
    rewriter.eraseOp(slice);
  }

  // Replace the original producer and collapse_shape.
  SmallVector<Value> replacements(newOp->getResults());
  rewriter.replaceOp(producerOp, replacements);
  rewriter.replaceOp(collapseOp, newOp->getResult(resultIdx));

  return success();
}

} // namespace

//===---------------------------------------------------------------------===//
// Public API Specializations
//===---------------------------------------------------------------------===//

LogicalResult
fuseExtractSliceIntoProducerLoop(RewriterBase &rewriter, PCF::LoopOp loopOp,
                                 tensor::ExtractSliceOp extractSliceOp) {
  return fuseExtractSliceIntoProducerImpl(rewriter, loopOp, extractSliceOp);
}

LogicalResult
fuseExtractSliceIntoProducerGeneric(RewriterBase &rewriter,
                                    PCF::GenericOp genericOp,
                                    tensor::ExtractSliceOp extractSliceOp) {
  return fuseExtractSliceIntoProducerImpl(rewriter, genericOp, extractSliceOp);
}

LogicalResult
fuseCollapseShapeIntoProducerLoop(RewriterBase &rewriter, PCF::LoopOp loopOp,
                                   tensor::CollapseShapeOp collapseOp) {
  return fuseCollapseShapeIntoProducerImpl(rewriter, loopOp, collapseOp);
}

LogicalResult
fuseCollapseShapeIntoProducerGeneric(RewriterBase &rewriter,
                                      PCF::GenericOp genericOp,
                                      tensor::CollapseShapeOp collapseOp) {
  return fuseCollapseShapeIntoProducerImpl(rewriter, genericOp, collapseOp);
}

LogicalResult matchTilableConsumer(RewriterBase &rewriter,
                                   PCF::GenericOp producerOp,
                                   TilingInterface target,
                                   ConsumerFusionParams &params) {
  return matchTilableConsumerImpl(rewriter, producerOp, target, params);
}

LogicalResult matchTilableConsumer(RewriterBase &rewriter,
                                   PCF::LoopOp producerOp,
                                   TilingInterface target,
                                   ConsumerFusionParams &params) {
  return matchTilableConsumerImpl(rewriter, producerOp, target, params);
}

void fuseTilableConsumer(RewriterBase &rewriter, PCF::GenericOp producerOp,
                         TilingInterface target,
                         const ConsumerFusionParams &params) {
  return fuseTilableConsumerImpl(rewriter, producerOp, target, params);
}

void fuseTilableConsumer(RewriterBase &rewriter, PCF::LoopOp producerOp,
                         TilingInterface target,
                         const ConsumerFusionParams &params) {
  return fuseTilableConsumerImpl(rewriter, producerOp, target, params);
}

void populateStreamKRecombineFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseStreamKRecombineIntoGeneric>(patterns.getContext());
}

void populateConsumerFusionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FuseIntoGenericOp, FuseIntoLoopOp>(ctx);
  patterns.add<FuseExtractSliceIntoLoopOp, FuseExtractSliceIntoGenericOp>(ctx);
  patterns.add<FuseStreamKRecombineIntoGeneric>(ctx);
  patterns.add<FuseCollapseShapeIntoGenericOp, FuseCollapseShapeIntoLoopOp>(
      ctx);
  populatePCFDropUnusedResultPatterns(patterns);
}

} // namespace mlir::iree_compiler::IREE::PCF
