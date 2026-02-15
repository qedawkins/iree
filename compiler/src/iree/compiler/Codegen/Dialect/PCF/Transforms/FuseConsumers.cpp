// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

/// Clones the body of the writeback region into the current insertion point.
/// The writeback region takes a single block argument (the final tile) and
/// ends with pcf.yield. This helper clones everything except the terminator,
/// replacing the block argument with |tileValue|.
static void cloneWritebackBody(OpBuilder &b, Location loc,
                               Region &writebackRegion, Value tileValue) {
  Block &srcBlock = writebackRegion.front();
  IRMapping mapping;
  mapping.map(srcBlock.getArgument(0), tileValue);
  for (Operation &op : srcBlock.without_terminator()) {
    b.clone(op, mapping);
  }
}

/// Decomposes `pcf.stream_k_recombine` into distributed control flow:
///
/// 1. Conditional scratch write (if tile is split among workgroups).
/// 2. Workgroup barrier for scratch visibility.
/// 3. Conditional control flow (only for split tiles):
///    a. Thread-0 only: release fence, atomic increment, store result to
///       broadcast dword (workgroup shared memory).
///    b. Workgroup barrier for broadcast dword visibility.
///    c. ALL threads: load broadcast dword, check last-contributor.
///    d. ALL threads (if last): acquire fence, read partial tiles,
///       accumulate via combiner, first copy of writeback.
/// 4. Else branch: second copy of writeback (non-split path).
/// 5. Final workgroup barrier.
/// Builds the accumulation loop for stream-k recombination.
/// Uses a plain OpBuilder to avoid greedy driver listener issues when
/// building ops inside newly-created regions.
static Value buildAccumulationLoop(
    OpBuilder &b, Location loc, Value c1, Value numInGroup, Value scratch,
    Value accInit, Value tileBaseOffset, RankedTensorType partialTileType,
    int64_t rank, ArrayRef<int64_t> tileShape,
    ArrayRef<OpFoldResult> readSizes, ArrayRef<OpFoldResult> readStrides,
    Region &combinerRegion) {
  Value tileHeight = arith::ConstantIndexOp::create(b, loc, tileShape[0]);

  // Create the for op without a body builder to avoid listener issues.
  auto accLoop = scf::ForOp::create(
      b, loc, /*lowerBound=*/c1,
      /*upperBound=*/numInGroup,
      /*step=*/c1,
      /*iterArgs=*/ValueRange{accInit});

  // Manually build the loop body.
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(accLoop.getBody());
    Value loopIdx = accLoop.getInductionVar();
    Value loopAcc = accLoop.getRegionIterArg(0);

    // Compute scratch offset for contributor i, including per-tile base.
    Value localOffset = arith::MulIOp::create(b, loc, loopIdx, tileHeight);
    Value offset =
        arith::AddIOp::create(b, loc, tileBaseOffset, localOffset);

    SmallVector<OpFoldResult> loopReadOffsets(rank, b.getIndexAttr(0));
    loopReadOffsets[0] = offset;

    Value partialRead = PCF::ReadSliceOp::create(
        b, loc, partialTileType, scratch, loopReadOffsets, readSizes,
        readStrides);

    // Build affine maps for the element-wise combiner.
    SmallVector<AffineMap> indexingMaps;
    AffineMap identityMap = b.getMultiDimIdentityMap(rank);
    indexingMaps.push_back(identityMap); // ins: partial.
    indexingMaps.push_back(identityMap); // outs: accumulator.

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // Apply combiner via linalg.generic.  Pass a body builder so that the
    // region is created with the correct block arguments.  Without a body
    // builder, GenericOp::create leaves the region empty, and accessing
    // getRegion().front() is undefined behaviour (the root cause of the
    // SEGV-during-print crash).
    Block &combinerBlock = combinerRegion.front();
    auto genericOp = linalg::GenericOp::create(
        b, loc, /*resultTypes=*/partialTileType,
        /*inputs=*/ValueRange{partialRead},
        /*outputs=*/ValueRange{loopAcc}, indexingMaps, iteratorTypes,
        [&](OpBuilder &bodyBuilder, Location bodyLoc,
            ValueRange blockArgs) {
          // blockArgs = {in_arg (partial), out_arg (accumulator)}.
          IRMapping combinerMapping;
          combinerMapping.map(combinerBlock.getArgument(0), blockArgs[0]);
          combinerMapping.map(combinerBlock.getArgument(1), blockArgs[1]);
          for (Operation &op : combinerBlock.without_terminator()) {
            bodyBuilder.clone(op, combinerMapping);
          }
          auto combinerYield =
              cast<PCF::YieldOp>(combinerBlock.getTerminator());
          Value yieldVal =
              combinerMapping.lookupOrDefault(combinerYield.getOperand(0));
          linalg::YieldOp::create(bodyBuilder, bodyLoc, yieldVal);
        });

    Value combined = genericOp.getResult(0);
    scf::YieldOp::create(b, loc, ValueRange{combined});
  }

  return accLoop.getResult(0);
}

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

/// Fuses a pcf.stream_k_recombine into its producer pcf.generic:
///
/// 1. Sets sync_on_return=true on the producer.
/// 2. Inside the producer body, adds conditional scratch writes before
///    each write_slice to the consumed result's ref arg.
/// 3. Outside the producer (after sync), creates conditional control flow:
///    - Split path: thread-0 predicated atomic + recombine + writeback.
///    - Non-split path: direct writeback of the producer's result.
/// 4. Final workgroup barrier.
struct FuseStreamKRecombineIntoGeneric final
    : public OpRewritePattern<StreamKRecombineOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(StreamKRecombineOp recombineOp,
                                PatternRewriter &rewriter) const override {
    // Match: partial_tile must come from a pcf.generic result.
    Value partialTile = recombineOp.getPartialTile();
    auto genericResult = dyn_cast<OpResult>(partialTile);
    if (!genericResult) {
      return rewriter.notifyMatchFailure(
          recombineOp, "partial_tile is not an op result");
    }
    auto producerOp =
        dyn_cast<PCF::GenericOp>(genericResult.getOwner());
    if (!producerOp) {
      return rewriter.notifyMatchFailure(
          recombineOp, "partial_tile not produced by pcf.generic");
    }

    // Gather recombine operands.
    Value scratch = recombineOp.getScratch();
    Value counter = recombineOp.getCounter();
    Value counterIndex = recombineOp.getCounterIndex();
    Value numInGroup = recombineOp.getNumInGroup();
    Value contributorOrdinal = recombineOp.getContributorOrdinal();

    // Hoist recombine operands (and their transitive dependencies) before
    // the producer if they do not already dominate it.  In the real
    // pipeline, numInGroup and contributorOrdinal are computed after the
    // producer pcf.generic, but their transitive leaves (constants, block
    // args, etc.) always dominate the producer.
    DominanceInfo dominanceInfo(producerOp->getParentOp());
    for (Value operand :
         {scratch, counterIndex, numInGroup, contributorOrdinal}) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        // Block argument -- always dominates.
        continue;
      }
      if (failed(hoistBeforeIfNeeded(defOp, producerOp, dominanceInfo))) {
        return rewriter.notifyMatchFailure(
            recombineOp,
            "recombine operand cannot be hoisted before producer");
      }
    }

    Location loc = recombineOp.getLoc();
    unsigned resultIdx = genericResult.getResultNumber();

    RankedTensorType partialTileType =
        cast<RankedTensorType>(partialTile.getType());
    int64_t rank = partialTileType.getRank();
    ArrayRef<int64_t> tileShape = partialTileType.getShape();

    PCF::ShapedRefType scratchType =
        cast<PCF::ShapedRefType>(scratch.getType());
    PCF::ShapedRefType counterType = recombineOp.getCounterType();

    // Step 1: Set sync_on_return=true on the producer generic.
    rewriter.modifyOpInPlace(producerOp, [&]() {
      producerOp.setSyncOnReturn(true);
    });

    // Step 2: Compute constants and conditions before the producer.
    // Since pcf.generic does not have IsolatedFromAbove, these values
    // are captured inside the producer body for the scratch write.
    OpBuilder b(producerOp);

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

    // Step 2b: Add conditional scratch write inside the producer body
    // before each write_slice to the consumed result's ref arg.
    Value resultRefArg = producerOp.getRegionRefArgs()[resultIdx];
    SmallVector<PCF::WriteSliceOp> producerSlices;
    for (Operation *user : resultRefArg.getUsers()) {
      if (auto writeSlice = dyn_cast<PCF::WriteSliceOp>(user)) {
        producerSlices.push_back(writeSlice);
      }
    }

    for (PCF::WriteSliceOp writeSlice : producerSlices) {
      // Use plain OpBuilder inside the producer body to avoid greedy
      // driver listener issues with ops in existing regions.
      OpBuilder bInner(writeSlice);
      auto scratchIf = scf::IfOp::create(
          bInner, loc, TypeRange{}, isSplit,
          /*withElseRegion=*/false);

      {
        OpBuilder::InsertionGuard ifGuard(bInner);
        bInner.setInsertionPointToStart(
            &scratchIf.getThenRegion().front());

        SmallVector<OpFoldResult> scratchOffsets(rank,
                                                 bInner.getIndexAttr(0));
        scratchOffsets[0] = scratchWriteOffset;
        SmallVector<OpFoldResult> scratchSizes;
        for (int64_t i = 0; i < rank; ++i) {
          scratchSizes.push_back(bInner.getIndexAttr(tileShape[i]));
        }
        SmallVector<OpFoldResult> scratchStrides(rank,
                                                 bInner.getIndexAttr(1));

        PCF::WriteSliceOp::create(bInner, loc, writeSlice.getSource(),
                                  scratch, scratchOffsets, scratchSizes,
                                  scratchStrides);
      }
    }

    // Step 3: Create post-producer control flow at the recombine's
    // location.  Use plain OpBuilder for all newly-created regions.
    b.setInsertionPoint(recombineOp);

    // Allocate a broadcast dword in workgroup shared memory.  Thread-0
    // stores the atomic result here; all threads load it after the
    // barrier to determine if this workgroup is the last contributor.
    auto broadcastSrefType = PCF::ShapedRefType::get(
        b.getContext(), {1}, b.getI32Type(), producerOp.getScope());
    Value broadcastSref =
        PCF::AllocOp::create(b, loc, broadcastSrefType, ValueRange{});

    auto outerIf = scf::IfOp::create(
        b, loc, TypeRange{}, isSplit, /*withElseRegion=*/true);

    // Then branch: split path.
    // The atomic is single-invocation (thread-0 only) but the recombine
    // accumulation and writeback are evaluated by ALL threads.  The atomic
    // result is broadcast via shared memory dword + barrier.
    {
      OpBuilder thenB(b.getContext());
      thenB.setInsertionPointToStart(
          &outerIf.getThenRegion().front());

      IndexType indexType = thenB.getIndexType();

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

        // Release fence before atomic.
        PCF::FenceOp::create(thenB, loc, /*is_release=*/true,
                             ValueRange{scratch});

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

        // Atomic increment.
        Value c1I32 = arith::ConstantOp::create(
            thenB, loc, thenB.getI32IntegerAttr(1));
        Value oldVal = memref::AtomicRMWOp::create(
            thenB, loc, arith::AtomicRMWKind::addi, c1I32,
            counterMemref, ValueRange{counterIndex});

        // Store atomic result to broadcast dword.
        memref::StoreOp::create(thenB, loc, oldVal, broadcastMemref,
                                ValueRange{c0});
      }
      // InsertionGuard restores thenB to after thread0If.

      // Barrier: all threads sync after thread-0 writes broadcast dword.
      gpu::BarrierOp::create(thenB, loc, gpu::AddressSpace::Workgroup);

      // ALL threads load the broadcast result.
      Value broadcastedOld = memref::LoadOp::create(
          thenB, loc, broadcastMemref, ValueRange{c0});

      // ALL threads check if this workgroup is the last contributor.
      Value oldIdx = arith::IndexCastOp::create(
          thenB, loc, indexType, broadcastedOld);
      Value expectedLast =
          arith::SubIOp::create(thenB, loc, numInGroup, c1);
      Value isLast = arith::CmpIOp::create(
          thenB, loc, arith::CmpIPredicate::eq, oldIdx, expectedLast);

      // If last contributor: ALL threads do recombine and write back.
      auto lastContribIf = scf::IfOp::create(
          thenB, loc, TypeRange{}, isLast,
          /*withElseRegion=*/false);

      {
        thenB.setInsertionPointToStart(
            &lastContribIf.getThenRegion().front());

        // Acquire fence.
        PCF::FenceOp::create(thenB, loc, /*is_release=*/false,
                             ValueRange{scratch});

        // Read the first partial tile from scratch (ordinal 0).
        SmallVector<OpFoldResult> readOffsets(rank,
                                             thenB.getIndexAttr(0));
        readOffsets[0] = tileBaseOffset;
        SmallVector<OpFoldResult> readSizes;
        for (int64_t i = 0; i < rank; ++i) {
          readSizes.push_back(thenB.getIndexAttr(tileShape[i]));
        }
        SmallVector<OpFoldResult> readStrides(rank,
                                              thenB.getIndexAttr(1));

        Value accInit = PCF::ReadSliceOp::create(
            thenB, loc, partialTileType, scratch, readOffsets,
            readSizes, readStrides);

        // Build the accumulation loop.
        Value accumulated = buildAccumulationLoop(
            thenB, loc, c1, numInGroup, scratch, accInit,
            tileBaseOffset, partialTileType, rank, tileShape,
            readSizes, readStrides, recombineOp.getCombiner());

        // Clone writeback body (first copy).
        cloneWritebackBody(thenB, loc,
                           recombineOp.getWriteback(), accumulated);
      }
    }

    // Else branch: non-split path (direct writeback).
    {
      OpBuilder elseB(b.getContext());
      elseB.setInsertionPointToStart(
          &outerIf.getElseRegion().front());
      cloneWritebackBody(elseB, loc,
                         recombineOp.getWriteback(), partialTile);
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

  // Verify that each write_slice can be collapsed. For each reassociation
  // group, all inner dimensions must be fully covered (offset=0, size=dim_size)
  // to ensure the written region is contiguous in the collapsed layout.
  for (PCF::WriteSliceOp slice : slices) {
    if (!slice.hasUnitStride()) {
      return rewriter.notifyMatchFailure(slice,
                                         "write_slice has non-unit stride");
    }
    SmallVector<OpFoldResult> offsets = slice.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = slice.getMixedSizes();
    for (const ReassociationIndices &group : reassociation) {
      if (group.size() <= 1) {
        continue;
      }
      for (size_t i = 1, e = group.size(); i < e; ++i) {
        int64_t dim = group[i];
        if (!isConstantIntValue(offsets[dim], 0)) {
          return rewriter.notifyMatchFailure(
              slice, "inner dimension offset is not zero for collapse");
        }
        if (!isConstantIntValue(sizes[dim], producerShape[dim])) {
          return rewriter.notifyMatchFailure(
              slice, "inner dimension size doesn't cover full dimension");
        }
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

    // Compute collapsed offsets and sizes.
    SmallVector<OpFoldResult> collapsedOffsets, collapsedSizes;
    computeCollapsedOffsetsAndSizes(rewriter, loc, slice.getMixedOffsets(),
                                    slice.getMixedSizes(), producerShape,
                                    reassociation, collapsedOffsets,
                                    collapsedSizes);

    // Collapse the source tensor to match the new sref shape.
    Value source = slice.getSource();
    Value collapsedSource = tensor::CollapseShapeOp::create(
        rewriter, loc, source, reassociation);

    // Create the new write_slice with collapsed offsets/sizes.
    SmallVector<OpFoldResult> strides(collapsedOffsets.size(),
                                      rewriter.getIndexAttr(1));
    PCF::WriteSliceOp::create(rewriter, loc, collapsedSource, slice.getDest(),
                              collapsedOffsets, collapsedSizes, strides);

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
