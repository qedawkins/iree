// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-pcf-lower-stream-k-recombine"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_LOWERSTREAMKRECOMBINEPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

/// Inline the writeback region at the current insertion point, substituting
/// the block argument with `finalTile`. The pcf.yield terminator is not
/// cloned.
static void inlineWritebackRegion(OpBuilder &builder, Location loc,
                                  Region &writeback, Value finalTile) {
  Block &block = writeback.front();
  IRMapping mapping;
  mapping.map(block.getArgument(0), finalTile);
  for (Operation &op : block.without_terminator()) {
    builder.clone(op, mapping);
  }
}

/// Create a linalg.generic that applies the combiner region element-wise,
/// accumulating `rhs` into `lhs` in-place. Using `lhs` as both input and
/// output ensures the result is "equivalent" to the iter_arg for
/// bufferization.
static Value createPointwiseCombine(OpBuilder &builder, Location loc,
                                    Region &combinerRegion, Value lhs,
                                    Value rhs) {
  auto tileType = cast<RankedTensorType>(lhs.getType());
  int64_t rank = tileType.getRank();

  // Identity maps: 1 input (rhs) + 1 output (lhs, also readable).
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
  SmallVector<AffineMap> indexingMaps(2, identityMap);
  SmallVector<utils::IteratorType> iteratorTypes(
      rank, utils::IteratorType::parallel);

  Block &combinerBlock = combinerRegion.front();

  auto genericOp = linalg::GenericOp::create(
      builder, loc, tileType, /*inputs=*/ValueRange{rhs},
      /*outputs=*/ValueRange{lhs}, indexingMaps, iteratorTypes,
      [&](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange bodyArgs) {
        // bodyArgs: [rhs_elem, lhs_elem (from output)].
        IRMapping mapping;
        mapping.map(combinerBlock.getArgument(0), bodyArgs[1]);
        mapping.map(combinerBlock.getArgument(1), bodyArgs[0]);
        for (Operation &op : combinerBlock.without_terminator()) {
          bodyBuilder.clone(op, mapping);
        }
        // Replace pcf.yield with linalg.yield.
        auto yieldOp = cast<YieldOp>(combinerBlock.getTerminator());
        Value yielded = mapping.lookupOrDefault(yieldOp.getOperand(0));
        linalg::YieldOp::create(bodyBuilder, bodyLoc, yielded);
      });

  return genericOp.getResult(0);
}

/// Return a statically-known constant integer value from an SSA value,
/// or std::nullopt if not statically known.
static std::optional<int64_t> getStaticValue(Value v) {
  if (auto constOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return constOp.value();
  }
  if (auto constOp = v.getDefiningOp<arith::ConstantIntOp>()) {
    return constOp.value();
  }
  return std::nullopt;
}

/// Compute scratch slot offsets/sizes/strides for a given slot index.
/// Slot layout: scratch rows are partitioned per output tile, and within each
/// partition, contributor `slotIndex` writes at row `base + slotIndex * dim0`.
static void buildScratchSlotAddressing(
    OpBuilder &builder, Location loc, ShapedType tileType,
    Value scratchBaseRow, Value slotIndex, Value partial,
    SmallVector<OpFoldResult> &offsets, SmallVector<OpFoldResult> &sizes,
    SmallVector<OpFoldResult> &strides) {
  int64_t tileRank = tileType.getRank();
  for (int64_t d = 0; d < tileRank; ++d) {
    if (d == 0) {
      Value dim0Size;
      if (tileType.isDynamicDim(0)) {
        Value c0Idx = arith::ConstantIndexOp::create(builder, loc, 0);
        dim0Size = tensor::DimOp::create(builder, loc, partial, c0Idx);
      } else {
        dim0Size = arith::ConstantIndexOp::create(builder, loc,
                                                   tileType.getDimSize(0));
      }
      Value localOffset = arith::MulIOp::create(
          builder, loc, slotIndex, dim0Size,
          arith::IntegerOverflowFlags::nsw);
      Value slotOffset = arith::AddIOp::create(
          builder, loc, scratchBaseRow, localOffset,
          arith::IntegerOverflowFlags::nsw);
      offsets.push_back(slotOffset);
    } else {
      offsets.push_back(
          arith::ConstantIndexOp::create(builder, loc, 0).getResult());
    }
    if (tileType.isDynamicDim(d)) {
      sizes.push_back(
          tensor::DimOp::create(
              builder, loc, partial,
              arith::ConstantIndexOp::create(builder, loc, d).getResult())
              .getResult());
    } else {
      sizes.push_back(builder.getI64IntegerAttr(tileType.getDimSize(d)));
    }
    strides.push_back(builder.getI64IntegerAttr(1));
  }
}

struct LowerStreamKRecombineOp final
    : OpRewritePattern<StreamKRecombineOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(StreamKRecombineOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value partial = op.getPartialTile();
    Value scratch = op.getScratch();
    Value counter = op.getCounter();
    Value counterIndex = op.getCounterIndex();
    Value numInGroup = op.getNumInGroup();
    Value contributorOrdinal = op.getContributorOrdinal();

    ShapedType tileType = op.getPartialTileType();
    Type counterElemType = op.getCounterType().getElementType();

    // Check for sole contributor optimization: if num_in_group is
    // statically 1, skip all scratch/atomic/branching.
    std::optional<int64_t> staticNumInGroup = getStaticValue(numInGroup);
    if (staticNumInGroup && *staticNumInGroup == 1) {
      // Sole contributor: inline writeback with partial directly.
      inlineWritebackRegion(rewriter, loc, op.getWriteback(), partial);
      rewriter.eraseOp(op);
      return success();
    }

    // ================================================================
    // General case: 3-zone distributed architecture.
    //
    // Zone A: Distributed scratch write (no thread guard).
    //         Writes partial tile to scratch BEFORE the atomic so other
    //         workgroups cannot read our slot prematurely.
    //
    // Zone B: Single-invocation atomic (thread_id == 0).
    //         Increments the completion counter. Only the last
    //         contributor proceeds to Zone C.
    //
    // Zone C: Recombine + writeback (inside isLast, thread_id == 0).
    //         Reads all scratch slots, combines them, inlines writeback.
    //
    // Non-split path: Direct writeback in else branch (distributed,
    //         fusable into the producer via 696 control flow fusion).
    // ================================================================

    // Constants.
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value c1Int = arith::ConstantOp::create(
        rewriter, loc, counterElemType,
        rewriter.getIntegerAttr(counterElemType, 1));

    // isSplit = numInGroup != 1 (true when output tile is shared by
    // multiple workgroups and scratch/atomic path is needed).
    Value isSplit = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, numInGroup, c1);

    // Compute per-output-tile scratch base row. Scratch is partitioned as
    // [numOutputTiles * maxNumInGroup * tileM, tileN]. Each output tile
    // gets `perTileScratchRows = scratchDim0 / numOutputTiles` rows.
    ShapedRefType scratchSrefType = op.getScratchType();
    ShapedRefType counterSrefType = op.getCounterType();
    Value scratchBaseRow;
    if (counterSrefType.getRank() > 0) {
      int64_t scratchDim0 = scratchSrefType.getShape()[0];
      int64_t numOutputTiles = counterSrefType.getShape()[0];
      int64_t perTileScratchRows = scratchDim0 / numOutputTiles;
      Value perTileScratchRowsVal =
          arith::ConstantIndexOp::create(rewriter, loc, perTileScratchRows);
      scratchBaseRow =
          arith::MulIOp::create(rewriter, loc, counterIndex,
                                perTileScratchRowsVal,
                                arith::IntegerOverflowFlags::nsw);
    } else {
      // Rank-0 counter: single output tile, no partitioning.
      scratchBaseRow = c0;
    }

    // ============================================================
    // Zone A: Distributed scratch write (no thread guard).
    // All threads participate. The scratch write uses
    // contributor_ordinal (computed deterministically from the work
    // distribution) instead of the atomic return value, because the
    // atomic has not happened yet.
    // ============================================================
    scf::IfOp::create(rewriter, loc, isSplit,
        [&](OpBuilder &zoneABuilder, Location zoneALoc) {
          SmallVector<OpFoldResult> writeOffsets, writeSizes, writeStrides;
          buildScratchSlotAddressing(zoneABuilder, zoneALoc, tileType,
                                     scratchBaseRow, contributorOrdinal,
                                     partial, writeOffsets, writeSizes,
                                     writeStrides);
          WriteSliceOp::create(zoneABuilder, zoneALoc, partial, scratch,
                               writeOffsets, writeSizes, writeStrides);
          scf::YieldOp::create(zoneABuilder, zoneALoc);
        });

    // Barrier: synchronize all threads within this workgroup. Ensures
    // every thread's scratch write is complete before thread 0 proceeds
    // with the release fence and atomic.
    gpu::BarrierOp::create(rewriter, loc, gpu::AddressSpace::Workgroup);

    // ============================================================
    // Zone B+C and non-split writeback.
    //
    // Split path (then): thread_id==0 performs release fence, atomic
    //   increment, isLast check, and recombine+writeback.
    // Non-split path (else): direct writeback runs distributed (no
    //   thread guard) so it can be fused into the producer via 696
    //   control flow fusion.
    // ============================================================
    scf::IfOp::create(rewriter, loc, isSplit,
        /*thenBuilder=*/[&](OpBuilder &splitBuilder, Location splitLoc) {
          // Thread-0 guard: atomic and recombine are single-invocation.
          Value tid = gpu::ThreadIdOp::create(splitBuilder, splitLoc,
                                              gpu::Dimension::x);
          Value isThread0 = arith::CmpIOp::create(
              splitBuilder, splitLoc, arith::CmpIPredicate::eq, tid, c0);
          scf::IfOp::create(splitBuilder, splitLoc, isThread0,
              [&](OpBuilder &t0Builder, Location t0Loc) {
                // Release fence: make this workgroup's scratch write
                // visible to other workgroups before the atomic signals
                // completion.
                FenceOp::create(t0Builder, t0Loc, /*is_release=*/true,
                                ValueRange{scratch});

                // Zone B: Atomic increment counter.
                int64_t counterRank = counterSrefType.getRank();
                SmallVector<int64_t> counterMemRefShape(
                    counterSrefType.getShape());
                auto stridedLayout = StridedLayoutAttr::get(
                    rewriter.getContext(), ShapedType::kDynamic,
                    SmallVector<int64_t>(counterRank, ShapedType::kDynamic));
                MemRefType counterMemRefType = MemRefType::get(
                    counterMemRefShape, counterElemType, stridedLayout);
                SmallVector<OpFoldResult> counterSizes;
                for (int64_t dim : counterMemRefShape) {
                  counterSizes.push_back(rewriter.getIndexAttr(dim));
                }
                Value counterMemRef = GetMemrefOp::create(
                    t0Builder, t0Loc, counterMemRefType, counter,
                    /*offsets=*/SmallVector<OpFoldResult>(
                        counterRank, rewriter.getIndexAttr(0)),
                    /*sizes=*/counterSizes,
                    /*strides=*/SmallVector<OpFoldResult>(
                        counterRank, rewriter.getIndexAttr(1)));

                // For rank-0 counters (single output tile), no subscript
                // indices are needed. For rank-1+ counters, index by
                // counter_index (the output tile index).
                SmallVector<Value> atomicIndices;
                if (counterRank > 0)
                  atomicIndices.push_back(counterIndex);
                Value oldInt = memref::AtomicRMWOp::create(
                    t0Builder, t0Loc, counterElemType,
                    arith::AtomicRMWKind::addi, c1Int, counterMemRef,
                    atomicIndices);
                Value old = arith::IndexCastOp::create(
                    t0Builder, t0Loc, rewriter.getIndexType(), oldInt);

                // isLast: this workgroup is the last contributor.
                Value numMinus1 = arith::SubIOp::create(
                    t0Builder, t0Loc, numInGroup, c1,
                    arith::IntegerOverflowFlags::nsw);
                Value isLast = arith::CmpIOp::create(
                    t0Builder, t0Loc, arith::CmpIPredicate::eq, old,
                    numMinus1);

                // Zone C: Recombine + writeback (last contributor only).
                scf::IfOp::create(t0Builder, t0Loc, isLast,
                    [&](OpBuilder &lastBuilder, Location lastLoc) {
                      // Acquire fence: see all other workgroups'
                      // scratch writes.
                      FenceOp::create(lastBuilder, lastLoc,
                                      /*is_release=*/false,
                                      ValueRange{scratch});

                      // Read slot 0 as initial accumulator.
                      SmallVector<OpFoldResult> readOffsets, readSizes,
                          readStrides;
                      buildScratchSlotAddressing(
                          lastBuilder, lastLoc, tileType, scratchBaseRow,
                          c0, partial, readOffsets, readSizes, readStrides);
                      Value acc = ReadSliceOp::create(
                          lastBuilder, lastLoc, tileType, scratch,
                          readOffsets, readSizes, readStrides);

                      // Combine remaining slots [1, numInGroup).
                      Value accResult =
                          scf::ForOp::create(lastBuilder, lastLoc,
                                  c1, numInGroup, c1, ValueRange{acc},
                                  [&](OpBuilder &loopBuilder,
                                      Location loopLoc, Value iv,
                                      ValueRange iterArgs) {
                                    SmallVector<OpFoldResult> loopOffsets,
                                        loopSizes, loopStrides;
                                    buildScratchSlotAddressing(
                                        loopBuilder, loopLoc, tileType,
                                        scratchBaseRow, iv, partial,
                                        loopOffsets, loopSizes,
                                        loopStrides);
                                    Value slotTile = ReadSliceOp::create(
                                        loopBuilder, loopLoc, tileType,
                                        scratch, loopOffsets, loopSizes,
                                        loopStrides);
                                    Value combined =
                                        createPointwiseCombine(
                                            loopBuilder, loopLoc,
                                            op.getCombiner(), iterArgs[0],
                                            slotTile);
                                    scf::YieldOp::create(loopBuilder,
                                                         loopLoc, combined);
                                  })
                              .getResult(0);

                      // Inline writeback with accumulated result.
                      inlineWritebackRegion(lastBuilder, lastLoc,
                                            op.getWriteback(), accResult);
                      scf::YieldOp::create(lastBuilder, lastLoc);
                    });
                scf::YieldOp::create(t0Builder, t0Loc);
              });
          scf::YieldOp::create(splitBuilder, splitLoc);
        },
        /*elseBuilder=*/[&](OpBuilder &nonSplitBuilder,
                            Location nonSplitLoc) {
          // Non-split path: direct writeback (distributed).
          // This write_slice will be fused into the producer through
          // the scf.if's else branch via 696 control flow fusion.
          inlineWritebackRegion(nonSplitBuilder, nonSplitLoc,
                                op.getWriteback(), partial);
          scf::YieldOp::create(nonSplitBuilder, nonSplitLoc);
        });

    // Final barrier: synchronize all threads before the next outer loop
    // iteration. Prevents race conditions on shared memory.
    gpu::BarrierOp::create(rewriter, loc, gpu::AddressSpace::Workgroup);

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerStreamKRecombinePass final
    : impl::LowerStreamKRecombinePassBase<LowerStreamKRecombinePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<iree_compiler::IREE::PCF::PCFDialect,
                    scf::SCFDialect, arith::ArithDialect,
                    gpu::GPUDialect, memref::MemRefDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

void LowerStreamKRecombinePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<LowerStreamKRecombineOp>(&getContext());
  walkAndApplyPatterns(getOperation(), std::move(patterns));

  // Verify no StreamKRecombineOps remain.
  WalkResult result =
      getOperation()->walk([](StreamKRecombineOp op) -> WalkResult {
        op.emitOpError("failed to lower stream_k_recombine");
        return WalkResult::interrupt();
      });
  if (result.wasInterrupted()) {
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
