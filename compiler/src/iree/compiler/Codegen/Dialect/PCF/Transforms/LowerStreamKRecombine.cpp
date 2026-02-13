// Copyright 2025 The IREE Authors
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

struct LowerStreamKRecombineOp final
    : OpRewritePattern<StreamKRecombineOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(StreamKRecombineOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value partial = op.getPartialTile();
    Value scratch = op.getScratch();
    Value counter = op.getCounter();
    Value numInGroup = op.getNumInGroup();

    ShapedType tileType = op.getPartialTileType();
    ShapedRefType counterSrefType = op.getCounterType();
    Type counterElemType = counterSrefType.getElementType();
    int64_t tileRank = tileType.getRank();

    // Check for sole contributor optimization: if num_in_group is
    // statically 1, skip all scratch/atomic/branching.
    std::optional<int64_t> staticNumInGroup = getStaticValue(numInGroup);
    if (staticNumInGroup && *staticNumInGroup == 1) {
      // Sole contributor: inline writeback with partial directly.
      inlineWritebackRegion(rewriter, loc, op.getWriteback(), partial);
      rewriter.eraseOp(op);
      return success();
    }

    // --- General case: atomic + branching + scratch + writeback ---

    // Constants.
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value c1Int = arith::ConstantOp::create(
        rewriter, loc, counterElemType,
        rewriter.getIntegerAttr(counterElemType, 1));

    // Step 1: Get memref from counter sref for atomic operation.
    // Counter is rank-0 (scalar sref), so offsets/sizes/strides are empty.
    // pcf.get_memref requires strided layout with dynamic offset.
    auto stridedLayout = StridedLayoutAttr::get(
        rewriter.getContext(), ShapedType::kDynamic, /*strides=*/{});
    MemRefType counterMemRefType =
        MemRefType::get({}, counterElemType, stridedLayout);
    Value counterMemRef = GetMemrefOp::create(
        rewriter, loc, counterMemRefType, counter,
        /*offsets=*/ArrayRef<OpFoldResult>{},
        /*sizes=*/ArrayRef<OpFoldResult>{},
        /*strides=*/ArrayRef<OpFoldResult>{});

    // Step 2: Atomic increment counter, get old value.
    Value oldInt = memref::AtomicRMWOp::create(
        rewriter, loc, counterElemType, arith::AtomicRMWKind::addi, c1Int,
        counterMemRef, ValueRange{});

    // Convert old count to index for comparisons.
    Value old = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), oldInt);

    // Step 3: Compute branch conditions.
    Value isOnly = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, numInGroup, c1);
    Value numMinus1 = arith::SubIOp::create(rewriter, loc, numInGroup, c1);
    Value isLast = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, old, numMinus1);

    // notOnly = !isOnly.
    Value trueVal = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    Value notOnly = arith::XOrIOp::create(rewriter, loc, isOnly, trueVal);

    // Step 4: Non-sole contributor path.
    scf::IfOp::create(rewriter,
        loc, notOnly,
        [&](OpBuilder &thenBuilder, Location thenLoc) {
          // Compute scratch slot offset for this contributor.
          // Slot layout: scratch is viewed as [K * dim0, dim1, ...].
          // Contributor `old` writes to offset [old * dim0, 0, ...].
          SmallVector<OpFoldResult> writeOffsets;
          SmallVector<OpFoldResult> writeSizes;
          SmallVector<OpFoldResult> writeStrides;
          for (int64_t d = 0; d < tileRank; ++d) {
            if (d == 0) {
              // Leading dimension: offset = old * dim0_size.
              Value dim0Size;
              if (tileType.isDynamicDim(0)) {
                Value c0Idx =
                    arith::ConstantIndexOp::create(thenBuilder, thenLoc, 0);
                dim0Size = tensor::DimOp::create(
                    thenBuilder, thenLoc, partial, c0Idx);
              } else {
                dim0Size = arith::ConstantIndexOp::create(
                    thenBuilder, thenLoc, tileType.getDimSize(0));
              }
              Value slotOffset =
                  arith::MulIOp::create(thenBuilder, thenLoc, old, dim0Size);
              writeOffsets.push_back(slotOffset);
            } else {
              writeOffsets.push_back(
                  arith::ConstantIndexOp::create(thenBuilder, thenLoc, 0)
                      .getResult());
            }
            if (tileType.isDynamicDim(d)) {
              // Use tensor.dim for dynamic dimensions.
              writeSizes.push_back(
                  tensor::DimOp::create(
                      thenBuilder, thenLoc, partial,
                      arith::ConstantIndexOp::create(thenBuilder, thenLoc, d).getResult())
                      .getResult());
            } else {
              writeSizes.push_back(
                  thenBuilder.getI64IntegerAttr(tileType.getDimSize(d)));
            }
            writeStrides.push_back(thenBuilder.getI64IntegerAttr(1));
          }

          // Write partial tile to scratch at computed slot.
          WriteSliceOp::create(
              thenBuilder, thenLoc, partial, scratch, writeOffsets, writeSizes,
              writeStrides);

          // Release fence: make our scratch write visible to other
          // workgroups.
          FenceOp::create(thenBuilder, thenLoc,
                          /*is_release=*/true,
                          ValueRange{scratch});

          // Last contributor: accumulate and writeback.
          scf::IfOp::create(thenBuilder,
              thenLoc, isLast,
              [&](OpBuilder &lastBuilder, Location lastLoc) {
                // Acquire fence: see all other workgroups' scratch
                // writes.
                FenceOp::create(lastBuilder, lastLoc,
                                /*is_release=*/false,
                                ValueRange{scratch});

                // Accumulate scratch slots.
                // Start with slot 0.
                SmallVector<OpFoldResult> readOffsets;
                SmallVector<OpFoldResult> readSizes;
                SmallVector<OpFoldResult> readStrides;
                for (int64_t d = 0; d < tileRank; ++d) {
                  readOffsets.push_back(
                      arith::ConstantIndexOp::create(lastBuilder, lastLoc, 0)
                          .getResult());
                  if (tileType.isDynamicDim(d)) {
                    readSizes.push_back(
                        tensor::DimOp::create(
                            lastBuilder, lastLoc, partial,
                            arith::ConstantIndexOp::create(
                                lastBuilder, lastLoc, d)
                                .getResult())
                            .getResult());
                  } else {
                    readSizes.push_back(
                        lastBuilder.getI64IntegerAttr(tileType.getDimSize(d)));
                  }
                  readStrides.push_back(
                      lastBuilder.getI64IntegerAttr(1));
                }

                Value acc = ReadSliceOp::create(
                    lastBuilder, lastLoc, tileType, scratch, readOffsets,
                    readSizes, readStrides);

                // Loop over remaining slots [1, old).
                // If old == 1, this loop doesn't execute.
                Value accResult =
                    scf::ForOp::create(lastBuilder,
                            lastLoc, c1, old, c1,
                            ValueRange{acc},
                            [&](OpBuilder &loopBuilder, Location loopLoc,
                                Value iv, ValueRange iterArgs) {
                              // Compute slot offset.
                              Value dim0Size;
                              if (tileType.isDynamicDim(0)) {
                                Value c0Idx =
                                    arith::ConstantIndexOp::create(
                                        loopBuilder, loopLoc, 0);
                                dim0Size =
                                    tensor::DimOp::create(
                                        loopBuilder, loopLoc, partial, c0Idx);
                              } else {
                                dim0Size =
                                    arith::ConstantIndexOp::create(
                                        loopBuilder, loopLoc,
                                        tileType.getDimSize(0));
                              }
                              Value slotOff =
                                  arith::MulIOp::create(
                                      loopBuilder, loopLoc, iv, dim0Size);

                              SmallVector<OpFoldResult> loopReadOffsets;
                              SmallVector<OpFoldResult> loopReadSizes(
                                  readSizes);
                              SmallVector<OpFoldResult> loopReadStrides(
                                  readStrides);
                              for (int64_t d = 0; d < tileRank; ++d) {
                                if (d == 0) {
                                  loopReadOffsets.push_back(slotOff);
                                } else {
                                  loopReadOffsets.push_back(
                                      arith::ConstantIndexOp::create(
                                          loopBuilder, loopLoc, 0)
                                          .getResult());
                                }
                              }

                              Value slotTile =
                                  ReadSliceOp::create(
                                      loopBuilder, loopLoc, tileType, scratch,
                                      loopReadOffsets, loopReadSizes,
                                      loopReadStrides);

                              // Combine current accumulator with this
                              // slot.
                              Value combined = createPointwiseCombine(
                                  loopBuilder, loopLoc,
                                  op.getCombiner(), iterArgs[0],
                                  slotTile);

                              scf::YieldOp::create(
                                  loopBuilder, loopLoc, combined);
                            })
                        .getResult(0);

                // Combine accumulated scratch with own partial tile.
                Value finalTile = createPointwiseCombine(
                    lastBuilder, lastLoc, op.getCombiner(), accResult,
                    partial);

                // Inline writeback with accumulated result.
                inlineWritebackRegion(lastBuilder, lastLoc,
                                      op.getWriteback(), finalTile);

                scf::YieldOp::create(lastBuilder, lastLoc);
              });

          scf::YieldOp::create(thenBuilder, thenLoc);
        });

    // Step 5: Sole contributor writeback.
    scf::IfOp::create(rewriter,
        loc, isOnly,
        [&](OpBuilder &soleBuilder, Location soleLoc) {
          inlineWritebackRegion(soleBuilder, soleLoc, op.getWriteback(),
                                partial);
          scf::YieldOp::create(soleBuilder, soleLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerStreamKRecombinePass final
    : impl::LowerStreamKRecombinePassBase<LowerStreamKRecombinePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<iree_compiler::IREE::PCF::PCFDialect,
                    scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
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
