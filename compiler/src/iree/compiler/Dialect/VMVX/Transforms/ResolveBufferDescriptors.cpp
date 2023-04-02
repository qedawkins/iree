// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VMVX {

namespace {
/// Helper struct to return the offset, sizes and strides
/// of a `source` of a `memref.extract_strided_metadata` op.
struct DescriptorInfo {
  OpFoldResult offset;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};
}  // namespace

/// Returns an AffineMap for an add or a mul.
static AffineMap getAddMap(MLIRContext *context) {
  AffineExpr s0, s1;
  bindSymbols(context, s0, s1);
  return AffineMap::get(0, 2, s0 + s1);
}
static AffineMap getMulMap(MLIRContext *context) {
  AffineExpr s0, s1;
  bindSymbols(context, s0, s1);
  return AffineMap::get(0, 2, s0 * s1);
}

static FailureOr<DescriptorInfo> resolveBufferDescriptorForSubview(
    memref::SubViewOp subview, RewriterBase &rewriter, Location loc,
    Value sourceOffset, ValueRange sourceSizes, ValueRange sourceStrides) {
  DescriptorInfo resultDescriptor;

  // For sizes, we just use the new ones.
  resultDescriptor.sizes = subview.getMixedSizes();

  // Apply stride multipliers.
  AffineMap mulMap = getMulMap(rewriter.getContext());
  for (auto [index, stride] : llvm::enumerate(subview.getMixedStrides())) {
    OpFoldResult currentStride = makeComposedFoldedAffineApply(
        rewriter, loc, mulMap, {sourceStrides[index], stride});
    resultDescriptor.strides.push_back(currentStride);
  }

  // Offsets.
  resultDescriptor.offset = sourceOffset;
  AffineMap addMap = getAddMap(rewriter.getContext());
  for (auto [index, offset] : llvm::enumerate(subview.getMixedOffsets())) {
    OpFoldResult physicalOffset = makeComposedFoldedAffineApply(
        rewriter, loc, mulMap, {offset, resultDescriptor.strides[index]});
    resultDescriptor.offset = makeComposedFoldedAffineApply(
        rewriter, loc, addMap, {resultDescriptor.offset, physicalOffset});
  }
  return resultDescriptor;
}

/// Returns the strides based on the sizes assuming that the `memref`
/// has default layout, i.e. it is not a result of a subview.
static SmallVector<OpFoldResult> getStridesFromSizes(
    RewriterBase &rewriter, Location loc, ArrayRef<OpFoldResult> sizes) {
  if (sizes.size() == 0) {
    return {};
  }
  SmallVector<OpFoldResult> strides(sizes.size());
  strides.back() = rewriter.getIndexAttr(1);
  if (sizes.size() == 1) {
    return strides;
  }
  AffineMap mulMap = getMulMap(rewriter.getContext());
  for (int i = sizes.size() - 2; i >= 0; --i) {
    strides[i] = makeComposedFoldedAffineApply(rewriter, loc, mulMap,
                                               {strides[i + 1], sizes[i + 1]});
  }
  return strides;
}

static FailureOr<DescriptorInfo> resolveBufferDescriptorForAllocation(
    memref::AllocaOp alloca, RewriterBase &rewriter, Location loc) {
  DescriptorInfo resultDescriptor;

  // Replace the op with values:
  //   base_buffer: The subspan result
  //   offset: byte offset from subspan divided by element type size
  //   sizes: static and dynamic sizes from the subspan
  //   strides: identity strides
  auto memRefType = alloca.getResult().getType().cast<MemRefType>();
  int rank = memRefType.getRank();

  // Compute sizes.
  auto dynamicDimIt = alloca.getDynamicSizes().begin();
  for (int i = 0; i < rank; ++i) {
    if (memRefType.isDynamicDim(i)) {
      resultDescriptor.sizes.push_back(*dynamicDimIt);
      dynamicDimIt++;
    } else {
      resultDescriptor.sizes.push_back(
          rewriter.getIndexAttr(memRefType.getDimSize(i)));
    }
  }

  // Strides (just creates identity strides).
  resultDescriptor.strides =
      getStridesFromSizes(rewriter, loc, resultDescriptor.sizes);

  resultDescriptor.offset = rewriter.getIndexAttr(0);
  return resultDescriptor;
}

static FailureOr<DescriptorInfo> resolveBufferDescriptorForGetGlobalOp(
    memref::GetGlobalOp global, RewriterBase &rewriter, Location loc) {
  IndexSet indexSet(loc, rewriter);
  DescriptorInfo resultDescriptor;

  // Replace the op with values:
  //   base_buffer: The subspan result
  //   offset: byte offset from subspan divided by element type size
  //   sizes: static and dynamic sizes from the subspan
  //   strides: identity strides
  auto memRefType = global.getResult().getType().cast<MemRefType>();
  int rank = memRefType.getRank();

  // Compute sizes.
  for (int i = 0; i < rank; ++i) {
    if (memRefType.isDynamicDim(i)) {
      return rewriter.notifyMatchFailure(
          global, "memref.get_global does not support dynamic dims");
    }
    resultDescriptor.sizes.push_back(
        rewriter.getIndexAttr(memRefType.getDimSize(i)));
  }

  // Strides (just creates identity strides).
  resultDescriptor.strides =
      getStridesFromSizes(rewriter, loc, resultDescriptor.sizes);

  // Offset.
  resultDescriptor.offset = rewriter.getIndexAttr(0);
  return resultDescriptor;
}

/// Replaces the offsets, sizes and strides based on values provided
/// by `DescriptorInfo` object.
template <typename OpTy>
static void replaceOffsetSizesAndStridesWith(
    RewriterBase &rewriter, OpTy op, const DescriptorInfo &resultDescriptor) {
  int rank = resultDescriptor.sizes.size();
  assert(rank == resultDescriptor.strides.size() &&
         "expected number of sizes and strides to match");
  assert(op.getSizes().size() == rank &&
         "expected as many size replacements as the number of sizes in the "
         "original operation");
  assert(op.getStrides().size() == rank &&
         "expected as many strides replacements as the number of strides in "
         "the original operation");
  Location loc = op.getLoc();
  for (int i = 0; i < rank; ++i) {
    // Sizes
    rewriter.replaceAllUsesWith(op.getSizes()[i],
                                getValueOrCreateConstantIndexOp(
                                    rewriter, loc, resultDescriptor.sizes[i]));
    // Strides
    rewriter.replaceAllUsesWith(
        op.getStrides()[i], getValueOrCreateConstantIndexOp(
                                rewriter, loc, resultDescriptor.strides[i]));
  }
  // Offset
  rewriter.replaceAllUsesWith(
      op.getOffset(),
      getValueOrCreateConstantIndexOp(rewriter, loc, resultDescriptor.offset));
}

namespace {

template <typename OpTy>
struct FromMemRefSubView : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto subview = op.getSource().template getDefiningOp<memref::SubViewOp>();
    if (!subview) return failure();
    auto loc = op.getLoc();
    IndexSet indexSet(loc, rewriter);

    // Get types.
    auto subType = subview.getResult().getType().template cast<MemRefType>();
    Value source = subview.getSource();
    auto sourceType = source.getType().template cast<MemRefType>();
    int sourceRank = sourceType.getRank();
    int subRank = subType.getRank();
    (void)subRank;

    // Create a descriptor for the source.
    IndexType indexType = rewriter.getIndexType();
    SmallVector<Type> sizeStrideTypes;
    for (int i = 0; i < sourceRank; i++) {
      sizeStrideTypes.push_back(indexType);
    }
    auto sourceDesc =
        rewriter.create<OpTy>(loc, op.getBaseBuffer().getType(), indexType,
                              sizeStrideTypes, sizeStrideTypes, source);

    FailureOr<DescriptorInfo> resultDescriptor =
        resolveBufferDescriptorForSubview(
            subview, rewriter, loc, sourceDesc.getOffset(),
            sourceDesc.getSizes(), sourceDesc.getStrides());

    if (failed(resultDescriptor)) {
      return rewriter.notifyMatchFailure(
          op, "failed to resolve descriptor with source being a subview op");
    }

    llvm::SmallBitVector droppedDims = subview.getDroppedDims();
    int targetIndex = 0;
    for (int i = 0; i < sourceRank; ++i) {
      if (droppedDims.test(i)) continue;
      rewriter.replaceAllUsesWith(
          op.getSizes()[targetIndex],
          getValueOrCreateConstantIndexOp(rewriter, loc,
                                          resultDescriptor->sizes[i]));
      rewriter.replaceAllUsesWith(
          op.getStrides()[targetIndex],
          getValueOrCreateConstantIndexOp(rewriter, loc,
                                          resultDescriptor->strides[i]));
      targetIndex++;
    }
    rewriter.replaceAllUsesWith(op.getOffset(),
                                getValueOrCreateConstantIndexOp(
                                    rewriter, loc, resultDescriptor->offset));

    // Base.
    rewriter.replaceAllUsesWith(op.getBaseBuffer(), sourceDesc.getBaseBuffer());
    rewriter.eraseOp(op);
    return success();
  }
};

struct FromHalInterfaceBindingSubspan
    : public OpRewritePattern<GetBufferDescriptorOp> {
  using OpRewritePattern<GetBufferDescriptorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GetBufferDescriptorOp op,
                                PatternRewriter &rewriter) const override {
    auto binding =
        op.getSource()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!binding) return failure();

    auto loc = op.getLoc();

    auto memRefType = binding.getResult().getType().cast<MemRefType>();
    int rank = memRefType.getRank();
    DescriptorInfo resultDescriptor;

    // Compute sizes.
    auto dynamicDimIt = binding.getDynamicDims().begin();
    for (int i = 0; i < rank; ++i) {
      if (memRefType.isDynamicDim(i)) {
        resultDescriptor.sizes.push_back(*dynamicDimIt);
        dynamicDimIt++;
      } else {
        resultDescriptor.sizes.push_back(
            rewriter.getIndexAttr(memRefType.getDimSize(i)));
      }
    }

    // Strides.
    resultDescriptor.strides =
        getStridesFromSizes(rewriter, loc, resultDescriptor.sizes);

    // Offset.
    auto elementSize =
        rewriter.create<IREE::Util::SizeOfOp>(loc, memRefType.getElementType());
    resultDescriptor.offset = rewriter.createOrFold<arith::DivUIOp>(
        loc, binding.getByteOffset(), elementSize);

    replaceOffsetSizesAndStridesWith(rewriter, op, resultDescriptor);

    // Base buffer.
    rewriter.replaceAllUsesWith(
        op.getBaseBuffer(),
        rewriter
            .create<IREE::VMVX::GetRawInterfaceBindingBufferOp>(
                loc, op.getBaseBuffer().getType(), binding.getSetAttr(),
                binding.getBindingAttr())
            .getResult());

    rewriter.eraseOp(op);
    return success();
  }
};

struct ResolveExtractMetadataFromHalInterfaceBindingSubspan
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
  using OpRewritePattern<memref::ExtractStridedMetadataOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    auto binding =
        op.getSource()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!binding) return failure();
    auto memRefType = binding.getResult().getType().template cast<MemRefType>();
    if (memRefType.getRank() < 1) return failure();

    auto loc = op.getLoc();
    int rank = memRefType.getRank();
    DescriptorInfo resultDescriptor;

    // Compute sizes.
    auto dynamicDimIt = binding.getDynamicDims().begin();
    for (int i = 0; i < rank; ++i) {
      if (memRefType.isDynamicDim(i)) {
        resultDescriptor.sizes.push_back(*dynamicDimIt);
        dynamicDimIt++;
      } else {
        resultDescriptor.sizes.push_back(
            rewriter.getIndexAttr(memRefType.getDimSize(i)));
      }
    }

    // Strides.
    resultDescriptor.strides =
        getStridesFromSizes(rewriter, loc, resultDescriptor.sizes);
    resultDescriptor.offset = rewriter.getIndexAttr(0);

    replaceOffsetSizesAndStridesWith(rewriter, op, resultDescriptor);

    // Base buffer. Use a 1D memref for hal.interface.binding.subspan.
    AffineMap mulMap = getMulMap(rewriter.getContext());
    OpFoldResult linearizedMemrefSize = rewriter.getIndexAttr(1);
    for (auto size : resultDescriptor.sizes) {
      linearizedMemrefSize = makeComposedFoldedAffineApply(
          rewriter, loc, mulMap, {linearizedMemrefSize, size});
    }
    SmallVector<int64_t> staticLinearShape;
    SmallVector<Value> dynamicLinearShape;
    dispatchIndexOpFoldResult(linearizedMemrefSize, dynamicLinearShape,
                              staticLinearShape);
    Value linearInterfaceBinding =
        rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
            loc, op.getBaseBuffer().getType(), binding.getSetAttr(),
            binding.getBindingAttr(), binding.getDescriptorTypeAttr(),
            binding.getByteOffset(),
            /*dynamicDims =*/ValueRange{}, binding.getAlignmentAttr(),
            binding.getDescriptorFlagsAttr());
    rewriter.replaceAllUsesWith(op.getBaseBuffer(), linearInterfaceBinding);

    rewriter.eraseOp(op);
    return success();
  }
};

// Allocations always return a non-offset memref and are matched by this
// pattern.
template <typename OpTy>
struct FromAllocation : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto alloca = op.getSource().template getDefiningOp<memref::AllocaOp>();
    if (!alloca) return failure();
    auto memRefType = alloca.getResult().getType().template cast<MemRefType>();
    if (!memRefType.getLayout().isIdentity()) {
      return rewriter.notifyMatchFailure(op, "not identity allocation");
    }

    auto loc = op.getLoc();
    FailureOr<DescriptorInfo> resultDescriptor =
        resolveBufferDescriptorForAllocation(alloca, rewriter, loc);
    if (failed(resultDescriptor)) {
      return rewriter.notifyMatchFailure(
          op, "failed to resolve descriptor for memref.alloca op");
    }

    replaceOffsetSizesAndStridesWith(rewriter, op, resultDescriptor.value());

    // Base buffer.
    rewriter.replaceAllUsesWith(
        op.getBaseBuffer(),
        rewriter
            .template create<UnrealizedConversionCastOp>(
                loc, op.getBaseBuffer().getType(), alloca.getResult())
            .getResult(0));

    rewriter.eraseOp(op);
    return success();
  }
};

// MemRef globals are always static shaped and reference a non-offset
// buffer.
template <typename OpTy>
struct FromGlobal : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto global = op.getSource().template getDefiningOp<memref::GetGlobalOp>();
    if (!global) return failure();
    auto memRefType = global.getResult().getType().template cast<MemRefType>();
    if (!memRefType.getLayout().isIdentity()) {
      return rewriter.notifyMatchFailure(op, "not identity allocation");
    }

    auto loc = op.getLoc();
    FailureOr<DescriptorInfo> resultDescriptor =
        resolveBufferDescriptorForGetGlobalOp(global, rewriter, loc);
    if (failed(resultDescriptor)) {
      return rewriter.notifyMatchFailure(
          op, "failed to resolve descriptor for memref.get_global source");
    }

    replaceOffsetSizesAndStridesWith(rewriter, op, resultDescriptor.value());

    // Base buffer.
    rewriter.replaceAllUsesWith(
        op.getBaseBuffer(),
        rewriter
            .template create<UnrealizedConversionCastOp>(
                loc, op.getBaseBuffer().getType(), global.getResult())
            .getResult(0));

    rewriter.eraseOp(op);
    return success();
  }
};

class ResolveBufferDescriptorsPass
    : public ResolveBufferDescriptorsBase<ResolveBufferDescriptorsPass> {
 public:
  ResolveBufferDescriptorsPass() = default;
  ResolveBufferDescriptorsPass(const ResolveBufferDescriptorsPass &) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::VMVX::VMVXDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<FromAllocation<GetBufferDescriptorOp>,
                    FromAllocation<memref::ExtractStridedMetadataOp>,
                    FromGlobal<GetBufferDescriptorOp>,
                    FromGlobal<memref::ExtractStridedMetadataOp>,
                    FromHalInterfaceBindingSubspan,
                    FromMemRefSubView<GetBufferDescriptorOp>,
                    FromMemRefSubView<memref::ExtractStridedMetadataOp>,
                    ResolveExtractMetadataFromHalInterfaceBindingSubspan>(
        &getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    // If any get_buffer_descriptor patterns remain, we fail.
    if (!allowUnresolved) {
      SmallVector<Operation *> remaining;
      getOperation()->walk([&](Operation *op) {
        if (isa<GetBufferDescriptorOp, memref::ExtractStridedMetadataOp>(op)) {
          remaining.push_back(op);
        }
      });

      if (!remaining.empty()) {
        auto diag = getOperation()->emitError()
                    << "Unable to resolve all strided buffer descriptors:";
        for (auto *op : remaining) {
          diag.attachNote(op->getLoc()) << "remaining live use";
        }
        signalPassFailure();
      }
    }
  }

  Option<bool> allowUnresolved{
      *this, "allow-unresolved",
      llvm::cl::desc("Allow unresolved descriptors (for testing)"),
      llvm::cl::init(false)};
};

}  // namespace

std::unique_ptr<mlir::OperationPass<>> createResolveBufferDescriptorsPass() {
  return std::make_unique<ResolveBufferDescriptorsPass>();
}

}  // namespace mlir::iree_compiler::IREE::VMVX
