// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-pcf-convert-sref-to-memref"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_CONVERTSREFTOMEMREFPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"
namespace {

struct ConvertGenericOp : public OpConversionPattern<PCF::GenericOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::GenericOp genericOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (llvm::any_of(genericOp.getResultTypes(),
                     [](Type t) { return !isa<MemRefType>(t); })) {
      return rewriter.notifyMatchFailure(
          genericOp, "expected all generic results to be of memref type");
    }

    Location loc = genericOp.getLoc();
    IntegerAttr alignment =
        genericOp.getScope().getPreferredAllocAlignment(rewriter.getContext());
    SmallVector<Value> replacements;

    // Init iterator.
    auto currInit = genericOp.getInits().begin();
    ValueRange dynamicSizes = genericOp.getDynamicSizes();
    for (auto [resultType, isTied] :
         llvm::zip_equal(genericOp.getResultTypes(), genericOp.getIsTied())) {
      if (isTied) {
        replacements.push_back(*currInit);
        ++currInit;
      } else {
        int64_t numDynamicDims =
            cast<ShapedType>(resultType).getNumDynamicDims();
        memref::AllocOp::create(rewriter, loc, resultType,
                                dynamicSizes.take_front(numDynamicDims),
                                /*symbolOperands=*/ValueRange(), alignment);
        dynamicSizes = dynamicSizes.drop_front(numDynamicDims);
      }
    }

    // Replace bbArg uses before moving the body over since the region args
    // will be removed.
    for (auto [bbArg, replacement] :
         llvm::zip_equal(genericOp.getRegionRefArgs(), replacements)) {
      rewriter.replaceAllUsesWith(bbArg, replacement);
    }

    PCF::GenericOp newGenericOp = PCF::GenericOp::create(
        rewriter, loc, genericOp.getScope(), adaptor.getCount());
    newGenericOp.getRegion().takeBody(genericOp.getRegion());

    // Conversion pattern rewriter overrides RAUW. Iterate over the replacements
    // manually.
    for (auto [result, replacement] :
         llvm::zip_equal(genericOp.getResults(), replacements)) {
      rewriter.replaceAllUsesWith(result, replacement);
    }
    return success();
  }
};

struct ConvertWriteSliceOp : public OpConversionPattern<PCF::WriteSliceOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::WriteSliceOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return failure();
  }
};

struct ConvertSRefToMemRefPass final
    : impl::ConvertSRefToMemRefPassBase<ConvertSRefToMemRefPass> {
  void runOnOperation() override;
};

void ConvertSRefToMemRefPass::runOnOperation() {
  auto *context = &getContext();

  TypeConverter typeConverter;
  ConversionTarget conversionTarget(getContext());
  RewritePatternSet patterns(&getContext());

  // Add a default type converter to a maximally generic strided memref.
  typeConverter.addConversion([](IREE::PCF::ShapedRefType type) -> Type {
    SmallVector<int64_t> strides(type.getRank(), ShapedType::kDynamic);
    return MemRefType::get(type.getShape(), type.getElementType(),
                           StridedLayoutAttr::get(type.getContext(),
                                                  ShapedType::kDynamic,
                                                  strides));
  });

  patterns.insert<ConvertGenericOp, ConvertWriteSliceOp>(typeConverter,
                                                         context);
  ConversionTarget target(*context);
  auto isIllegalType = [&](Type t) { return !isa<PCF::ShapedRefType>(t); };

  // Verify that all operand, result, and region argument types have been
  // converted.
  auto isLegallyTypedOp = [&](Operation *op) -> bool {
    for (Type type : op->getResultTypes()) {
      if (isIllegalType(type))
        return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (isIllegalType(type))
        return false;
    }
    for (auto &region : op->getRegions()) {
      for (auto type : region.getArgumentTypes()) {
        if (isIllegalType(type))
          return false;
      }
    }
    return true;
  };
  target.markUnknownOpDynamicallyLegal(isLegallyTypedOp);
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
