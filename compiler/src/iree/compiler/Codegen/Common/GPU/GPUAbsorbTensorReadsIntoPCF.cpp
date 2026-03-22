// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUABSORBTENSORREADSINTOPCFPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

using namespace IREE::PCF;
using namespace IREE::VectorExt;

namespace {

/// Returns true if `val` is defined outside of `region`.
static bool isDefinedOutside(Value val, Region &region) {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    return !region.isAncestor(blockArg.getOwner()->getParent());
  }
  return !region.isAncestor(val.getDefiningOp()->getParentRegion());
}

/// Creates a pcf.to_sref op for a tensor value, caching to avoid duplicates.
static Value getOrCreateSref(OpBuilder &builder, Location loc, Value tensor,
                             ScopeAttrInterface scope,
                             DenseMap<Value, Value> &srefCache) {
  auto it = srefCache.find(tensor);
  if (it != srefCache.end()) {
    return it->second;
  }

  auto tensorType = cast<RankedTensorType>(tensor.getType());
  ShapedRefType srefType = ShapedRefType::get(
      builder.getContext(), tensorType.getShape(),
      tensorType.getElementType(), scope);
  Value sref = ToSrefOp::create(builder, loc, srefType, tensor);
  srefCache[tensor] = sref;
  return sref;
}

/// Convert vector.transfer_read on a captured tensor to
/// iree_vector_ext.transfer_read on a pcf.sref.
static void convertInputReads(SharedExecutorOp sharedExec,
                              ScopeAttrInterface scope,
                              IRRewriter &rewriter) {
  Region &region = sharedExec.getRegion();
  DenseMap<Value, Value> srefCache;

  // Collect reads first to avoid modifying the IR while iterating.
  SmallVector<vector::TransferReadOp> readsToConvert;
  region.walk([&](vector::TransferReadOp readOp) {
    Value source = readOp.getBase();
    if (isa<RankedTensorType>(source.getType()) &&
        isDefinedOutside(source, region)) {
      readsToConvert.push_back(readOp);
    }
  });

  for (vector::TransferReadOp readOp : readsToConvert) {
    Value source = readOp.getBase();
    rewriter.setInsertionPoint(readOp);

    // Create pcf.to_sref at the start of the region body.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&region.front());
    Value sref = getOrCreateSref(rewriter, readOp.getLoc(), source, scope,
                                 srefCache);

    // Create iree_vector_ext.transfer_read on the sref.
    rewriter.setInsertionPoint(readOp);
    VectorType vecType = readOp.getVectorType();

    // Extract in_bounds as bool array.
    SmallVector<bool> inBounds;
    for (Attribute attr : readOp.getInBounds()) {
      inBounds.push_back(cast<BoolAttr>(attr).getValue());
    }

    AffineMap permMap = readOp.getPermutationMap();
    Value newRead = TransferReadOp::create(
        rewriter, readOp.getLoc(), vecType, sref, readOp.getIndices(),
        readOp.getPadding(), inBounds, permMap, readOp.getMask());
    rewriter.replaceOp(readOp, newRead);
  }
}

/// Convert vector.transfer_write + pcf.write_slice to
/// iree_vector_ext.transfer_write on sref.
static void convertOutputWrites(SharedExecutorOp sharedExec,
                                IRRewriter &rewriter) {
  Region &region = sharedExec.getRegion();

  // Collect write_slice ops whose source is a vector.transfer_write result.
  SmallVector<WriteSliceOp> writeSlicesToConvert;
  region.walk([&](WriteSliceOp writeSlice) {
    Value source = writeSlice.getSource();
    if (source.getDefiningOp<vector::TransferWriteOp>()) {
      writeSlicesToConvert.push_back(writeSlice);
    }
  });

  for (WriteSliceOp writeSlice : writeSlicesToConvert) {
    auto transferWrite =
        writeSlice.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!transferWrite) {
      continue;
    }

    rewriter.setInsertionPoint(writeSlice);
    Location loc = writeSlice.getLoc();

    // Get the vector being written.
    Value vec = transferWrite.getVector();
    // Get the destination sref.
    Value destSref = writeSlice.getDest();

    // Compose offsets: transfer_write offsets + write_slice offsets.
    // The transfer_write typically writes to [0, 0, ...] of a temporary
    // tensor, so the composed offset is just the write_slice offset.
    SmallVector<OpFoldResult> writeSliceOffsets =
        writeSlice.getMixedOffsets();
    SmallVector<Value> transferWriteIndices(transferWrite.getIndices());

    // Compose: add transfer_write indices to write_slice offsets.
    SmallVector<Value> composedIndices;
    for (auto [twIdx, wsOffset] :
         llvm::zip_equal(transferWriteIndices, writeSliceOffsets)) {
      if (auto constOffset = getConstantIntValue(wsOffset)) {
        if (*constOffset == 0) {
          composedIndices.push_back(twIdx);
          continue;
        }
      }
      // Check if the transfer_write index is zero.
      if (auto constIdx = getConstantIntValue(twIdx)) {
        if (*constIdx == 0) {
          composedIndices.push_back(
              getValueOrCreateConstantIndexOp(rewriter, loc, wsOffset));
          continue;
        }
      }
      // General case: add both.
      Value wsVal = getValueOrCreateConstantIndexOp(rewriter, loc, wsOffset);
      Value sum = arith::AddIOp::create(rewriter, loc, twIdx, wsVal);
      composedIndices.push_back(sum);
    }

    // Extract in_bounds from the transfer_write.
    SmallVector<bool> inBounds;
    for (Attribute attr : transferWrite.getInBounds()) {
      inBounds.push_back(cast<BoolAttr>(attr).getValue());
    }

    AffineMap permMap = transferWrite.getPermutationMap();
    TransferWriteOp::create(rewriter, loc, vec, destSref, composedIndices,
                            inBounds, permMap, transferWrite.getMask());

    // Erase the write_slice and the transfer_write if it has no other uses.
    rewriter.eraseOp(writeSlice);
    if (transferWrite->use_empty()) {
      rewriter.eraseOp(transferWrite);
    }
  }
}

struct GPUAbsorbTensorReadsIntoPCFPass final
    : impl::GPUAbsorbTensorReadsIntoPCFPassBase<
          GPUAbsorbTensorReadsIntoPCFPass> {
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    getOperation()->walk([&](SharedExecutorOp sharedExec) {
      ScopeAttrInterface scope = sharedExec.getScope();
      convertInputReads(sharedExec, scope, rewriter);
      convertOutputWrites(sharedExec, rewriter);
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
