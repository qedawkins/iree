// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

/// Traces through view-like tensor ops (extract_slice, expand_shape) to find
/// the root tensor defined outside the region. Returns the chain of ops from
/// root to the immediate source of the transfer_read (outermost first).
/// Returns std::nullopt if the chain does not trace to an outside tensor.
static std::optional<SmallVector<Operation *>>
traceToOutsideTensor(Value source, Region &region) {
  SmallVector<Operation *> chain;
  Value current = source;
  while (!isDefinedOutside(current, region)) {
    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      return std::nullopt;
    }
    if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
      chain.push_back(defOp);
      current = extractSlice.getSource();
    } else if (auto expandShape = dyn_cast<tensor::ExpandShapeOp>(defOp)) {
      chain.push_back(defOp);
      current = expandShape.getSrc();
    } else {
      // Unsupported op in the chain.
      return std::nullopt;
    }
  }
  // Reverse so chain goes from root outward to the transfer_read source.
  std::reverse(chain.begin(), chain.end());
  return chain;
}

/// Convert vector.transfer_read on a captured tensor (or a view-like chain
/// from a captured tensor) to iree_vector_ext.transfer_read on a pcf.sref.
/// Tensor extract_slice/expand_shape ops in the chain are converted to
/// pcf.subview/pcf.expand_shape on the sref.
static void convertInputReads(SharedExecutorOp sharedExec,
                              ScopeAttrInterface scope,
                              IRRewriter &rewriter) {
  Region &region = sharedExec.getRegion();
  DenseMap<Value, Value> srefCache;

  // Collect reads first to avoid modifying the IR while iterating.
  SmallVector<vector::TransferReadOp> readsToConvert;
  region.walk([&](vector::TransferReadOp readOp) {
    Value source = readOp.getBase();
    if (!isa<RankedTensorType>(source.getType())) {
      return;
    }
    // Direct outside read or traceable chain to outside.
    if (isDefinedOutside(source, region) ||
        traceToOutsideTensor(source, region).has_value()) {
      readsToConvert.push_back(readOp);
    }
  });

  for (vector::TransferReadOp readOp : readsToConvert) {
    Value source = readOp.getBase();
    Location loc = readOp.getLoc();

    // Find the root tensor and chain of view-like ops.
    Value rootTensor = source;
    SmallVector<Operation *> chain;
    if (isDefinedOutside(source, region)) {
      rootTensor = source;
    } else {
      auto maybeChain = traceToOutsideTensor(source, region);
      assert(maybeChain && "should have been filtered above");
      chain = std::move(*maybeChain);
      // Walk to the root.
      Value current = source;
      while (!isDefinedOutside(current, region)) {
        Operation *defOp = current.getDefiningOp();
        if (auto es = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
          current = es.getSource();
        } else if (auto exp = dyn_cast<tensor::ExpandShapeOp>(defOp)) {
          current = exp.getSrc();
        } else {
          break;
        }
      }
      rootTensor = current;
    }

    // Create pcf.to_sref for the root tensor.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&region.front());
      getOrCreateSref(rewriter, loc, rootTensor, scope, srefCache);
    }
    Value sref = srefCache[rootTensor];

    // Replay the chain as PCF ops on the sref.
    rewriter.setInsertionPoint(readOp);
    for (Operation *op : chain) {
      if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(op)) {
        // Create pcf.subview.
        auto resultTensorType =
            cast<RankedTensorType>(extractSlice.getResult().getType());
        ShapedRefType resultSrefType = ShapedRefType::get(
            rewriter.getContext(), resultTensorType.getShape(),
            resultTensorType.getElementType(), scope);
        sref = SubviewOp::create(
            rewriter, loc, resultSrefType, sref,
            extractSlice.getMixedOffsets(), extractSlice.getMixedSizes(),
            extractSlice.getMixedStrides());
      } else if (auto expandShape = dyn_cast<tensor::ExpandShapeOp>(op)) {
        // Create pcf.expand_shape.
        auto resultTensorType =
            cast<RankedTensorType>(expandShape.getResult().getType());
        ShapedRefType resultSrefType = ShapedRefType::get(
            rewriter.getContext(), resultTensorType.getShape(),
            resultTensorType.getElementType(), scope);
        SmallVector<Value> outputShape;
        for (int64_t dim : resultTensorType.getShape()) {
          outputShape.push_back(
              rewriter.create<arith::ConstantIndexOp>(loc, dim));
        }
        sref = ExpandShapeOp::create(
            rewriter, loc, resultSrefType, sref,
            expandShape.getReassociation(), outputShape);
      }
    }

    // Create iree_vector_ext.transfer_read on the final sref.
    VectorType vecType = readOp.getVectorType();
    SmallVector<bool> inBounds;
    for (Attribute attr : readOp.getInBounds()) {
      inBounds.push_back(cast<BoolAttr>(attr).getValue());
    }
    AffineMap permMap = readOp.getPermutationMap();
    Value newRead = TransferReadOp::create(
        rewriter, loc, vecType, sref, readOp.getIndices(),
        readOp.getPadding(), inBounds, permMap, readOp.getMask());
    rewriter.replaceOp(readOp, newRead);

    // Clean up dead tensor ops in the chain.
    for (Operation *op : llvm::reverse(chain)) {
      if (op->use_empty()) {
        rewriter.eraseOp(op);
      }
    }
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
