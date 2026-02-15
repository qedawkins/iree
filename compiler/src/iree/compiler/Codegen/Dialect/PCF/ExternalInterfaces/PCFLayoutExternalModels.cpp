// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/ExternalInterfaces/PCFLayoutExternalModels.h"

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir::iree_compiler::IREE::PCF {

namespace {

/// External model implementing PCF::LayoutAttrInterface on
/// VectorExt::NestedLayoutAttr. This bridges the VectorExt layout
/// representation into the PCF constraint system without adding a
/// compile-time dependency from PCF/IR to VectorExt/IR.
struct NestedLayoutAsPCFLayout final
    : public LayoutAttrInterface::ExternalModel<NestedLayoutAsPCFLayout,
                                                VectorExt::NestedLayoutAttr> {

  LogicalResult isValidLayout(Attribute attr, ShapedType tensorType,
                              Location loc) const {
    return cast<VectorExt::NestedLayoutAttr>(attr).isValidLayout(tensorType,
                                                                 loc);
  }

  int64_t getRank(Attribute attr) const {
    return cast<VectorExt::NestedLayoutAttr>(attr).getRank();
  }

  SmallVector<int64_t> getUndistributedShape(Attribute attr) const {
    return cast<VectorExt::NestedLayoutAttr>(attr).getUndistributedShape();
  }

  SmallVector<int64_t> getDistributedShape(Attribute attr) const {
    return cast<VectorExt::NestedLayoutAttr>(attr).getDistributedShape();
  }

  SmallVector<int64_t> getElementTile(Attribute attr) const {
    VectorExt::NestedLayoutAttr layout =
        cast<VectorExt::NestedLayoutAttr>(attr);
    return SmallVector<int64_t>(layout.getElementTile());
  }

  SmallVector<int64_t> getThreadTile(Attribute attr) const {
    VectorExt::NestedLayoutAttr layout =
        cast<VectorExt::NestedLayoutAttr>(attr);
    return SmallVector<int64_t>(layout.getThreadTile());
  }

  SmallVector<int64_t> getSubgroupTile(Attribute attr) const {
    VectorExt::NestedLayoutAttr layout =
        cast<VectorExt::NestedLayoutAttr>(attr);
    return SmallVector<int64_t>(layout.getSubgroupTile());
  }

  SmallVector<Value> computeThreadSliceOffsets(Attribute attr,
                                               OpBuilder &builder, Location loc,
                                               Value threadId, Value subgroupId,
                                               int64_t subgroupSize) const {
    // TODO: Implement full thread slice offset computation.
    // This requires delinearizing thread/subgroup IDs via
    // NestedLayoutAttr::computeThreadIds() and then computing per-dimension
    // offsets from the virtual IDs, strides, and tile sizes. The full
    // implementation should be completed before the distribution pass (Pair C)
    // starts, as it is the only consumer.
    llvm_unreachable("computeThreadSliceOffsets not yet implemented");
  }

  Attribute permute(Attribute attr, ArrayRef<int64_t> permutation) const {
    // NestedLayoutAttr::permute() returns VectorLayoutInterface.
    // Cast back to Attribute for the PCF interface.
    VectorExt::VectorLayoutInterface permuted =
        cast<VectorExt::NestedLayoutAttr>(attr).permute(permutation);
    return cast<Attribute>(permuted);
  }

  Attribute project(Attribute attr, ArrayRef<bool> droppedDims) const {
    // NestedLayoutAttr::project() returns VectorLayoutInterface.
    // Cast back to Attribute for the PCF interface.
    VectorExt::VectorLayoutInterface projected =
        cast<VectorExt::NestedLayoutAttr>(attr).project(droppedDims);
    return cast<Attribute>(projected);
  }

  bool isFullyDistributed(Attribute attr) const {
    VectorExt::NestedLayoutAttr layout =
        cast<VectorExt::NestedLayoutAttr>(attr);
    ArrayRef<int64_t> threadStrides = layout.getThreadStrides();
    ArrayRef<int64_t> subgroupStrides = layout.getSubgroupStrides();
    // A dimension is undistributed if both its thread and subgroup strides
    // are 0 and its thread/subgroup tiles are 1.
    for (int64_t i = 0, e = layout.getRank(); i < e; ++i) {
      bool threadDist =
          (threadStrides[i] != 0 || layout.getThreadTile()[i] != 1);
      bool sgDist =
          (subgroupStrides[i] != 0 || layout.getSubgroupTile()[i] != 1);
      if (!threadDist && !sgDist) {
        return false;
      }
    }
    return true;
  }

  SmallVector<bool> getUnconstrainedDims(Attribute attr) const {
    VectorExt::NestedLayoutAttr layout =
        cast<VectorExt::NestedLayoutAttr>(attr);
    SmallVector<bool> result(layout.getRank(), false);
    ArrayRef<int64_t> threadStrides = layout.getThreadStrides();
    ArrayRef<int64_t> subgroupStrides = layout.getSubgroupStrides();
    for (int64_t i = 0, e = layout.getRank(); i < e; ++i) {
      bool threadDist =
          (threadStrides[i] != 0 || layout.getThreadTile()[i] != 1);
      bool sgDist =
          (subgroupStrides[i] != 0 || layout.getSubgroupTile()[i] != 1);
      result[i] = !threadDist && !sgDist;
    }
    return result;
  }

  bool isCompatibleWith(Attribute attr, Attribute other) const {
    // Two NestedLayoutAttrs are compatible if they have the same
    // subgroup_tile, thread_tile, element_tile, and strides.
    // Batch and outer tiles may differ (they affect local computation
    // but not physical distribution).
    VectorExt::NestedLayoutAttr lhs = cast<VectorExt::NestedLayoutAttr>(attr);
    VectorExt::NestedLayoutAttr rhs =
        dyn_cast<VectorExt::NestedLayoutAttr>(other);
    if (!rhs) {
      return false;
    }
    return lhs.getSubgroupTile() == rhs.getSubgroupTile() &&
           lhs.getThreadTile() == rhs.getThreadTile() &&
           lhs.getElementTile() == rhs.getElementTile() &&
           lhs.getSubgroupStrides() == rhs.getSubgroupStrides() &&
           lhs.getThreadStrides() == rhs.getThreadStrides();
  }
};

} // namespace

void registerPCFLayoutExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::VectorExt::IREEVectorExtDialect *dialect) {
    VectorExt::NestedLayoutAttr::attachInterface<NestedLayoutAsPCFLayout>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::PCF
