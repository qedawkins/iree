// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/ExternalInterfaces/GPUScopeExternalModels.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/ConversionDialectInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::IREE::GPU {

namespace {

/// Dialect interface implementation for IREE GPU dialect that loads dependent
/// dialects for structured lowerings (which generate counts/ids).
class IREEGPUConversionDialectInterface final
    : public PCFConversionDialectInterface {
public:
  using PCFConversionDialectInterface::PCFConversionDialectInterface;

  void
  loadStructuralLoweringDependentDialects(MLIRContext *context) const override {
    // Load the upstream GPU dialect for gpu.subgroup_id, gpu.lane_id, etc.
    context->loadDialect<gpu::GPUDialect>();
  }
  void loadSRefLoweringDependentDialects(MLIRContext *context) const override {
    // Load the upstream GPU dialect for shared memory space attribute.
    context->loadDialect<gpu::GPUDialect>();
  }
};

/// External model for SubgroupScopeAttr implementing ScopeAttrInterface.
/// Uses gpu.num_subgroups for worker count and gpu.subgroup_id for worker ID.
/// Allocations at this scope use GPU shared memory space.
struct SubgroupScopeModel
    : PCF::ScopeAttrInterface::ExternalModel<SubgroupScopeModel,
                                             GPU::SubgroupScopeAttr> {
  SmallVector<Value> getWorkerCounts(Attribute attr, OpBuilder &builder,
                                     Location loc, int64_t numIds) const {
    assert(numIds >= 1 && "expected at least one requested worker count");
    SmallVector<Value> counts(numIds, Value());
    // There is only a single id, so set the front to the number of subgroups.
    counts.front() =
        gpu::NumSubgroupsOp::create(builder, loc, /*upper_bound=*/nullptr);

    // Pad remaining counts with 1.
    if (numIds > 1) {
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      llvm::fill(drop_begin(counts), one);
    }
    return counts;
  }

  SmallVector<Value> getWorkerIDs(Attribute attr, OpBuilder &builder,
                                  Location loc, int64_t numIds) const {
    assert(numIds >= 1 && "expected at least one requested worker id");
    SmallVector<Value> ids(numIds, Value());
    // The fastest varying ID is just the subgroup id.
    ids.front() =
        gpu::SubgroupIdOp::create(builder, loc, /*upper_bound=*/nullptr);

    // Pad remaining ids with 0.
    if (numIds > 1) {
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      llvm::fill(drop_begin(ids), zero);
    }
    return ids;
  }

  LogicalResult addBarrier(Attribute attr, OpBuilder &builder) const {
    gpu::BarrierOp::create(builder, builder.getUnknownLoc(),
                           gpu::AddressSpace::Workgroup);
    return success();
  }

  FailureOr<Attribute> getAllocMemSpace(Attribute attr,
                                        MLIRContext *context) const {
    return gpu::AddressSpaceAttr::get(context, gpu::AddressSpace::Workgroup);
  }

  int64_t getNativeNumProcessorIds(Attribute attr) const {
    // SubgroupScope natively provides a single 1D processor ID (subgroup_id).
    return 1;
  }
};

/// External model for ThreadScopeAttr implementing ScopeAttrInterface.
/// Uses linearized 3D thread IDs for worker ID and product of block
/// dimensions for worker count. Allocations use workgroup shared memory.
struct ThreadScopeModel
    : PCF::ScopeAttrInterface::ExternalModel<ThreadScopeModel,
                                             GPU::ThreadScopeAttr> {
  SmallVector<Value> getWorkerCounts(Attribute attr, OpBuilder &builder,
                                     Location loc, int64_t numIds) const {
    assert(numIds >= 1 && "expected at least one requested worker count");
    SmallVector<Value> counts(numIds, Value());
    // Total thread count = dimX * dimY * dimZ.
    Value dimX = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::x);
    Value dimY = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::y);
    Value dimZ = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::z);
    Value total = arith::MulIOp::create(builder, loc, dimX, dimY);
    total = arith::MulIOp::create(builder, loc, total, dimZ);
    counts.front() = total;

    // Pad remaining counts with 1.
    if (numIds > 1) {
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      llvm::fill(drop_begin(counts), one);
    }
    return counts;
  }

  SmallVector<Value> getWorkerIDs(Attribute attr, OpBuilder &builder,
                                  Location loc, int64_t numIds) const {
    assert(numIds >= 1 && "expected at least one requested worker id");
    SmallVector<Value> ids(numIds, Value());
    // Linearize 3D thread IDs: tid = z * (dimX * dimY) + y * dimX + x.
    Value tidX = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
    Value tidY = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::y);
    Value tidZ = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::z);
    Value dimX = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::x);
    Value dimY = gpu::BlockDimOp::create(builder, loc, gpu::Dimension::y);
    Value xySize = arith::MulIOp::create(builder, loc, dimX, dimY);
    Value linearId = arith::MulIOp::create(builder, loc, tidZ, xySize);
    Value yOff = arith::MulIOp::create(builder, loc, tidY, dimX);
    linearId = arith::AddIOp::create(builder, loc, linearId, yOff);
    linearId = arith::AddIOp::create(builder, loc, linearId, tidX);
    ids.front() = linearId;

    // Pad remaining ids with 0.
    if (numIds > 1) {
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      llvm::fill(drop_begin(ids), zero);
    }
    return ids;
  }

  LogicalResult addBarrier(Attribute attr, OpBuilder &builder) const {
    gpu::BarrierOp::create(builder, builder.getUnknownLoc(),
                           gpu::AddressSpace::Workgroup);
    return success();
  }

  FailureOr<Attribute> getAllocMemSpace(Attribute attr,
                                        MLIRContext *context) const {
    return gpu::AddressSpaceAttr::get(context, gpu::AddressSpace::Workgroup);
  }

  int64_t getNativeNumProcessorIds(Attribute attr) const {
    // Thread scope provides a single linearized ID.
    return 1;
  }
};

/// External model for LaneScopeAttr implementing ScopeAttrInterface.
/// Uses gpu.subgroup_size for worker count and gpu.lane_id for worker ID.
/// Allocations at this scope are not yet supported.
struct LaneScopeModel
    : PCF::ScopeAttrInterface::ExternalModel<LaneScopeModel,
                                             GPU::LaneScopeAttr> {
  SmallVector<Value> getWorkerCounts(Attribute attr, OpBuilder &builder,
                                     Location loc, int64_t numIds) const {
    assert(numIds >= 1 && "expected at least one requested worker count");
    SmallVector<Value> counts(numIds, Value());
    // There is only a single id, so set the front to the subgroup size.
    counts.front() =
        gpu::SubgroupSizeOp::create(builder, loc, /*upper_bound=*/nullptr);

    // Pad remaining counts with 1.
    if (numIds > 1) {
      Value one = arith::ConstantIndexOp::create(builder, loc, 1);
      llvm::fill(drop_begin(counts), one);
    }
    return counts;
  }

  SmallVector<Value> getWorkerIDs(Attribute attr, OpBuilder &builder,
                                  Location loc, int64_t numIds) const {
    assert(numIds >= 1 && "expected at least one requested worker id");
    SmallVector<Value> ids(numIds, Value());
    // The fastest varying ID is just the lane id.
    ids.front() = gpu::LaneIdOp::create(builder, loc, /*upper_bound=*/nullptr);

    // Pad remaining ids with 0.
    if (numIds > 1) {
      Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
      llvm::fill(drop_begin(ids), zero);
    }
    return ids;
  }

  LogicalResult addBarrier(Attribute attr, OpBuilder &builder) const {
    // Lane-level barriers are not yet supported (missing a generic fence).
    return failure();
  }

  FailureOr<Attribute> getAllocMemSpace(Attribute attr,
                                        MLIRContext *context) const {
    // Lane scope allocations are not supported - need custom allocation
    // logic to allocate + subview.
    return failure();
  }

  int64_t getNativeNumProcessorIds(Attribute attr) const {
    // LaneScope natively provides a single 1D processor ID (lane_id).
    return 1;
  }
};

} // namespace

void registerGPUScopeExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, GPU::IREEGPUDialect *dialect) {
        GPU::SubgroupScopeAttr::attachInterface<SubgroupScopeModel>(*context);
        GPU::LaneScopeAttr::attachInterface<LaneScopeModel>(*context);
        GPU::ThreadScopeAttr::attachInterface<ThreadScopeModel>(*context);

        // Register the PCF conversion dialect interface to load required
        // dialects during PCF lowering passes.
        dialect->addInterfaces<IREEGPUConversionDialectInterface>();
      });
}

} // namespace mlir::iree_compiler::IREE::GPU
