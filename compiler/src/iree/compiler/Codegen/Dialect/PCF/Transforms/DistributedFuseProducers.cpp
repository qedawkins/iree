// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTilingInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-pcf-distributed-fuse-producers"

namespace mlir::iree_compiler::IREE::PCF {

namespace {

//===----------------------------------------------------------------------===//
// Distributed producer match impl
//===----------------------------------------------------------------------===//

/// For a given result of a scoped op, find all pcf.read_slice ops on the
/// corresponding sref region argument that read the init value.
template <typename OpTy>
static LogicalResult
lookupConsumerSlices(OpResult result,
                     SmallVectorImpl<PCF::ReadSliceOp> &slices) {
  OpTy owner = cast<OpTy>(result.getOwner());
  Value tiedArg = owner.getRegionRefArgs()[result.getResultNumber()];

  // Collect all read_slice ops on this sref arg. Write_slice users are
  // allowed but not tracked. Any other user type prevents fusion.
  SmallVector<PCF::ReadSliceOp> reads;
  for (Operation *user : tiedArg.getUsers()) {
    if (isa<PCF::WriteSliceOp>(user)) {
      continue;
    }
    auto readOp = dyn_cast<PCF::ReadSliceOp>(user);
    if (!readOp || !readOp.hasUnitStride()) {
      return failure();
    }
    reads.push_back(readOp);
  }

  // With SyncOnReturn semantics, the only case where a read does not see the
  // init value is when there are overlapping reads and writes, which is
  // explicitly undefined behavior. We may therefore optimize as though all
  // reads see the init value.
  auto srefType = cast<PCF::ShapedRefType>(tiedArg.getType());
  if (!srefType.isReturnOnlySync()) {
    return failure();
  }

  llvm::append_range(slices, reads);
  return success();
}

template <typename OpTy>
static LogicalResult
matchDistributedProducerImpl(RewriterBase &rewriter, OpTy scopedOp,
                             DistributedProducerFusionParams &params) {
  for (int64_t i = 0, e = scopedOp->getNumResults(); i < e; ++i) {
    OpOperand *tiedInit = scopedOp.getTiedInit(i);
    if (!tiedInit) {
      continue;
    }

    Value initValue = tiedInit->get();
    Operation *defOp = initValue.getDefiningOp();
    if (!defOp) {
      continue;
    }

    // Must implement PCFTilingOpInterface.
    auto producer = dyn_cast<PCFTilingOpInterface>(defOp);
    if (!producer) {
      continue;
    }

    // Must be DPS with single result.
    auto dpsProducer = dyn_cast<DestinationStyleOpInterface>(defOp);
    if (!dpsProducer || defOp->getNumResults() != 1) {
      continue;
    }

    // Verify dominance of producer's operands.
    DominanceInfo dominanceInfo(scopedOp->getParentOp());
    bool allDominate = llvm::all_of(defOp->getOperands(), [&](Value v) {
      return dominanceInfo.dominates(v, scopedOp);
    });
    if (!allDominate) {
      continue;
    }

    // Find read_slice ops on the corresponding sref arg.
    SmallVector<PCF::ReadSliceOp> readSlices;
    if (failed(
            lookupConsumerSlices<OpTy>(scopedOp->getOpResult(i), readSlices))) {
      continue;
    }

    if (readSlices.empty()) {
      continue;
    }

    params.resultIndex = i;
    params.producer = defOp;
    params.readSlices = std::move(readSlices);
    return success();
  }
  return rewriter.notifyMatchFailure(scopedOp,
                                     "no fusable distributed producer found");
}

//===----------------------------------------------------------------------===//
// Distributed producer fusion impl
//===----------------------------------------------------------------------===//

template <typename OpTy>
static void
fuseDistributedProducerImpl(RewriterBase &rewriter, OpTy scopedOp,
                            const DistributedProducerFusionParams &params) {
  auto producer = cast<PCFTilingOpInterface>(params.producer);
  Operation *producerOp = params.producer;

  // Step 1: Replace the tied init with the producer's DPS init.
  auto dpsProducer = cast<DestinationStyleOpInterface>(producerOp);
  Value producerDPSInit = dpsProducer.getDpsInits()[0];

  OpOperand *tiedInit = scopedOp.getTiedInit(params.resultIndex);
  rewriter.modifyOpInPlace(scopedOp,
                           [&]() { tiedInit->assign(producerDPSInit); });

  // Step 2: Identify the producer's non-fused input operands that need
  // readonly sref args.
  SmallVector<unsigned> tileableIndices = producer.getTileableOperandIndices();
  DenseSet<unsigned> tileableSet(tileableIndices.begin(),
                                 tileableIndices.end());

  // The DPS init operand index (result 0 init).
  DenseSet<unsigned> dpsInitIndices;
  for (OpOperand &init : dpsProducer.getDpsInitsMutable()) {
    dpsInitIndices.insert(init.getOperandNumber());
  }

  // Collect readonly inits for non-DPS-init tileable operands.
  SmallVector<Value> newReadonlyInits;
  SmallVector<int64_t> operandToNewReadonlyIdx(producerOp->getNumOperands(),
                                               -1);
  for (int64_t i = 0, e = producerOp->getNumOperands(); i < e; ++i) {
    unsigned idx = static_cast<unsigned>(i);
    if (dpsInitIndices.contains(idx)) {
      // DPS init is already represented by the existing readwrite sref.
      continue;
    }
    if (!tileableSet.contains(idx)) {
      // Not tileable; will pass through original value.
      continue;
    }
    // Only shaped (tensor) operands become sref args. Scalar operands
    // (e.g. the fill value in linalg.fill) are passed through directly.
    if (!isa<ShapedType>(producerOp->getOperand(i).getType())) {
      continue;
    }
    operandToNewReadonlyIdx[i] = static_cast<int64_t>(newReadonlyInits.size());
    newReadonlyInits.push_back(producerOp->getOperand(i));
  }

  // Step 3: If there are new readonly inits, create a new scoped op with
  // additional readonly sref args.
  SmallVector<BlockArgument> newReadonlyBlockArgs;
  OpTy newScopedOp = scopedOp;
  if (!newReadonlyInits.empty()) {
    newScopedOp = addReadonlyArgs(rewriter, scopedOp, newReadonlyInits,
                                  newReadonlyBlockArgs);
  }

  // Step 4: At each read_slice site, generate the distributed producer.
  for (PCF::ReadSliceOp readSlice : params.readSlices) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(readSlice);
    Location loc = readSlice.getLoc();

    SmallVector<OpFoldResult> offsets = readSlice.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = readSlice.getMixedSizes();

    // Build DistributedOperandInfo for the producer's operands.
    SmallVector<DistributedOperandInfo> operandInfo;
    operandInfo.reserve(producerOp->getNumOperands());
    for (int64_t i = 0, e = producerOp->getNumOperands(); i < e; ++i) {
      unsigned idx = static_cast<unsigned>(i);

      if (dpsInitIndices.contains(idx)) {
        // DPS init operand: read from the existing readwrite sref using the
        // read_slice's offsets/sizes.
        Value sref = newScopedOp.getRegionRefArgs()[params.resultIndex];
        auto srefType = cast<ShapedRefType>(sref.getType());
        int64_t srefRank = srefType.getRank();

        // Build read type from the slice sizes.
        SmallVector<int64_t> staticSizes;
        staticSizes.reserve(srefRank);
        for (int64_t dim = 0, de = std::min(srefRank,
                                            static_cast<int64_t>(sizes.size()));
             dim < de; ++dim) {
          if (auto attr = dyn_cast<Attribute>(sizes[dim])) {
            staticSizes.push_back(cast<IntegerAttr>(attr).getInt());
          } else {
            staticSizes.push_back(ShapedType::kDynamic);
          }
        }
        for (int64_t dim = staticSizes.size(); dim < srefRank; ++dim) {
          staticSizes.push_back(srefType.getDimSize(dim));
        }
        RankedTensorType readType =
            RankedTensorType::get(staticSizes, srefType.getElementType());
        SmallVector<OpFoldResult> strides(srefRank, rewriter.getIndexAttr(1));

        // Pad offsets/sizes to match sref rank.
        SmallVector<OpFoldResult> readOffsets(offsets);
        SmallVector<OpFoldResult> readSizes(sizes);
        while (static_cast<int64_t>(readOffsets.size()) < srefRank) {
          readOffsets.push_back(rewriter.getIndexAttr(0));
        }
        while (static_cast<int64_t>(readSizes.size()) < srefRank) {
          readSizes.push_back(
              rewriter.getIndexAttr(srefType.getDimSize(readSizes.size())));
        }

        Value tile = ReadSliceOp::create(rewriter, loc, readType, sref,
                                         readOffsets, readSizes, strides);
        operandInfo.push_back({tile, /*isTile=*/true});
        continue;
      }

      if (operandToNewReadonlyIdx[i] >= 0) {
        // Non-fused tileable input: read from the new readonly sref arg.
        BlockArgument sref = newReadonlyBlockArgs[operandToNewReadonlyIdx[i]];
        auto srefType = cast<ShapedRefType>(sref.getType());
        int64_t srefRank = srefType.getRank();

        // Use the read_slice's offsets/sizes as the iteration domain tile.
        // The producer's getDistributedImplementation will compute
        // operand-specific tile positions from the iteration domain tile.
        SmallVector<OpFoldResult> readOffsets(offsets);
        SmallVector<OpFoldResult> readSizes(sizes);

        // Build read type.
        SmallVector<int64_t> staticSizes;
        staticSizes.reserve(srefRank);
        for (int64_t dim = 0, de = std::min(srefRank, static_cast<int64_t>(
                                                          readSizes.size()));
             dim < de; ++dim) {
          if (auto attr = dyn_cast<Attribute>(readSizes[dim])) {
            staticSizes.push_back(cast<IntegerAttr>(attr).getInt());
          } else {
            staticSizes.push_back(ShapedType::kDynamic);
          }
        }
        for (int64_t dim = staticSizes.size(); dim < srefRank; ++dim) {
          staticSizes.push_back(srefType.getDimSize(dim));
        }
        RankedTensorType readType =
            RankedTensorType::get(staticSizes, srefType.getElementType());
        SmallVector<OpFoldResult> strides(srefRank, rewriter.getIndexAttr(1));

        // Pad offsets/sizes to match sref rank.
        while (static_cast<int64_t>(readOffsets.size()) < srefRank) {
          readOffsets.push_back(rewriter.getIndexAttr(0));
        }
        while (static_cast<int64_t>(readSizes.size()) < srefRank) {
          readSizes.push_back(
              rewriter.getIndexAttr(srefType.getDimSize(readSizes.size())));
        }

        Value tile = ReadSliceOp::create(rewriter, loc, readType, sref,
                                         readOffsets, readSizes, strides);
        operandInfo.push_back({tile, /*isTile=*/true});
        continue;
      }

      // Non-tileable operand: pass through original value.
      operandInfo.push_back({producerOp->getOperand(i), /*isTile=*/false});
    }

    // Build DistributedResultInfo: null destSref means return a tile.
    SmallVector<DistributedResultInfo> resultInfo;
    resultInfo.push_back({/*destSref=*/Value()});

    // Check feasibility, then call getDistributedImplementation.
    assert(succeeded(producer.canDistribute(offsets, sizes, operandInfo,
                                            resultInfo)) &&
           "canDistribute check failed unexpectedly during producer fusion");
    FailureOr<TilingResult> tiledResult = producer.getDistributedImplementation(
        rewriter, offsets, sizes, operandInfo, resultInfo);
    assert(succeeded(tiledResult) &&
           "unexpected distributed implementation failure");

    Value replacement = tiledResult->tiledValues[0];

    // If the read_slice returns a vector, wrap the tensor result in
    // vector.transfer_read.
    if (auto vectorType = dyn_cast<VectorType>(readSlice.getType())) {
      auto tensorType = cast<RankedTensorType>(replacement.getType());
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      SmallVector<Value> indices(tensorType.getRank(), zero);
      SmallVector<bool> inBounds(tensorType.getRank(), true);
      replacement = vector::TransferReadOp::create(
          rewriter, loc, vectorType, replacement, indices,
          /*padding=*/std::nullopt, inBounds);
    }

    rewriter.replaceOp(readSlice, replacement);
  }

  // Step 5: Erase the original producer if it has no remaining uses.
  if (params.producer->use_empty()) {
    rewriter.eraseOp(params.producer);
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API Specializations
//===----------------------------------------------------------------------===//

LogicalResult
matchDistributedProducer(RewriterBase &rewriter, PCF::GenericOp genericOp,
                         DistributedProducerFusionParams &params) {
  return matchDistributedProducerImpl(rewriter, genericOp, params);
}

LogicalResult
matchDistributedProducer(RewriterBase &rewriter, PCF::LoopOp loopOp,
                         DistributedProducerFusionParams &params) {
  return matchDistributedProducerImpl(rewriter, loopOp, params);
}

void fuseDistributedProducer(RewriterBase &rewriter, PCF::GenericOp genericOp,
                             const DistributedProducerFusionParams &params) {
  fuseDistributedProducerImpl(rewriter, genericOp, params);
}

void fuseDistributedProducer(RewriterBase &rewriter, PCF::LoopOp loopOp,
                             const DistributedProducerFusionParams &params) {
  fuseDistributedProducerImpl(rewriter, loopOp, params);
}

} // namespace mlir::iree_compiler::IREE::PCF
