// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Utils/RewriteUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "iree-pcf-convert-forall-to-pcf"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_TESTCONVERTFORALLTOLOOPSPASS
#define GEN_PASS_DEF_TESTCONVERTFORALLTOGENERICNESTPASS
#define GEN_PASS_DEF_TESTFOLDFORALLINTOPCFLOOPPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Shared Utilities
//===----------------------------------------------------------------------===//

/// Returns true if the forall op has LocalMappingAttr mapping attributes,
/// or the mapping is empty/not present.
static bool hasEmptyOrLocalMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  if (!mapping || mapping->empty()) {
    return true;
  }
  return llvm::all_of(mapping.value(),
                      llvm::IsaPred<IREE::Codegen::LocalMappingAttr>);
}

/// Returns the permutation from mapping attributes based on their relative
/// processor IDs. Lower IDs indicate faster varying dimensions. The returned
/// permutation maps from the fastest-to-slowest order back to the original
/// forall dimension order.
///
/// Example: mapping = [local_mapping<1>, local_mapping<0>]
/// - local_mapping<0> has lower ID than local_mapping<1>
/// - So dimension 1 (id 0) is faster, dimension 0 (id 1) is slower
/// - Returns [1, 0]: position 0 in linearized order maps to dim 1, etc.
static SmallVector<int64_t> getMappingPermutation(ArrayAttr mapping) {
  auto mappingRange = mapping.getAsRange<DeviceMappingAttrInterface>();
  int64_t mappingBase =
      cast<DeviceMappingAttrInterface>(
          *std::min_element(mappingRange.begin(), mappingRange.end(),
                            [](auto a, auto b) {
                              return a.getMappingId() < b.getMappingId();
                            }))
          .getMappingId();
  return llvm::map_to_vector(
      mappingRange, [&](auto a) { return a.getMappingId() - mappingBase; });
}

/// Returns the permutation for processor ID ordering from the forall's mapping.
/// For unspecified or empty mappings, returns a reversed sequence (assumes
/// natural fastest-to-slowest is reverse of dimension order).
static FailureOr<SmallVector<int64_t>>
getProcessorIdPermutation(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mappingAttr = forallOp.getMapping();
  // Unspecified mappings indicate sequential foralls which we can choose the
  // iteration order for.
  if (!mappingAttr) {
    return llvm::to_vector(llvm::reverse(llvm::seq(forallOp.getRank())));
  }
  // Empty mappings are unsupported at the moment. It's unclear when a forall
  // with an empty mapping would be useful or important.
  if (mappingAttr.value().empty()) {
    return SmallVector<int64_t>{};
  }

  SmallVector<int64_t> perm = getMappingPermutation(mappingAttr.value());
  if (!isPermutationVector(perm)) {
    return failure();
  }
  return perm;
}

//===----------------------------------------------------------------------===//
// scf.forall + pcf.loop -> pcf.generic fold helpers
//===----------------------------------------------------------------------===//

/// Validates that the forall terminator has the expected structure for folding:
/// - All ops are tensor.parallel_insert_slice
/// - All insert sources come from the same pcf.loop result
/// - All insert destinations are forall shared_outs
/// Returns the found pcf.loop on success.
static FailureOr<PCF::LoopOp> matchFoldTerminator(scf::ForallOp forallOp) {
  auto terminator =
      cast<scf::InParallelOp>(forallOp.getRegion().front().getTerminator());

  PCF::LoopOp foundLoop = nullptr;

  for (Operation &op : terminator.getYieldingOps()) {
    // All ops must be tensor.parallel_insert_slice.
    auto insertSliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
    if (!insertSliceOp) {
      return failure();
    }

    // Source must be from a pcf.loop result.
    auto loopResult = dyn_cast<OpResult>(insertSliceOp.getSource());
    if (!loopResult) {
      return failure();
    }

    auto currentLoop = dyn_cast<PCF::LoopOp>(loopResult.getOwner());
    if (!currentLoop) {
      return failure();
    }

    // All inserts must reference the same pcf.loop.
    if (foundLoop && foundLoop != currentLoop) {
      return failure();
    }
    foundLoop = currentLoop;

    // Destination must be a shared_out of the forall.
    auto destArg = dyn_cast<BlockArgument>(insertSliceOp.getDest());
    if (!destArg || destArg.getOwner() != &forallOp.getRegion().front()) {
      return failure();
    }

    // Verify it's a shared_out (comes after induction vars).
    if (destArg.getArgNumber() < forallOp.getRank()) {
      return failure();
    }
  }

  if (!foundLoop) {
    return failure();
  }

  return foundLoop;
}

/// Validates the pcf.loop structure for folding:
/// - Single count argument (linearized)
/// - Count value dominates the forall
/// - Loop is last op before terminator
static LogicalResult matchFoldPCFLoop(scf::ForallOp forallOp,
                                      PCF::LoopOp loopOp) {
  // Single count argument required.
  if (loopOp.getCount().size() != 1) {
    return failure();
  }

  // Count must dominate the forall.
  Value countValue = loopOp.getCount()[0];
  if (auto defOp = countValue.getDefiningOp()) {
    if (!defOp->isBeforeInBlock(forallOp)) {
      return failure();
    }
  }

  // Loop must be last op before terminator.
  Operation *lastOp =
      forallOp.getRegion().front().getTerminator()->getPrevNode();
  if (lastOp != loopOp) {
    return failure();
  }

  return success();
}

/// Validates pcf.loop region ref args:
/// - All users are pcf.write_slice ops
/// - Ref args have SyncOnReturnAttr sync scope
/// Populates writesByResult with write_slice ops grouped by result index.
static LogicalResult matchFoldWriteSlices(
    PCF::LoopOp loopOp,
    DenseMap<unsigned, SmallVector<PCF::WriteSliceOp>> &writesByResult) {
  for (auto [idx, refArg] : llvm::enumerate(loopOp.getRegionRefArgs())) {
    // Check sync scope is sync_on_return.
    auto srefType = cast<PCF::ShapedRefType>(refArg.getType());
    auto syncScope = srefType.getSyncScope();
    if (!isa_and_nonnull<PCF::SyncOnReturnAttr>(syncScope)) {
      return failure();
    }

    // All users must be write_slice ops.
    for (Operation *user : refArg.getUsers()) {
      auto writeOp = dyn_cast<PCF::WriteSliceOp>(user);
      if (!writeOp) {
        return failure();
      }
      writesByResult[idx].push_back(writeOp);
    }
  }

  return success();
}

/// Core implementation of foldForallIntoPCFLoop after matching succeeds.
/// Note: writesByResult is validated during matching but not directly used here
/// since we iterate over ref arg users to find write_slice ops in their final
/// positions after the IR has been restructured.
static PCF::GenericOp
foldForallIntoPCFLoopImpl(RewriterBase &rewriter, scf::ForallOp forallOp,
                          PCF::LoopOp loopOp) {
  Location loc = forallOp.getLoc();
  scf::InParallelOp terminator = forallOp.getTerminator();

  // Replace RegionIterArgs with initial values (except in terminator).
  for (auto [iterArg, init] :
       llvm::zip(forallOp.getRegionIterArgs(), forallOp.getOutputs())) {
    rewriter.replaceUsesWithIf(iterArg, init, [&](OpOperand &use) {
      return use.getOwner()->getParentOp() != terminator;
    });
  }

  Value loopCount = loopOp.getCount()[0];

  // Create pcf.generic.
  auto genericOp = PCF::GenericOp::create(
      rewriter, loc,
      /*resultTypes=*/forallOp.getResultTypes(),
      /*scope=*/loopOp.getScope(),
      /*inits=*/forallOp.getOutputs(),
      /*dynamic_sizes=*/ValueRange{},
      /*is_tied=*/SmallVector<bool>(forallOp.getNumResults(), true),
      /*num_iterators=*/1);

  // Set sync scope to SyncOnReturn for the pcf.generic sref arguments.
  Attribute syncScope = PCF::SyncOnReturnAttr::get(rewriter.getContext());
  for (auto regionRefArg : genericOp.getRegionRefArgs()) {
    auto srefType = cast<PCF::ShapedRefType>(regionRefArg.getType());
    auto newSrefType = PCF::ShapedRefType::get(
        rewriter.getContext(), srefType.getShape(), srefType.getElementType(),
        srefType.getScope(), syncScope);
    regionRefArg.setType(newSrefType);
  }

  Block *forallBody = &forallOp.getRegion().front();
  Block *genericBody = &genericOp.getRegion().front();
  rewriter.setInsertionPointToStart(genericBody);

  // Compute scf.forall iteration counts inside generic.
  // Handle both normalized form (lb/ub/step explicit) and short form (just ub).
  SmallVector<Value> iterCounts;
  Value totalIters = nullptr;

  SmallVector<OpFoldResult> lowerBounds = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> upperBounds = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> steps = forallOp.getMixedStep();

  // Use upper bound count as canonical size.
  size_t numDims = upperBounds.size();

  for (size_t i = 0; i < numDims; ++i) {
    // Default lb=0, step=1 if not explicitly provided.
    OpFoldResult lb = i < lowerBounds.size()
                          ? lowerBounds[i]
                          : rewriter.getIndexAttr(0);
    OpFoldResult ub = upperBounds[i];
    OpFoldResult step =
        i < steps.size() ? steps[i] : rewriter.getIndexAttr(1);

    Value lbVal = getValueOrCreateConstantIndexOp(rewriter, loc, lb);
    Value ubVal = getValueOrCreateConstantIndexOp(rewriter, loc, ub);
    Value stepVal = getValueOrCreateConstantIndexOp(rewriter, loc, step);

    Value range = arith::SubIOp::create(rewriter, loc, ubVal, lbVal);
    Value count = arith::CeilDivSIOp::create(rewriter, loc, range, stepVal);
    iterCounts.push_back(count);

    if (!totalIters) {
      totalIters = count;
    } else {
      totalIters = arith::MulIOp::create(rewriter, loc, totalIters, count);
    }
  }

  // Delinearize pcf.generic id into 2D (forall linear id, pcf loop id).
  BlockArgument linearId = genericOp.getIdArgs()[0];
  Value forallLinearId =
      arith::DivSIOp::create(rewriter, loc, linearId, loopCount);
  Value pcfLoopLinearId =
      arith::RemSIOp::create(rewriter, loc, linearId, loopCount);

  // Create outer scf.for for the forall dimensions.
  Value totalWorkers = genericOp.getCountArgs()[0];
  Value outerStep =
      arith::CeilDivSIOp::create(rewriter, loc, totalWorkers, loopCount);

  auto outerFor =
      scf::ForOp::create(rewriter, loc, forallLinearId, totalIters, outerStep);

  // Compute actual forall induction vars.
  rewriter.setInsertionPointToStart(outerFor.getBody());

  auto forallIvDelinOp = affine::AffineDelinearizeIndexOp::create(
      rewriter, loc, outerFor.getInductionVar(), iterCounts);
  ValueRange delinearizedIvs = forallIvDelinOp.getMultiIndex();

  // Apply steps and offsets: iv = delinearized * step + lb.
  // Use the same bounds arrays computed earlier to handle short form.
  IRMapping forallMapping;
  SmallVector<Value> forallIvs(forallOp.getInductionVars());
  for (size_t i = 0; i < numDims; ++i) {
    Value delinIv = delinearizedIvs[i];
    OpFoldResult lb = i < lowerBounds.size()
                          ? lowerBounds[i]
                          : rewriter.getIndexAttr(0);
    OpFoldResult step =
        i < steps.size() ? steps[i] : rewriter.getIndexAttr(1);
    Value forallIv = forallIvs[i];

    Value lbVal = getValueOrCreateConstantIndexOp(rewriter, loc, lb);
    Value stepVal = getValueOrCreateConstantIndexOp(rewriter, loc, step);
    Value scaled = arith::MulIOp::create(rewriter, loc, delinIv, stepVal);
    Value actualIv = arith::AddIOp::create(rewriter, loc, scaled, lbVal);
    forallMapping.map(forallIv, actualIv);
  }

  // NOTE: We do NOT map iter args to ref args here. The iter args in the
  // terminator's parallel_insert_slice ops are handled separately - we erase
  // those ops after composing write_slice ops. Any other uses of iter args
  // have already been replaced with init values at line 225-230.

  // Collect info about parallel_insert_slice ops before moving body.
  struct InsertSliceInfo {
    unsigned resultIdx;
    unsigned argIdx;
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
  };
  SmallVector<InsertSliceInfo> insertSliceInfos;

  for (Operation &op : terminator.getYieldingOps()) {
    auto insertOp = cast<tensor::ParallelInsertSliceOp>(&op);
    auto loopResult = cast<OpResult>(insertOp.getSource());
    unsigned resultIdx = loopResult.getResultNumber();
    auto destArg = cast<BlockArgument>(insertOp.getDest());
    unsigned argIdx = destArg.getArgNumber() - forallOp.getRank();

    InsertSliceInfo info;
    info.resultIdx = resultIdx;
    info.argIdx = argIdx;
    info.offsets = insertOp.getMixedOffsets();
    info.sizes = insertOp.getMixedSizes();
    info.strides = insertOp.getMixedStrides();
    insertSliceInfos.push_back(info);
  }

  // Move forall body operations into outer for (except terminator).
  Block *outerForBody = outerFor.getBody();
  Operation *outerForTerminator = outerForBody->getTerminator();

  for (Operation &op :
       llvm::make_early_inc_range(forallBody->without_terminator())) {
    op.moveBefore(outerForTerminator);
  }

  // Remap induction var block arguments in moved operations.
  // Only replace induction vars - iter args were already replaced with init
  // values at line 225-230, and terminator uses are handled when we erase
  // parallel_insert_slice ops.
  for (Value iv : forallOp.getInductionVars()) {
    Value mapped = forallMapping.lookup(iv);
    iv.replaceAllUsesWith(mapped);
  }

  // Find the moved pcf.loop.
  PCF::LoopOp movedLoopOp = nullptr;
  for (Operation &op : outerForBody->without_terminator()) {
    if (auto loop = dyn_cast<PCF::LoopOp>(&op)) {
      movedLoopOp = loop;
      break;
    }
  }
  assert(movedLoopOp && "Failed to find moved pcf.loop");

  // Compose tensor.parallel_insert_slice with pcf.write_slice.
  for (Operation &op :
       llvm::make_early_inc_range(terminator.getYieldingOps())) {
    auto insertOp = cast<tensor::ParallelInsertSliceOp>(&op);
    auto loopResult = cast<OpResult>(insertOp.getSource());
    unsigned resultIdx = loopResult.getResultNumber();

    // Find matching insertSliceInfo.
    const InsertSliceInfo *matchingInfo = nullptr;
    for (const InsertSliceInfo &info : insertSliceInfos) {
      if (info.resultIdx == resultIdx) {
        matchingInfo = &info;
        break;
      }
    }
    assert(matchingInfo && "Failed to find matching insert slice info");

    BlockArgument genericRefArg =
        genericOp.getRegionRefArgs()[matchingInfo->argIdx];
    BlockArgument movedRefArg = movedLoopOp.getRegionRefArgs()[resultIdx];

    // Compose all write_slice ops that write to this ref arg.
    for (Operation *user : llvm::make_early_inc_range(movedRefArg.getUsers())) {
      auto writeOp = dyn_cast<PCF::WriteSliceOp>(user);
      if (!writeOp) {
        continue;
      }

      rewriter.setInsertionPoint(writeOp);

      SmallVector<OpFoldResult> writeOffsets = writeOp.getMixedOffsets();
      SmallVector<OpFoldResult> writeStrides = writeOp.getMixedStrides();
      SmallVector<OpFoldResult> writeSizes = writeOp.getMixedSizes();

      // Expand source to match sref rank by adding unit dims.
      auto sourceType = cast<RankedTensorType>(writeOp.getSource().getType());
      auto srefType = cast<PCF::ShapedRefType>(genericRefArg.getType());
      int64_t srefRank = srefType.getRank();
      int64_t sourceRank = sourceType.getRank();

      Value expandedSource = writeOp.getSource();
      if (srefRank > sourceRank) {
        SmallVector<int64_t> expandedShape;
        SmallVector<ReassociationIndices> reassociation;

        // First sourceRank-1 dimensions map 1:1.
        for (int64_t i = 0; i < sourceRank - 1; ++i) {
          expandedShape.push_back(sourceType.getDimSize(i));
          reassociation.push_back({i});
        }

        // Last input dimension expands to include itself plus unit dims.
        ReassociationIndices lastGroup;
        if (sourceRank > 0) {
          expandedShape.push_back(sourceType.getDimSize(sourceRank - 1));
          lastGroup.push_back(sourceRank - 1);
        }

        // Add unit dimensions to reach sref rank.
        int64_t numUnitDims = srefRank - sourceRank;
        for (int64_t i = 0; i < numUnitDims; ++i) {
          expandedShape.push_back(1);
          lastGroup.push_back(sourceRank + i);
        }

        if (!lastGroup.empty()) {
          reassociation.push_back(lastGroup);
        }

        auto expandedType =
            RankedTensorType::get(expandedShape, sourceType.getElementType());
        expandedSource = tensor::ExpandShapeOp::create(
            rewriter, writeOp.getLoc(), expandedType, writeOp.getSource(),
            reassociation);
      }

      // Compose offsets, sizes, and strides.
      SmallVector<OpFoldResult> composedOffsets;
      SmallVector<OpFoldResult> composedSizes;
      SmallVector<OpFoldResult> composedStrides;

      // Helper to remap OpFoldResult Values through forallMapping.
      // This is needed because insertSliceInfos was captured before we
      // replaced forall IVs, so any Value operands still reference the
      // original forall IVs.
      auto remapOFR = [&](OpFoldResult ofr) -> OpFoldResult {
        if (auto val = dyn_cast<Value>(ofr))
          return forallMapping.lookupOrDefault(val);
        return ofr;
      };

      // First writeRank dimensions: add write and insert offsets.
      for (size_t i = 0; i < writeOffsets.size(); ++i) {
        Value writeOff = getValueOrCreateConstantIndexOp(
            rewriter, writeOp.getLoc(), writeOffsets[i]);
        Value insertOff = getValueOrCreateConstantIndexOp(
            rewriter, writeOp.getLoc(), remapOFR(matchingInfo->offsets[i]));
        Value composedOff =
            arith::AddIOp::create(rewriter, writeOp.getLoc(), writeOff,
                                  insertOff);
        composedOffsets.push_back(composedOff);

        composedSizes.push_back(remapOFR(writeSizes[i]));

        Value writeStride = getValueOrCreateConstantIndexOp(
            rewriter, writeOp.getLoc(), writeStrides[i]);
        Value insertStride = getValueOrCreateConstantIndexOp(
            rewriter, writeOp.getLoc(), remapOFR(matchingInfo->strides[i]));
        Value composedStride = arith::MulIOp::create(
            rewriter, writeOp.getLoc(), writeStride, insertStride);
        composedStrides.push_back(composedStride);
      }

      // Remaining dimensions from insert.
      for (size_t i = writeOffsets.size(); i < matchingInfo->offsets.size();
           ++i) {
        composedOffsets.push_back(remapOFR(matchingInfo->offsets[i]));
        composedSizes.push_back(remapOFR(matchingInfo->sizes[i]));
        composedStrides.push_back(remapOFR(matchingInfo->strides[i]));
      }

      // Replace old write_slice with composed one.
      rewriter.replaceOpWithNewOp<PCF::WriteSliceOp>(
          writeOp, expandedSource, genericRefArg, composedOffsets,
          composedSizes, composedStrides);
    }

    rewriter.eraseOp(insertOp);
  }

  // Replace forall results with generic results.
  for (auto [forallResult, genericResult] :
       llvm::zip(forallOp.getResults(), genericOp.getResults())) {
    rewriter.replaceAllUsesWith(forallResult, genericResult);
  }

  // At this point, all parallel_insert_slice ops should be erased, so iter args
  // should have no uses. Induction vars were replaced earlier. Verify this.
  for (BlockArgument arg : forallBody->getArguments()) {
    assert(arg.use_empty() && "forall block arg still has uses after fold!");
  }

  rewriter.eraseOp(forallOp);

  // Convert moved pcf.loop to inner scf.for.
  rewriter.setInsertionPoint(movedLoopOp);

  Value product =
      arith::MulIOp::create(rewriter, loc, loopCount, forallLinearId);
  Value diff = arith::SubIOp::create(rewriter, loc, totalWorkers, product);
  Value innerStep = arith::MinSIOp::create(rewriter, loc, diff, loopCount);

  auto innerFor =
      scf::ForOp::create(rewriter, loc, pcfLoopLinearId, loopCount, innerStep);

  // Replace loop's id arg with inner for's induction variable.
  rewriter.replaceAllUsesWith(movedLoopOp.getIdArgs()[0],
                              innerFor.getInductionVar());

  // Move operations from loop body to inner for.
  Block *loopBody = &movedLoopOp.getRegion().front();
  Block *innerForBody = innerFor.getBody();

  innerForBody->getOperations().splice(
      std::prev(innerForBody->end()), loopBody->getOperations(),
      loopBody->begin(), std::prev(loopBody->end()));

  rewriter.eraseOp(movedLoopOp);

  // Add terminator to generic's region.
  rewriter.setInsertionPointToEnd(genericBody);
  PCF::ReturnOp::create(rewriter, loc);

  return genericOp;
}

} // namespace (anonymous)

//===----------------------------------------------------------------------===//
// Public API: foldForallIntoPCFLoop
//===----------------------------------------------------------------------===//

FailureOr<GenericOp> foldForallIntoPCFLoop(RewriterBase &rewriter,
                                           scf::ForallOp forallOp) {
  // Step 1: Validate terminator structure.
  FailureOr<LoopOp> loopOpOrFailure = matchFoldTerminator(forallOp);
  if (failed(loopOpOrFailure)) {
    return failure();
  }
  LoopOp loopOp = *loopOpOrFailure;

  // Step 2: Validate pcf.loop structure.
  if (failed(matchFoldPCFLoop(forallOp, loopOp))) {
    return failure();
  }

  // Step 3: Validate write_slice ops.
  DenseMap<unsigned, SmallVector<WriteSliceOp>> writesByResult;
  if (failed(matchFoldWriteSlices(loopOp, writesByResult))) {
    return failure();
  }

  // All validations passed, perform the fold.
  return foldForallIntoPCFLoopImpl(rewriter, forallOp, loopOp);
}

namespace {

/// Validates that the forall op can be converted.
static LogicalResult matchForallConversion(scf::ForallOp forallOp) {
  scf::InParallelOp terminator = forallOp.getTerminator();
  for (Operation &op : terminator.getBody()->getOperations()) {
    // Bail on terminator ops other than parallel insert slice since we don't
    // know how to convert it.
    auto insertSliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
    if (!insertSliceOp) {
      return failure();
    }

    // Bail on non-shared outs destinations.
    auto bbArgDest = dyn_cast<BlockArgument>(insertSliceOp.getDest());
    if (!bbArgDest || bbArgDest.getOwner()->getParentOp() != forallOp) {
      return failure();
    }
  }

  for (BlockArgument bbArg : forallOp.getRegionIterArgs()) {
    for (OpOperand &use : bbArg.getUses()) {
      // Skip users outside of the terminator. These are replaced with the init.
      if (use.getOwner()->getParentOp() != terminator) {
        continue;
      }

      // Bail if the use is not on the dest of the insert slice.
      auto insertSliceUser =
          cast<tensor::ParallelInsertSliceOp>(use.getOwner());
      if (use != insertSliceUser.getDestMutable()) {
        return failure();
      }
    }
  }
  // Validate that the mapping permutation is valid.
  if (failed(getProcessorIdPermutation(forallOp))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// scf.forall -> pcf.loop Implementation
//===----------------------------------------------------------------------===//

static PCF::LoopOp convertForallToPCFLoopImpl(RewriterBase &rewriter,
                                              scf::ForallOp forallOp,
                                              PCF::ScopeAttrInterface scope,
                                              int64_t numIds) {
  assert(succeeded(matchForallConversion(forallOp)) &&
         "converting unsupported forall op");

  // Maps from fastest -> slowest to current order.
  SmallVector<int64_t> perm = *getProcessorIdPermutation(forallOp);

  // Maps from current order to fastest -> slowest.
  SmallVector<int64_t> invPerm = invertPermutationVector(perm);

  // Get the permuted ubs/lbs/steps and save them for later since we need them
  // to reconstruct the correct ids.
  SmallVector<OpFoldResult> mixedUbs = forallOp.getMixedUpperBound();
  applyPermutationToVector(mixedUbs, invPerm);
  SmallVector<OpFoldResult> mixedLbs = forallOp.getMixedLowerBound();
  applyPermutationToVector(mixedLbs, invPerm);
  SmallVector<OpFoldResult> mixedStep = forallOp.getMixedStep();
  applyPermutationToVector(mixedStep, invPerm);
  // Permute the ivs of the body to match the original mapping order by
  // permuting the uses before we move it over to the new op.
  permuteValues(rewriter, forallOp.getLoc(), forallOp.getInductionVars(), perm);

  scf::InParallelOp terminator = forallOp.getTerminator();
  MutableArrayRef<BlockArgument> bodySharedOuts = forallOp.getRegionIterArgs();

  // Replace non-insert slice users outside of the `scf.forall.in_parallel` with
  // the init values.
  ValueRange inits = forallOp.getDpsInits();
  for (auto [init, bbArg] : llvm::zip_equal(inits, bodySharedOuts)) {
    rewriter.replaceUsesWithIf(bbArg, init, [&](OpOperand &use) {
      return use.getOwner()->getParentOp() != terminator;
    });
  }

  Location loc = forallOp.getLoc();

  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr numIters = (s0 - s1).ceilDiv(s2);
  SmallVector<Value> iterationCounts;
  for (auto [ub, lb, step] : llvm::zip_equal(mixedUbs, mixedLbs, mixedStep)) {
    OpFoldResult iterCount = affine::makeComposedFoldedAffineApply(
        rewriter, loc, numIters, ArrayRef<OpFoldResult>{ub, lb, step});
    iterationCounts.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, iterCount));
  }

  int64_t numDelinIds = numIds > 0 && iterationCounts.size() > numIds
                            ? iterationCounts.size() - numIds + 1
                            : 0;
  // Reverse the delinearization basis because affine.delinearize_index is from
  // slowest to fastest varying.
  SmallVector<Value> delinearizationBasis(
      llvm::reverse(ArrayRef<Value>(iterationCounts).take_back(numDelinIds)));
  if (!delinearizationBasis.empty()) {
    AffineExpr mul = s0 * s1;
    Value total = delinearizationBasis.front();
    total = std::accumulate(
        delinearizationBasis.begin() + 1, delinearizationBasis.end(), total,
        [&](Value l, Value r) {
          OpFoldResult acc =
              affine::makeComposedFoldedAffineApply(rewriter, loc, mul, {l, r});
          return getValueOrCreateConstantIndexOp(rewriter, loc, acc);
        });
    // Replace the first |numDelinIds| entries with their product.
    iterationCounts.erase(iterationCounts.end() - numDelinIds,
                          iterationCounts.end());
    iterationCounts.push_back(total);
  }

  auto loopOp = PCF::LoopOp::create(rewriter, loc, scope, iterationCounts,
                                    forallOp.getDpsInits());
  SmallVector<Value> argReplacements;
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(loopOp.getBody());
    // If/else here to avoid popping off the front of a vector.
    if (!delinearizationBasis.empty()) {
      auto delin = affine::AffineDelinearizeIndexOp::create(
          rewriter, loc, loopOp.getIdArgs().back(), delinearizationBasis,
          /*hasOuterBound=*/true);
      argReplacements.append(loopOp.getIdArgs().begin(),
                             loopOp.getIdArgs().end() - 1);
      // Add replacements in reverse to get fastest -> slowest.
      auto delinResultsReverse = llvm::reverse(delin.getResults());
      argReplacements.append(delinResultsReverse.begin(),
                             delinResultsReverse.end());
    } else {
      argReplacements.append(loopOp.getIdArgs().begin(),
                             loopOp.getIdArgs().end());
    }

    // id * step + lb.
    AffineExpr applyLbAndStep = s0 * s1 + s2;
    for (auto [id, step, lb] :
         llvm::zip_equal(argReplacements, mixedStep, mixedLbs)) {
      OpFoldResult newId = affine::makeComposedFoldedAffineApply(
          rewriter, loc, applyLbAndStep, {id, step, lb});
      id = getValueOrCreateConstantIndexOp(rewriter, loc, newId);
    }
  }

  // Add parent only sync scope to the body arg types.
  Attribute syncScope = PCF::SyncOnReturnAttr::get(rewriter.getContext());
  for (auto regionRefArg : loopOp.getRegionRefArgs()) {
    auto srefType = cast<PCF::ShapedRefType>(regionRefArg.getType());
    auto newSrefType = PCF::ShapedRefType::get(
        rewriter.getContext(), srefType.getShape(), srefType.getElementType(),
        srefType.getScope(), syncScope);
    regionRefArg.setType(newSrefType);
  }

  rewriter.setInsertionPoint(terminator);
  llvm::SmallDenseMap<Value, Value> argToReplacementMap;
  for (auto [bbArg, refArg] :
       llvm::zip_equal(bodySharedOuts, loopOp.getRegionRefArgs())) {
    argToReplacementMap[bbArg] = refArg;
  }

  // Iterate the insert_slice ops in the order to retain the order of writes.
  SmallVector<tensor::ParallelInsertSliceOp> insertOps(
      terminator.getBody()->getOps<tensor::ParallelInsertSliceOp>());
  for (tensor::ParallelInsertSliceOp insertSliceOp : insertOps) {
    PCF::WriteSliceOp::create(
        rewriter, insertSliceOp.getLoc(), insertSliceOp.getSource(),
        argToReplacementMap[insertSliceOp.getDest()],
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());
    rewriter.eraseOp(insertSliceOp);
  }

  // Replace the terminator with the new terminator kind.
  rewriter.replaceOpWithNewOp<PCF::ReturnOp>(terminator);

  // Use the inits as the replacements for the shared outs bbargs to appease
  // `inlineBlockBefore`. By this point all of their users have been replaced
  // or erased so it doesn't matter what goes here.
  argReplacements.append(inits.begin(), inits.end());
  rewriter.inlineBlockBefore(forallOp.getBody(), loopOp.getBody(),
                             loopOp.getBody()->end(), argReplacements);

  rewriter.replaceOp(forallOp, loopOp);
  return loopOp;
}

//===----------------------------------------------------------------------===//
// scf.forall + pcf.loop -> pcf.generic (fold)
//===----------------------------------------------------------------------===//

struct TestFoldForallIntoPCFLoopPass final
    : impl::TestFoldForallIntoPCFLoopPassBase<TestFoldForallIntoPCFLoopPass> {
  void runOnOperation() override {
    SmallVector<scf::ForallOp> forallOps;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      // Only match foralls with local_mapping attribute.
      if (!hasEmptyOrLocalMapping(forallOp)) {
        return;
      }
      // Check if there's a pcf.loop with sequential scope inside.
      scf::InParallelOp terminator = forallOp.getTerminator();
      Operation *lastOp = terminator->getPrevNode();
      auto loopOp = dyn_cast_or_null<PCF::LoopOp>(lastOp);
      if (!loopOp) {
        return;
      }
      // Check for sequential scope.
      if (!isa<PCF::SequentialAttr>(loopOp.getScope())) {
        return;
      }
      forallOps.push_back(forallOp);
    });

    IRRewriter rewriter(&getContext());
    for (scf::ForallOp forallOp : forallOps) {
      rewriter.setInsertionPoint(forallOp);
      FailureOr<PCF::GenericOp> result = foldForallIntoPCFLoop(rewriter, forallOp);
      if (failed(result)) {
        forallOp.emitError("failed to fold forall into pcf.loop");
        return signalPassFailure();
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// scf.forall -> pcf.generic nest Implementation
//===----------------------------------------------------------------------===//

static PCF::GenericOp
convertForallToGenericNestImpl(RewriterBase &rewriter, scf::ForallOp forallOp,
                               ArrayRef<PCF::ScopeAttrInterface> scopes) {
  assert(succeeded(matchForallConversion(forallOp)) &&
         "converting unsupported forall op");
  assert(!scopes.empty() && "at least one scope required");

  Location loc = forallOp.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Get forall outputs to base the tied results on.
  ValueRange outputs = forallOp.getOutputs();
  TypeRange resultTypes = forallOp.getResultTypes();
  scf::InParallelOp terminator = forallOp.getTerminator();
  MutableArrayRef<BlockArgument> bodySharedOuts = forallOp.getRegionIterArgs();

  // Replace non-insert slice users outside of the `scf.forall.in_parallel` with
  // the init values.
  ValueRange inits = forallOp.getDpsInits();
  for (auto [init, bbArg] : llvm::zip_equal(inits, bodySharedOuts)) {
    rewriter.replaceUsesWithIf(bbArg, init, [&](OpOperand &use) {
      return use.getOwner()->getParentOp() != terminator;
    });
  }

  // Create nested pcf.generic ops for each scope.
  SmallVector<bool> isTied(outputs.size(), true);
  Attribute syncScope = PCF::SyncOnReturnAttr::get(ctx);

  SmallVector<Value> allIds, allCounts;

  // Create outermost generic with inits and results. Use the scope's native
  // number of processor IDs to determine iteration dimensions.
  int64_t outermostNumIds = scopes[0].getNativeNumProcessorIds();
  PCF::GenericOp outermostGeneric = PCF::GenericOp::create(
      rewriter, loc, resultTypes, scopes[0], outputs, /*dynamicSizes=*/{},
      isTied, outermostNumIds, /*syncOnReturn=*/false);

  // Add sync scope to sref types for outermost generic.
  for (BlockArgument regionRefArg : outermostGeneric.getRegionRefArgs()) {
    auto srefType = cast<PCF::ShapedRefType>(regionRefArg.getType());
    auto newSrefType = PCF::ShapedRefType::get(ctx, srefType.getShape(),
                                               srefType.getElementType(),
                                               srefType.getScope(), syncScope);
    regionRefArg.setType(newSrefType);
  }

  // Collect id/count args from outermost generic.
  allIds.append(outermostGeneric.getIdArgs().begin(),
                outermostGeneric.getIdArgs().end());
  allCounts.append(outermostGeneric.getCountArgs().begin(),
                   outermostGeneric.getCountArgs().end());

  // Set insertion point to end of execute block for next generic or body.
  Block &outermostBlock = outermostGeneric.getRegion().front();
  rewriter.setInsertionPointToEnd(&outermostBlock);

  // Create inner generics (no inits, no results).
  for (size_t i = 1; i < scopes.size(); ++i) {
    PCF::ScopeAttrInterface scope = scopes[i];
    int64_t numIds = scope.getNativeNumProcessorIds();
    PCF::GenericOp generic = PCF::GenericOp::create(
        rewriter, loc, TypeRange{}, scope, ValueRange{}, /*dynamicSizes=*/{},
        SmallVector<bool>{}, numIds, /*syncOnReturn=*/false);

    // Same here, collect id/count args from this generic and update the
    // insertion point for the next one.
    allIds.append(generic.getIdArgs().begin(), generic.getIdArgs().end());
    allCounts.append(generic.getCountArgs().begin(),
                     generic.getCountArgs().end());

    Block &executeBlock = generic.getRegion().front();
    rewriter.setInsertionPointToEnd(&executeBlock);
  }

  // In innermost block, linearize all ids if there are multiple.
  // linearId = id[0] * count[1] * ... * count[n-1] + id[1] * count[2] * ... +
  // ... + id[n-1] totalWorkers = count[0] * count[1] * ... * count[n-1]
  Value linearId;
  if (allIds.size() == 1) {
    // Shortcut single id to avoid creating IR.
    linearId = allIds[0];
  } else {
    // Use affine.linearize_index with the |counts| as the linearization basis.
    SmallVector<OpFoldResult> countOfrs = llvm::map_to_vector(
        allCounts, [](Value v) -> OpFoldResult { return v; });
    linearId =
        affine::AffineLinearizeIndexOp::create(rewriter, loc, allIds, countOfrs,
                                               /*disjoint=*/false);
  }

  // Compute total workers as product of all counts.
  Value totalWorkers = allCounts[0];
  for (size_t i = 1; i < allCounts.size(); ++i) {
    totalWorkers =
        arith::MulIOp::create(rewriter, loc, totalWorkers, allCounts[i]);
  }

  // Compute total iteration count from forall bounds.
  SmallVector<OpFoldResult> mixedUbs = forallOp.getMixedUpperBound();
  Value totalIters = arith::ConstantIndexOp::create(rewriter, loc, 1);
  for (OpFoldResult ub : mixedUbs) {
    Value dim = getValueOrCreateConstantIndexOp(rewriter, loc, ub);
    totalIters = arith::MulIOp::create(rewriter, loc, totalIters, dim);
  }

  // Compute chunk bounds: chunkSize = ceildiv(total, totalWorkers). We'll
  // create a spillover scf.forall that iterates from the start of the current
  // chunk (call it start) to min(start + chunkSize, total). This greedily
  // allocates an equal number of iterations to all workers except the last.
  Value chunkSize =
      arith::CeilDivUIOp::create(rewriter, loc, totalIters, totalWorkers);
  Value linearLb = arith::MulIOp::create(rewriter, loc, linearId, chunkSize);
  Value linearUbRaw = arith::AddIOp::create(rewriter, loc, linearLb, chunkSize);
  Value linearUb =
      arith::MinUIOp::create(rewriter, loc, linearUbRaw, totalIters);

  // Create the inner scf.forall with linearized bounds (no body builder).
  SmallVector<OpFoldResult> lbs = {linearLb};
  SmallVector<OpFoldResult> ubs = {linearUb};
  SmallVector<OpFoldResult> steps = {rewriter.getIndexAttr(1)};

  auto innerForall = scf::ForallOp::create(rewriter, loc, lbs, ubs, steps,
                                           /*outputs=*/ValueRange{},
                                           /*mapping=*/std::nullopt);

  // The scf.forall without body builder creates an empty block with a
  // terminator. Remove the terminator so we can populate the body.
  rewriter.eraseOp(innerForall.getBody()->getTerminator());

  // Get the permutation from mapping. The permutation value at position i
  // indicates the "speed rank" of dimension i (lower = faster varying).
  // For example, [#iree_codegen.local_mapping<1>,
  // #iree_codegen.local_mapping<0>] gives perm = [1, 0]:
  //   - dim 0 has rank 1 (slower)
  //   - dim 1 has rank 0 (faster)
  SmallVector<int64_t> perm = *getProcessorIdPermutation(forallOp);

  // Compute bounds ordered from slowest to fastest for delinearization.
  // affine.delinearize_index expects bounds from slowest to fastest varying.
  // Sort dimensions by decreasing speed rank (slowest first).
  SmallVector<std::pair<int64_t, int64_t>> rankAndDim;
  for (size_t i = 0; i < perm.size(); ++i) {
    rankAndDim.emplace_back(perm[i], i);
  }
  // Sort by rank descending (highest rank = slowest first).
  llvm::sort(rankAndDim, [](auto &a, auto &b) { return a.first > b.first; });

  // Build permuted upper bounds (slowest to fastest).
  SmallVector<OpFoldResult> permutedUbs;
  SmallVector<int64_t> delinToOrigDim; // Maps delinearized result index to
                                       // original forall dim.
  for (auto [rank, dim] : rankAndDim) {
    permutedUbs.push_back(mixedUbs[dim]);
    delinToOrigDim.push_back(dim);
  }

  // Map old induction variables to new linearized index.
  // For 1D: directly use the induction variable.
  // For multi-D: need delinearization with permutation handling.
  IRMapping mapping;
  if (forallOp.getRank() == 1) {
    // Shortcut rank-1 foralls to avoid creating the delinearize_index.
    mapping.map(forallOp.getInductionVars()[0],
                innerForall.getInductionVars()[0]);
  } else {
    // For multi-dimensional forall, we need to delinearize.
    // Delinearize the iteration index back to multi-D indices.
    rewriter.setInsertionPointToStart(innerForall.getBody());
    Value linearIdx = innerForall.getInductionVars()[0];
    auto delinearized = affine::AffineDelinearizeIndexOp::create(
        rewriter, loc, linearIdx, permutedUbs, /*hasOuterBound=*/true);
    // The delinearized results are in permuted order (slowest to fastest).
    // Map them back to the corresponding forall induction variables.
    // delinToOrigDim[i] tells us which original forall dim corresponds to
    // delinearized result i.
    for (size_t i = 0; i < delinToOrigDim.size(); ++i) {
      int64_t origDim = delinToOrigDim[i];
      mapping.map(forallOp.getInductionVars()[origDim],
                  delinearized.getResult(i));
    }
  }

  // Clone body operations (except terminator) into inner forall.
  rewriter.setInsertionPointToEnd(innerForall.getBody());

  for (Operation &op : forallOp.getBody()->without_terminator()) {
    rewriter.clone(op, mapping);
  }

  // Convert parallel_insert_slice to pcf.write_slice.
  llvm::SmallDenseMap<Value, Value> bbArgToSref;
  for (auto [bbArg, refArg] :
       llvm::zip_equal(bodySharedOuts, outermostGeneric.getRegionRefArgs())) {
    bbArgToSref[bbArg] = refArg;
  }

  for (Operation &op : terminator.getBody()->getOperations()) {
    if (auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op)) {
      Value src = mapping.lookupOrDefault(insertOp.getSource());
      Value destRef = bbArgToSref[insertOp.getDest()];

      // Map offsets, sizes, and strides through the mapping.
      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (OpFoldResult offset : insertOp.getMixedOffsets()) {
        if (auto val = dyn_cast<Value>(offset)) {
          offsets.push_back(mapping.lookupOrDefault(val));
        } else {
          offsets.push_back(offset);
        }
      }
      for (OpFoldResult size : insertOp.getMixedSizes()) {
        if (auto val = dyn_cast<Value>(size)) {
          sizes.push_back(mapping.lookupOrDefault(val));
        } else {
          sizes.push_back(size);
        }
      }
      for (OpFoldResult stride : insertOp.getMixedStrides()) {
        if (auto val = dyn_cast<Value>(stride)) {
          strides.push_back(mapping.lookupOrDefault(val));
        } else {
          strides.push_back(stride);
        }
      }

      PCF::WriteSliceOp::create(rewriter, loc, src, destRef, offsets, sizes,
                                strides);
    }
  }

  // Add an empty in_parallel terminator to the inner forall.
  scf::InParallelOp::create(rewriter, loc);

  // Add pcf.return terminators to all generic blocks, from innermost to
  // outermost. We need to walk from innermost out, adding returns.
  // The innermost is where we currently are.
  Operation *current = innerForall.getOperation();
  while (current) {
    Operation *parent = current->getParentOp();
    if (auto generic = dyn_cast_or_null<PCF::GenericOp>(parent)) {
      Block &block = generic.getRegion().front();
      rewriter.setInsertionPointToEnd(&block);
      PCF::ReturnOp::create(rewriter, loc);
      current = generic.getOperation();
    } else {
      // Break on the first non-generic (should be the parent of all the IR we
      // just created).
      break;
    }
  }

  return outermostGeneric;
}

//===----------------------------------------------------------------------===//
// scf.forall -> pcf.loop Pass
//===----------------------------------------------------------------------===//

struct TestConvertForallToLoopsPass final
    : impl::TestConvertForallToLoopsPassBase<TestConvertForallToLoopsPass> {
  void runOnOperation() override {
    SmallVector<scf::ForallOp> opsToConvert;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      // Empty mapping, no mapping, and local mapping all map to
      // `pcf.sequential`. If it is a local mapping, then the lowering pattern
      // will automatically handle any mapping permutation based on the mapping
      // attribute's relative id.
      if (hasEmptyOrLocalMapping(forallOp)) {
        opsToConvert.push_back(forallOp);
      }
    });

    IRRewriter rewriter(getOperation());
    PCF::ScopeAttrInterface sequentialScope =
        PCF::SequentialAttr::get(&getContext());
    for (auto forallOp : opsToConvert) {
      rewriter.setInsertionPoint(forallOp);
      if (failed(convertForallToPCFLoop(rewriter, forallOp, sequentialScope))) {
        forallOp->emitOpError("failed to convert forall");
        return signalPassFailure();
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// scf.forall -> pcf.generic nest Pass
//===----------------------------------------------------------------------===//

struct TestConvertForallToGenericNestPass final
    : impl::TestConvertForallToGenericNestPassBase<
          TestConvertForallToGenericNestPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Build scope list based on numSequentialScopes.
    SmallVector<PCF::ScopeAttrInterface> scopeAttrs;
    for (int64_t i = 0; i < numSequentialScopes; ++i) {
      scopeAttrs.push_back(PCF::SequentialAttr::get(ctx));
    }

    if (scopeAttrs.empty()) {
      emitError(getOperation()->getLoc()) << "no scopes specified";
      return signalPassFailure();
    }

    IRRewriter rewriter(ctx);
    SmallVector<scf::ForallOp> forallOps;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      // Only convert foralls with empty mapping or local_mapping attributes.
      if (hasEmptyOrLocalMapping(forallOp)) {
        forallOps.push_back(forallOp);
      }
    });

    for (scf::ForallOp forallOp : forallOps) {
      rewriter.setInsertionPoint(forallOp);
      FailureOr<PCF::GenericOp> result =
          convertForallToGenericNest(rewriter, forallOp, scopeAttrs);
      if (failed(result)) {
        forallOp.emitError("failed to convert forall to generic nest");
        return signalPassFailure();
      }
      // Replace forall results with generic results.
      rewriter.replaceOp(forallOp, result->getResults());
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API: scf.forall -> pcf.loop
//===----------------------------------------------------------------------===//

FailureOr<PCF::LoopOp> convertForallToPCFLoop(RewriterBase &rewriter,
                                              scf::ForallOp forallOp,
                                              PCF::ScopeAttrInterface scope,
                                              int64_t numIds) {
  if (failed(matchForallConversion(forallOp))) {
    return failure();
  }
  return convertForallToPCFLoopImpl(rewriter, forallOp, scope, numIds);
}

//===----------------------------------------------------------------------===//
// Public API: scf.forall -> pcf.generic nest
//===----------------------------------------------------------------------===//

FailureOr<PCF::GenericOp>
convertForallToGenericNest(RewriterBase &rewriter, scf::ForallOp forallOp,
                           ArrayRef<PCF::ScopeAttrInterface> scopes) {
  if (scopes.empty()) {
    return forallOp.emitError("at least one scope required");
  }

  if (failed(matchForallConversion(forallOp))) {
    return failure();
  }

  return convertForallToGenericNestImpl(rewriter, forallOp, scopes);
}

} // namespace mlir::iree_compiler::IREE::PCF
