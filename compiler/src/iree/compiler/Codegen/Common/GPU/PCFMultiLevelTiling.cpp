// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTilingInterface.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-pcf-multi-level-tiling"

namespace mlir::iree_compiler::IREE::PCF {

/// Returns true if the OpFoldResult is a constant zero.
static bool isZeroTileSize(OpFoldResult ofr) {
  std::optional<int64_t> val = getConstantIntValue(ofr);
  return val && *val == 0;
}

/// Computes tile offsets and sizes for a set of tiled dimensions given
/// worker IDs and tile sizes. Only dimensions with non-zero tile sizes
/// are tiled; the corresponding worker ID is used.
///
/// `workerIds` has one entry per tiled dimension (in order).
/// `tileSizes` has one entry per iteration domain dimension.
/// `domainSizes` has one entry per iteration domain dimension.
static void computeTileOffsetsAndSizes(OpBuilder &b, Location loc,
                                       ArrayRef<OpFoldResult> tileSizes,
                                       ArrayRef<Range> iterDomain,
                                       ArrayRef<BlockArgument> workerIds,
                                       SmallVectorImpl<OpFoldResult> &offsets,
                                       SmallVectorImpl<OpFoldResult> &sizes) {
  int64_t numDims = tileSizes.size();
  offsets.resize(numDims);
  sizes.resize(numDims);

  AffineExpr s0, s1, s2;
  bindSymbols(b.getContext(), s0, s1, s2);
  AffineMap minMap3 = AffineMap::get(0, 3, {s0, s1 - s2}, b.getContext());

  // Add domain offset: offset = domainOffset + id * tileSize.
  AffineMap addMulMap = AffineMap::get(0, 3, s0 + s1 * s2, b.getContext());

  int64_t tiledIdx = 0;
  for (int64_t i = 0; i < numDims; ++i) {
    if (isZeroTileSize(tileSizes[i])) {
      offsets[i] = iterDomain[i].offset;
      sizes[i] = iterDomain[i].size;
    } else {
      OpFoldResult idVal = workerIds[tiledIdx];
      OpFoldResult offset = affine::makeComposedFoldedAffineApply(
          b, loc, addMulMap, {iterDomain[i].offset, idVal, tileSizes[i]});
      offsets[i] = offset;
      OpFoldResult size = affine::makeComposedFoldedAffineMin(
          b, loc, minMap3, {tileSizes[i], iterDomain[i].size, offset});
      sizes[i] = size;
      ++tiledIdx;
    }
  }
}

/// Counts the number of non-zero tile sizes.
static int64_t countTiledDims(ArrayRef<OpFoldResult> tileSizes) {
  return llvm::count_if(tileSizes,
                        [](OpFoldResult ts) { return !isZeroTileSize(ts); });
}

/// Returns the promotion attribute for the given operand index.
/// If promotionTypes is set, uses the corresponding entry.
/// Otherwise falls back to DerivedThreadConfig.
static IREE::GPU::PromotionAttr
getPromotionAttr(MLIRContext *ctx, const MultiLevelTilingParams &params,
                 unsigned operandIdx) {
  // Find the position of operandIdx in operandsToPromote.
  for (int64_t i = 0, e = params.operandsToPromote.size(); i < e; ++i) {
    if (params.operandsToPromote[i] == operandIdx) {
      if (i < static_cast<int64_t>(params.promotionTypes.size())) {
        return cast<IREE::GPU::PromotionAttr>(params.promotionTypes[i]);
      }
      break;
    }
  }
  // Default.
  return cast<IREE::GPU::PromotionAttr>(
      IREE::GPU::DerivedThreadConfigAttr::get(ctx));
}

/// Builds DistributedOperandInfo for all operands, handling promotion and
/// sref routing. Used by both MMA and regular tiling paths.
static SmallVector<DistributedOperandInfo>
buildOperandInfo(RewriterBase &rewriter, Location loc, Operation *op,
                 const DenseSet<unsigned> &tileableSet,
                 const DenseSet<unsigned> &dpsInitIndices,
                 ArrayRef<int64_t> operandToReadonlyIdx,
                 ArrayRef<int64_t> operandToReadwriteIdx,
                 ArrayRef<BlockArgument> sgReadonlyRefs, ValueRange iterArgs,
                 const MultiLevelTilingParams &params) {
  SmallVector<DistributedOperandInfo> operandInfo;
  for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    unsigned idx = static_cast<unsigned>(i);
    if (!tileableSet.contains(idx) ||
        !isa<ShapedType>(op->getOperand(i).getType())) {
      operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
      continue;
    }

    if (dpsInitIndices.contains(idx)) {
      // DPS init: use the iter_arg from the reduction loop.
      int64_t rwIdx = operandToReadwriteIdx[i];
      assert(rwIdx >= 0 && rwIdx < static_cast<int64_t>(iterArgs.size()) &&
             "DPS init should be mapped to a valid readwrite sref or iter arg");
      operandInfo.push_back({iterArgs[rwIdx], /*isTile=*/true});
    } else {
      // Input: use the readonly sref from outer generic.
      int64_t roIdx = operandToReadonlyIdx[i];
      if (roIdx >= 0) {
        Value sref = sgReadonlyRefs[roIdx];

        // Check if this operand should be promoted.
        if (llvm::is_contained(params.operandsToPromote, idx)) {
          ShapedRefType srefType = cast<ShapedRefType>(sref.getType());
          int64_t rank = srefType.getRank();
          SmallVector<Attribute> symbolNames;
          std::string nsPrefix = "n" + std::to_string(i) + ".";
          for (int64_t d = 0; d < rank; ++d) {
            symbolNames.push_back(
                rewriter.getStringAttr(nsPrefix + "d" + std::to_string(d)));
          }
          ShapedRefType promotedType = ShapedRefType::get(
              rewriter.getContext(), srefType.getShape(),
              srefType.getElementType(),
              cast<ScopeAttrInterface>(params.subgroup.scope));
          IREE::GPU::PromotionAttr promotion =
              getPromotionAttr(rewriter.getContext(), params, idx);
          Value promoted = IREE::GPU::PromoteOperandOp::create(
              rewriter, loc, promotedType, promotion, sref,
              rewriter.getArrayAttr(symbolNames));
          operandInfo.push_back({promoted, /*isTile=*/false});
        } else {
          operandInfo.push_back({sref, /*isTile=*/false});
        }
      } else {
        operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
      }
    }
  }
  return operandInfo;
}

static FailureOr<SmallVector<OpFoldResult>>
getPackedSizesForMma(linalg::LinalgOp linalgOp, RewriterBase &rewriter,
                     IREE::Codegen::InnerTileDescAttrInterface kind) {
  auto createPackedSizes =
      [&rewriter, &linalgOp](SmallVector<int64_t> dims,
                             SmallVector<SmallVector<unsigned, 2>> indices)
      -> FailureOr<SmallVector<OpFoldResult>> {
    auto zero = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> packedSizes(linalgOp.getNumLoops(), zero);
    for (auto [dim, index] : llvm::zip_equal(dims, indices)) {
      if (index.empty()) {
        return failure();
      }
      packedSizes[index.back()] = rewriter.getIndexAttr(dim);
    }
    return packedSizes;
  };

  SmallVector<int64_t> dims;
  SmallVector<SmallVector<unsigned, 2>> indices;
  if (auto smmaKind = dyn_cast<IREE::GPU::ScaledMMAAttr>(kind)) {
    FailureOr<IREE::LinalgExt::ScaledContractionDimensions> scaledContrDims =
        IREE::LinalgExt::inferScaledContractionDims(linalgOp);
    if (succeeded(scaledContrDims)) {
      auto [m, n, k, kB] = smmaKind.getScaledMNKShape();
      indices = {scaledContrDims->m, scaledContrDims->n, scaledContrDims->k,
                 scaledContrDims->kB};
      dims = {m, n, k, kB};
    }
  }
  if (auto mmaKind = dyn_cast<IREE::GPU::MmaInterfaceAttr>(kind)) {
    FailureOr<linalg::ContractionDimensions> contractionDims =
        linalg::inferContractionDims(linalgOp);
    if (succeeded(contractionDims)) {
      if (mmaKind.isBlockIntrinsic()) {
        auto [b, m, n, k] = mmaKind.getBMNKShape();
        indices = {contractionDims->batch, contractionDims->m,
                   contractionDims->n, contractionDims->k};
        dims = {b, m, n, k};
      } else {
        auto [m, n, k] = mmaKind.getMNKShape();
        indices = {contractionDims->m, contractionDims->n, contractionDims->k};
        dims = {m, n, k};
      }
    }
  }

  if (dims.empty() || indices.empty()) {
    return failure();
  }
  return createPackedSizes(dims, indices);
}

struct MmaOperandPackSpec {
  SmallVector<int64_t> innerDimsPos;
  SmallVector<OpFoldResult> innerTiles;
};

static FailureOr<SmallVector<MmaOperandPackSpec>>
getMmaPackSpecsForOperands(linalg::LinalgOp linalgOp,
                           ArrayRef<OpFoldResult> packedLoopSizes) {
  int64_t numLoops = static_cast<int64_t>(packedLoopSizes.size());
  SmallVector<MmaOperandPackSpec> packSpecs;
  packSpecs.reserve(linalgOp->getNumOperands());

  for (OpOperand &operand : linalgOp->getOpOperands()) {
    MmaOperandPackSpec spec;
    if (!isa<ShapedType>(operand.get().getType())) {
      packSpecs.push_back(std::move(spec));
      continue;
    }
    AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&operand);
    for (int64_t loopDim : llvm::seq<int64_t>(0, numLoops)) {
      if (isZeroTileSize(packedLoopSizes[loopDim])) {
        continue;
      }
      for (auto [operandDim, expr] :
           llvm::enumerate(indexingMap.getResults())) {
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr || dimExpr.getPosition() != loopDim) {
          continue;
        }
        spec.innerDimsPos.push_back(operandDim);
        spec.innerTiles.push_back(packedLoopSizes[loopDim]);
        break;
      }
    }
    packSpecs.push_back(std::move(spec));
  }

  return packSpecs;
}

/// Computes the offsets and sizes for a single operand tile given the
/// iteration domain offsets/sizes and the operand's indexing map.
static std::pair<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>
computeOperandTilePosition(OpBuilder &b, Location loc,
                           linalg::LinalgOp linalgOp, OpOperand &opOperand,
                           ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes) {
  AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&opOperand);
  int64_t rank = indexingMap.getNumResults();
  SmallVector<OpFoldResult> operandOffsets;
  SmallVector<OpFoldResult> operandSizes;
  operandOffsets.reserve(rank);
  operandSizes.reserve(rank);

  for (auto [r, expr] : llvm::enumerate(indexingMap.getResults())) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      unsigned pos = dimExpr.getPosition();
      operandOffsets.push_back(offsets[pos]);
      operandSizes.push_back(sizes[pos]);
      continue;
    }

    AffineMap subMap = indexingMap.getSubMap({static_cast<unsigned>(r)});
    SmallVector<Attribute> zeros(offsets.size(), b.getIndexAttr(0));
    SmallVector<Attribute> mAtZero;
    [[maybe_unused]] LogicalResult foldRes =
        subMap.constantFold(zeros, mAtZero);
    assert(succeeded(foldRes) &&
           "affine_map with only dims must be evaluable at zero.");
    int64_t mAtZeroInt =
        cast<IntegerAttr>(mAtZero[0]).getValue().getSExtValue();
    OpFoldResult offset = affine::makeComposedFoldedAffineApply(
        b, loc, subMap.getResult(0) - mAtZeroInt, offsets);
    operandOffsets.push_back(offset);

    SmallVector<OpFoldResult> upperBounds;
    upperBounds.reserve(offsets.size());
    AffineExpr s0 = getAffineSymbolExpr(0, b.getContext());
    AffineExpr s1 = getAffineSymbolExpr(1, b.getContext());
    AffineMap addMinusOneMap =
        AffineMap::get(0, 2, s0 + s1 - 1, b.getContext());
    for (auto [off, sz] : llvm::zip_equal(offsets, sizes)) {
      upperBounds.push_back(affine::makeComposedFoldedAffineApply(
          b, loc, addMinusOneMap, {off, sz}));
    }
    OpFoldResult maxIndex = affine::makeComposedFoldedAffineApply(
        b, loc, subMap.getResult(0) - mAtZeroInt, upperBounds);

    AffineExpr d0 = getAffineDimExpr(0, b.getContext());
    AffineExpr d1 = getAffineDimExpr(1, b.getContext());
    AffineMap sizeMap = AffineMap::get(2, 0, d0 - d1 + 1, b.getContext());
    OpFoldResult size = affine::makeComposedFoldedAffineApply(
        b, loc, sizeMap, {maxIndex, offset});
    operandSizes.push_back(size);
  }

  return std::make_pair(operandOffsets, operandSizes);
}

struct MmaPackedSrefInfo {
  Value packedSref;
  SmallVector<OpFoldResult> packedOuterSizes;
};

struct MmaDistributedSliceInfo {
  Value transformedSref;
  SmallVector<OpFoldResult> sliceOffsets;
  SmallVector<OpFoldResult> sliceSizes;
  SmallVector<OpFoldResult> sliceStrides;
  RankedTensorType readTensorType;
  RankedTensorType logicalTensorType;
  SmallVector<int64_t> readToLogicalPermutation;
  SmallVector<int64_t> logicalToReadPermutation;
};

static ShapedRefType
getSubviewResultTypeFromTileSizes(OpBuilder &builder, ShapedRefType sourceType,
                                  ArrayRef<OpFoldResult> sizes) {
  SmallVector<int64_t> shape;
  shape.reserve(sizes.size());
  for (OpFoldResult size : sizes) {
    if (auto cst = getConstantIntValue(size)) {
      shape.push_back(*cst);
    } else {
      shape.push_back(ShapedType::kDynamic);
    }
  }
  return ShapedRefType::get(builder.getContext(), shape,
                            sourceType.getElementType(), sourceType.getScope(),
                            sourceType.getSyncScope());
}

static RankedTensorType
getReadResultTypeFromSliceSizes(OpBuilder &builder, Type elementType,
                                ArrayRef<OpFoldResult> sizes) {
  SmallVector<int64_t> shape;
  shape.reserve(sizes.size());
  for (OpFoldResult size : sizes) {
    if (auto cst = getConstantIntValue(size)) {
      shape.push_back(*cst);
    } else {
      shape.push_back(ShapedType::kDynamic);
    }
  }
  return RankedTensorType::get(shape, elementType);
}

static bool isIdentityPermutation(ArrayRef<int64_t> permutation) {
  for (auto [idx, value] : llvm::enumerate(permutation)) {
    if (value != static_cast<int64_t>(idx)) {
      return false;
    }
  }
  return true;
}

static SmallVector<int64_t> invertPermutation(ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> inverse(permutation.size(), -1);
  for (auto [idx, value] : llvm::enumerate(permutation)) {
    if (value < 0 || value >= static_cast<int64_t>(permutation.size())) {
      return {};
    }
    inverse[value] = static_cast<int64_t>(idx);
  }
  if (llvm::any_of(inverse, [](int64_t v) { return v < 0; })) {
    return {};
  }
  return inverse;
}

static Value transposeTensor(RewriterBase &rewriter, Location loc, Value tensor,
                             ArrayRef<int64_t> permutation) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  if (permutation.empty() || isIdentityPermutation(permutation)) {
    return tensor;
  }
  assert(static_cast<int64_t>(permutation.size()) == tensorType.getRank() &&
         "transpose permutation must match tensor rank");
  SmallVector<OpFoldResult> outputShape =
      tensor::getMixedSizes(rewriter, loc, tensor);
  applyPermutationToVector(outputShape, permutation);
  Value empty = tensor::EmptyOp::create(rewriter, loc, outputShape,
                                        tensorType.getElementType());
  return linalg::TransposeOp::create(rewriter, loc, tensor, empty, permutation)
      .getResult()[0];
}

static OpFoldResult ceilDivOpFoldResult(OpBuilder &builder, Location loc,
                                        OpFoldResult lhs, OpFoldResult rhs) {
  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);
  AffineMap ceilDivMap =
      AffineMap::get(0, 2, (s0 + s1 - 1).floorDiv(s1), builder.getContext());
  return affine::makeComposedFoldedAffineApply(builder, loc, ceilDivMap,
                                               {lhs, rhs});
}

static FailureOr<MmaPackedSrefInfo>
createPackedSrefView(RewriterBase &rewriter, Location loc, Value sourceSref,
                     ArrayRef<OpFoldResult> tileOffsets,
                     ArrayRef<OpFoldResult> tileSizes,
                     const MmaOperandPackSpec &packSpec) {
  auto sourceType = dyn_cast<ShapedRefType>(sourceSref.getType());
  if (!sourceType) {
    return failure();
  }
  int64_t rank = sourceType.getRank();
  SmallVector<OpFoldResult> unitStrides(rank, rewriter.getIndexAttr(1));

  ShapedRefType subviewType =
      getSubviewResultTypeFromTileSizes(rewriter, sourceType, tileSizes);
  Value tileSubview = SubviewOp::create(rewriter, loc, subviewType, sourceSref,
                                        tileOffsets, tileSizes, unitStrides);

  if (packSpec.innerDimsPos.empty()) {
    return MmaPackedSrefInfo{tileSubview, SmallVector<OpFoldResult>(tileSizes)};
  }

  SmallVector<OpFoldResult> packedOuterSizes(tileSizes.begin(),
                                             tileSizes.end());
  for (auto [innerDimPos, innerTile] :
       llvm::zip_equal(packSpec.innerDimsPos, packSpec.innerTiles)) {
    packedOuterSizes[innerDimPos] = ceilDivOpFoldResult(
        rewriter, loc, packedOuterSizes[innerDimPos], innerTile);
  }

  SmallVector<int64_t> dimToPackPos(rank, -1);
  for (auto [packPos, dim] : llvm::enumerate(packSpec.innerDimsPos)) {
    dimToPackPos[dim] = static_cast<int64_t>(packPos);
  }

  SmallVector<int64_t> packedShape;
  packedShape.reserve(rank + packSpec.innerTiles.size());
  SmallVector<ReassociationIndices> reassociations;
  reassociations.reserve(rank);
  SmallVector<Value> outputShape;
  outputShape.reserve(rank + packSpec.innerTiles.size());

  int64_t reassocIndex = 0;
  for (int64_t dim : llvm::seq<int64_t>(0, rank)) {
    ReassociationIndices group = {reassocIndex++};
    OpFoldResult outerSize = packedOuterSizes[dim];
    if (auto cst = getConstantIntValue(outerSize)) {
      packedShape.push_back(*cst);
    } else {
      packedShape.push_back(ShapedType::kDynamic);
    }
    outputShape.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, outerSize));

    int64_t packPos = dimToPackPos[dim];
    if (packPos >= 0) {
      OpFoldResult innerSize = packSpec.innerTiles[packPos];
      if (auto cst = getConstantIntValue(innerSize)) {
        packedShape.push_back(*cst);
      } else {
        packedShape.push_back(ShapedType::kDynamic);
      }
      outputShape.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, innerSize));
      group.push_back(reassocIndex++);
    }
    reassociations.push_back(std::move(group));
  }

  ShapedRefType packedType = ShapedRefType::get(
      rewriter.getContext(), packedShape, sourceType.getElementType(),
      sourceType.getScope(), sourceType.getSyncScope());

  SmallVector<Attribute> reassociationAttrs;
  reassociationAttrs.reserve(reassociations.size());
  for (ArrayRef<int64_t> group : reassociations) {
    SmallVector<Attribute> groupAttrs;
    groupAttrs.reserve(group.size());
    for (int64_t idx : group) {
      groupAttrs.push_back(rewriter.getI64IntegerAttr(idx));
    }
    reassociationAttrs.push_back(rewriter.getArrayAttr(groupAttrs));
  }
  Value packed = ExpandShapeOp::create(
      rewriter, loc, packedType, tileSubview,
      rewriter.getArrayAttr(reassociationAttrs), outputShape);
  return MmaPackedSrefInfo{packed, std::move(packedOuterSizes)};
}

static FailureOr<MmaDistributedSliceInfo>
createMmaDistributedSlice(RewriterBase &rewriter, Location loc,
                          linalg::LinalgOp linalgOp, uint32_t operandIndex,
                          Value sourceSref,
                          ArrayRef<OpFoldResult> iterationOffsets,
                          ArrayRef<OpFoldResult> iterationSizes, Value laneId,
                          Codegen::InnerTileDescAttrInterface mmaKind,
                          const MmaOperandPackSpec &packSpec) {
  auto sourceType = dyn_cast<ShapedRefType>(sourceSref.getType());
  if (!sourceType) {
    return failure();
  }

  OpOperand &opOperand = linalgOp->getOpOperand(operandIndex);
  auto [tileOffsets, tileSizes] = computeOperandTilePosition(
      rewriter, loc, linalgOp, opOperand, iterationOffsets, iterationSizes);

  FailureOr<MmaPackedSrefInfo> maybePacked = createPackedSrefView(
      rewriter, loc, sourceSref, tileOffsets, tileSizes, packSpec);
  if (failed(maybePacked)) {
    return failure();
  }

  AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&opOperand);
  int64_t outerRank = indexingMap.getNumResults();
  if (outerRank != static_cast<int64_t>(maybePacked->packedOuterSizes.size())) {
    return failure();
  }

  SmallVector<OpFoldResult> offsets(outerRank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes(maybePacked->packedOuterSizes.begin(),
                                  maybePacked->packedOuterSizes.end());
  SmallVector<OpFoldResult> strides(outerRank, rewriter.getIndexAttr(1));

  SmallVector<int64_t> permutation;
  if (!packSpec.innerDimsPos.empty()) {
    permutation = packSpec.innerDimsPos;
  } else {
    int64_t packedRank =
        cast<ShapedRefType>(maybePacked->packedSref.getType()).getRank();
    permutation =
        llvm::to_vector(llvm::seq<int64_t>(0, packedRank - outerRank));
  }

  if (failed(mmaKind.populateOperandOffsetsSizesStrides(
          rewriter, loc, operandIndex, laneId, permutation, offsets, sizes,
          strides))) {
    return failure();
  }

  // The interface appends inner tile dims after all outer dims.
  SmallVector<OpFoldResult> logicalOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> logicalSizes(sizes.begin(), sizes.end());
  SmallVector<OpFoldResult> logicalStrides(strides.begin(), strides.end());

  // The packed sref view interleaves [outer, inner] per source dim to keep
  // reassociation groups contiguous. Reorder coordinates into that interleaved
  // layout for pcf.read_slice/pcf.write_slice, and later transpose the loaded
  // tensor back to logical [outer..., inner...] order.
  SmallVector<int64_t> dimToPackPos(outerRank, -1);
  for (auto [packPos, dim] : llvm::enumerate(packSpec.innerDimsPos)) {
    dimToPackPos[dim] = static_cast<int64_t>(packPos);
  }
  SmallVector<OpFoldResult> reorderedOffsets;
  SmallVector<OpFoldResult> reorderedSizes;
  SmallVector<OpFoldResult> reorderedStrides;
  SmallVector<int64_t> physicalToLogical;
  reorderedOffsets.reserve(outerRank + packSpec.innerDimsPos.size());
  reorderedSizes.reserve(outerRank + packSpec.innerDimsPos.size());
  reorderedStrides.reserve(outerRank + packSpec.innerDimsPos.size());
  physicalToLogical.reserve(outerRank + packSpec.innerDimsPos.size());
  for (int64_t dim : llvm::seq<int64_t>(0, outerRank)) {
    reorderedOffsets.push_back(logicalOffsets[dim]);
    reorderedSizes.push_back(logicalSizes[dim]);
    reorderedStrides.push_back(logicalStrides[dim]);
    physicalToLogical.push_back(dim);
    int64_t packPos = dimToPackPos[dim];
    if (packPos < 0) {
      continue;
    }
    int64_t innerIdx = outerRank + packPos;
    reorderedOffsets.push_back(logicalOffsets[innerIdx]);
    reorderedSizes.push_back(logicalSizes[innerIdx]);
    reorderedStrides.push_back(logicalStrides[innerIdx]);
    physicalToLogical.push_back(innerIdx);
  }
  SmallVector<int64_t> logicalToPhysical = invertPermutation(physicalToLogical);
  if (logicalToPhysical.empty()) {
    return failure();
  }

  RankedTensorType readType = getReadResultTypeFromSliceSizes(
      rewriter, sourceType.getElementType(), reorderedSizes);
  RankedTensorType logicalType = getReadResultTypeFromSliceSizes(
      rewriter, sourceType.getElementType(), logicalSizes);
  return MmaDistributedSliceInfo{
      maybePacked->packedSref,
      std::move(reorderedOffsets),
      std::move(reorderedSizes),
      std::move(reorderedStrides),
      readType,
      logicalType,
      std::move(logicalToPhysical),
      std::move(physicalToLogical)};
}

/// Builds a nested reduction loop (one scf.for per reduction dimension).
///
/// For the regular path, calls getDistributedImplementation inside the
/// innermost loop body. For the MMA path, builds lane-distributed
/// `iree_codegen.inner_tiled` directly with PCF views/slices.
///
/// Uses scf::buildLoopNest to create a perfect nest over all reduction
/// dimensions in |reductionDims|. Emits result writes after the outermost loop.
///
/// Returns failure if getDistributedImplementation fails.
static FailureOr<SmallVector<Value>> buildReductionLoop(
    RewriterBase &rewriter, Location loc, PCFTilingOpInterface target,
    Operation *op, ArrayRef<Range> iterDomain,
    ArrayRef<utils::IteratorType> iterTypes,
    ArrayRef<OpFoldResult> redTileSizes, ArrayRef<OpFoldResult> laneOffsets,
    ArrayRef<OpFoldResult> laneSizes, ArrayRef<int64_t> reductionDims,
    const DenseSet<unsigned> &tileableSet,
    const DenseSet<unsigned> &dpsInitIndices,
    ArrayRef<int64_t> operandToReadonlyIdx,
    ArrayRef<int64_t> operandToReadwriteIdx,
    ArrayRef<BlockArgument> sgReadonlyRefs,
    ArrayRef<BlockArgument> sgReadwriteRefs, ValueRange initValues,
    const MultiLevelTilingParams &params, Value laneId) {
  std::optional<Codegen::InnerTileDescAttrInterface> mmaKind;
  linalg::LinalgOp mmaLinalgOp;
  DestinationStyleOpInterface mmaDpsOp;
  SmallVector<MmaOperandPackSpec> mmaPackSpecs;
  SmallVector<MmaDistributedSliceInfo, 0> mmaInitSliceInfos;
  SmallVector<Value> loopInitValues(initValues.begin(), initValues.end());

  if (params.mmaKind) {
    if (!laneId) {
      return rewriter.notifyMatchFailure(
          target, "mma_kind requires a valid lane id in lane scope");
    }
    mmaLinalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!mmaLinalgOp) {
      return rewriter.notifyMatchFailure(
          target, "mma_kind requires a linalg op for lane distribution");
    }
    mmaDpsOp = dyn_cast<DestinationStyleOpInterface>(op);
    if (!mmaDpsOp) {
      return rewriter.notifyMatchFailure(
          target, "mma_kind requires destination-style semantics");
    }

    mmaKind = dyn_cast<Codegen::InnerTileDescAttrInterface>(params.mmaKind);
    if (!mmaKind) {
      return rewriter.notifyMatchFailure(target, "invalid mma_kind type");
    }

    FailureOr<SmallVector<OpFoldResult>> packedLoopSizes =
        getPackedSizesForMma(mmaLinalgOp, rewriter, *mmaKind);
    if (failed(packedLoopSizes)) {
      return rewriter.notifyMatchFailure(
          target, "failed to derive MMA packing tile sizes");
    }

    FailureOr<SmallVector<MmaOperandPackSpec>> maybeSpecs =
        getMmaPackSpecsForOperands(mmaLinalgOp, *packedLoopSizes);
    if (failed(maybeSpecs) ||
        maybeSpecs->size() != static_cast<size_t>(op->getNumOperands())) {
      return rewriter.notifyMatchFailure(
          target, "failed to derive MMA operand packing specs");
    }
    mmaPackSpecs = std::move(*maybeSpecs);

    // For MMA, loop-carried values are lane-distributed inner_tiled init tiles,
    // not subgroup-uniform tiles.
    loopInitValues.clear();
    mmaInitSliceInfos.clear();
    for (auto [resultIdx, init] :
         llvm::enumerate(mmaDpsOp.getDpsInitsMutable())) {
      uint32_t operandIndex = init.getOperandNumber();
      int64_t rwIdx = operandToReadwriteIdx[operandIndex];
      if (rwIdx < 0 || rwIdx >= static_cast<int64_t>(sgReadwriteRefs.size())) {
        return rewriter.notifyMatchFailure(target,
                                           "invalid readwrite operand mapping");
      }
      FailureOr<MmaDistributedSliceInfo> maybeSliceInfo =
          createMmaDistributedSlice(rewriter, loc, mmaLinalgOp, operandIndex,
                                    sgReadwriteRefs[rwIdx], laneOffsets,
                                    laneSizes, laneId, *mmaKind,
                                    mmaPackSpecs[operandIndex]);
      if (failed(maybeSliceInfo)) {
        return rewriter.notifyMatchFailure(
            target, "failed to derive distributed MMA init slice");
      }
      Value initTile = ReadSliceOp::create(
          rewriter, loc, maybeSliceInfo->readTensorType,
          maybeSliceInfo->transformedSref, maybeSliceInfo->sliceOffsets,
          maybeSliceInfo->sliceSizes, maybeSliceInfo->sliceStrides);
      initTile = transposeTensor(rewriter, loc, initTile,
                                 maybeSliceInfo->readToLogicalPermutation);
      loopInitValues.push_back(initTile);
      mmaInitSliceInfos.push_back(std::move(*maybeSliceInfo));
      (void)resultIdx;
    }
  }

  // Build lower bounds, upper bounds, and steps for each reduction dim.
  SmallVector<Value> lbs, ubs, steps;
  for (int64_t redDim : reductionDims) {
    lbs.push_back(getValueOrCreateConstantIndexOp(rewriter, loc,
                                                  rewriter.getIndexAttr(0)));
    ubs.push_back(getValueOrCreateConstantIndexOp(rewriter, loc,
                                                  iterDomain[redDim].size));
    steps.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, redTileSizes[redDim]));
  }

  // Track whether the inner body failed. The buildLoopNest callback
  // cannot return failure, so we capture it and check afterwards.
  bool innerFailed = false;

  scf::LoopNest loopNest = scf::buildLoopNest(
      rewriter, loc, lbs, ubs, steps, loopInitValues,
      [&](OpBuilder &b, Location nestLoc, ValueRange ivs,
          ValueRange iterArgs) -> scf::ValueVector {
        // Compute tile offsets including all reduction dimensions.
        SmallVector<OpFoldResult> tileOffsets(laneOffsets);
        SmallVector<OpFoldResult> tileSizes(laneSizes);
        for (auto [idx, redDim] : llvm::enumerate(reductionDims)) {
          tileOffsets[redDim] = ivs[idx];
          tileSizes[redDim] = redTileSizes[redDim];
        }

        if (params.mmaKind) {
          if (!mmaKind || !mmaLinalgOp || !mmaDpsOp) {
            innerFailed = true;
            return scf::ValueVector(iterArgs.begin(), iterArgs.end());
          }

          SmallVector<Value> mmaOperands(op->getNumOperands());
          for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
            unsigned idx = static_cast<unsigned>(i);
            if (!tileableSet.contains(idx) ||
                !isa<ShapedType>(op->getOperand(i).getType())) {
              mmaOperands[i] = op->getOperand(i);
              continue;
            }

            if (dpsInitIndices.contains(idx)) {
              int64_t rwIdx = operandToReadwriteIdx[i];
              if (rwIdx < 0 || rwIdx >= static_cast<int64_t>(iterArgs.size())) {
                innerFailed = true;
                return scf::ValueVector(iterArgs.begin(), iterArgs.end());
              }
              mmaOperands[i] = iterArgs[rwIdx];
              continue;
            }

            int64_t roIdx = operandToReadonlyIdx[i];
            if (roIdx < 0 ||
                roIdx >= static_cast<int64_t>(sgReadonlyRefs.size())) {
              innerFailed = true;
              return scf::ValueVector(iterArgs.begin(), iterArgs.end());
            }
            Value sref = sgReadonlyRefs[roIdx];

            if (llvm::is_contained(params.operandsToPromote, idx)) {
              auto srefType = cast<ShapedRefType>(sref.getType());
              int64_t rank = srefType.getRank();
              SmallVector<Attribute> symbolNames;
              std::string nsPrefix = "n" + std::to_string(i) + ".";
              for (int64_t d = 0; d < rank; ++d) {
                symbolNames.push_back(
                    rewriter.getStringAttr(nsPrefix + "d" + std::to_string(d)));
              }
              ShapedRefType promotedType = ShapedRefType::get(
                  rewriter.getContext(), srefType.getShape(),
                  srefType.getElementType(),
                  cast<ScopeAttrInterface>(params.subgroup.scope));
              IREE::GPU::PromotionAttr promotion =
                  getPromotionAttr(rewriter.getContext(), params, idx);
              sref = IREE::GPU::PromoteOperandOp::create(
                  rewriter, nestLoc, promotedType, promotion, sref,
                  rewriter.getArrayAttr(symbolNames));
            }

            FailureOr<MmaDistributedSliceInfo> maybeSliceInfo =
                createMmaDistributedSlice(rewriter, nestLoc, mmaLinalgOp, idx,
                                          sref, tileOffsets, tileSizes, laneId,
                                          *mmaKind, mmaPackSpecs[idx]);
            if (failed(maybeSliceInfo)) {
              innerFailed = true;
              return scf::ValueVector(iterArgs.begin(), iterArgs.end());
            }
            Value readTile = ReadSliceOp::create(
                rewriter, nestLoc, maybeSliceInfo->readTensorType,
                maybeSliceInfo->transformedSref, maybeSliceInfo->sliceOffsets,
                maybeSliceInfo->sliceSizes, maybeSliceInfo->sliceStrides);
            mmaOperands[i] =
                transposeTensor(rewriter, nestLoc, readTile,
                                maybeSliceInfo->readToLogicalPermutation);
          }

          SmallVector<Value> mmaInputs, mmaInits;
          for (OpOperand *input : mmaDpsOp.getDpsInputOperands()) {
            mmaInputs.push_back(mmaOperands[input->getOperandNumber()]);
          }
          for (OpOperand &init : mmaDpsOp.getDpsInitsMutable()) {
            mmaInits.push_back(mmaOperands[init.getOperandNumber()]);
          }

          std::optional<SmallVector<SmallVector<int64_t>>> permutations;
          permutations.emplace();
          for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
            permutations->push_back(mmaPackSpecs[i].innerDimsPos);
          }

          auto distributedSemantics = IREE::GPU::InnerTiledSemanticsAttr::get(
              rewriter.getContext(), /*distributed=*/true, /*opaque=*/true);
          auto innerTiled = Codegen::InnerTiledOp::create(
              rewriter, nestLoc, mmaInputs, mmaInits,
              mmaLinalgOp.getIndexingMapsArray(),
              mmaLinalgOp.getIteratorTypesArray(), *mmaKind,
              distributedSemantics, permutations);
          if (auto config = getLoweringConfig(mmaLinalgOp)) {
            setLoweringConfig(innerTiled, config);
          }

          SmallVector<Value> reducedValues(innerTiled.getResults().begin(),
                                           innerTiled.getResults().end());
          return scf::ValueVector(reducedValues.begin(), reducedValues.end());
        }

        // Build operand info using shared helper.
        SmallVector<DistributedOperandInfo> operandInfo =
            buildOperandInfo(rewriter, nestLoc, op, tileableSet, dpsInitIndices,
                             operandToReadonlyIdx, operandToReadwriteIdx,
                             sgReadonlyRefs, iterArgs, params);

        // Inside the reduction loop, results are returned as tiles (not
        // written to srefs) since they become iter_args.
        SmallVector<DistributedResultInfo> resultInfo;
        for (int64_t i = 0, e = op->getNumResults(); i < e; ++i) {
          resultInfo.push_back({Value()});
        }

        if (failed(target.canDistribute(tileOffsets, tileSizes, operandInfo,
                                        resultInfo))) {
          innerFailed = true;
          return scf::ValueVector(iterArgs.begin(), iterArgs.end());
        }
        FailureOr<TilingResult> tiledResult =
            target.getDistributedImplementation(b, tileOffsets, tileSizes,
                                                operandInfo, resultInfo);
        if (failed(tiledResult)) {
          innerFailed = true;
          return scf::ValueVector(iterArgs.begin(), iterArgs.end());
        }

        SmallVector<Value> reducedValues(tiledResult->tiledValues.begin(),
                                         tiledResult->tiledValues.end());
        return scf::ValueVector(reducedValues.begin(), reducedValues.end());
      });

  if (innerFailed) {
    return rewriter.notifyMatchFailure(target,
                                       "getDistributedImplementation failed");
  }

  SmallVector<Value> loopResults(loopNest.results.begin(),
                                 loopNest.results.end());

  if (params.mmaKind) {
    if (loopResults.size() != mmaInitSliceInfos.size()) {
      return rewriter.notifyMatchFailure(
          target, "mismatch between MMA loop results and init slice metadata");
    }
    for (auto [result, sliceInfo] :
         llvm::zip_equal(loopResults, mmaInitSliceInfos)) {
      Value writeValue = transposeTensor(rewriter, loc, result,
                                         sliceInfo.logicalToReadPermutation);
      WriteSliceOp::create(rewriter, loc, writeValue, sliceInfo.transformedSref,
                           sliceInfo.sliceOffsets, sliceInfo.sliceSizes,
                           sliceInfo.sliceStrides);
    }
    return loopResults;
  }

  // Emit writeback after the reduction loop nest.
  target.emitResultTileStore(rewriter, loopResults, sgReadwriteRefs,
                             laneOffsets, laneSizes, params);

  return loopResults;
}

FailureOr<GenericOp>
applyMultiLevelTiling(RewriterBase &rewriter, PCFTilingOpInterface target,
                      const MultiLevelTilingParams &params) {
  Location loc = target.getLoc();
  Operation *op = target.getOperation();

  // Get iteration domain and iterator types.
  auto tilingIface = cast<TilingInterface>(op);
  SmallVector<Range> iterDomain = tilingIface.getIterationDomain(rewriter);
  SmallVector<utils::IteratorType> iterTypes =
      tilingIface.getLoopIteratorTypes();
  int64_t numDims = iterDomain.size();

  // Classify operands: tileable inputs → readonly, DPS inits → readwrite.
  SmallVector<unsigned> tileableIndices = target.getTileableOperandIndices();
  DenseSet<unsigned> tileableSet(tileableIndices.begin(),
                                 tileableIndices.end());
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);

  DenseSet<unsigned> dpsInitIndices;
  if (dpsOp) {
    for (OpOperand &init : dpsOp.getDpsInitsMutable()) {
      dpsInitIndices.insert(init.getOperandNumber());
    }
  }

  SmallVector<Value> readonlyInits;
  SmallVector<Value> readwriteInits;
  SmallVector<int64_t> operandToReadonlyIdx(op->getNumOperands(), -1);
  SmallVector<int64_t> operandToReadwriteIdx(op->getNumOperands(), -1);

  for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    unsigned idx = static_cast<unsigned>(i);
    if (!tileableSet.contains(idx)) {
      continue;
    }
    if (!isa<ShapedType>(op->getOperand(i).getType())) {
      continue;
    }
    if (dpsInitIndices.contains(idx)) {
      operandToReadwriteIdx[i] = static_cast<int64_t>(readwriteInits.size());
      readwriteInits.push_back(op->getOperand(i));
    } else {
      operandToReadonlyIdx[i] = static_cast<int64_t>(readonlyInits.size());
      readonlyInits.push_back(op->getOperand(i));
    }
  }

  // Pad tile sizes to iteration domain size.
  SmallVector<OpFoldResult> sgTileSizes(numDims, rewriter.getIndexAttr(0));
  for (int64_t i = 0,
               e = std::min(numDims, static_cast<int64_t>(
                                         params.subgroup.tileSizes.size()));
       i < e; ++i) {
    sgTileSizes[i] = params.subgroup.tileSizes[i];
  }
  SmallVector<OpFoldResult> laneTileSizes(numDims, rewriter.getIndexAttr(0));
  for (int64_t i = 0, e = std::min(numDims, static_cast<int64_t>(
                                                params.lane.tileSizes.size()));
       i < e; ++i) {
    laneTileSizes[i] = params.lane.tileSizes[i];
  }
  SmallVector<OpFoldResult> redTileSizes(numDims, rewriter.getIndexAttr(0));
  for (int64_t i = 0,
               e = std::min(numDims, static_cast<int64_t>(
                                         params.reductionTileSizes.size()));
       i < e; ++i) {
    redTileSizes[i] = params.reductionTileSizes[i];
  }

  // Count subgroup-tiled dimensions (non-zero subgroup tiles on parallel dims).
  int64_t numSgIterators = countTiledDims(sgTileSizes);
  if (numSgIterators == 0) {
    return rewriter.notifyMatchFailure(target, "no subgroup-tiled dimensions");
  }

  // Count lane-tiled dimensions.
  int64_t numLaneIterators = countTiledDims(laneTileSizes);
  if (numLaneIterators == 0 && !params.mmaKind) {
    return rewriter.notifyMatchFailure(
        target, "ill-formed lowering config: subgroup tiling specified but no "
                "lane tiling mechanism (lane tile sizes or mma_kind)");
  }

  // === Create outer pcf.generic (subgroup scope) ===
  rewriter.setInsertionPoint(op);
  auto sgScope = cast<ScopeAttrInterface>(params.subgroup.scope);
  GenericOp outerGeneric = GenericOp::create(
      rewriter, loc, sgScope, readonlyInits, readwriteInits, numSgIterators);

  // === Populate initializer with promotion symbols ===
  if (!params.operandsToPromote.empty()) {
    OpBuilder::InsertionGuard initGuard(rewriter);
    Region &initRegion = outerGeneric.getInitializer();
    Block *initBlock = rewriter.createBlock(&initRegion);
    rewriter.setInsertionPointToStart(initBlock);

    // For each promoted operand, define symbols for the tile sizes of each
    // dimension. The tile size is determined by the indexing map: parallel
    // dims use subgroup tile sizes, reduction dims use reduction tile sizes.
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    for (unsigned promIdx : params.operandsToPromote) {
      if (!linalgOp) {
        break;
      }
      OpOperand &opOperand = op->getOpOperand(promIdx);
      AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&opOperand);

      std::string initNsPrefix = "n" + std::to_string(promIdx) + ".";
      for (auto [d, expr] : llvm::enumerate(indexingMap.getResults())) {
        std::string symName = initNsPrefix + "d" + std::to_string(d);
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        OpFoldResult tileSize;
        if (dimExpr) {
          unsigned pos = dimExpr.getPosition();
          // Use subgroup tile for parallel dims, reduction tile for
          // reduction dims.
          if (iterTypes[pos] == utils::IteratorType::reduction) {
            tileSize = (pos < redTileSizes.size()) ? redTileSizes[pos]
                                                   : rewriter.getIndexAttr(0);
          } else {
            tileSize = (pos < sgTileSizes.size()) ? sgTileSizes[pos]
                                                  : rewriter.getIndexAttr(0);
          }
        } else {
          // Non-simple indexing — use 0 as placeholder.
          tileSize = rewriter.getIndexAttr(0);
        }
        Value tileSizeVal =
            getValueOrCreateConstantIndexOp(rewriter, loc, tileSize);
        IndexSymbolOp::create(rewriter, loc, rewriter.getStringAttr(symName),
                              tileSizeVal);
      }
    }

    // Yield nothing from the initializer.
    YieldOp::create(rewriter, loc, ValueRange{});
  }

  // === Build outer body ===
  {
    OpBuilder::InsertionGuard outerGuard(rewriter);
    rewriter.setInsertionPointToStart(&outerGeneric.getRegion().front());

    ArrayRef<BlockArgument> sgReadonlyRefs = outerGeneric.getReadonlyRefArgs();
    ArrayRef<BlockArgument> sgReadwriteRefs = outerGeneric.getRegionRefArgs();
    ArrayRef<BlockArgument> sgIdArgs = outerGeneric.getIdArgs();

    // Compute subgroup tile offsets/sizes.
    SmallVector<OpFoldResult> sgOffsets, sgSizes;
    computeTileOffsetsAndSizes(rewriter, loc, sgTileSizes, iterDomain, sgIdArgs,
                               sgOffsets, sgSizes);

    // Build a sub-domain for the lane level (subgroup-relative).
    SmallVector<Range> sgDomain(numDims);
    for (int64_t i = 0; i < numDims; ++i) {
      sgDomain[i].offset = sgOffsets[i];
      sgDomain[i].size = sgSizes[i];
      sgDomain[i].stride = rewriter.getIndexAttr(1);
    }

    // === Create inner pcf.generic (lane scope) ===
    // No inits/results — captures outer srefs directly.
    //
    // When numLaneIterators == 0, the only path here is the MMA path
    // (mmaKind is set). In this case, lane distribution is handled by
    // the MMA conversion pass rather than explicit lane tiling. The
    // inner generic with 1 iterator provides the lane scope context
    // that the MMA conversion needs to operate within.
    auto laneScope = cast<ScopeAttrInterface>(params.lane.scope);
    GenericOp innerGeneric =
        numLaneIterators > 0
            ? GenericOp::create(rewriter, loc, laneScope, numLaneIterators)
            : GenericOp::create(rewriter, loc, laneScope, /*numIterators=*/1);

    {
      OpBuilder::InsertionGuard innerGuard(rewriter);
      rewriter.setInsertionPointToStart(&innerGeneric.getRegion().front());

      ArrayRef<BlockArgument> laneIdArgs = innerGeneric.getIdArgs();

      // Compute lane tile offsets/sizes (relative to subgroup tile).
      SmallVector<OpFoldResult> laneOffsets, laneSizes;
      if (numLaneIterators > 0) {
        computeTileOffsetsAndSizes(rewriter, loc, laneTileSizes, sgDomain,
                                   laneIdArgs, laneOffsets, laneSizes);
      } else {
        // MMA path: no explicit lane tiling, use full subgroup tile. The MMA
        // conversion will distribute work across lanes internally.
        laneOffsets = llvm::to_vector(sgOffsets);
        laneSizes = llvm::to_vector(sgSizes);
      }

      // Emit reduction init and determine reduction dimensions. Both MMA
      // and regular paths share the same reduction loop structure.
      SmallVector<Value> initValues = target.emitInitTileLoad(
          rewriter, sgReadwriteRefs, laneOffsets, laneSizes, params);

      SmallVector<int64_t> reductionDims;
      for (int64_t i = 0; i < numDims; ++i) {
        if (iterTypes[i] == utils::IteratorType::reduction &&
            !isZeroTileSize(redTileSizes[i])) {
          reductionDims.push_back(i);
        }
      }

      if (!reductionDims.empty()) {
        Value laneId = laneIdArgs.empty() ? Value() : Value(laneIdArgs.front());
        FailureOr<SmallVector<Value>> loopResults = buildReductionLoop(
            rewriter, loc, target, op, iterDomain, iterTypes, redTileSizes,
            laneOffsets, laneSizes, reductionDims, tileableSet, dpsInitIndices,
            operandToReadonlyIdx, operandToReadwriteIdx, sgReadonlyRefs,
            sgReadwriteRefs, initValues, params, laneId);
        if (failed(loopResults)) {
          rewriter.eraseOp(innerGeneric);
          return failure();
        }
      } else {
        // No reduction -- call getDistributedImplementation directly.
        SmallVector<DistributedOperandInfo> operandInfo;
        for (int64_t i = 0, e = op->getNumOperands(); i < e; ++i) {
          unsigned idx = static_cast<unsigned>(i);
          if (!tileableSet.contains(idx) ||
              !isa<ShapedType>(op->getOperand(i).getType())) {
            operandInfo.push_back({op->getOperand(i), /*isTile=*/false});
            continue;
          }
          if (dpsInitIndices.contains(idx)) {
            int64_t rwIdx = operandToReadwriteIdx[i];
            assert(rwIdx >= 0 &&
                   "DPS init should have been mapped to readwrite sref");
            operandInfo.push_back({sgReadwriteRefs[rwIdx], /*isTile=*/false});
          } else {
            int64_t roIdx = operandToReadonlyIdx[i];
            assert(roIdx >= 0 &&
                   "tileable input should have been mapped to readonly sref");
            operandInfo.push_back({sgReadonlyRefs[roIdx], /*isTile=*/false});
          }
        }
        // Build result info: map each result to its readwrite sref via the
        // DPS init operand, not via result index directly.
        SmallVector<DistributedResultInfo> resultInfo;
        if (dpsOp) {
          for (auto [i, init] : llvm::enumerate(dpsOp.getDpsInitsMutable())) {
            int64_t rwIdx = operandToReadwriteIdx[init.getOperandNumber()];
            assert(rwIdx >= 0 &&
                   "DPS init should have been mapped to readwrite sref");
            resultInfo.push_back({sgReadwriteRefs[rwIdx]});
          }
        } else {
          for (int64_t i = 0, e = op->getNumResults(); i < e; ++i) {
            resultInfo.push_back({sgReadwriteRefs[i]});
          }
        }
        if (failed(target.canDistribute(laneOffsets, laneSizes, operandInfo,
                                        resultInfo))) {
          rewriter.eraseOp(innerGeneric);
          return rewriter.notifyMatchFailure(target,
                                             "canDistribute check failed");
        }
        FailureOr<TilingResult> tiledResult =
            target.getDistributedImplementation(
                rewriter, laneOffsets, laneSizes, operandInfo, resultInfo);
        if (failed(tiledResult)) {
          rewriter.eraseOp(innerGeneric);
          return rewriter.notifyMatchFailure(
              target, "getDistributedImplementation failed unexpectedly");
        }
      }

      // Create pcf.return for inner generic.
      ReturnOp::create(rewriter, loc);
    }

    // Create pcf.return for outer generic.
    ReturnOp::create(rewriter, loc);
  }

  // Replace original op with outer generic's results.
  rewriter.replaceOp(op, outerGeneric.getResults());

  return outerGeneric;
}

} // namespace mlir::iree_compiler::IREE::PCF
