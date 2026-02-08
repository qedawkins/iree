// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/Template/IR/Template.h"
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::GPU {
//===----------------------------------------------------------------------===//
// BarrierRegionOp
//===----------------------------------------------------------------------===//

// Build a BarrierRegionOp with an empty.
void BarrierRegionOp::build(OpBuilder &b, OperationState &result,
                            TypeRange resultTypes, ValueRange inputs) {
  result.addOperands(inputs);
  (void)result.addRegion();
  result.addTypes(resultTypes);
  SmallVector<Location> blockArgLocs(inputs.size(), result.location);

  Region *region = result.regions[0].get();

  // `builder.createBlock` changes the insertion point within the block. Create
  // a guard to reset the insertion point of the builder after it is destroyed.
  OpBuilder::InsertionGuard guard(b);
  b.createBlock(region, region->end(), inputs.getTypes(), blockArgLocs);
}

LogicalResult BarrierRegionOp::verify() { return success(); }

LogicalResult BarrierRegionOp::verifyRegions() {
  auto &region = getRegion();
  Block &block = region.front();
  if (block.getNumArguments() != getNumOperands()) {
    return emitError(
        "expected the block argument count to match operand count");
  }

  if (!llvm::all_of_zip(block.getArgumentTypes(), getOperandTypes(),
                        [](Type a, Type b) { return a == b; })) {
    return emitError("expected block argument types to match operand types");
  }

  // Ensure that the region yields an element of the right type.
  auto yieldOp = cast<GPU::YieldOp>(block.getTerminator());
  if (yieldOp->getNumOperands() != getNumResults()) {
    return emitOpError(
        "expected body to yield same number of values as results");
  }

  if (!llvm::all_of_zip(yieldOp->getOperandTypes(), getResultTypes(),
                        [](Type a, Type b) { return a == b; })) {
    return emitError("expected yielded value types to match result types");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ValueBarrierOp
//===----------------------------------------------------------------------===//

void ValueBarrierOp::build(OpBuilder &builder, OperationState &result,
                           ValueRange input) {
  result.addOperands(input);
  result.addTypes(llvm::map_range(input, [](Value v) { return v.getType(); }));
}

LogicalResult ValueBarrierOp::verify() {
  if (getNumOperands() == 0) {
    return emitOpError("Atleast one input required");
  }

  // Make sure we either have all tensors or all vectors.
  if (hasTensorSemantics()) {
    bool allTensor =
        llvm::all_of(getInputTypes(), llvm::IsaPred<RankedTensorType>);
    if (!allTensor) {
      return emitOpError(
          "All inputs should be either of tensor or vector type");
    }
    return success();
  }

  bool allVector = llvm::all_of(getInputTypes(), llvm::IsaPred<VectorType>);
  if (!allVector) {
    return emitOpError("All inputs should be either of tensor or vector type");
  }

  return success();
}

// AMD Specific Operations

//===----------------------------------------------------------------------===//
// BufferResourceCastOp
//===----------------------------------------------------------------------===//

static RankedTensorType getMaximumStaticType(tensor::CastOp castOp) {
  auto inputType = dyn_cast<RankedTensorType>(castOp.getSource().getType());
  auto resultType = dyn_cast<RankedTensorType>(castOp.getType());
  if (!inputType || !resultType) {
    return RankedTensorType();
  }

  assert(inputType.getRank() == resultType.getRank() &&
         "Rank must match for ranked -> ranked cast");

  SmallVector<int64_t> join;
  join.reserve(inputType.getRank());
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    if (inputType.isDynamicDim(i)) {
      join.push_back(resultType.getDimSize(i));
      continue;
    }
    if (resultType.isDynamicDim(i)) {
      join.push_back(inputType.getDimSize(i));
      continue;
    }

    // Cast verifier requires that static sizes match.
    join.push_back(inputType.getDimSize(i));
  }
  return RankedTensorType::get(join, inputType.getElementType());
}

struct FoldBufferCastOfTensorCast final
    : OpRewritePattern<BufferResourceCastOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(BufferResourceCastOp castOp,
                                PatternRewriter &rewriter) const override {
    // Check whether the cast increases the amount of available static info.
    auto tensorCast = castOp.getInput().getDefiningOp<tensor::CastOp>();
    if (!tensorCast) {
      return failure();
    }

    RankedTensorType maxStaticType = getMaximumStaticType(tensorCast);
    if (!maxStaticType || maxStaticType == castOp.getInput().getType()) {
      return failure();
    }

    Value newSource = tensorCast.getSource();
    if (newSource.getType() != maxStaticType) {
      // Cast to the type with maximum static information if the input and
      // result types contain different static info.
      newSource = tensor::CastOp::create(rewriter, castOp.getLoc(),
                                         maxStaticType, newSource);
    }
    auto newBufferCast = IREE::GPU::BufferResourceCastOp::create(
        rewriter, castOp.getLoc(), maxStaticType, newSource,
        castOp.getCacheSwizzleStride());
    newBufferCast->setDiscardableAttrs(castOp->getDiscardableAttrDictionary());

    // Cast back to the original result type.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        castOp, castOp.getResult().getType(), newBufferCast);
    return success();
  };
};

void BufferResourceCastOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *ctx) {
  results.add<FoldBufferCastOfTensorCast>(ctx);
}

//===----------------------------------------------------------------------===//
// CoalescedGatherDMAOp
//===----------------------------------------------------------------------===//

// ParallelCombiningOpInterface implementation
MutableOperandRange CoalescedGatherDMAOp::getUpdatedDestinations() {
  // Only relevant for tensor operands
  if (!isa<RankedTensorType>(getInit().getType())) {
    return MutableOperandRange(getOperation(), /*start=*/0, /*length=*/0);
  }
  // Return the init operand as the destination being updated
  return getInitMutable();
}

Operation *CoalescedGatherDMAOp::getIteratingParent() {
  // Only relevant for tensor operands
  if (!isa<RankedTensorType>(getInit().getType())) {
    return nullptr;
  }
  // Return the parent scf.forall operation
  return getOperation()->getParentOfType<scf::ForallOp>();
}

LogicalResult CoalescedGatherDMAOp::verify() {
  TypedValue<ShapedType> init = getInit();
  auto initType = init.getType();

  bool hasTensor = isa<RankedTensorType>(initType);
  bool hasMemRef = isa<MemRefType>(initType);

  if (!hasTensor && !hasMemRef) {
    return emitOpError("init type must either be a tensor or a memref");
  }

  auto initShapedType = cast<ShapedType>(initType);
  auto sourceType = cast<ShapedType>(getSource().getType());
  ArrayRef<int64_t> initShape = initShapedType.getShape();
  ArrayRef<int64_t> sourceShape = sourceType.getShape();

  if (hasTensor && !isa<RankedTensorType>(sourceType)) {
    return emitOpError("source must be tensor when init is tensor");
  }
  if (hasMemRef && !isa<MemRefType>(sourceType)) {
    return emitOpError("source must be memref when init is memref");
  }

  OperandRange indices = getIndices();

  if (indices.size() > initShape.size()) {
    return emitOpError("number of indices (")
           << indices.size() << ") cannot exceed destination rank ("
           << initShape.size() << ")";
  }

  if (indices.size() > sourceShape.size()) {
    return emitOpError("number of indices (")
           << indices.size() << ") cannot exceed source rank ("
           << sourceShape.size() << ")";
  }

  // Make sure indices have no dynamic shapes.
  for (auto [i, indexVal] : llvm::enumerate(indices)) {
    auto indexType = cast<ShapedType>(indexVal.getType());
    for (auto dim : indexType.getShape()) {
      if (ShapedType::isDynamic(dim)) {
        return emitOpError("expected index ") << i << " to have static shape";
      }
    }
  }

  // For gather operations with indices, all index vectors should have the same
  // length equal to the batch size (first dimension of destination). This is
  // validated here so that lowering passes can rely on these constraints
  // without duplicating the checks.
  if (!indices.empty()) {
    // Verify all index vectors are 1D and have the same length.
    auto firstIndexShape = cast<ShapedType>(indices[0].getType()).getShape();
    if (firstIndexShape.size() != 1) {
      return emitOpError("expected index 0 to be a 1-D tensor or vector");
    }
    int64_t batchSize = firstIndexShape.front();

    for (auto [i, indexVal] : llvm::enumerate(indices)) {
      auto indexShape = cast<ShapedType>(indexVal.getType()).getShape();
      if (indexShape.size() != 1) {
        return emitOpError("expected index ")
               << i << " to be a 1-D tensor or vector";
      }
      if (indexShape.front() != batchSize) {
        return emitOpError(
                   "expected all index vectors to have the same length; ")
               << "index " << i << " has length " << indexShape.front()
               << " but expected " << batchSize;
      }
    }

    // The batch size should match the first dimension of the destination.
    if (!initShape.empty() && batchSize != initShape[0]) {
      return emitOpError("expected batch size (length of index vectors: ")
             << batchSize << ") to match first destination dimension ("
             << initShape[0] << ")";
    }
  }

  // Verify the contiguous (non-indexed) dimensions match between source and
  // dest.
  for (auto [dim, size] : llvm::enumerate(initShape)) {
    if (dim >= sourceShape.size()) {
      return emitOpError("expected source to have at least ")
             << (dim + 1) << " dimensions when destination has rank "
             << initShape.size();
    }

    // Skip indexed dimensions - they're validated above.
    if (dim < indices.size()) {
      continue;
    }

    // Check the suffix (hidden) gathering dimensions are the same in `source`
    // and `init`.
    int64_t sourceDim = sourceShape[dim];
    if (sourceDim != size) {
      return emitOpError("expected unindexed dimension ")
             << dim << " to have same length in source (" << sourceDim
             << ") and destination (" << size << ')';
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ProcessInnerTileOp
//===----------------------------------------------------------------------===//

// Parser for the custom assembly format:
//   bounds(%m, %n, %k : index, index, index)
//   kind(#iree_gpu.mma_layout<...>)
//   indexing_maps = [...]
//   iterator_types = [...]
//   outer_dim_distribution = [...]
//   ins(%lhs, %rhs : ...)
//   outs(%init : ...)
//   @template_func -> result_types
ParseResult ProcessInnerTileOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  MLIRContext *context = parser.getContext();

  // Parse bounds(%m, %n, %k : index, index, index)
  SmallVector<OpAsmParser::UnresolvedOperand> boundsOperands;
  SmallVector<Type> boundsTypes;
  if (parser.parseKeyword("bounds") || parser.parseLParen()) {
    return failure();
  }
  if (parser.parseOperandList(boundsOperands) || parser.parseColon() ||
      parser.parseTypeList(boundsTypes) || parser.parseRParen()) {
    return failure();
  }
  if (parser.resolveOperands(boundsOperands, boundsTypes, parser.getNameLoc(),
                             result.operands)) {
    return failure();
  }

  // Parse kind(...)
  Codegen::InnerTileDescAttrInterface kindAttr;
  if (parser.parseKeyword("kind") || parser.parseLParen() ||
      parser.parseAttribute(kindAttr) || parser.parseRParen()) {
    return failure();
  }
  result.addAttribute("kind", kindAttr);

  // Parse indexing_maps = [...]
  SmallVector<Attribute> indexingMaps;
  if (parser.parseKeyword("indexing_maps") || parser.parseEqual() ||
      parser.parseLSquare()) {
    return failure();
  }
  do {
    AffineMapAttr mapAttr;
    if (parser.parseAttribute(mapAttr)) {
      return failure();
    }
    indexingMaps.push_back(mapAttr);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) {
    return failure();
  }
  result.addAttribute("indexing_maps", ArrayAttr::get(context, indexingMaps));

  // Parse iterator_types = [...]
  SmallVector<Attribute> iteratorTypes;
  if (parser.parseKeyword("iterator_types") || parser.parseEqual() ||
      parser.parseLSquare()) {
    return failure();
  }
  do {
    StringAttr iterAttr;
    if (parser.parseAttribute(iterAttr)) {
      return failure();
    }
    iteratorTypes.push_back(linalg::IteratorTypeAttr::get(
        context, llvm::StringSwitch<utils::IteratorType>(iterAttr.getValue())
                     .Case("parallel", utils::IteratorType::parallel)
                     .Case("reduction", utils::IteratorType::reduction)));
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) {
    return failure();
  }
  result.addAttribute("iterator_types", ArrayAttr::get(context, iteratorTypes));

  // Parse outer_dim_distribution = [...]
  SmallVector<int64_t> outerDimDist;
  if (parser.parseKeyword("outer_dim_distribution") || parser.parseEqual() ||
      parser.parseLSquare()) {
    return failure();
  }
  do {
    int64_t val;
    if (parser.parseInteger(val)) {
      return failure();
    }
    outerDimDist.push_back(val);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) {
    return failure();
  }
  result.addAttribute("outer_dim_distribution",
                      DenseI64ArrayAttr::get(context, outerDimDist));

  // Parse ins(...) - inputs
  SmallVector<OpAsmParser::UnresolvedOperand> inputOperands;
  SmallVector<Type> inputTypes;
  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }
  if (parser.parseOperandList(inputOperands) || parser.parseColon() ||
      parser.parseTypeList(inputTypes) || parser.parseRParen()) {
    return failure();
  }

  // Parse outs(...) - outputs
  SmallVector<OpAsmParser::UnresolvedOperand> outputOperands;
  SmallVector<Type> outputTypes;
  if (parser.parseKeyword("outs") || parser.parseLParen()) {
    return failure();
  }
  if (parser.parseOperandList(outputOperands) || parser.parseColon() ||
      parser.parseTypeList(outputTypes) || parser.parseRParen()) {
    return failure();
  }

  // Resolve input and output operands.
  if (parser.resolveOperands(inputOperands, inputTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperands(outputOperands, outputTypes, parser.getNameLoc(),
                             result.operands)) {
    return failure();
  }

  // Record segment sizes.
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(boundsOperands.size()),
                           static_cast<int32_t>(inputOperands.size()),
                           static_cast<int32_t>(outputOperands.size())}));

  // Parse @callee
  FlatSymbolRefAttr calleeAttr;
  if (parser.parseAttribute(calleeAttr)) {
    return failure();
  }
  result.addAttribute("callee", calleeAttr);

  // Parse -> result_types
  SmallVector<Type> resultTypes;
  if (parser.parseArrow() || parser.parseTypeList(resultTypes)) {
    return failure();
  }
  result.addTypes(resultTypes);

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void ProcessInnerTileOp::print(OpAsmPrinter &p) {
  // bounds(%m, %n, %k : index, index, index)
  p << " bounds(";
  llvm::interleaveComma(getBounds(), p, [&](Value v) { p.printOperand(v); });
  p << " : ";
  llvm::interleaveComma(getBounds().getTypes(), p);
  p << ")";

  // kind(...)
  p << " kind(" << getKind() << ")";

  // indexing_maps = [...]
  p << " indexing_maps = [";
  llvm::interleaveComma(getIndexingMaps(), p);
  p << "]";

  // iterator_types = [...]
  p << " iterator_types = [";
  llvm::interleaveComma(getIteratorTypes(), p, [&](Attribute attr) {
    auto iterType = cast<linalg::IteratorTypeAttr>(attr).getValue();
    p << "\"" << utils::stringifyIteratorType(iterType) << "\"";
  });
  p << "]";

  // outer_dim_distribution = [...]
  p << " outer_dim_distribution = [";
  llvm::interleaveComma(getOuterDimDistribution(), p);
  p << "]";

  // ins(...)
  p << " ins(";
  llvm::interleaveComma(getInputs(), p, [&](Value v) { p.printOperand(v); });
  p << " : ";
  llvm::interleaveComma(getInputs().getTypes(), p);
  p << ")";

  // outs(...)
  p << " outs(";
  llvm::interleaveComma(getOutputs(), p, [&](Value v) { p.printOperand(v); });
  p << " : ";
  llvm::interleaveComma(getOutputs().getTypes(), p);
  p << ")";

  // @callee
  p << " ";
  p.printAttribute(getCalleeAttr());

  // -> result_types
  p << " -> ";
  llvm::interleaveComma(getResultTypes(), p);

  // Print any extra attributes (excluding known ones).
  SmallVector<StringRef> elidedAttrs = {
      "kind",           "indexing_maps",
      "iterator_types", "outer_dim_distribution",
      "callee",         "operandSegmentSizes"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

LogicalResult ProcessInnerTileOp::verify() {
  // Verify the number of indexing maps matches inputs + outputs.
  size_t numOperands = getInputs().size() + getOutputs().size();
  if (getIndexingMaps().size() != numOperands) {
    return emitOpError("expected ") << numOperands << " indexing maps but got "
                                    << getIndexingMaps().size();
  }

  // Verify the number of iterator types matches bounds.
  if (getIteratorTypes().size() != getBounds().size()) {
    return emitOpError("expected ")
           << getBounds().size() << " iterator types but got "
           << getIteratorTypes().size();
  }

  // Verify result types match output types.
  if (getResults().size() != getOutputs().size()) {
    return emitOpError("expected ")
           << getOutputs().size() << " results but got " << getResults().size();
  }
  for (auto [result, output] : llvm::zip(getResults(), getOutputs())) {
    if (result.getType() != output.getType()) {
      return emitOpError("result type ")
             << result.getType() << " does not match output type "
             << output.getType();
    }
  }

  // Verify indexing map dimensions match bounds count.
  int64_t numIterators = getBounds().size();
  for (auto [i, mapAttr] : llvm::enumerate(getIndexingMaps())) {
    AffineMap map = cast<AffineMapAttr>(mapAttr).getValue();
    if (map.getNumDims() != static_cast<unsigned>(numIterators)) {
      return emitOpError("indexing map ")
             << i << " has " << map.getNumDims() << " dims but expected "
             << numIterators;
    }
  }

  // Verify kind interface constraints.
  Codegen::InnerTileDescAttrInterface kind = getKind();
  if (static_cast<int64_t>(getInputs().size()) != kind.getExpectedNumInputs()) {
    return emitOpError("expected ")
           << kind.getExpectedNumInputs() << " inputs for kind but got "
           << getInputs().size();
  }
  if (static_cast<int64_t>(getOutputs().size()) !=
      kind.getExpectedNumOutputs()) {
    return emitOpError("expected ")
           << kind.getExpectedNumOutputs() << " outputs for kind but got "
           << getOutputs().size();
  }

  // Verify indexing maps using the kind interface.
  SmallVector<AffineMap> maps;
  for (Attribute attr : getIndexingMaps()) {
    maps.push_back(cast<AffineMapAttr>(attr).getValue());
  }
  if (failed(kind.verifyIndexingMaps(maps))) {
    return emitOpError("indexing maps failed kind verification");
  }

  return success();
}

// TemplateCallOpInterface implementation

FlatSymbolRefAttr ProcessInnerTileOp::getCalledSymbol() {
  return getCalleeAttr();
}

SmallVector<Value> ProcessInnerTileOp::getCallOperands() {
  // The template.func signature is:
  //   (%output: tensor, %k: index, %lhs: tensor, %rhs: tensor) -> tensor
  // We return: outputs + reduction bounds + inputs.
  SmallVector<Value> operands;

  // First: output operands.
  for (Value output : getOutputs()) {
    operands.push_back(output);
  }

  // Second: reduction dimension bounds.
  ArrayAttr iteratorTypeAttrs = getIteratorTypes();
  ValueRange bounds = getBounds();
  for (auto [bound, iterTypeAttr] : llvm::zip(bounds, iteratorTypeAttrs)) {
    auto iterType = cast<linalg::IteratorTypeAttr>(iterTypeAttr).getValue();
    if (iterType == utils::IteratorType::reduction) {
      operands.push_back(bound);
    }
  }

  // Third: input operands.
  for (Value input : getInputs()) {
    operands.push_back(input);
  }

  return operands;
}

SmallVector<SmallVector<Type>> ProcessInnerTileOp::getTemplateTypes() {
  // Three type bindings with outer-then-inner distributed layout:
  //   type<0>: Distributed accumulator (per-subgroup).
  //   type<1>: Distributed LHS input (per-subgroup).
  //   type<2>: Distributed RHS input (per-subgroup).
  //
  // Shape = [outer_dims..., inner_dims...] where:
  //   outer[d] = (tile_bound[iterDim] / intrinsic[d]) / dist[iterDim]
  //   inner[d] = distributed_inner_tile_shape[d]
  SmallVector<SmallVector<Type>> bindings;

  Codegen::InnerTileDescAttrInterface kind = getKind();

  // Get undistributed and distributed tile types from MMA layout.
  SmallVector<VectorType> undistributedTypes;
  kind.getUndistributedTileTypes(undistributedTypes);
  SmallVector<VectorType> distributedTypes;
  kind.getDistributedTileTypes(distributedTypes);

  // Get indexing maps.
  SmallVector<AffineMap> indexingMaps;
  for (Attribute attr : getIndexingMaps()) {
    indexingMaps.push_back(cast<AffineMapAttr>(attr).getValue());
  }

  // Get iterator types.
  SmallVector<utils::IteratorType> iteratorTypes;
  for (Attribute attr : getIteratorTypes()) {
    iteratorTypes.push_back(cast<linalg::IteratorTypeAttr>(attr).getValue());
  }

  // Get bounds as static values.
  SmallVector<int64_t> staticBounds;
  for (Value bound : getBounds()) {
    auto constOp = bound.getDefiningOp<arith::ConstantIndexOp>();
    if (!constOp) {
      return bindings;
    }
    staticBounds.push_back(constOp.value());
  }

  // Build mapping from iterator dim -> distribution index.
  ArrayRef<int64_t> outerDimDist = getOuterDimDistribution();
  SmallVector<int64_t> iterDimToDistIdx(iteratorTypes.size(), -1);
  int64_t distIdx = 0;
  for (unsigned i = 0; i < iteratorTypes.size(); ++i) {
    if (iteratorTypes[i] == utils::IteratorType::parallel) {
      iterDimToDistIdx[i] = distIdx++;
    }
  }

  int64_t numInputs = getInputs().size();

  // Compute distributed type for one operand.
  // The shape is [outer_dims..., inner_dims...] where:
  //   outer[d] = (tile_bound / intrinsic_size) / distribution_factor
  //   inner = canonical per-lane element shape from MMA layout
  auto computeDistType = [&](int64_t operandIdx, AffineMap indexingMap,
                             Type elemType) -> RankedTensorType {
    VectorType undistType = undistributedTypes[operandIdx];
    ArrayRef<int64_t> intrinsicShape = undistType.getShape();

    SmallVector<int64_t> outerDims;
    for (unsigned d = 0; d < indexingMap.getNumResults(); ++d) {
      auto dimExpr = cast<AffineDimExpr>(indexingMap.getResult(d));
      unsigned iterDim = dimExpr.getPosition();
      int64_t tileBound = staticBounds[iterDim];
      int64_t intrinsicSize = intrinsicShape[d];
      int64_t totalOuter = tileBound / intrinsicSize;

      if (iteratorTypes[iterDim] == utils::IteratorType::parallel) {
        int64_t di = iterDimToDistIdx[iterDim];
        if (di >= 0 && di < static_cast<int64_t>(outerDimDist.size())) {
          totalOuter /= outerDimDist[di];
        }
      }
      outerDims.push_back(totalOuter);
    }

    // Compute canonical per-lane inner shape from MMA layout.
    // This matches the rank-reduced shape used by
    // populateOperandOffsetsSizesStrides, ensuring consistent dimensions
    // for sref read/write operations.
    SmallVector<int64_t> innerDims;
    if (auto mmaAttr = dyn_cast<MMAAttr>(kind)) {
      MMASingleSubgroupLayout layout =
          getSingleSubgroupLayout(mmaAttr.getIntrinsic(), operandIdx);
      for (auto [outer, element] :
           llvm::zip(layout.outer, layout.element)) {
        if (outer != 1)
          innerDims.push_back(outer);
        innerDims.push_back(element);
      }
    } else {
      // Fallback: use distributed vector shape.
      VectorType distType = distributedTypes[operandIdx];
      innerDims.assign(distType.getShape().begin(),
                       distType.getShape().end());
    }

    // Shape = [outer_dims..., inner_dims...]
    SmallVector<int64_t> shape;
    shape.append(outerDims.begin(), outerDims.end());
    shape.append(innerDims.begin(), innerDims.end());
    return RankedTensorType::get(shape, elemType);
  };

  // type<0>: Distributed accumulator.
  // ACC is the last operand (operandIdx = numInputs).
  {
    int64_t accIdx = numInputs;
    AffineMap accMap = indexingMaps.back();
    Type accElemType =
        cast<RankedTensorType>(getOutputs().front().getType()).getElementType();
    bindings.push_back({computeDistType(accIdx, accMap, accElemType)});
  }

  // type<1+i>: Distributed input i.
  for (int64_t i = 0; i < numInputs; ++i) {
    Type inputElemType =
        cast<RankedTensorType>(getInputs()[i].getType()).getElementType();
    bindings.push_back({computeDistType(i, indexingMaps[i], inputElemType)});
  }

  return bindings;
}

/// Helper: Read a distributed tensor from a sref by looping over outer tiles.
///
/// Constructs a tensor with shape [outerDims..., innerDims...] by iterating
/// over outer tile positions. For each tile, computes:
///   sref_offset = sg_base + outer_pos * intrinsic_size + per_lane_inner_offset
/// and reads the per-lane inner tile via pcf.read_slice.
///
/// \param extraLeadingOffsets - Extra leading sref dimensions (e.g., buf_idx)
static Value readDistributedFromSref(
    OpBuilder &builder, Location loc,
    Codegen::InnerTileDescAttrInterface kind, uint32_t operandIdx,
    AffineMap indexingMap, Value sref, Value sgId, Value laneId,
    RankedTensorType distType, ArrayRef<OpFoldResult> extraLeadingOffsets,
    ArrayRef<utils::IteratorType> iteratorTypes,
    ArrayRef<int64_t> outerDimDist,
    ArrayRef<int64_t> iterDimToDistIdx) {
  // Get intrinsic shapes.
  SmallVector<VectorType> undistTypes;
  kind.getUndistributedTileTypes(undistTypes);

  ArrayRef<int64_t> intrinsicShape = undistTypes[operandIdx].getShape();

  // Compute outer/inner dim split from the indexing map:
  // outer dims = number of indexing map results (one per logical dimension).
  int64_t numOuterDims = indexingMap.getNumResults();
  int64_t numInnerDims = distType.getRank() - numOuterDims;
  ArrayRef<int64_t> outerShape = distType.getShape().slice(0, numOuterDims);

  // Get per-lane inner offsets/sizes/strides.
  // The perm size must match the rank-reduced canonical layout dimensions.
  SmallVector<OpFoldResult> innerOffsets, innerSizes, innerStrides;
  SmallVector<int64_t> perm =
      llvm::to_vector(llvm::seq<int64_t>(0, numInnerDims));
  (void)kind.populateOperandOffsetsSizesStrides(builder, loc, operandIdx,
                                                 laneId, perm, innerOffsets,
                                                 innerSizes, innerStrides);

  // Delinearize sg_id into per-parallel-dim indices.
  SmallVector<Value> sgParallelIndices;
  {
    Value remaining = sgId;
    for (int d = static_cast<int>(outerDimDist.size()) - 1; d >= 0; --d) {
      Value dimSize =
          arith::ConstantIndexOp::create(builder, loc, outerDimDist[d]);
      sgParallelIndices.push_back(
          arith::RemUIOp::create(builder, loc, remaining, dimSize));
      remaining = arith::DivUIOp::create(builder, loc, remaining, dimSize);
    }
    std::reverse(sgParallelIndices.begin(), sgParallelIndices.end());
  }

  // Compute per-operand-dim subgroup base offsets in the sref.
  SmallVector<Value> sgBaseOffset(numOuterDims);
  for (int64_t d = 0; d < numOuterDims; ++d) {
    unsigned iterDim =
        cast<AffineDimExpr>(indexingMap.getResult(d)).getPosition();
    if (iteratorTypes[iterDim] == utils::IteratorType::parallel) {
      int64_t di = iterDimToDistIdx[iterDim];
      Value sgIdx = sgParallelIndices[di];
      int64_t scale = outerShape[d] * intrinsicShape[d];
      Value scaleVal = arith::ConstantIndexOp::create(builder, loc, scale);
      sgBaseOffset[d] = arith::MulIOp::create(builder, loc, sgIdx, scaleVal);
    } else {
      sgBaseOffset[d] = arith::ConstantIndexOp::create(builder, loc, 0);
    }
  }

  // Inner tile shape for read_slice result.
  SmallVector<int64_t> innerTileStaticShape;
  for (OpFoldResult size : innerSizes) {
    if (auto attr = dyn_cast<Attribute>(size)) {
      innerTileStaticShape.push_back(cast<IntegerAttr>(attr).getInt());
    } else {
      innerTileStaticShape.push_back(ShapedType::kDynamic);
    }
  }

  // Full distributed vector type for the accumulator.
  VectorType distVecType =
      VectorType::get(distType.getShape(), distType.getElementType());

  // Inner tile vector type (without extra leading dims).
  VectorType innerTileVecType =
      VectorType::get(innerTileStaticShape, distType.getElementType());

  // Flat loop over outer tiles.
  int64_t totalOuterTiles = 1;
  for (int64_t d : outerShape)
    totalOuterTiles *= d;

  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
  Value totalOuterVal =
      arith::ConstantIndexOp::create(builder, loc, totalOuterTiles);

  // Initialize accumulator vector to zero.
  Value zeroVec = arith::ConstantOp::create(builder, loc, distVecType,
                                            builder.getZeroAttr(distVecType));

  auto forOp =
      scf::ForOp::create(builder, loc, c0, totalOuterVal, c1, zeroVec);
  // Mark loop for unrolling so vector.insert chain becomes pure SSA.
  forOp->setAttr("unroll_loop", UnitAttr::get(builder.getContext()));

  {
    Block *body = forOp.getBody();
    if (!body->empty() && body->back().hasTrait<OpTrait::IsTerminator>())
      body->back().erase();
    builder.setInsertionPointToEnd(body);

    Value flatIdx = forOp.getInductionVar();
    Value loopVec = forOp.getRegionIterArg(0);

    // Delinearize flatIdx into per-dim outer positions.
    SmallVector<Value> outerPos(numOuterDims);
    {
      Value remainingFlat = flatIdx;
      for (int d = numOuterDims - 1; d >= 0; --d) {
        Value dimSizeVal =
            arith::ConstantIndexOp::create(builder, loc, outerShape[d]);
        outerPos[d] =
            arith::RemUIOp::create(builder, loc, remainingFlat, dimSizeVal);
        remainingFlat =
            arith::DivUIOp::create(builder, loc, remainingFlat, dimSizeVal);
      }
    }

    // Compute sref offsets.
    SmallVector<OpFoldResult> srefOffsets;
    srefOffsets.append(extraLeadingOffsets.begin(), extraLeadingOffsets.end());
    for (int64_t d = 0; d < numOuterDims; ++d) {
      Value tileOffset = arith::MulIOp::create(
          builder, loc, outerPos[d],
          arith::ConstantIndexOp::create(builder, loc, intrinsicShape[d]));
      Value withSg =
          arith::AddIOp::create(builder, loc, sgBaseOffset[d], tileOffset);
      Value innerOff;
      if (auto attr = dyn_cast<Attribute>(innerOffsets[d])) {
        innerOff = arith::ConstantIndexOp::create(
            builder, loc, cast<IntegerAttr>(attr).getInt());
      } else {
        innerOff = cast<Value>(innerOffsets[d]);
      }
      Value totalOff = arith::AddIOp::create(builder, loc, withSg, innerOff);
      srefOffsets.push_back(totalOff);
    }

    SmallVector<OpFoldResult> srefSizes;
    for (unsigned i = 0; i < extraLeadingOffsets.size(); ++i) {
      srefSizes.push_back(builder.getIndexAttr(1));
    }
    srefSizes.append(innerSizes.begin(), innerSizes.end());

    int64_t srefReadRank =
        static_cast<int64_t>(extraLeadingOffsets.size()) + numInnerDims;
    SmallVector<OpFoldResult> srefStrides(srefReadRank,
                                          builder.getIndexAttr(1));

    // Read type must match sref rank (AllRanksMatch constraint).
    SmallVector<int64_t> readShape;
    for (unsigned i = 0; i < extraLeadingOffsets.size(); ++i) {
      readShape.push_back(1);
    }
    readShape.append(innerTileStaticShape.begin(), innerTileStaticShape.end());
    VectorType readVecType =
        VectorType::get(readShape, distType.getElementType());

    Value readResult = PCF::ReadSliceOp::create(builder, loc, readVecType, sref,
                                                srefOffsets, srefSizes,
                                                srefStrides);

    // Remove extra leading unit dims via vector.shape_cast if present.
    Value innerTile = readResult;
    if (!extraLeadingOffsets.empty()) {
      innerTile = vector::ShapeCastOp::create(builder, loc, innerTileVecType,
                                               readResult);
    }

    // Insert inner tile into accumulator vector at [outerPos...].
    // vector.insert inserts an (n-k)-D sub-vector at k-D position.
    SmallVector<OpFoldResult> insertPos;
    for (int64_t d = 0; d < numOuterDims; ++d) {
      insertPos.push_back(outerPos[d]);
    }
    Value updated =
        vector::InsertOp::create(builder, loc, innerTile, loopVec, insertPos);

    scf::YieldOp::create(builder, loc, updated);
  }

  builder.setInsertionPointAfter(forOp);

  // Convert vector result back to tensor via tensor.empty + transfer_write.
  // This is a simple pattern that bufferizes correctly (unlike the previous
  // tensor.empty + tensor.insert_slice loop that confused EliminateEmptyTensors).
  Value emptyTensor = tensor::EmptyOp::create(builder, loc, distType.getShape(),
                                               distType.getElementType());
  SmallVector<Value> zeros(distType.getRank(), c0);
  auto writeOp = vector::TransferWriteOp::create(
      builder, loc, forOp.getResult(0), emptyTensor, zeros,
      SmallVector<bool>(distType.getRank(), true));
  return writeOp.getResult();
}

/// Helper: Write a distributed tensor to a sref by looping over outer tiles.
/// Reverse of readDistributedFromSref.
static void writeDistributedToSref(
    OpBuilder &builder, Location loc,
    Codegen::InnerTileDescAttrInterface kind, uint32_t operandIdx,
    AffineMap indexingMap, Value source, Value sref, Value sgId, Value laneId,
    RankedTensorType distType, ArrayRef<utils::IteratorType> iteratorTypes,
    ArrayRef<int64_t> outerDimDist,
    ArrayRef<int64_t> iterDimToDistIdx) {
  // Get intrinsic shapes.
  SmallVector<VectorType> undistTypes;
  kind.getUndistributedTileTypes(undistTypes);

  ArrayRef<int64_t> intrinsicShape = undistTypes[operandIdx].getShape();

  // Compute outer/inner dim split from the indexing map.
  int64_t numOuterDims = indexingMap.getNumResults();
  int64_t numInnerDims = distType.getRank() - numOuterDims;
  ArrayRef<int64_t> outerShape = distType.getShape().slice(0, numOuterDims);

  // Get per-lane inner offsets/sizes/strides.
  SmallVector<OpFoldResult> innerOffsets, innerSizes, innerStrides;
  SmallVector<int64_t> perm =
      llvm::to_vector(llvm::seq<int64_t>(0, numInnerDims));
  (void)kind.populateOperandOffsetsSizesStrides(builder, loc, operandIdx,
                                                 laneId, perm, innerOffsets,
                                                 innerSizes, innerStrides);

  // Delinearize sg_id.
  SmallVector<Value> sgParallelIndices;
  {
    Value remaining = sgId;
    for (int d = static_cast<int>(outerDimDist.size()) - 1; d >= 0; --d) {
      Value dimSize =
          arith::ConstantIndexOp::create(builder, loc, outerDimDist[d]);
      sgParallelIndices.push_back(
          arith::RemUIOp::create(builder, loc, remaining, dimSize));
      remaining = arith::DivUIOp::create(builder, loc, remaining, dimSize);
    }
    std::reverse(sgParallelIndices.begin(), sgParallelIndices.end());
  }

  // Compute sg base offsets.
  SmallVector<Value> sgBaseOffset(numOuterDims);
  for (int64_t d = 0; d < numOuterDims; ++d) {
    unsigned iterDim =
        cast<AffineDimExpr>(indexingMap.getResult(d)).getPosition();
    if (iteratorTypes[iterDim] == utils::IteratorType::parallel) {
      int64_t di = iterDimToDistIdx[iterDim];
      Value sgIdx = sgParallelIndices[di];
      int64_t scale = outerShape[d] * intrinsicShape[d];
      Value scaleVal = arith::ConstantIndexOp::create(builder, loc, scale);
      sgBaseOffset[d] = arith::MulIOp::create(builder, loc, sgIdx, scaleVal);
    } else {
      sgBaseOffset[d] = arith::ConstantIndexOp::create(builder, loc, 0);
    }
  }

  // Convert source tensor to vector to avoid alloca+subview after
  // bufferization (which OptimizeVectorTransferPass would incorrectly remove).
  VectorType distVecType =
      VectorType::get(distType.getShape(), distType.getElementType());
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
  SmallVector<Value> readIndices(distType.getRank(), c0);
  Value padValue;
  Type elemType = distType.getElementType();
  if (isa<FloatType>(elemType)) {
    padValue = arith::ConstantOp::create(builder, loc, elemType,
                                         builder.getFloatAttr(elemType, 0.0));
  } else {
    padValue = arith::ConstantOp::create(builder, loc, elemType,
                                         builder.getIntegerAttr(elemType, 0));
  }
  Value srcVec = vector::TransferReadOp::create(
      builder, loc, distVecType, source, readIndices, padValue,
      SmallVector<bool>(distType.getRank(), true));

  // Flat loop over outer tiles (no iter_args - writes are side effects).
  int64_t totalOuterTiles = 1;
  for (int64_t d : outerShape)
    totalOuterTiles *= d;

  Value totalOuterVal =
      arith::ConstantIndexOp::create(builder, loc, totalOuterTiles);

  auto forOp = scf::ForOp::create(builder, loc, c0, totalOuterVal, c1);
  // Mark for unrolling so vector.extract positions become constants.
  forOp->setAttr("unroll_loop", UnitAttr::get(builder.getContext()));
  {
    Block *body = forOp.getBody();
    if (!body->empty() && body->back().hasTrait<OpTrait::IsTerminator>())
      body->back().erase();
    builder.setInsertionPointToEnd(body);

    Value flatIdx = forOp.getInductionVar();

    // Delinearize flatIdx.
    SmallVector<Value> outerPos(numOuterDims);
    {
      Value remainingFlat = flatIdx;
      for (int d = numOuterDims - 1; d >= 0; --d) {
        Value dimSizeVal =
            arith::ConstantIndexOp::create(builder, loc, outerShape[d]);
        outerPos[d] =
            arith::RemUIOp::create(builder, loc, remainingFlat, dimSizeVal);
        remainingFlat =
            arith::DivUIOp::create(builder, loc, remainingFlat, dimSizeVal);
      }
    }

    // Extract inner tile from the source vector at outer positions.
    SmallVector<OpFoldResult> extractPos;
    for (int64_t d = 0; d < numOuterDims; ++d) {
      extractPos.push_back(outerPos[d]);
    }
    Value innerTile =
        vector::ExtractOp::create(builder, loc, srcVec, extractPos);

    // Compute sref offsets.
    SmallVector<OpFoldResult> srefOffsets;
    for (int64_t d = 0; d < numOuterDims; ++d) {
      Value tileOffset = arith::MulIOp::create(
          builder, loc, outerPos[d],
          arith::ConstantIndexOp::create(builder, loc, intrinsicShape[d]));
      Value withSg =
          arith::AddIOp::create(builder, loc, sgBaseOffset[d], tileOffset);
      Value innerOff;
      if (auto attr = dyn_cast<Attribute>(innerOffsets[d])) {
        innerOff = arith::ConstantIndexOp::create(
            builder, loc, cast<IntegerAttr>(attr).getInt());
      } else {
        innerOff = cast<Value>(innerOffsets[d]);
      }
      Value totalOff = arith::AddIOp::create(builder, loc, withSg, innerOff);
      srefOffsets.push_back(totalOff);
    }

    SmallVector<OpFoldResult> srefSizes;
    srefSizes.append(innerSizes.begin(), innerSizes.end());
    SmallVector<OpFoldResult> srefStrides(numInnerDims,
                                          builder.getIndexAttr(1));

    PCF::WriteSliceOp::create(builder, loc, innerTile, sref, srefOffsets,
                              srefSizes, srefStrides);

    scf::YieldOp::create(builder, loc, ValueRange{});
  }

  builder.setInsertionPointAfter(forOp);
}

LogicalResult ProcessInnerTileOp::inlineImplementationBlocks(
    OpBuilder &builder, ArrayRef<Block *> blocksToPopulate) {
  // Pingpong template has 7 blocks. We populate those with unimplemented:
  //   Block 0: Init accumulators (sg_id, lane_id, dest: sref) -> type<0>
  //   Block 1: Copy LHS (SKIP - already concrete from PingpongConfig)
  //   Block 2: Copy RHS (SKIP - already concrete from PingpongConfig)
  //   Block 3: Read LHS from shared (buf_idx, sg_id, lane_id, alloc) -> type<1>
  //   Block 4: Read RHS from shared (buf_idx, sg_id, lane_id, alloc) -> type<2>
  //   Block 5: Compute MMA (acc, lhs, rhs) -> type<0>
  //   Block 6: Write results (sg_id, lane_id, result, dest: sref)

  Location loc = getLoc();
  MLIRContext *context = getContext();

  // Get indexing maps and iterator types.
  SmallVector<AffineMap> indexingMaps;
  for (Attribute attr : getIndexingMaps()) {
    indexingMaps.push_back(cast<AffineMapAttr>(attr).getValue());
  }

  SmallVector<utils::IteratorType> iteratorTypes;
  for (Attribute attr : getIteratorTypes()) {
    iteratorTypes.push_back(cast<linalg::IteratorTypeAttr>(attr).getValue());
  }

  Codegen::InnerTileDescAttrInterface kind = getKind();
  auto semantics = InnerTiledSemanticsAttr::get(context, /*distributed=*/true,
                                                /*opaque=*/true);

  int64_t numInputs = getInputs().size();

  // Get type bindings: type<0>=acc, type<1>=lhs, type<2>=rhs, ...
  SmallVector<SmallVector<Type>> typeBindings = getTemplateTypes();
  if (typeBindings.size() < 1 + static_cast<size_t>(numInputs)) {
    return emitOpError("insufficient type bindings: expected at least ")
           << (1 + numInputs) << ", got " << typeBindings.size();
  }

  // Get static bounds.
  SmallVector<int64_t> staticBounds;
  for (Value bound : getBounds()) {
    auto constOp = bound.getDefiningOp<arith::ConstantIndexOp>();
    if (!constOp) {
      return emitOpError("dynamic bounds not supported");
    }
    staticBounds.push_back(constOp.value());
  }

  // Build iterator dim -> distribution index mapping.
  ArrayRef<int64_t> outerDimDist = getOuterDimDistribution();
  SmallVector<int64_t> iterDimToDistIdx(iteratorTypes.size(), -1);
  int64_t distIdx = 0;
  for (unsigned i = 0; i < iteratorTypes.size(); ++i) {
    if (iteratorTypes[i] == utils::IteratorType::parallel) {
      iterDimToDistIdx[i] = distIdx++;
    }
  }

  // Process each unimplemented block. The blocksToPopulate array only contains
  // blocks that have template.unimplemented terminators (concrete blocks like
  // copy blocks 1-2 are excluded by ConcretizeTemplateCalls). So the enumerate
  // indices are sequential (0,1,2,...) for the unimplemented blocks only:
  //   idx 0 = template block 0: Init accumulators
  //   idx 1 = template block 3: Read LHS from shared memory
  //   idx 2 = template block 4: Read RHS from shared memory
  //   idx 3 = template block 5: Compute MMA
  //   idx 4 = template block 6: Write results
  for (auto [blockIdx, destBlock] : llvm::enumerate(blocksToPopulate)) {
    if (destBlock->empty()) {
      continue;
    }

    Operation *terminator = destBlock->getTerminator();
    auto unimplOp = dyn_cast<Template::UnimplementedOp>(terminator);
    if (!unimplOp) {
      continue;
    }

    builder.setInsertionPoint(terminator);

    switch (blockIdx) {
    case 0: {
      // Init accumulators from destination sref.
      // Args: (sg_id, lane_id, dest: sref). Returns: type<0>.
      if (destBlock->getNumArguments() < 3) {
        return emitOpError("init-acc block: expected at least 3 arguments");
      }
      Value sgId = destBlock->getArgument(0);
      Value laneId = destBlock->getArgument(1);
      Value destArg = destBlock->getArgument(2);

      RankedTensorType accDistType =
          cast<RankedTensorType>(typeBindings[0][0]);
      AffineMap accMap = indexingMaps.back();

      Value result = readDistributedFromSref(
          builder, loc, kind, /*operandIdx=*/numInputs, accMap, destArg, sgId,
          laneId, accDistType, /*extraLeadingOffsets=*/{}, iteratorTypes,
          outerDimDist, iterDimToDistIdx);

      Template::ReturnOp::create(builder, loc, result);
      break;
    }
    case 1: {
      // Read LHS from shared memory.
      // Args: (buf_idx, sg_id, lane_id, lhs_alloc: sref). Returns: type<1>.
      if (destBlock->getNumArguments() < 4) {
        return emitOpError("read-LHS block: expected at least 4 arguments");
      }
      Value bufIdx = destBlock->getArgument(0);
      Value sgId = destBlock->getArgument(1);
      Value laneId = destBlock->getArgument(2);
      Value allocArg = destBlock->getArgument(3);

      RankedTensorType lhsDistType =
          cast<RankedTensorType>(typeBindings[1][0]);
      AffineMap lhsMap = indexingMaps[0];

      Value result = readDistributedFromSref(
          builder, loc, kind, /*operandIdx=*/0, lhsMap, allocArg, sgId, laneId,
          lhsDistType, /*extraLeadingOffsets=*/{OpFoldResult(bufIdx)},
          iteratorTypes, outerDimDist, iterDimToDistIdx);

      Template::ReturnOp::create(builder, loc, result);
      break;
    }
    case 2: {
      // Read RHS from shared memory.
      // Args: (buf_idx, sg_id, lane_id, rhs_alloc: sref). Returns: type<2>.
      if (destBlock->getNumArguments() < 4) {
        return emitOpError("read-RHS block: expected at least 4 arguments");
      }
      Value bufIdx = destBlock->getArgument(0);
      Value sgId = destBlock->getArgument(1);
      Value laneId = destBlock->getArgument(2);
      Value allocArg = destBlock->getArgument(3);

      RankedTensorType rhsDistType =
          cast<RankedTensorType>(typeBindings[2][0]);
      AffineMap rhsMap = indexingMaps[1];

      Value result = readDistributedFromSref(
          builder, loc, kind, /*operandIdx=*/1, rhsMap, allocArg, sgId, laneId,
          rhsDistType, /*extraLeadingOffsets=*/{OpFoldResult(bufIdx)},
          iteratorTypes, outerDimDist, iterDimToDistIdx);

      Template::ReturnOp::create(builder, loc, result);
      break;
    }
    case 3: {
      // Compute MMA with outer tiling.
      // Args: (acc: type<0>, lhs: type<1>, rhs: type<2>). Returns: type<0>.
      if (destBlock->getNumArguments() < 3) {
        return emitOpError("compute block: expected at least 3 arguments");
      }
      Value accArg = destBlock->getArgument(0);
      SmallVector<Value> inputArgs;
      for (int64_t i = 0; i < numInputs; ++i) {
        inputArgs.push_back(destBlock->getArgument(1 + i));
      }

      // Use the original indexing maps and iterator types for outer tiling.
      // The inner_tiled op describes the outer tile iteration pattern.
      auto innerTiledOp = Codegen::InnerTiledOp::create(
          builder, loc, inputArgs, {accArg}, indexingMaps, iteratorTypes, kind,
          semantics);

      Template::ReturnOp::create(builder, loc, innerTiledOp.getResults());
      break;
    }
    case 4: {
      // Write results to destination sref.
      // Args: (sg_id, lane_id, result: type<0>, dest: sref). Returns: void.
      if (destBlock->getNumArguments() < 4) {
        return emitOpError("write-results block: expected at least 4 arguments");
      }
      Value sgId = destBlock->getArgument(0);
      Value laneId = destBlock->getArgument(1);
      Value resultArg = destBlock->getArgument(2);
      Value destArg = destBlock->getArgument(3);

      RankedTensorType accDistType =
          cast<RankedTensorType>(typeBindings[0][0]);
      AffineMap accMap = indexingMaps.back();

      writeDistributedToSref(builder, loc, kind,
                             /*operandIdx=*/numInputs, accMap, resultArg,
                             destArg, sgId, laneId, accDistType, iteratorTypes,
                             outerDimDist, iterDimToDistIdx);

      Template::ReturnOp::create(builder, loc, ValueRange{});
      break;
    }
    default: {
      // Fallback for unexpected blocks.
      SmallVector<Value> returnValues;
      for (Type resultType : unimplOp.getResultTypes()) {
        if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
          Value emptyTensor = tensor::EmptyOp::create(
              builder, loc, tensorType.getShape(), tensorType.getElementType());
          returnValues.push_back(emptyTensor);
        } else if (isa<IndexType>(resultType)) {
          Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
          returnValues.push_back(zero);
        } else {
          return emitOpError("unsupported return type in block ")
                 << blockIdx << ": " << resultType;
        }
      }
      Template::ReturnOp::create(builder, loc, returnValues);
      break;
    }
    }

    terminator->erase();
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::GPU
