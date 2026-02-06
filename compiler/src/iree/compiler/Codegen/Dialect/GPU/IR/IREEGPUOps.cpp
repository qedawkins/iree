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
  // The template.func signature is (%inits: type<0>, %k: index) -> type<0>.
  // We return: outputs (dest operands) + reduction dimension bounds.
  SmallVector<Value> operands;

  // First: output operands (correspond to type<0> = dest/inits).
  for (Value output : getOutputs()) {
    operands.push_back(output);
  }

  // Second: reduction dimension bounds (correspond to %k: index).
  // Extract bounds for reduction dimensions from iterator_types.
  ArrayAttr iteratorTypeAttrs = getIteratorTypes();
  ValueRange bounds = getBounds();
  for (auto [bound, iterTypeAttr] : llvm::zip(bounds, iteratorTypeAttrs)) {
    auto iterType = cast<linalg::IteratorTypeAttr>(iterTypeAttr).getValue();
    if (iterType == utils::IteratorType::reduction) {
      operands.push_back(bound);
    }
  }

  return operands;
}

SmallVector<SmallVector<Type>> ProcessInnerTileOp::getTemplateTypes() {
  // Per design doc section 2.2, we provide 3 + numInputs type bindings:
  //   type<0>: Result types (same as output tensor types)
  //   type<1>: pcf.sref with same shape/element as results (for destinations)
  //   type<2>: Per-thread accumulator tensors (distributed inner tile shape)
  //   type<3+i>: pcf.sref for input i's shared memory
  SmallVector<SmallVector<Type>> bindings;
  MLIRContext *context = getContext();

  Codegen::InnerTileDescAttrInterface kind = getKind();

  // Get undistributed and distributed tile types.
  SmallVector<VectorType> undistributedTypes;
  kind.getUndistributedTileTypes(undistributedTypes);
  SmallVector<VectorType> distributedTypes;
  kind.getDistributedTileTypes(distributedTypes);

  // Create scope attributes for shared memory allocations.
  // type<1> and type<3> are srefs at the subgroup scope level.
  auto subgroupScope =
      cast<PCF::ScopeAttrInterface>(SubgroupScopeAttr::get(context));

  // type<0>: Output/result tensor type (full tensor shape).
  if (!getOutputs().empty()) {
    bindings.push_back({getOutputs().front().getType()});
  } else {
    bindings.push_back({});
  }

  // type<1>: pcf.sref for destination (same shape/element as output).
  if (!getOutputs().empty()) {
    auto outputType = cast<RankedTensorType>(getOutputs().front().getType());
    auto srefType = PCF::ShapedRefType::get(
        context, outputType.getShape(), outputType.getElementType(),
        subgroupScope);
    bindings.push_back({srefType});
  } else {
    bindings.push_back({});
  }

  // type<2>: Per-thread accumulator tensor (distributed shape).
  // Shape = outer_dims + distributed_inner_dims
  // For MFMA_F32_16x16x16_F16 with tensor<4x4x16x16xf32>:
  //   - outer_dims = [4, 4]
  //   - undistributed inner = [16, 16]
  //   - distributed inner = [4, 1] (per-thread portion)
  //   - result type<2> = tensor<4x4x4x1xf32>
  if (!getOutputs().empty() && !undistributedTypes.empty() &&
      !distributedTypes.empty()) {
    auto outputType = cast<RankedTensorType>(getOutputs().front().getType());
    int64_t numInputs = getInputs().size();
    // Accumulator is after inputs in the operand list.
    int64_t accIndex = numInputs;

    if (accIndex < undistributedTypes.size() &&
        accIndex < distributedTypes.size()) {
      VectorType undistType = undistributedTypes[accIndex];
      VectorType distType = distributedTypes[accIndex];

      int64_t innerRank = undistType.getRank();
      int64_t totalRank = outputType.getRank();
      int64_t outerRank = totalRank - innerRank;

      // Build shape: outer dims from output + distributed inner dims.
      SmallVector<int64_t> shape;
      for (int64_t i = 0; i < outerRank; ++i) {
        shape.push_back(outputType.getDimSize(i));
      }
      for (int64_t dim : distType.getShape()) {
        shape.push_back(dim);
      }

      auto type2 = RankedTensorType::get(shape, outputType.getElementType());
      bindings.push_back({type2});
    } else {
      bindings.push_back({getOutputs().front().getType()});
    }
  } else {
    bindings.push_back({});
  }

  // type<3+i>: pcf.sref for input i's shared memory (one binding per input).
  for (Value input : getInputs()) {
    auto inputType = cast<RankedTensorType>(input.getType());
    auto srefType = PCF::ShapedRefType::get(
        context, inputType.getShape(), inputType.getElementType(),
        subgroupScope);
    bindings.push_back({srefType});
  }

  return bindings;
}

/// Helper to compute offsets/sizes/strides for an operand using the kind
/// interface, given the lane_id for distribution.
static LogicalResult computeDistributedSliceParams(
    OpBuilder &builder, Location loc, Codegen::InnerTileDescAttrInterface kind,
    uint32_t operandIndex, Value operand, Value laneId,
    SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) {
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);

  // Get tensor type for computing outer rank.
  auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorType) {
    return failure();
  }

  // Get undistributed tile types to determine inner rank.
  SmallVector<VectorType> undistributedTypes;
  kind.getUndistributedTileTypes(undistributedTypes);
  if (operandIndex >= undistributedTypes.size()) {
    return failure();
  }
  int64_t innerRank = undistributedTypes[operandIndex].getRank();
  int64_t totalRank = tensorType.getRank();
  int64_t outerRank = totalRank - innerRank;

  // Initialize outer dimensions: offset=0, size=dim, stride=1.
  for (int64_t i = 0; i < outerRank; ++i) {
    offsets.push_back(zero);
    sizes.push_back(tensor::getMixedSize(builder, loc, operand, i));
    strides.push_back(one);
  }

  // Compute permutation (identity permutation for inner dims).
  SmallVector<int64_t> permutation =
      llvm::to_vector(llvm::seq<int64_t>(0, innerRank));

  // Use the kind interface to populate inner offsets/sizes/strides based on
  // lane_id distribution.
  if (failed(kind.populateOperandOffsetsSizesStrides(
          builder, loc, operandIndex, laneId, permutation, offsets, sizes,
          strides))) {
    return failure();
  }

  return success();
}

LogicalResult ProcessInnerTileOp::inlineImplementationBlocks(
    OpBuilder &builder, ArrayRef<Block *> blocksToPopulate) {
  // Per design doc section 2.3, ProcessInnerTileOp generates IR for 5 blocks:
  //   Block 0: Allocate shared memory - () -> type<3>
  //   Block 1: Initialize accumulators - (sg_id, lane_id, dest) -> type<2>
  //   Block 2: Copy inputs to shared memory - (sg_id, lane_id, k_idx, allocs)
  //   Block 3: Perform inner_tiled - (sg_id, lane_id, acc, allocs) -> type<2>
  //   Block 4: Write results - (sg_id, lane_id, result, dest) -> ()
  //
  // Block argument layout (for blocks 1-4):
  //   arg 0: subgroup_id (index) - for distributing outer dims
  //   arg 1: lane_id (index) - for distributing inner tile across lanes
  //   arg 2+: block-specific data arguments
  //
  // Inputs (lhs, rhs) are captured from the enclosing scope. When the
  // template.instance is created inside a func.func, the func arguments
  // are accessible from within the implementation blocks.

  Location loc = getLoc();
  MLIRContext *context = getContext();

  // Get indexing maps and iterator types from this op.
  SmallVector<AffineMap> indexingMaps;
  for (Attribute attr : getIndexingMaps()) {
    indexingMaps.push_back(cast<AffineMapAttr>(attr).getValue());
  }

  SmallVector<utils::IteratorType> iteratorTypes;
  for (Attribute attr : getIteratorTypes()) {
    iteratorTypes.push_back(cast<linalg::IteratorTypeAttr>(attr).getValue());
  }

  // Get the kind (MMA layout descriptor) and create DISTRIBUTED semantics.
  // We slice operands using lane_id and create a distributed inner_tiled op.
  Codegen::InnerTileDescAttrInterface kind = getKind();
  auto semantics = InnerTiledSemanticsAttr::get(context, /*distributed=*/true,
                                                /*opaque=*/true);

  // Get the inputs from this op - these are captured values.
  SmallVector<Value> capturedInputs(getInputs().begin(), getInputs().end());
  int64_t numInputs = capturedInputs.size();

  // Get concrete type bindings. type<3+i> contains the sref for input i.
  SmallVector<SmallVector<Type>> typeBindings = getTemplateTypes();
  // Collect alloc types from type<3> through type<3+numInputs-1>.
  SmallVector<Type> allocTypes;
  for (int64_t i = 0; i < numInputs; ++i) {
    allocTypes.push_back(typeBindings[3 + i][0]);
  }

  // Process each block by index.
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
      // Block 0: Allocate shared memory.
      // Arguments: () - no arguments.
      // Returns one pcf.sref per input operand (type<3+i> bindings).
      SmallVector<Value> returnValues;
      for (auto [i, allocType] : llvm::enumerate(allocTypes)) {
        auto srefType = dyn_cast<PCF::ShapedRefType>(allocType);
        if (!srefType) {
          return emitOpError("block 0: expected pcf.sref for type<")
                 << (3 + i) << ">, got " << allocType;
        }
        Value allocResult = PCF::AllocOp::create(builder, loc, srefType);
        returnValues.push_back(allocResult);
      }
      Template::ReturnOp::create(builder, loc, returnValues);
      break;
    }
    case 1: {
      // Block 1: Initialize accumulators from destination.
      // Arguments: (subgroup_id, lane_id, dest).
      // dest is pcf.sref (type<1>). Uses lane_id to read this thread's portion
      // of the destination via pcf.read_slice.
      if (destBlock->getNumArguments() < 3) {
        return emitOpError("block 1: expected at least 3 arguments");
      }
      Value laneId = destBlock->getArgument(1);
      Value destArg = destBlock->getArgument(2);

      // Verify dest is pcf.sref.
      auto srefType = dyn_cast<PCF::ShapedRefType>(destArg.getType());
      if (!srefType) {
        return emitOpError("block 1: expected pcf.sref for dest argument, got ")
               << destArg.getType();
      }

      SmallVector<Value> returnValues;
      for (Type resultType : unimplOp.getResultTypes()) {
        auto tensorType = dyn_cast<RankedTensorType>(resultType);
        if (!tensorType) {
          return emitOpError("block 1: unsupported return type ") << resultType;
        }

        // Create a placeholder tensor to compute slice params (we need a
        // RankedTensorType for the helper function).
        auto placeholderType = RankedTensorType::get(srefType.getShape(),
                                                     srefType.getElementType());
        Value placeholder =
            tensor::EmptyOp::create(builder, loc, placeholderType.getShape(),
                                    placeholderType.getElementType());

        // Compute this thread's slice of the destination using lane_id.
        // The accumulator operand index is after all inputs.
        SmallVector<OpFoldResult> offsets, sizes, strides;
        if (failed(computeDistributedSliceParams(builder, loc, kind, numInputs,
                                                 placeholder, laneId, offsets,
                                                 sizes, strides))) {
          return emitOpError("block 1: failed to compute slice params");
        }

        // Build the full-rank slice shape from sizes.
        // pcf.read_slice requires source and result to have the same rank.
        SmallVector<int64_t> sliceShape;
        for (OpFoldResult size : sizes) {
          if (auto attr = dyn_cast<Attribute>(size)) {
            sliceShape.push_back(cast<IntegerAttr>(attr).getInt());
          } else {
            sliceShape.push_back(ShapedType::kDynamic);
          }
        }
        auto fullRankType =
            RankedTensorType::get(sliceShape, srefType.getElementType());

        // Use pcf.read_slice to read the full-rank slice from dest (pcf.sref).
        Value fullSlice = PCF::ReadSliceOp::create(
            builder, loc, fullRankType, destArg, offsets, sizes, strides);

        // If the result type has fewer dimensions (e.g., trailing 1s dropped),
        // use rank-reducing extract_slice instead of collapse_shape. This
        // bufferizes to memref.subview which correctly preserves address spaces.
        Value resultSlice = fullSlice;
        if (fullRankType.getRank() != tensorType.getRank()) {
          SmallVector<OpFoldResult> extractOffsets(fullRankType.getRank(),
                                                   builder.getIndexAttr(0));
          SmallVector<OpFoldResult> extractSizes;
          for (int64_t dim : sliceShape) {
            extractSizes.push_back(builder.getIndexAttr(dim));
          }
          SmallVector<OpFoldResult> extractStrides(fullRankType.getRank(),
                                                    builder.getIndexAttr(1));
          resultSlice = tensor::ExtractSliceOp::create(
              builder, loc, tensorType, fullSlice, extractOffsets, extractSizes,
              extractStrides);
        }

        returnValues.push_back(resultSlice);
      }
      Template::ReturnOp::create(builder, loc, returnValues);
      break;
    }
    case 2: {
      // Block 2: Copy inputs to shared memory.
      // Arguments: (subgroup_id, lane_id, k_idx, alloc0, alloc1, ...).
      // Each alloc_i is a pcf.sref (type<3+i>) for one input operand.
      // Uses lane_id to compute which portion of the input each lane copies.
      if (destBlock->getNumArguments() < 3 + numInputs) {
        return emitOpError("block 2: expected at least ")
               << (3 + numInputs) << " arguments, got "
               << destBlock->getNumArguments();
      }
      Value laneId = destBlock->getArgument(1);
      // Value kIdx = destBlock->getArgument(2);

      // Extract per-lane slice from each captured input using lane_id
      // and write to the corresponding shared memory alloc via pcf.write_slice.
      for (auto [opIndex, input] : llvm::enumerate(capturedInputs)) {
        Value allocArg = destBlock->getArgument(3 + opIndex);

        // Verify alloc is pcf.sref.
        if (!isa<PCF::ShapedRefType>(allocArg.getType())) {
          return emitOpError(
                     "block 2: expected pcf.sref for allocs argument ")
                 << opIndex << ", got " << allocArg.getType();
        }

        SmallVector<OpFoldResult> offsets, sizes, strides;
        if (failed(computeDistributedSliceParams(builder, loc, kind, opIndex,
                                                 input, laneId, offsets, sizes,
                                                 strides))) {
          return emitOpError("block 2: failed to compute slice for input ")
                 << opIndex;
        }

        // Extract per-lane slice from input (tensor).
        Value inputSlice = tensor::ExtractSliceOp::create(
            builder, loc, input, offsets, sizes, strides);

        // Write to shared memory (pcf.sref) using pcf.write_slice.
        PCF::WriteSliceOp::create(builder, loc, inputSlice, allocArg, offsets,
                                  sizes, strides);
      }

      // Block 2 returns void.
      Template::ReturnOp::create(builder, loc, ValueRange{});
      break;
    }
    case 3: {
      // Block 3: Perform inner_tiled computation.
      // Arguments: (subgroup_id, lane_id, acc, alloc0, alloc1, ...).
      // The acc argument is type<2> (per-thread distributed accumulator).
      // Uses lane_id to slice captured inputs for per-thread computation.
      if (destBlock->getNumArguments() < 3 + numInputs) {
        return emitOpError("block 3: expected at least ")
               << (3 + numInputs) << " arguments, got "
               << destBlock->getNumArguments();
      }
      Value laneId = destBlock->getArgument(1);
      Value accArg = destBlock->getArgument(2);

      // Slice each captured input using lane_id distribution.
      SmallVector<Value> inputSlices;
      for (auto [opIndex, input] : llvm::enumerate(capturedInputs)) {
        SmallVector<OpFoldResult> offsets, sizes, strides;
        if (failed(computeDistributedSliceParams(builder, loc, kind, opIndex,
                                                 input, laneId, offsets, sizes,
                                                 strides))) {
          return emitOpError("block 3: failed to compute slice for input ")
                 << opIndex;
        }
        Value slice = tensor::ExtractSliceOp::create(builder, loc, input,
                                                     offsets, sizes, strides);
        inputSlices.push_back(slice);
      }

      // The acc argument is already the per-thread distributed shape (type<2>).
      // Create inner_tiled op with distributed semantics using sliced inputs.
      //
      // At the per-lane level, we're performing a single MMA with no outer
      // tiling dimensions. The indexing maps must describe only the outer
      // dimensions (of which there are none), so we use empty maps () -> ()
      // with empty iterator types. This matches the single_multi_mma pattern
      // in iree_gpu_inner_tiled_ops.mlir.
      int64_t numOperands = inputSlices.size() + 1; // inputs + acc
      SmallVector<AffineMap> singleMmaMaps(
          numOperands, AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                                      /*results=*/{}, context));
      SmallVector<utils::IteratorType> singleMmaIteratorTypes;
      auto innerTiledOp = Codegen::InnerTiledOp::create(
          builder, loc, inputSlices, {accArg}, singleMmaMaps,
          singleMmaIteratorTypes, kind, semantics);

      Template::ReturnOp::create(builder, loc, innerTiledOp.getResults());
      break;
    }
    case 4: {
      // Block 4: Write results back to destinations.
      // Arguments: (subgroup_id, lane_id, result, dest).
      // dest is pcf.sref (type<1>). Uses lane_id to compute where to write
      // this thread's result via pcf.write_slice.
      if (destBlock->getNumArguments() < 4) {
        return emitOpError("block 4: expected at least 4 arguments");
      }
      Value laneId = destBlock->getArgument(1);
      Value resultArg = destBlock->getArgument(2);
      Value destArg = destBlock->getArgument(3);

      // Verify dest is pcf.sref.
      auto srefType = dyn_cast<PCF::ShapedRefType>(destArg.getType());
      if (!srefType) {
        return emitOpError("block 4: expected pcf.sref for dest argument, got ")
               << destArg.getType();
      }

      // Create a placeholder tensor to compute slice params (we need a
      // RankedTensorType for the helper function).
      auto placeholderType =
          RankedTensorType::get(srefType.getShape(), srefType.getElementType());
      Value placeholder =
          tensor::EmptyOp::create(builder, loc, placeholderType.getShape(),
                                  placeholderType.getElementType());

      // Compute offsets for writing this thread's result into destination.
      SmallVector<OpFoldResult> offsets, sizes, strides;
      if (failed(computeDistributedSliceParams(builder, loc, kind, numInputs,
                                               placeholder, laneId, offsets,
                                               sizes, strides))) {
        return emitOpError("block 4: failed to compute slice params");
      }

      // pcf.write_slice requires source and dest to have the same rank.
      // If result has fewer dimensions (collapsed from full rank), expand it.
      auto resultType = cast<RankedTensorType>(resultArg.getType());
      int64_t destRank = srefType.getRank();
      Value sourceToWrite = resultArg;

      if (resultType.getRank() != destRank) {
        // Build the full-rank slice shape from sizes.
        SmallVector<int64_t> sliceShape;
        for (OpFoldResult size : sizes) {
          if (auto attr = dyn_cast<Attribute>(size)) {
            sliceShape.push_back(cast<IntegerAttr>(attr).getInt());
          } else {
            sliceShape.push_back(ShapedType::kDynamic);
          }
        }
        auto fullRankType =
            RankedTensorType::get(sliceShape, resultType.getElementType());

        // Use rank-expanding insert_slice instead of expand_shape. This
        // bufferizes to memref.subview which correctly preserves address spaces.
        Value empty = tensor::EmptyOp::create(
            builder, loc, fullRankType.getShape(),
            fullRankType.getElementType());
        SmallVector<OpFoldResult> insertOffsets(destRank,
                                                builder.getIndexAttr(0));
        SmallVector<OpFoldResult> insertSizes;
        for (int64_t dim : sliceShape) {
          insertSizes.push_back(builder.getIndexAttr(dim));
        }
        SmallVector<OpFoldResult> insertStrides(destRank,
                                                 builder.getIndexAttr(1));
        sourceToWrite = tensor::InsertSliceOp::create(
            builder, loc, resultArg, empty, insertOffsets, insertSizes,
            insertStrides);
      }

      // Write result to dest (pcf.sref) using pcf.write_slice.
      PCF::WriteSliceOp::create(builder, loc, sourceToWrite, destArg, offsets,
                                sizes, strides);

      // Block 4 returns void - result flows through the main region's return.
      Template::ReturnOp::create(builder, loc, ValueRange{});
      break;
    }
    default: {
      // Handle additional blocks with generic fallback.
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
