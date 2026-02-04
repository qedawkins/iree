// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateInterfaces.h"
#include "llvm/ADT/STLExtras.h"
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

SmallVector<SmallVector<Type>> ProcessInnerTileOp::getTemplateTypes() {
  MLIRContext *context = getContext();
  SmallVector<SmallVector<Type>> typeBindings;

  // type<0>: Result types (same as output tensor types)
  SmallVector<Type> resultTypes;
  for (Type t : getOutputs().getTypes()) {
    resultTypes.push_back(t);
  }
  typeBindings.push_back(std::move(resultTypes));

  // type<1>: pcf.sref with result shapes using return_only_sync_scope
  // This is for the output shared memory.
  SmallVector<Type> outputSrefTypes;
  for (Type t : getOutputs().getTypes()) {
    auto tensorType = cast<RankedTensorType>(t);
    auto syncAttr = PCF::SyncOnReturnAttr::get(context);
    auto scopeAttr = PCF::SequentialAttr::get(context);
    auto srefType = PCF::ShapedRefType::get(context, tensorType.getShape(),
                                            tensorType.getElementType(),
                                            scopeAttr, syncAttr);
    outputSrefTypes.push_back(srefType);
  }
  typeBindings.push_back(std::move(outputSrefTypes));

  // type<2>: Per-thread accumulator tensors (outer dims / distribution / inner)
  // For now, we compute this based on the inner tile description.
  SmallVector<Type> accumulatorTypes;
  SmallVector<VectorType> undistributedTileTypes;
  getKind().getUndistributedTileTypes(undistributedTileTypes);
  // The accumulator types are the output operand tile types.
  for (int64_t i = 0; i < getKind().getExpectedNumOutputs(); ++i) {
    int64_t operandIdx = getKind().getExpectedNumInputs() + i;
    VectorType tileType = undistributedTileTypes[operandIdx];
    // Create a tensor type from the vector type dimensions.
    auto tensorType =
        RankedTensorType::get(tileType.getShape(), tileType.getElementType());
    accumulatorTypes.push_back(tensorType);
  }
  typeBindings.push_back(std::move(accumulatorTypes));

  // type<3>: pcf.sref for input shared memory
  SmallVector<Type> inputSrefTypes;
  for (Type t : getInputs().getTypes()) {
    auto tensorType = cast<RankedTensorType>(t);
    auto syncAttr = PCF::SyncOnReturnAttr::get(context);
    auto scopeAttr = PCF::SequentialAttr::get(context);
    auto srefType = PCF::ShapedRefType::get(context, tensorType.getShape(),
                                            tensorType.getElementType(),
                                            scopeAttr, syncAttr);
    inputSrefTypes.push_back(srefType);
  }
  typeBindings.push_back(std::move(inputSrefTypes));

  return typeBindings;
}

LogicalResult ProcessInnerTileOp::inlineImplementationBlocks(
    OpBuilder &builder, ArrayRef<Block *> blocksToPopulate) {
  // For ProcessInnerTileOp, the blocks are:
  // 0. Allocate shared memory
  // 1. Initialize accumulators
  // 2. Copy inputs to shared memory
  // 3. Perform inner-tiled operation
  // 4. Write results to destinations
  //
  // The actual implementation is highly target-specific and will be
  // populated by the lowering passes. For now, we emit placeholder IR
  // that can be further lowered.
  //
  // This method is called by ConcretizeTemplateCallsPass to populate
  // the blocks with the implementation from this high-level op.

  // For the initial implementation, we return failure to indicate
  // that the template should be resolved by the template.func's own
  // implementation blocks rather than generating them here.
  //
  // In the future, this could generate the full implementation based
  // on the inner tile description and indexing maps.
  return failure();
}

} // namespace mlir::iree_compiler::IREE::GPU
