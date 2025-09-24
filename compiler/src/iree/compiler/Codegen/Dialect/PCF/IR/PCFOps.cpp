// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include <numeric>
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// StructuralOps
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

static ParseResult parseParallelBody(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inits,
    SmallVectorImpl<Type> &initTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicSizes,
    SmallVectorImpl<Type> &resultTypes, SmallVectorImpl<bool> &isTied,
    SmallVectorImpl<bool> &hasToken, Region &body) {
  if (failed(parser.parseKeyword("initialize")))
    return failure();
  SmallVector<OpAsmParser::Argument> regionRefArgs;
  SmallVector<OpAsmParser::Argument> regionTokenArgs;
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      // Reserve entries in the lists.
      regionRefArgs.emplace_back();
      if (failed(parser.parseArgument(regionRefArgs.back(),
                                      /*allowType=*/false,
                                      /*allowAttrs=*/true))) {
        return failure();
      }

      if (succeeded(parser.parseOptionalLSquare())) {
        regionTokenArgs.emplace_back();
        if (failed(parser.parseArgument(regionTokenArgs.back(),
                                        /*allowType=*/true,
                                        /*allowAttrs=*/true))) {
          return failure();
        }
        if (failed(parser.parseOptionalRSquare())) {
          return failure();
        }
        hasToken.push_back(true);
      } else {
        hasToken.push_back(false);
      }

      // Parse the tied init if present.
      if (succeeded(parser.parseOptionalEqual())) {
        inits.emplace_back();
        if (failed(parser.parseOperand(inits.back()))) {
          return failure();
        }
        isTied.push_back(true);
      } else {
        isTied.push_back(false);
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }

  OpAsmParser::Argument numThreadsArg;
  if (failed(parser.parseLSquare())) {
    return failure();
  }

  if (failed(parser.parseArgument(numThreadsArg,
                                  /*allowType=*/true, /*allowAttrs=*/true))) {
    return failure();
  }

  if (failed(parser.parseRSquare())) {
    return failure();
  }

  // If there is at least one region arg the arg types and op result types need
  // to be parsed.
  if (!regionRefArgs.empty()) {
    if (failed(parser.parseColon()) || failed(parser.parseLParen())) {
      return failure();
    }

    // Parse all types except the last followed by commas.
    for (OpAsmParser::Argument &arg :
         MutableArrayRef<OpAsmParser::Argument>(regionRefArgs.begin(),
                                                regionRefArgs.end())
             .drop_back()) {
      if (failed(parser.parseType(arg.type)) || failed(parser.parseComma())) {
        return failure();
      }
    }

    // Parse the last type.
    if (failed(parser.parseType(regionRefArgs.back().type))) {
      return failure();
    }

    if (failed(parser.parseRParen()) || failed(parser.parseArrow()) ||
        failed(parser.parseLParen())) {
      return failure();
    }

    int64_t numResults = isTied.size();
    resultTypes.resize(numResults);
    for (auto [i, isTied] : llvm::enumerate(isTied)) {
      if (failed(parser.parseType(resultTypes[i]))) {
        return failure();
      }

      auto shapedType = dyn_cast<ShapedType>(resultTypes[i]);
      if (!shapedType) {
        return failure();
      }

      if (isTied) {
        initTypes.push_back(resultTypes[i]);
      } else if (succeeded(parser.parseOptionalLBrace())) {
        // Only parse dynamic dims for non-tied operands.
        SmallVector<OpAsmParser::UnresolvedOperand> dims;
        if (failed(parser.parseOperandList(dims))) {
          return failure();
        }
        size_t numDynamicDims = shapedType.getNumDynamicDims();
        if (dims.size() != numDynamicDims) {
          return failure();
        }
        if (failed(parser.parseRBrace())) {
          return failure();
        }
        dynamicSizes.append(dims);
      }

      if (i < numResults - 1 && failed(parser.parseComma())) {
        return failure();
      }
    }

    if (failed(parser.parseRParen())) {
      return failure();
    }
  }

  // The printed argument order is for readability. The stored argument order
  // (num threads) (result tied refs) (tokens) to optimize for the common
  // pattern of requesting the ref tied to a result.
  SmallVector<OpAsmParser::Argument> args = {numThreadsArg};
  args.append(regionRefArgs);
  args.append(regionTokenArgs);
  return parser.parseRegion(body, args, /*enableNameShadowing=*/false);
}

static void printParallelBody(OpAsmPrinter &p, Operation *op,
                              OperandRange inits, TypeRange initTypes,
                              OperandRange dynamicSizes, TypeRange resultTypes,
                              ArrayRef<bool> isTied, ArrayRef<bool> hasToken,
                              Region &body) {
  p.printNewline();
  p << "  initialize";

  int64_t numResults = resultTypes.size();

  BlockArgument numThreadsArg = body.getArguments().front();
  MutableArrayRef<BlockArgument> refArgRange =
      body.getArguments().drop_front().take_front(numResults);
  // 1 for the thread count. The rest is split between shaped refs and tokens
  // where each result has a single associated ref.
  MutableArrayRef<BlockArgument> tokenArgRange =
      body.getArguments().drop_front(1 + numResults);

  if (numResults != 0) {
    p << "(";
    int64_t currInitIndex = 0;
    int64_t currTokenIndex = 0;
    for (int64_t i = 0, e = numResults; i < e; ++i) {
      p << refArgRange[i];
      if (hasToken[i]) {
        p << "[";
        p << tokenArgRange[currTokenIndex];
        p << ": ";
        p << tokenArgRange[currTokenIndex].getType();
        p << "]";
        ++currTokenIndex;
      }
      if (isTied[i]) {
        p << " = ";
        p << inits[currInitIndex];
        ++currInitIndex;
      }
      if (i < numResults - 1) {
        p << ", ";
      }
    }
    p << ")";
  }
  p << "[" << numThreadsArg << ": " << numThreadsArg.getType() << "]";

  // Now print the function type.
  if (numResults != 0) {
    p.printNewline();
    // Whitespace to line up parentheses.
    //   |--initialize(
    //   |--________: (
    p << "          : (";
    llvm::interleaveComma(refArgRange, p,
                          [&](BlockArgument arg) { p << arg.getType(); });
    p << ")";
    p.printNewline();
    //   |--initialize(
    //   |--_______-> (
    p << "         -> (";
    OperandRange currSizes = dynamicSizes;
    for (int64_t i = 0, e = numResults; i < e; ++i) {
      ShapedType resultType = cast<ShapedType>(resultTypes[i]);
      bool isResultTied = isTied[i];
      p << resultType;
      if (!isResultTied && !resultType.hasStaticShape()) {
        int64_t numDynamicDims = resultType.getNumDynamicDims();
        p << "{";
        llvm::interleaveComma(currSizes.take_front(numDynamicDims), p,
                              [&](Value dim) { p << dim; });
        currSizes = currSizes.drop_front(numDynamicDims);
        p << "}";
      }
      if (i < numResults - 1) {
        p << ", ";
      }
    }
    p << ") ";
  } else {
    // Print a space before the region brace if there are no loop results.
    p << " ";
  }
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

void GenericOp::getAsmBlockArgumentNames(Region &region,
                                         OpAsmSetValueNameFn setNameFn) {
  setNameFn(getNumThreadsArg(), "count");
  for (Value v : getRegionRefArgs()) {
    setNameFn(v, "ref");
  }
  for (Value v : getRegionTokenArgs()) {
    setNameFn(v, "token");
  }
}

LogicalResult verifyParallelBodyArgs() { return success(); }

LogicalResult GenericOp::verify() {
  // Verify tied/token array lengths.
  ArrayRef<bool> isTied = getIsTied();
  ArrayRef<bool> hasToken = getHasToken();
  int64_t numResults = getNumResults();
  if (isTied.size() != numResults) {
    return emitOpError(
               "`is_tied` mask length expected to match number of results ")
           << numResults;
  }

  int64_t numInits =
      std::accumulate(getIsTied().begin(), getIsTied().end(), (int64_t)(0));
  if (getInits().size() != numInits) {
    return emitOpError("number of inits ")
           << getInits().size()
           << " does not match the number of results marked as tied "
           << numInits;
  }

  if (hasToken.size() != numResults) {
    return emitOpError(
               "`has_token` mask length expected to match number of results ")
           << numResults;
  }

  int64_t numTokens =
      std::accumulate(getHasToken().begin(), getHasToken().end(), (int64_t)(0));

  if (getRegion().getArguments().size() != numResults + 1 + numTokens) {
    return emitOpError("expected region to have |numResults| + 1 + |numTokens| "
                       "total arguments");
  }

  if (!getNumThreadsArg().getType().isIndex()) {
    return emitOpError(
        "expected index type for first (thread count) region argument");
  }

  PCF::ScopeAttr scope = getScope();
  int64_t currIsTiedIndex = 0;
  int64_t currResultIndex = 0;
  for (auto [resultType, refArg, isTied] :
       llvm::zip_equal(getResultTypes(), getRegionRefArgs(), getIsTied())) {
    auto srefType = dyn_cast<PCF::ShapedRefType>(refArg.getType());
    if (!srefType || srefType.getScope() != scope) {
      return emitOpError("expected region ref argument to be of type !pcf.sref "
                         "with scope ")
             << scope;
    }

    // Traits guarantee this cast to be valid.
    auto shapedResultType = cast<ShapedType>(resultType);
    if (shapedResultType.getShape() != srefType.getShape()) {
      return emitOpError("region arg at index ")
             << currResultIndex << " with type " << srefType
             << " shape mismatch with tied result of type " << resultType;
    }

    if (shapedResultType.getElementType() != srefType.getElementType()) {
      return emitOpError("region arg at index ")
             << currResultIndex << " element type mismatch of "
             << srefType.getElementType() << " vs "
             << shapedResultType.getElementType();
    }

    if (isTied) {
      Value init = getInits()[currIsTiedIndex];
      if (init.getType() != resultType) {
        return emitOpError("tied init at index ")
               << currIsTiedIndex << " does not match the type " << resultType
               << " at result index " << currResultIndex;
      }
      ++currIsTiedIndex;
    }
    ++currResultIndex;
  }

  // Verify token types and scopes.
  for (auto tokenArg : getRegionTokenArgs()) {
    auto tokenType = dyn_cast<PCF::TokenType>(tokenArg.getType());
    if (!tokenType || tokenType.getScope() != scope) {
      return emitOpError("expected region token argument to be of type "
                         "!pcf.token with scope ")
             << scope;
    }
  }
  return success();
}

void GenericOp::getSuccessorRegions(RegionBranchPoint point,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the GenericOp, branch into the body.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getRegion()));
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.push_back(RegionSuccessor(getResults()));
}

//===----------------------------------------------------------------------===//
// WriteOps
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ParallelInsertSliceOp
//===----------------------------------------------------------------------===//

// Build a WriteSliceOp with mixed static and dynamic entries.
void WriteSliceOp::build(OpBuilder &b, OperationState &result, Value source,
                         Value dest, ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes,
                         ArrayRef<OpFoldResult> strides,
                         ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  result.addAttributes(attrs);
  build(b, result, {}, source, dest, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
}

/// Build an WriteSliceOp with mixed static and dynamic entries
/// packed into a Range vector.
void WriteSliceOp::build(OpBuilder &b, OperationState &result, Value source,
                         Value dest, ArrayRef<Range> ranges,
                         ArrayRef<NamedAttribute> attrs) {
  auto [offsets, sizes, strides] = getOffsetsSizesAndStrides(ranges);
  build(b, result, source, dest, offsets, sizes, strides, attrs);
}

// Build a WriteSliceOp with dynamic entries.
void WriteSliceOp::build(OpBuilder &b, OperationState &result, Value source,
                         Value dest, ValueRange offsets, ValueRange sizes,
                         ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, source, dest, offsetValues, sizeValues, strideValues);
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void PCFDialect::registerOperations() {
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::PCF

//===----------------------------------------------------------------------===//
// TableGen definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp.inc" // IWYU pragma: keep
