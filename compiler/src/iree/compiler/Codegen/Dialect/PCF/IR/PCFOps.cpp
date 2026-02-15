// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include <numeric>
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// AllocOps
//===----------------------------------------------------------------------===//

LogicalResult AllocOp::verify() {
  if (getDynamicSizes().size() != getResultType().getNumDynamicDims()) {
    return emitOpError(
        "dimension operand count does not equal sref dynamic dimension count");
  }

  return success();
}

SmallVector<OpFoldResult> AllocOp::getMixedSizes() {
  Builder b(getContext());
  return getMixedValues(getResultType().getShape(), getDynamicSizes(), b);
}

//===----------------------------------------------------------------------===//
// StructuralOps
//===----------------------------------------------------------------------===//

template <typename OpTy>
static LogicalResult verifyParallelBodyOp(OpTy op, int64_t numLeadingArgs,
                                          int64_t numIndexBodyArgs,
                                          ArrayRef<BlockArgument> indexArgs) {
  // Verify tied/token array lengths.
  ArrayRef<bool> isTied = op.getIsTied();
  int64_t numResults = op.getNumResults();
  if (isTied.size() != numResults) {
    return op.emitOpError(
               "`is_tied` mask length expected to match number of results ")
           << numResults;
  }

  int64_t numInits = llvm::sum_of(isTied, (int64_t)(0));
  if (op.getInits().size() != numInits) {
    return op.emitOpError("number of inits ")
           << op.getInits().size()
           << " does not match the number of results marked as tied "
           << numInits;
  }

  if (op.getRegion().getArguments().size() !=
      numLeadingArgs + numResults + numIndexBodyArgs) {
    return op.emitOpError("expected region to have |numLeadingArgs| + "
                          "|numIndexArgs| + |numResults| "
                          "total arguments");
  }

  for (BlockArgument countArg : indexArgs) {
    if (!countArg.getType().isIndex()) {
      return op.emitOpError(
          "expected index type for thread count/id region arguments");
    }
  }

  PCF::ScopeAttrInterface scope = op.getScope();
  int64_t currIsTiedIndex = 0;
  int64_t currResultIndex = 0;
  for (auto [resultType, refArg, isTied] : llvm::zip_equal(
           op.getResultTypes(), op.getRegionRefArgs(), op.getIsTied())) {
    auto srefType = dyn_cast<PCF::ShapedRefType>(refArg.getType());
    if (!srefType || srefType.getScope() != scope) {
      return op.emitOpError(
                 "expected region ref argument to be of type !pcf.sref "
                 "with scope ")
             << scope;
    }
    if (!srefType.isReturnOnlySync() && srefType.getSyncScope()) {
      return op.emitOpError(
          "expected region ref argument to sync on return or is unspecified");
    }

    // Traits guarantee this cast to be valid.
    auto shapedResultType = cast<ShapedType>(resultType);
    if (shapedResultType.getShape() != srefType.getShape()) {
      return op.emitOpError("region arg at index ")
             << currResultIndex << " with type " << srefType
             << " shape mismatch with tied result of type " << resultType;
    }

    if (shapedResultType.getElementType() != srefType.getElementType()) {
      return op.emitOpError("region arg at index ")
             << currResultIndex << " element type mismatch of "
             << srefType.getElementType() << " vs "
             << shapedResultType.getElementType();
    }

    if (isTied) {
      Value init = op.getInits()[currIsTiedIndex];
      if (init.getType() != resultType) {
        return op.emitOpError("tied init at index ")
               << currIsTiedIndex << " does not match the type " << resultType
               << " at result index " << currResultIndex;
      }
      ++currIsTiedIndex;
    }
    ++currResultIndex;
  }
  return success();
}

static ParseResult parseParallelExecutionBody(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inits,
    SmallVectorImpl<Type> &initTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicSizes,
    SmallVectorImpl<Type> &resultTypes, SmallVectorImpl<bool> &isTied,
    Region &body, int64_t &numLeadingArgs, bool parseOptionalLeadingArgs) {
  SmallVector<OpAsmParser::Argument> regionLeadingArgs;
  if (parseOptionalLeadingArgs) {
    if (succeeded(parser.parseOptionalArrow())) {
      SMLoc leadingArgsLoc = parser.getCurrentLocation();
      if (failed(parser.parseArgumentList(regionLeadingArgs,
                                          OpAsmParser::Delimiter::Paren,
                                          /*allowType=*/true))) {
        return parser.emitError(leadingArgsLoc,
                                "failed to parse leading arguments");
      }
    }
    numLeadingArgs = regionLeadingArgs.size();
  }

  if (failed(parser.parseKeyword("execute"))) {
    return failure();
  }

  SmallVector<OpAsmParser::Argument> regionRefArgs;
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      // Reserve entries in the lists.
      regionRefArgs.emplace_back();
      SMLoc argLoc = parser.getCurrentLocation();
      if (failed(parser.parseArgument(regionRefArgs.back(),
                                      /*allowType=*/false,
                                      /*allowAttrs=*/true))) {
        return parser.emitError(argLoc, "failed to parse region ref argument");
      }

      // Parse the tied init if present.
      if (succeeded(parser.parseOptionalEqual())) {
        inits.emplace_back();
        SMLoc initLoc = parser.getCurrentLocation();
        if (failed(parser.parseOperand(inits.back()))) {
          return parser.emitError(initLoc, "failed to parse tied init operand");
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

  SMLoc indexArgsLoc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::Argument> indexArgs;
  if (failed(parser.parseArgumentList(
          indexArgs, /*delimiter=*/OpAsmParser::Delimiter::Square,
          /*allowType=*/true, /*allowAttrs=*/true))) {
    return parser.emitError(indexArgsLoc,
                            "failed to parse index arguments list");
  }

  // If there is at least one region arg the arg types and op result types need
  // to be parsed.
  if (!regionRefArgs.empty()) {
    if (failed(parser.parseColon())) {
      return failure();
    }

    // Parse "(<type list>)" directly into the type fields of `regionRefArgs`.
    auto it = regionRefArgs.begin();
    SMLoc refTypesLoc = parser.getCurrentLocation();
    if (failed(parser.parseCommaSeparatedList(
            OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
              if (it == regionRefArgs.end()) {
                return failure();
              }
              ParseResult p = parser.parseType(it->type);
              ++it;
              return p;
            }))) {
      return parser.emitError(refTypesLoc,
                              "failed to parse region ref argument types");
    }

    if (failed(parser.parseArrow()) || failed(parser.parseLParen())) {
      return failure();
    }

    int64_t numResults = isTied.size();
    resultTypes.resize(numResults);
    for (auto [i, isTied] : llvm::enumerate(isTied)) {
      SMLoc resultTypeLoc = parser.getCurrentLocation();
      if (failed(parser.parseType(resultTypes[i]))) {
        return parser.emitError(resultTypeLoc, "failed to parse result type");
      }

      ShapedType shapedType = dyn_cast<ShapedType>(resultTypes[i]);
      if (!shapedType) {
        return parser.emitError(resultTypeLoc,
                                "result type must be a shaped type");
      }

      if (isTied) {
        initTypes.push_back(resultTypes[i]);
      } else if (!shapedType.hasStaticShape()) {
        if (failed(parser.parseLBrace())) {
          return failure();
        }
        // Only parse dynamic dims for non-tied operands.
        SmallVector<OpAsmParser::UnresolvedOperand> dims;
        if (failed(parser.parseOperandList(dims))) {
          return failure();
        }
        size_t numDynamicDims = shapedType.getNumDynamicDims();
        if (dims.size() != numDynamicDims) {
          return parser.emitError(resultTypeLoc, "expected ")
                 << numDynamicDims << " dynamic dimension operands for type "
                 << shapedType << ", but got " << dims.size();
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

  // The stored argument order is:
  // (initialized vals) (result tied refs) (num threads).
  SmallVector<OpAsmParser::Argument> args;
  args.append(regionLeadingArgs);
  args.append(regionRefArgs);
  args.append(indexArgs);
  return parser.parseRegion(body, args, /*enableNameShadowing=*/false);
}

static ParseResult parseParallelExecutionBody(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inits,
    SmallVectorImpl<Type> &initTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicSizes,
    SmallVectorImpl<Type> &resultTypes, SmallVectorImpl<bool> &isTied,
    Region &body) {
  int64_t numLeadingArgs = 0;
  return parseParallelExecutionBody(parser, inits, initTypes, dynamicSizes,
                                    resultTypes, isTied, body, numLeadingArgs,
                                    false);
}

static void printParallelExecutionBody(
    OpAsmPrinter &p, Operation *op, OperandRange inits, TypeRange initTypes,
    OperandRange dynamicSizes, TypeRange resultTypes, ArrayRef<bool> isTied,
    Region &body, int64_t numLeadingArgs, bool printOptionalLeadingArgs) {
  if (printOptionalLeadingArgs && numLeadingArgs > 0) {
    p << "-> (";
    MutableArrayRef<BlockArgument> leadingArgRange =
        body.getArguments().take_front(numLeadingArgs);
    llvm::interleaveComma(leadingArgRange, p, [&](BlockArgument arg) {
      p.printRegionArgument(arg);
    });
    p << ")";
  }

  p.printNewline();
  p << "  execute";

  int64_t numResults = resultTypes.size();
  int64_t numIndexArgs = body.getNumArguments() - numResults - numLeadingArgs;
  MutableArrayRef<BlockArgument> threadCountArgRange =
      body.getArguments().take_back(numIndexArgs);
  MutableArrayRef<BlockArgument> refArgRange =
      body.getArguments().drop_back(numIndexArgs).take_back(numResults);

  if (numResults != 0) {
    p << "(";
    int64_t currInitIndex = 0;
    for (int64_t i = 0, e = numResults; i < e; ++i) {
      p << refArgRange[i];
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
  p << "[";
  llvm::interleaveComma(threadCountArgRange, p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << "]";

  // Now print the function type.
  if (numResults != 0) {
    p.printNewline();
    // Whitespace to line up parentheses.
    //   |--execute(
    //   |--_____: (
    p << "       : (";
    llvm::interleaveComma(refArgRange, p,
                          [&](BlockArgument arg) { p << arg.getType(); });
    p << ")";
    p.printNewline();
    //   |--execute(
    //   |--____-> (
    p << "      -> (";
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

static void printParallelExecutionBody(OpAsmPrinter &p, Operation *op,
                                       OperandRange inits, TypeRange initTypes,
                                       OperandRange dynamicSizes,
                                       TypeRange resultTypes,
                                       ArrayRef<bool> isTied, Region &body) {
  return printParallelExecutionBody(p, op, inits, initTypes, dynamicSizes,
                                    resultTypes, isTied, body, 0, false);
}

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

static ParseResult parseInferNumIndexArgs(OpAsmParser &parser, Region &body,
                                          int64_t &numLeadingArgs,
                                          int64_t &numIndexArgs) {
  numIndexArgs = 0;
  for (BlockArgument bbArg :
       llvm::reverse(body.getArguments().drop_front(numLeadingArgs))) {
    if (!bbArg.getType().isIndex()) {
      return success();
    }
    ++numIndexArgs;
  }
  return success();
}

static void printInferNumIndexArgs(OpAsmPrinter &, Operation *, Region &,
                                   int64_t &, int64_t) {
  // Nothing to do. The number of count args gets parsed solely from the region.
}

void GenericOp::getAsmBlockArgumentNames(Region &region,
                                         OpAsmSetValueNameFn setNameFn) {
  if (&region == &getInitializer()) {
    return;
  }

  assert(&region == &getRegion() && "Unexpected region");
  for (Value v : getIdArgs()) {
    setNameFn(v, "id");
  }
  for (Value v : getCountArgs()) {
    setNameFn(v, "count");
  }
  for (Value v : getRegionRefArgs()) {
    setNameFn(v, "ref");
  }
}

LogicalResult GenericOp::verify() {
  if (getNumIndexArgs() % 2 != 0) {
    return emitOpError("expected even number of id + count args");
  }
  if (getRegion().front().getNumArguments() < getNumIndexArgs()) {
    return emitOpError(
        "fewer body arguments than specified number of counts/ids.");
  }
  return verifyParallelBodyOp(*this, getNumLeadingArgs(), getNumIndexArgs(),
                              getIdAndCountArgs());
}

void GenericOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                      ScopeAttrInterface scope, int64_t numIterators,
                      bool syncOnReturn) {
  GenericOp::build(b, result, TypeRange(), scope, ArrayRef<Value>{},
                   ArrayRef<Value>{}, ArrayRef<bool>{}, numIterators,
                   syncOnReturn);
}

void GenericOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                      ScopeAttrInterface scope, ValueRange inits,
                      int64_t numIterators, bool syncOnReturn) {
  SmallVector<bool> isTied(inits.size(), true);
  SmallVector<Type> resultTypes =
      llvm::map_to_vector(inits, [](Value v) -> Type { return v.getType(); });
  GenericOp::build(b, result, resultTypes, scope, inits, ArrayRef<Value>{},
                   isTied, numIterators, syncOnReturn);
}

void GenericOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                      TypeRange resultTypes, ScopeAttrInterface scope,
                      ValueRange dynamicSizes, int64_t numIterators,
                      bool syncOnReturn) {
  SmallVector<bool> isTied(resultTypes.size(), false);
  GenericOp::build(b, result, resultTypes, scope, ArrayRef<Value>{},
                   dynamicSizes, isTied, numIterators, syncOnReturn);
}

void GenericOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                      TypeRange resultTypes, ScopeAttrInterface scope,
                      ValueRange inits, ValueRange dynamicSizes,
                      ArrayRef<bool> isTied, int64_t numIterators,
                      bool syncOnReturn) {

  result.addAttribute(GenericOp::getScopeAttrName(result.name), scope);
  result.addOperands(inits);
  result.addOperands(dynamicSizes);
  result.addTypes(resultTypes);

  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setOperandSegmentSizes(
      {static_cast<int32_t>(inits.size()),
       static_cast<int32_t>(dynamicSizes.size())});
  inherentAttrs.setIsTied(isTied);
  inherentAttrs.setSyncOnReturn(syncOnReturn);
  inherentAttrs.setNumIndexArgs(2 * numIterators);

  // Add the initializer region.
  result.addRegion();

  // Add the main region.
  Region *region = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(region);
  Block &entryBlock = region->front();

  // Add block arguments.

  // sref args.
  for (Type resultType : resultTypes) {
    auto shapedType = cast<ShapedType>(resultType);
    entryBlock.addArgument(
        PCF::ShapedRefType::get(b.getContext(), shapedType.getShape(),
                                shapedType.getElementType(), scope),
        result.location);
  }

  // Thread count/id args.
  Type indexType = b.getIndexType();
  for (int64_t i = 0; i < 2 * numIterators; ++i) {
    entryBlock.addArgument(indexType, result.location);
  }
}

bool GenericOp::isRegionRefArg(BlockArgument b) {
  assert(b.getOwner() == &getRegion().front() &&
         "unexpected non-entry block arg");
  int64_t rangeBegin = getNumLeadingArgs();
  int64_t rangeEnd = getNumLeadingArgs() + getNumResults();
  return b.getArgNumber() >= rangeBegin && b.getArgNumber() < rangeEnd;
}

SmallVector<int64_t> GenericOp::getInitTiedResultIndices() {
  SmallVector<int64_t> tiedResults;
  for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
    if (isTied) {
      tiedResults.push_back(i);
    }
  }
  return tiedResults;
}

OpResult GenericOp::getTiedResult(OpOperand &operand) {
  int64_t beginIndex = getInits().getBeginOperandIndex();
  int64_t operandIndex = operand.getOperandNumber();
  if (operandIndex < beginIndex ||
      operandIndex >= getInits().size() + beginIndex) {
    return OpResult();
  }

  int64_t initIndex = operandIndex - beginIndex;
  for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
    if (isTied) {
      if (initIndex == 0) {
        return (*this)->getOpResult(i);
      }
      --initIndex;
    }
  }

  return OpResult();
}

OpOperand *GenericOp::getTiedInit(int64_t i) {
  if (i < 0 || i >= getNumResults() || !getIsTied()[i]) {
    return nullptr;
  }

  int64_t initIndex = llvm::count(getIsTied().take_front(i), true);
  return &getInitsMutable()[initIndex];
}

ValueRange GenericOp::getResultDims(int64_t i) {
  if (getIsTied()[i]) {
    return {};
  }

  int64_t startIndex = 0;
  for (auto [curr, isTied] : llvm::enumerate(getIsTied())) {
    if (curr == i) {
      break;
    }
    if (!isTied) {
      startIndex += getResultType(curr).getNumDynamicDims();
    }
  }

  return ValueRange(getDynamicSizes().slice(
      startIndex, startIndex + getResultType(i).getNumDynamicDims()));
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

void LoopOp::getAsmBlockArgumentNames(Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
  for (Value v : getIdArgs()) {
    setNameFn(v, "id");
  }
  for (Value v : getRegionRefArgs()) {
    setNameFn(v, "ref");
  }
}

LogicalResult LoopOp::verify() {
  if (getCount().empty()) {
    return emitOpError("expected at least one iteration count argument");
  }
  if (getBody()->getNumArguments() < getNumIdArgs()) {
    return emitOpError("fewer body arguments than specified number of ids.");
  }
  return verifyParallelBodyOp(*this, /*numLeadingArgs=*/0, getNumIdArgs(),
                              getIdArgs());
}

void LoopOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                   ScopeAttrInterface scope, ValueRange count,
                   bool syncOnReturn) {
  LoopOp::build(b, result, TypeRange(), scope, count, ArrayRef<Value>{},
                ArrayRef<Value>{}, ArrayRef<bool>{}, syncOnReturn);
}

void LoopOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                   ScopeAttrInterface scope, ValueRange count, ValueRange inits,
                   bool syncOnReturn) {
  SmallVector<bool> isTied(inits.size(), true);
  SmallVector<Type> resultTypes =
      llvm::map_to_vector(inits, [](Value v) -> Type { return v.getType(); });
  LoopOp::build(b, result, resultTypes, scope, count, inits, ArrayRef<Value>{},
                isTied, syncOnReturn);
}

void LoopOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                   TypeRange resultTypes, ScopeAttrInterface scope,
                   ValueRange count, ValueRange dynamicSizes,
                   bool syncOnReturn) {
  SmallVector<bool> isTied(resultTypes.size(), false);
  LoopOp::build(b, result, resultTypes, scope, count, ArrayRef<Value>{},
                dynamicSizes, isTied, syncOnReturn);
}

void LoopOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                   TypeRange resultTypes, ScopeAttrInterface scope,
                   ValueRange count, ValueRange inits, ValueRange dynamicSizes,
                   ArrayRef<bool> isTied, bool syncOnReturn) {

  result.addAttribute(LoopOp::getScopeAttrName(result.name), scope);
  result.addOperands(count);
  result.addOperands(inits);
  result.addOperands(dynamicSizes);
  result.addTypes(resultTypes);

  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setOperandSegmentSizes(
      {static_cast<int32_t>(count.size()), static_cast<int32_t>(inits.size()),
       static_cast<int32_t>(dynamicSizes.size())});
  inherentAttrs.setIsTied(isTied);
  inherentAttrs.setSyncOnReturn(syncOnReturn);

  Region *region = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(region);
  Block &entryBlock = region->front();

  // Add block arguments.

  // sref args.
  for (Type resultType : resultTypes) {
    auto shapedType = cast<ShapedType>(resultType);
    entryBlock.addArgument(
        PCF::ShapedRefType::get(b.getContext(), shapedType.getShape(),
                                shapedType.getElementType(), scope),
        result.location);
  }

  // Thread count args.
  Type indexType = b.getIndexType();
  int64_t numCountArgs = count.empty() ? 1 : count.size();
  for (int64_t i = 0; i < numCountArgs; ++i) {
    entryBlock.addArgument(indexType, result.location);
  }
}

ValueRange LoopOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isParent() ? getOperation()->getResults() : ValueRange();
}

void LoopOp::getSuccessorRegions(RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the GenericOp, branch into the body.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getRegion()));
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.push_back(RegionSuccessor::parent());
}

SmallVector<int64_t> LoopOp::getInitTiedResultIndices() {
  SmallVector<int64_t> tiedResults;
  for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
    if (isTied) {
      tiedResults.push_back(i);
    }
  }
  return tiedResults;
}

OpResult LoopOp::getTiedResult(OpOperand &operand) {
  int64_t beginIndex = getInits().getBeginOperandIndex();
  int64_t operandIndex = operand.getOperandNumber();
  if (operandIndex < beginIndex ||
      operandIndex >= getInits().size() + beginIndex) {
    return OpResult();
  }

  int64_t initIndex = operandIndex - beginIndex;
  for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
    if (isTied) {
      if (initIndex == 0) {
        return (*this)->getOpResult(i);
      }
      --initIndex;
    }
  }

  return OpResult();
}

OpOperand *LoopOp::getTiedInit(int64_t i) {
  if (i < 0 || i >= getNumResults() || !getIsTied()[i]) {
    return nullptr;
  }

  int64_t initIndex = llvm::count(getIsTied().take_front(i), true);
  return &getInitsMutable()[initIndex];
}

ValueRange LoopOp::getResultDims(int64_t i) {
  if (getIsTied()[i]) {
    return {};
  }

  int64_t startIndex = 0;
  for (auto [curr, isTied] : llvm::enumerate(getIsTied())) {
    if (curr == i) {
      break;
    }
    if (!isTied) {
      startIndex += getResultType(curr).getNumDynamicDims();
    }
  }

  return ValueRange(getDynamicSizes().slice(
      startIndex, startIndex + getResultType(i).getNumDynamicDims()));
}

//===----------------------------------------------------------------------===//
// FormBundlesOp
//===----------------------------------------------------------------------===//

LogicalResult FormBundlesOp::verify() {
  // Check that the form_bundles scope matches the parent's scope.
  Operation *parentOp = getOperation()->getParentOp();
  if (auto genericOp = dyn_cast<GenericOp>(parentOp)) {
    if (getScope() != genericOp.getScope()) {
      return emitOpError("scope does not match parent generic scope");
    }
  } else if (auto sharedExecOp = dyn_cast<SharedExecutorOp>(parentOp)) {
    bool found = llvm::any_of(sharedExecOp.getScopesAttr(),
                              [&](Attribute a) { return a == getScope(); });
    if (!found) {
      return emitOpError("scope is not among parent shared_executor scopes");
    }
  }

  ArrayRef<int64_t> sizes = getSizes();
  if (sizes.empty()) {
    return emitOpError("sizes must have at least one entry");
  }
  for (int64_t i = 0, e = sizes.size(); i < e; ++i) {
    if (sizes[i] < 1) {
      return emitOpError("bundle size at index ")
             << i << " must be >= 1, got " << sizes[i];
    }
  }

  Block &body = getBody().front();
  int64_t numBundleArgs = static_cast<int64_t>(sizes.size());
  if (static_cast<int64_t>(body.getNumArguments()) != numBundleArgs) {
    return emitOpError("expected ")
           << numBundleArgs << " block arguments (one per bundle) but got "
           << body.getNumArguments();
  }

  ScopeAttrInterface scope = getScope();
  for (int64_t i = 0, e = body.getNumArguments(); i < e; ++i) {
    BundleType bundleTy = dyn_cast<BundleType>(body.getArgument(i).getType());
    if (!bundleTy) {
      return emitOpError("block argument ")
             << i << " must have !pcf.bundle type";
    }
    if (bundleTy.getScope() != scope) {
      return emitOpError("block argument ")
             << i << " has bundle scope '" << bundleTy.getScope()
             << "' but expected '" << scope << "'";
    }
    if (bundleTy.getId() != i) {
      return emitOpError("block argument ")
             << i << " has bundle ID " << bundleTy.getId() << " but expected "
             << i;
    }
  }

  return success();
}

ParseResult FormBundlesOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse scope attribute.
  Attribute scope;
  if (parser.parseAttribute(scope)) {
    return failure();
  }
  result.addAttribute("scope", scope);

  // Parse "sizes" keyword and integer array.
  if (parser.parseKeyword("sizes")) {
    return failure();
  }
  SmallVector<int64_t> sizes;
  if (parser.parseLSquare()) {
    return failure();
  }
  if (parser.parseCommaSeparatedList([&]() -> ParseResult {
        int64_t val;
        if (parser.parseInteger(val)) {
          return failure();
        }
        sizes.push_back(val);
        return success();
      })) {
    return failure();
  }
  if (parser.parseRSquare()) {
    return failure();
  }
  result.addAttribute("sizes",
                      DenseI64ArrayAttr::get(parser.getContext(), sizes));

  // Parse region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body)) {
    return failure();
  }

  // Parse optional attribute dict.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void FormBundlesOp::print(OpAsmPrinter &printer) {
  printer << " " << getScope();
  printer << " sizes [";
  llvm::interleaveComma(getSizes(), printer.getStream());
  printer << "] ";
  printer.printRegion(getBody());
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"scope", "sizes"});
}

//===----------------------------------------------------------------------===//
// ExecuteAsOp
//===----------------------------------------------------------------------===//

LogicalResult ExecuteAsOp::verify() {
  // Check that bundle operands are block arguments of the parent form_bundles.
  auto *parentOp = getOperation()->getParentOp();
  if (auto formBundles = dyn_cast<FormBundlesOp>(parentOp)) {
    Block &parentBlock = formBundles.getBody().front();
    for (auto [i, bundle] : llvm::enumerate(getBundles())) {
      auto blockArg = dyn_cast<BlockArgument>(bundle);
      if (!blockArg || blockArg.getOwner() != &parentBlock) {
        return emitOpError("bundle operand at index ")
               << i << " must be a block argument of the parent form_bundles";
      }
    }
  }

  // Check for duplicate bundle IDs.
  DenseSet<int64_t> seenIds;
  for (Value bundle : getBundles()) {
    BundleType ty = cast<BundleType>(bundle.getType());
    if (!seenIds.insert(ty.getId()).second) {
      return emitOpError("duplicate bundle operand with ID ") << ty.getId();
    }
  }

  return success();
}

ParseResult ExecuteAsOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse "[" bundle_list "]".
  if (parser.parseLSquare()) {
    return failure();
  }
  SmallVector<OpAsmParser::UnresolvedOperand> bundles;
  SmallVector<Type> bundleTypes;
  if (failed(parser.parseOptionalRSquare())) {
    do {
      OpAsmParser::UnresolvedOperand bundle;
      Type bundleType;
      if (parser.parseOperand(bundle) || parser.parseColonType(bundleType)) {
        return failure();
      }
      bundles.push_back(bundle);
      bundleTypes.push_back(bundleType);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare()) {
      return failure();
    }
  }

  // Resolve bundle operands.
  if (parser.resolveOperands(bundles, bundleTypes, parser.getNameLoc(),
                             result.operands)) {
    return failure();
  }

  // Parse region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body)) {
    return failure();
  }

  // Parse optional attribute dict.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void ExecuteAsOp::print(OpAsmPrinter &printer) {
  printer << " [";
  llvm::interleaveComma(getBundles(), printer, [&](Value bundle) {
    printer << bundle << " : " << bundle.getType();
  });
  printer << "] ";
  printer.printRegion(getBody());
  printer.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// SharedExecutorOp
//===----------------------------------------------------------------------===//

void SharedExecutorOp::build(OpBuilder &builder, OperationState &result,
                             ArrayAttr scopes, ValueRange inits,
                             ValueRange captures,
                             DenseI64ArrayAttr countDimsPerScope,
                             ArrayRef<bool> isCapture, TypeRange resultTypes,
                             int64_t numCountArgs) {
  result.addOperands(inits);
  result.addOperands(captures);
  result.addTypes(resultTypes);
  result.addAttribute("scopes", scopes);
  result.addAttribute("count_dims_per_scope", countDimsPerScope);
  result.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(inits.size()),
                                    static_cast<int32_t>(captures.size()),
                                    /*dynamic_sizes=*/0}));
  auto &props = result.getOrAddProperties<SharedExecutorOp::Properties>();
  props.is_capture = SmallVector<bool>(isCapture);
  props.num_leading_args = 0;

  // Add the initializer and body regions.
  result.addRegion(); // Initializer (empty).
  Region *body = result.addRegion();
  Block *bodyBlock = &body->emplaceBlock();
  // Add ref args with proper sref types.
  ScopeAttrInterface firstScope = cast<ScopeAttrInterface>(scopes[0]);
  int64_t captureIdx = 0;
  int64_t tiedIdx = 0;
  for (bool cap : isCapture) {
    if (cap) {
      ShapedType capType = cast<ShapedType>(captures[captureIdx].getType());
      bodyBlock->addArgument(
          ShapedRefType::get(builder.getContext(), capType.getShape(),
                             capType.getElementType(), firstScope,
                             AccessorMode::ReadOnly),
          result.location);
      ++captureIdx;
    } else {
      ShapedType resType = cast<ShapedType>(resultTypes[tiedIdx]);
      bodyBlock->addArgument(
          ShapedRefType::get(builder.getContext(), resType.getShape(),
                             resType.getElementType(), firstScope,
                             AccessorMode::ReadWrite),
          result.location);
      ++tiedIdx;
    }
  }
  // Add count args.
  for (int64_t i = 0; i < numCountArgs; ++i) {
    bodyBlock->addArgument(builder.getIndexType(), result.location);
  }
}

void SharedExecutorOp::getAsmBlockArgumentNames(Region &region,
                                                OpAsmSetValueNameFn setNameFn) {
  if (&region == &getInitializer()) {
    return;
  }
  // Body region: [leading_args][ref_args][count_args].
  int64_t numLeading = getNumLeadingArgs();
  int64_t numRefs = getNumRefArgs();
  int64_t numCounts = getTotalCountDims();
  auto args = region.getArguments();
  for (int64_t i = 0; i < numLeading; ++i) {
    setNameFn(args[i], "init_arg");
  }
  for (int64_t i = 0; i < numRefs; ++i) {
    setNameFn(args[numLeading + i], "ref");
  }
  for (int64_t i = 0; i < numCounts; ++i) {
    setNameFn(args[numLeading + numRefs + i], "count");
  }
}

LogicalResult SharedExecutorOp::verify() {
  // Check at least one scope.
  ArrayAttr scopesAttr = getScopesAttr();
  if (scopesAttr.empty()) {
    return emitOpError("expected at least one scope");
  }

  // Check count_dims_per_scope length matches number of scopes.
  ArrayRef<int64_t> countDims = getCountDimsPerScope();
  if (static_cast<int64_t>(countDims.size()) != getNumScopes()) {
    return emitOpError("count_dims_per_scope length (")
           << countDims.size() << ") must match number of scopes ("
           << getNumScopes() << ")";
  }

  // Check each entry >= 1.
  for (auto [i, d] : llvm::enumerate(countDims)) {
    if (d < 1) {
      return emitOpError("count_dims_per_scope[")
             << i << "] must be >= 1, got " << d;
    }
  }

  // Check total count dims matches count block args.
  int64_t totalCountDims = getTotalCountDims();
  int64_t numLeading = getNumLeadingArgs();
  int64_t numRefs = getNumRefArgs();
  int64_t expectedBodyArgs = numLeading + numRefs + totalCountDims;
  int64_t actualBodyArgs = getBody().getNumArguments();
  if (actualBodyArgs != expectedBodyArgs) {
    return emitOpError("expected ")
           << expectedBodyArgs << " body block arguments (" << numLeading
           << " leading + " << numRefs << " refs + " << totalCountDims
           << " counts), but got " << actualBodyArgs;
  }

  // Check capture sref args are readonly, tied args are readwrite.
  ArrayRef<bool> isCapture = getIsCapture();
  auto bodyArgs = getBody().getArguments();
  for (auto [i, cap] : llvm::enumerate(isCapture)) {
    BlockArgument arg = bodyArgs[numLeading + i];
    ShapedRefType srefType = dyn_cast<ShapedRefType>(arg.getType());
    if (!srefType) {
      continue;
    }
    if (cap && srefType.hasAccessorMode() && !srefType.isReadOnly()) {
      return emitOpError("capture ref arg at index ")
             << i << " must have readonly accessor mode";
    }
    if (!cap && srefType.hasAccessorMode() && !srefType.isReadWrite()) {
      return emitOpError("tied ref arg at index ")
             << i << " must have readwrite accessor mode";
    }
  }

  // Check number of results matches number of non-capture (tied) ref args.
  int64_t numTied = llvm::count(isCapture, false);
  if (static_cast<int64_t>(getResults().size()) != numTied) {
    return emitOpError("expected ")
           << numTied << " results (one per tied ref), got "
           << getResults().size();
  }

  return success();
}

ParseResult SharedExecutorOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // Parse scope(s).
  // Either: scope(#attr) or scopes(#attr, #attr, ...).
  SmallVector<Attribute> scopes;
  if (succeeded(parser.parseOptionalKeyword("scope"))) {
    Attribute scope;
    if (parser.parseLParen() || parser.parseAttribute(scope) ||
        parser.parseRParen()) {
      return failure();
    }
    scopes.push_back(scope);
  } else if (succeeded(parser.parseOptionalKeyword("scopes"))) {
    if (parser.parseLParen()) {
      return failure();
    }
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          Attribute scope;
          if (parser.parseAttribute(scope)) {
            return failure();
          }
          scopes.push_back(scope);
          return success();
        })) {
      return failure();
    }
    if (parser.parseRParen()) {
      return failure();
    }
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'scope' or 'scopes' keyword");
  }
  result.addAttribute("scopes", ArrayAttr::get(parser.getContext(), scopes));

  // Parse optional initializer region.
  Region *initializer = result.addRegion();
  int64_t numLeadingArgs = 0;
  SmallVector<OpAsmParser::Argument> leadingArgs;
  if (succeeded(parser.parseOptionalKeyword("initialize"))) {
    if (parser.parseRegion(*initializer)) {
      return failure();
    }
    // Parse -> (arg_list) for leading args.
    if (parser.parseArrow() || parser.parseLParen()) {
      return failure();
    }
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          leadingArgs.emplace_back();
          if (parser.parseArgument(leadingArgs.back(), /*allowType=*/false,
                                   /*allowAttrs=*/false) ||
              parser.parseColonType(leadingArgs.back().type)) {
            return failure();
          }
          return success();
        })) {
      return failure();
    }
    if (parser.parseRParen()) {
      return failure();
    }
    numLeadingArgs = leadingArgs.size();
  }

  // Parse "execute" keyword.
  if (parser.parseKeyword("execute")) {
    return failure();
  }

  // Parse ref arg list: (ref_arg from capture, ref_arg = init, ...).
  SmallVector<OpAsmParser::Argument> refArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> captureOperands;
  SmallVector<Type> captureTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> initOperands;
  SmallVector<Type> initTypes;
  SmallVector<bool> isCapture;

  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          refArgs.emplace_back();
          if (parser.parseArgument(refArgs.back(), /*allowType=*/false,
                                   /*allowAttrs=*/false)) {
            return failure();
          }
          // Check for `from` (capture) or `=` (tied init).
          if (succeeded(parser.parseOptionalKeyword("from"))) {
            isCapture.push_back(true);
            captureOperands.emplace_back();
            if (parser.parseOperand(captureOperands.back())) {
              return failure();
            }
          } else if (succeeded(parser.parseOptionalEqual())) {
            isCapture.push_back(false);
            initOperands.emplace_back();
            if (parser.parseOperand(initOperands.back())) {
              return failure();
            }
          } else {
            return parser.emitError(
                parser.getCurrentLocation(),
                "expected 'from' or '=' after ref argument");
          }
          return success();
        })) {
      return failure();
    }
    if (parser.parseRParen()) {
      return failure();
    }
  }

  // Parse count arg groups: [args][args]... one per scope.
  SmallVector<int64_t> countDimsPerScope;
  SmallVector<OpAsmParser::Argument> countArgs;
  for (size_t i = 0, e = scopes.size(); i < e; ++i) {
    SmallVector<OpAsmParser::Argument> scopeCountArgs;
    if (parser.parseArgumentList(scopeCountArgs,
                                 /*delimiter=*/OpAsmParser::Delimiter::Square,
                                 /*allowType=*/true, /*allowAttrs=*/false)) {
      return failure();
    }
    countDimsPerScope.push_back(scopeCountArgs.size());
    countArgs.append(scopeCountArgs);
  }
  result.addAttribute(
      "count_dims_per_scope",
      DenseI64ArrayAttr::get(parser.getContext(), countDimsPerScope));

  // Parse ref arg types and result types if we have ref args.
  SmallVector<Type> resultTypes;
  if (!refArgs.empty()) {
    if (parser.parseColon()) {
      return failure();
    }
    // Parse (type_list) for ref arg types.
    auto it = refArgs.begin();
    if (parser.parseCommaSeparatedList(
            OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
              if (it == refArgs.end()) {
                return failure();
              }
              ParseResult p = parser.parseType(it->type);
              ++it;
              return p;
            })) {
      return failure();
    }

    // Parse -> (result_types).
    if (parser.parseArrow() || parser.parseLParen()) {
      return failure();
    }
    // Count the number of tied (non-capture) refs to know how many results.
    int64_t numTied = llvm::count(isCapture, false);
    for (int64_t i = 0; i < numTied; ++i) {
      Type ty;
      if (parser.parseType(ty)) {
        return failure();
      }
      resultTypes.push_back(ty);
      if (i < numTied - 1 && parser.parseComma()) {
        return failure();
      }
    }
    if (parser.parseRParen()) {
      return failure();
    }
  }
  result.addTypes(resultTypes);

  // Set up operand segment sizes.
  // Resolve capture types from ref arg types for capture entries.
  for (auto [i, cap] : llvm::enumerate(isCapture)) {
    if (cap) {
      ShapedRefType srefType = dyn_cast<ShapedRefType>(refArgs[i].type);
      if (srefType) {
        // The capture operand type is a tensor matching the sref shape.
        captureTypes.push_back(RankedTensorType::get(
            srefType.getShape(), srefType.getElementType()));
      }
    } else {
      // Tied init type matches the corresponding result type.
      initTypes.push_back(resultTypes[initTypes.size()]);
    }
  }

  // Parse the body region.
  Region *body = result.addRegion();
  SmallVector<OpAsmParser::Argument> allBodyArgs;
  allBodyArgs.append(leadingArgs);
  allBodyArgs.append(refArgs);
  allBodyArgs.append(countArgs);
  if (parser.parseRegion(*body, allBodyArgs)) {
    return failure();
  }

  // Parse optional attribute dict.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Resolve operands.
  if (parser.resolveOperands(initOperands, initTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperands(captureOperands, captureTypes, parser.getNameLoc(),
                             result.operands)) {
    return failure();
  }

  // Set operand segment sizes: [inits, captures, dynamic_sizes].
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(initOperands.size()),
                           static_cast<int32_t>(captureOperands.size()),
                           /*dynamic_sizes=*/0}));

  // Set properties.
  SharedExecutorOp::Properties &props =
      result.getOrAddProperties<SharedExecutorOp::Properties>();
  props.is_capture = isCapture;
  props.num_leading_args = numLeadingArgs;

  return success();
}

void SharedExecutorOp::print(OpAsmPrinter &p) {
  // Print scope(s).
  ArrayAttr scopesAttr = getScopesAttr();
  if (scopesAttr.size() == 1) {
    p << " scope(" << scopesAttr[0] << ")";
  } else {
    p << " scopes(";
    llvm::interleaveComma(scopesAttr, p);
    p << ")";
  }

  // Print optional initializer.
  if (!getInitializer().empty()) {
    p.printNewline();
    p << "    initialize ";
    p.printRegion(getInitializer());
    p << " -> (";
    int64_t numLeading = getNumLeadingArgs();
    auto bodyArgs = getBody().getArguments();
    llvm::interleaveComma(
        bodyArgs.take_front(numLeading), p,
        [&](BlockArgument arg) { p.printRegionArgument(arg); });
    p << ")";
  }

  p.printNewline();
  p << "    execute";

  // Print ref args.
  int64_t numLeading = getNumLeadingArgs();
  int64_t numRefs = getNumRefArgs();
  ArrayRef<bool> isCapture = getIsCapture();
  auto bodyArgs = getBody().getArguments();
  auto refArgRange = bodyArgs.slice(numLeading, numRefs);

  if (numRefs > 0) {
    p << "(";
    int64_t captureIdx = 0;
    int64_t initIdx = 0;
    for (int64_t i = 0; i < numRefs; ++i) {
      p << refArgRange[i];
      if (isCapture[i]) {
        p << " from " << getCaptures()[captureIdx];
        ++captureIdx;
      } else {
        p << " = " << getInits()[initIdx];
        ++initIdx;
      }
      if (i < numRefs - 1) {
        p << ", ";
      }
    }
    p << ")";
  }

  // Print count args grouped by scope.
  ArrayRef<int64_t> countDims = getCountDimsPerScope();
  int64_t countOffset = numLeading + numRefs;
  for (int64_t d : countDims) {
    p << "[";
    for (int64_t j = 0; j < d; ++j) {
      if (j > 0) {
        p << ", ";
      }
      p.printRegionArgument(bodyArgs[countOffset + j]);
    }
    p << "]";
    countOffset += d;
  }

  // Print ref types and result types.
  if (numRefs > 0) {
    p.printNewline();
    p << "        : (";
    llvm::interleaveComma(refArgRange, p,
                          [&](BlockArgument arg) { p << arg.getType(); });
    p << ")";
    p.printNewline();
    p << "        -> (";
    llvm::interleaveComma(getResultTypes(), p);
    p << ")";
  }

  // Print body region.
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"scopes", "count_dims_per_scope",
                                           "operandSegmentSizes"});
}

//===----------------------------------------------------------------------===//
// Control Flow Ops
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BranchCondReturnOp
//===----------------------------------------------------------------------===//

void BranchCondReturnOp::setDest(Block *block) { return setSuccessor(block); }

void BranchCondReturnOp::eraseOperand(unsigned index) {
  (*this)->eraseOperand(index);
}

SuccessorOperands BranchCondReturnOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  // Single index operand produced by this op.
  return SuccessorOperands(getDestOperandsMutable());
}

Block *
BranchCondReturnOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  if (IntegerAttr condAttr =
          llvm::dyn_cast_or_null<IntegerAttr>(operands.front())) {
    return condAttr.getValue().isOne() ? nullptr : getDest();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// FenceOp
//===----------------------------------------------------------------------===//

ParseResult FenceOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *context = parser.getContext();
  bool isRelease = false;
  if (succeeded(parser.parseOptionalKeyword("release"))) {
    isRelease = true;
  } else if (failed(parser.parseKeyword("acquire"))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'release' or 'acquire'");
  }
  result.addAttribute("is_release", BoolAttr::get(context, isRelease));

  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> types;
  if (failed(parser.parseOperandList(operands))) {
    return failure();
  }
  if (!operands.empty()) {
    if (failed(parser.parseColonTypeList(types))) {
      return failure();
    }
    if (failed(parser.resolveOperands(operands, types, parser.getNameLoc(),
                                      result.operands))) {
      return failure();
    }
  }
  return parser.parseOptionalAttrDict(result.attributes);
}

void FenceOp::print(OpAsmPrinter &p) {
  p << (getIsRelease() ? " release" : " acquire");
  if (!getSrefs().empty()) {
    p << " ";
    llvm::interleaveComma(getSrefs(), p);
    p << " : ";
    llvm::interleaveComma(getSrefs().getTypes(), p);
  }
  SmallVector<StringRef> elidedAttrs = {"is_release"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
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
  auto offsetValues =
      llvm::map_to_vector(offsets, llvm::StaticCastTo<OpFoldResult>);
  auto sizeValues =
      llvm::map_to_vector(sizes, llvm::StaticCastTo<OpFoldResult>);
  auto strideValues =
      llvm::map_to_vector(strides, llvm::StaticCastTo<OpFoldResult>);
  build(b, result, source, dest, offsetValues, sizeValues, strideValues);
}

//===----------------------------------------------------------------------===//
// ReadSliceOp
//===----------------------------------------------------------------------===//

void ReadSliceOp::build(OpBuilder &b, OperationState &result, Type resultType,
                        Value source, ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        ArrayRef<OpFoldResult> strides,
                        ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  result.addAttributes(attrs);
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
}

void ReadSliceOp::build(OpBuilder &b, OperationState &result, Type resultType,
                        Value source, ArrayRef<Range> ranges,
                        ArrayRef<NamedAttribute> attrs) {
  auto [offsets, sizes, strides] = getOffsetsSizesAndStrides(ranges);
  build(b, result, resultType, source, offsets, sizes, strides, attrs);
}

void ReadSliceOp::build(OpBuilder &b, OperationState &result, Type resultType,
                        Value source, ValueRange offsets, ValueRange sizes,
                        ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  auto offsetValues =
      llvm::map_to_vector(offsets, llvm::StaticCastTo<OpFoldResult>);
  auto sizeValues =
      llvm::map_to_vector(sizes, llvm::StaticCastTo<OpFoldResult>);
  auto strideValues =
      llvm::map_to_vector(strides, llvm::StaticCastTo<OpFoldResult>);
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues);
}

//===----------------------------------------------------------------------===//
// GetMemrefOp
//===----------------------------------------------------------------------===//

void GetMemrefOp::build(OpBuilder &b, OperationState &result, Type resultType,
                        Value source, ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        ArrayRef<OpFoldResult> strides,
                        ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  result.addAttributes(attrs);
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
}

void GetMemrefOp::build(OpBuilder &b, OperationState &result, Type resultType,
                        Value source, ArrayRef<Range> ranges,
                        ArrayRef<NamedAttribute> attrs) {
  auto [offsets, sizes, strides] = getOffsetsSizesAndStrides(ranges);
  build(b, result, resultType, source, offsets, sizes, strides, attrs);
}

void GetMemrefOp::build(OpBuilder &b, OperationState &result, Type resultType,
                        Value source, ValueRange offsets, ValueRange sizes,
                        ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  auto offsetValues =
      llvm::map_to_vector(offsets, llvm::StaticCastTo<OpFoldResult>);
  auto sizeValues =
      llvm::map_to_vector(sizes, llvm::StaticCastTo<OpFoldResult>);
  auto strideValues =
      llvm::map_to_vector(strides, llvm::StaticCastTo<OpFoldResult>);
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues);
}

LogicalResult WriteSliceOp::verify() {
  // Check accessor mode on destination sref.
  ShapedRefType destType = getDestType();
  if (destType.hasAccessorMode() && destType.isReadOnly()) {
    return emitOpError("cannot write to readonly sref");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Folders
//===----------------------------------------------------------------------===//

LogicalResult WriteSliceOp::fold(FoldAdaptor adaptor,
                                 SmallVectorImpl<OpFoldResult> &results) {
  SmallVector<OpFoldResult> mixedOffsets = getMixedOffsets();
  SmallVector<OpFoldResult> mixedStrides = getMixedStrides();

  // Try to fold dynamic offsets/strides to static.
  if (failed(foldDynamicIndexList(mixedOffsets, /*onlyNonNegative=*/true)) &&
      failed(foldDynamicIndexList(mixedStrides))) {
    return failure();
  }

  OpBuilder builder(getContext());

  // Dispatch back to static/dynamic.
  SmallVector<int64_t> staticOffsets, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicStrides;
  dispatchIndexOpFoldResults(mixedOffsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

  // Update the op's attributes in-place.
  setStaticOffsetsAttr(builder.getDenseI64ArrayAttr(staticOffsets));
  setStaticStridesAttr(builder.getDenseI64ArrayAttr(staticStrides));
  getOffsetsMutable().assign(dynamicOffsets);
  getStridesMutable().assign(dynamicStrides);

  return success();
}

OpFoldResult ReadSliceOp::fold(FoldAdaptor adaptor) {
  SmallVector<OpFoldResult> mixedOffsets = getMixedOffsets();
  SmallVector<OpFoldResult> mixedStrides = getMixedStrides();

  // Try to fold dynamic offsets/strides to static.
  if (failed(foldDynamicIndexList(mixedOffsets, /*onlyNonNegative=*/true)) &&
      failed(foldDynamicIndexList(mixedStrides))) {
    return {};
  }

  OpBuilder builder(getContext());

  // Dispatch back to static/dynamic.
  SmallVector<int64_t> staticOffsets, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicStrides;
  dispatchIndexOpFoldResults(mixedOffsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

  // Update the op's attributes in-place.
  setStaticOffsetsAttr(builder.getDenseI64ArrayAttr(staticOffsets));
  setStaticStridesAttr(builder.getDenseI64ArrayAttr(staticStrides));
  getOffsetsMutable().assign(dynamicOffsets);
  getStridesMutable().assign(dynamicStrides);

  return {};
}

OpFoldResult GetMemrefOp::fold(FoldAdaptor adaptor) {
  SmallVector<OpFoldResult> mixedOffsets = getMixedOffsets();
  SmallVector<OpFoldResult> mixedStrides = getMixedStrides();

  // Try to fold dynamic offsets/strides to static.
  if (failed(foldDynamicIndexList(mixedOffsets, /*onlyNonNegative=*/true)) &&
      failed(foldDynamicIndexList(mixedStrides))) {
    return {};
  }

  OpBuilder builder(getContext());

  // Dispatch back to static/dynamic.
  SmallVector<int64_t> staticOffsets, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicStrides;
  dispatchIndexOpFoldResults(mixedOffsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

  // Update the op's attributes in-place.
  setStaticOffsetsAttr(builder.getDenseI64ArrayAttr(staticOffsets));
  setStaticStridesAttr(builder.getDenseI64ArrayAttr(staticStrides));
  getOffsetsMutable().assign(dynamicOffsets);
  getStridesMutable().assign(dynamicStrides);

  return {};
}

LogicalResult GetMemrefOp::verify() {
  MemRefType resultType = getResultType();

  // Check that the result has no memory space.
  if (resultType.getMemorySpace()) {
    return emitOpError("result memref must have no memory space, got ")
           << resultType;
  }

  // Check that the result has a strided layout.
  auto layout = dyn_cast_or_null<StridedLayoutAttr>(resultType.getLayout());
  if (!layout) {
    return emitOpError(
               "result memref must have a strided layout attribute, got ")
           << resultType;
  }

  // Check that all strides and offset are dynamic.
  if (layout.getOffset() != ShapedType::kDynamic) {
    return emitOpError("result memref layout must have dynamic offset, got ")
           << resultType;
  }

  for (auto stride : layout.getStrides()) {
    if (stride != ShapedType::kDynamic) {
      return emitOpError(
                 "result memref layout must have all dynamic strides, got ")
             << resultType;
    }
  }

  return success();
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
