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
// SharedExecutorOp
//===----------------------------------------------------------------------===//

ParseResult SharedExecutorOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // Parse optional "sync" keyword.
  bool syncOnReturn = succeeded(parser.parseOptionalKeyword("sync"));

  // Parse "scope" "(" #attr ")".
  ScopeAttrInterface scope;
  if (parser.parseKeyword("scope") || parser.parseLParen() ||
      parser.parseAttribute(scope) || parser.parseRParen()) {
    return failure();
  }
  result.addAttribute(SharedExecutorOp::getScopeAttrName(result.name), scope);

  // Parse optional initializer region + leading args binding.
  Region *initializer = result.addRegion();
  SmallVector<OpAsmParser::Argument> leadingArgs;
  if (succeeded(parser.parseOptionalKeyword("initialize"))) {
    if (parser.parseRegion(*initializer)) {
      return failure();
    }
    if (succeeded(parser.parseOptionalArrow())) {
      if (parser.parseArgumentList(leadingArgs, OpAsmParser::Delimiter::Paren,
                                   /*allowType=*/true)) {
        return failure();
      }
    }
  }

  // Parse "execute".
  if (parser.parseKeyword("execute")) {
    return failure();
  }

  // Parse ref args: "(" %name "<-" %operand | %name "=" %operand ")".
  // "<-" denotes readonly, "=" denotes readwrite (tied to a result).
  SmallVector<OpAsmParser::Argument> readonlyRefArgs;
  SmallVector<OpAsmParser::Argument> readwriteRefArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> readonlyInits;
  SmallVector<OpAsmParser::UnresolvedOperand> readwriteInits;
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      OpAsmParser::Argument refArg;
      if (parser.parseArgument(refArg)) {
        return failure();
      }
      // "<-" is parsed as two tokens: '<' then '-'.
      if (succeeded(parser.parseOptionalLess())) {
        if (parser.parseMinus()) {
          return failure();
        }
        readonlyRefArgs.push_back(refArg);
        readonlyInits.emplace_back();
        if (parser.parseOperand(readonlyInits.back())) {
          return failure();
        }
      } else if (succeeded(parser.parseOptionalEqual())) {
        readwriteRefArgs.push_back(refArg);
        readwriteInits.emplace_back();
        if (parser.parseOperand(readwriteInits.back())) {
          return failure();
        }
      } else {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected '<-' or '=' after ref argument");
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen()) {
      return failure();
    }
  }

  // Parse "[%tg: !pcf.threadgroup<...>]".
  SmallVector<OpAsmParser::Argument> tgArgs;
  if (parser.parseArgumentList(tgArgs, OpAsmParser::Delimiter::Square,
                               /*allowType=*/true)) {
    return failure();
  }
  if (tgArgs.size() != 1) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected exactly one threadgroup argument");
  }

  // Parse sref types ": (sref_types)" and optional result types "-> (types)".
  int64_t numRefs = readonlyRefArgs.size() + readwriteRefArgs.size();
  SmallVector<Type> srefTypes;
  SmallVector<Type> resultTypes;
  if (numRefs != 0) {
    if (parser.parseColon()) {
      return failure();
    }
    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                       [&]() -> ParseResult {
                                         Type srefType;
                                         if (parser.parseType(srefType)) {
                                           return failure();
                                         }
                                         srefTypes.push_back(srefType);
                                         return success();
                                       })) {
      return failure();
    }
    if (srefTypes.size() != static_cast<size_t>(numRefs)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "sref type count does not match ref arg count");
    }

    // Assign sref types to ref block args.
    for (auto [i, arg] : llvm::enumerate(readonlyRefArgs)) {
      arg.type = srefTypes[i];
    }
    for (auto [i, arg] : llvm::enumerate(readwriteRefArgs)) {
      arg.type = srefTypes[readonlyRefArgs.size() + i];
    }

    // Parse optional result types.
    if (succeeded(parser.parseOptionalArrow())) {
      if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                         [&]() -> ParseResult {
                                           Type resType;
                                           if (parser.parseType(resType)) {
                                             return failure();
                                           }
                                           resultTypes.push_back(resType);
                                           return success();
                                         })) {
        return failure();
      }
    }
  }

  // Parse the region body.
  Region *body = result.addRegion();
  SmallVector<OpAsmParser::Argument> allArgs;
  allArgs.append(leadingArgs);
  allArgs.append(readonlyRefArgs);
  allArgs.append(readwriteRefArgs);
  allArgs.append(tgArgs);
  if (parser.parseRegion(*body, allArgs)) {
    return failure();
  }

  // Parse optional attr-dict.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Resolve operands. Readonly inits are resolved using types inferred from
  // the corresponding sref types (shape + element type as RankedTensorType).
  // Readwrite init types equal result types.
  SmallVector<Type> readonlyInitTypes;
  readonlyInitTypes.reserve(readonlyRefArgs.size());
  for (int64_t i = 0, e = readonlyRefArgs.size(); i < e; ++i) {
    ShapedRefType srefType = cast<ShapedRefType>(srefTypes[i]);
    readonlyInitTypes.push_back(
        RankedTensorType::get(srefType.getShape(), srefType.getElementType()));
  }
  if (parser.resolveOperands(readonlyInits, readonlyInitTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }
  if (parser.resolveOperands(readwriteInits, resultTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  result.addTypes(resultTypes);

  // Set properties.
  Properties &props = result.getOrAddProperties<Properties>();
  props.setOperandSegmentSizes({static_cast<int32_t>(readonlyInits.size()),
                                static_cast<int32_t>(readwriteInits.size())});
  props.setSyncOnReturn(syncOnReturn);
  props.setNumLeadingArgs(leadingArgs.size());
  props.setNumReadonlyRefs(readonlyRefArgs.size());

  return success();
}

void SharedExecutorOp::print(OpAsmPrinter &p) {
  if (getSyncOnReturn()) {
    p << " sync";
  }
  p << " scope(";
  p.printAttribute(getScope());
  p << ")";

  // Print optional initializer region.
  if (!getInitializer().empty()) {
    p << " initialize ";
    p.printRegion(getInitializer(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }

  int64_t numLeading = getNumLeadingArgs();
  if (numLeading > 0) {
    p << " -> (";
    MutableArrayRef<BlockArgument> leadingArgRange =
        getRegion().getArguments().take_front(numLeading);
    llvm::interleaveComma(leadingArgRange, p, [&](BlockArgument arg) {
      p.printRegionArgument(arg);
    });
    p << ")";
  }

  p.printNewline();
  p << "  execute";

  int64_t numReadonly = getNumReadonlyRefs();
  int64_t numReadwrite = getNumResults();
  int64_t numRefs = numReadonly + numReadwrite;

  ArrayRef<BlockArgument> readonlyArgs = getReadonlyRefArgs();
  ArrayRef<BlockArgument> readwriteArgs = getReadwriteRefArgs();
  BlockArgument tgArg = getThreadGroup();

  if (numRefs != 0) {
    p << "(";
    int64_t currReadonlyInit = 0;
    int64_t currReadwriteInit = 0;
    for (int64_t i = 0, e = numRefs; i < e; ++i) {
      if (i > 0) {
        p << ", ";
      }
      if (i < numReadonly) {
        p << readonlyArgs[i] << " <- ";
        p << getReadonlyInits()[currReadonlyInit];
        ++currReadonlyInit;
      } else {
        p << readwriteArgs[i - numReadonly] << " = ";
        p << getReadwriteInits()[currReadwriteInit];
        ++currReadwriteInit;
      }
    }
    p << ")";
  }

  p << "[";
  p.printRegionArgument(tgArg);
  p << "]";

  // Print sref types and result types.
  if (numRefs != 0) {
    p.printNewline();
    p << "       : (";
    SmallVector<BlockArgument> allRefArgs;
    allRefArgs.append(readonlyArgs.begin(), readonlyArgs.end());
    allRefArgs.append(readwriteArgs.begin(), readwriteArgs.end());
    llvm::interleaveComma(allRefArgs, p,
                          [&](BlockArgument arg) { p << arg.getType(); });
    p << ")";

    if (numReadwrite > 0) {
      p.printNewline();
      p << "      -> (";
      llvm::interleaveComma(getResultTypes(), p, [&](Type type) { p << type; });
      p << ") ";
    } else {
      p << " ";
    }
  } else {
    p << " ";
  }

  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getScopeAttrName(), getOperandSegmentSizesAttrName()});
}

LogicalResult SharedExecutorOp::verify() {
  int64_t numLeading = getNumLeadingArgs();
  int64_t numReadonly = getNumReadonlyRefs();
  int64_t numReadwrite = getNumResults();
  int64_t expectedArgs = numLeading + numReadonly + numReadwrite + 1;

  if (static_cast<int64_t>(getRegion().getNumArguments()) != expectedArgs) {
    return emitOpError("expected region to have ")
           << expectedArgs << " arguments (leading=" << numLeading
           << ", readonly=" << numReadonly << ", readwrite=" << numReadwrite
           << ", threadgroup=1), got " << getRegion().getNumArguments();
  }

  ScopeAttrInterface scope = getScope();

  // Verify threadgroup arg is last and has matching scope.
  BlockArgument tgArg = getThreadGroup();
  ThreadGroupType tgType = dyn_cast<ThreadGroupType>(tgArg.getType());
  if (!tgType) {
    return emitOpError("expected last region argument to be "
                       "!pcf.threadgroup, got ")
           << tgArg.getType();
  }
  if (tgType.getScope() != scope) {
    return emitOpError("threadgroup scope must match op scope");
  }

  // Verify readonly sref args.
  for (BlockArgument refArg : getReadonlyRefArgs()) {
    ShapedRefType srefType = dyn_cast<ShapedRefType>(refArg.getType());
    if (!srefType) {
      return emitOpError("expected readonly region argument to be !pcf.sref");
    }
    if (srefType.getScope() != scope) {
      return emitOpError("readonly sref scope must match op scope");
    }
  }

  // Verify readwrite sref args and result type consistency.
  for (auto [i, refArg] : llvm::enumerate(getReadwriteRefArgs())) {
    ShapedRefType srefType = dyn_cast<ShapedRefType>(refArg.getType());
    if (!srefType) {
      return emitOpError("expected readwrite region argument to be !pcf.sref");
    }
    if (srefType.getScope() != scope) {
      return emitOpError("readwrite sref scope must match op scope");
    }

    ShapedType resultType = getResultType(i);
    if (resultType.getShape() != srefType.getShape()) {
      return emitOpError("readwrite sref at index ")
             << i << " shape mismatch with result type";
    }
    if (resultType.getElementType() != srefType.getElementType()) {
      return emitOpError("readwrite sref at index ")
             << i << " element type mismatch with result type";
    }
  }

  // Verify readwrite inits match result types.
  for (auto [i, init] : llvm::enumerate(getReadwriteInits())) {
    if (init.getType() != getResultType(i)) {
      return emitOpError("readwrite init at index ")
             << i << " type " << init.getType()
             << " does not match result type " << getResultType(i);
    }
  }

  // Verify initializer yield matches leading args.
  if (!getInitializer().empty()) {
    Block &initBlock = getInitializer().front();
    Operation *terminator = initBlock.getTerminator();
    if (auto yieldOp = dyn_cast<YieldOp>(terminator)) {
      if (static_cast<int64_t>(yieldOp.getNumOperands()) != numLeading) {
        return emitOpError("initializer yield operand count (")
               << yieldOp.getNumOperands()
               << ") does not match num_leading_args (" << numLeading << ")";
      }
      for (auto [i, pair] : llvm::enumerate(
               llvm::zip(yieldOp.getOperandTypes(), getLeadingArgs()))) {
        Type yieldType = std::get<0>(pair);
        Type argType = std::get<1>(pair).getType();
        if (yieldType != argType) {
          return emitOpError("initializer yield type ")
                 << yieldType << " at index " << i
                 << " does not match leading arg type " << argType;
        }
      }
    }
  }

  return success();
}

void SharedExecutorOp::build(OpBuilder &b, OperationState &result,
                             ScopeAttrInterface scope,
                             ValueRange readwriteInits, bool syncOnReturn) {
  SharedExecutorOp::build(b, result, scope, /*readonlyInits=*/ValueRange(),
                          readwriteInits, syncOnReturn);
}

void SharedExecutorOp::build(OpBuilder &b, OperationState &result,
                             ScopeAttrInterface scope, ValueRange readonlyInits,
                             ValueRange readwriteInits, bool syncOnReturn) {
  result.addAttribute(SharedExecutorOp::getScopeAttrName(result.name), scope);
  result.addOperands(readonlyInits);
  result.addOperands(readwriteInits);

  SmallVector<Type> resultTypes = llvm::map_to_vector(
      readwriteInits, [](Value v) -> Type { return v.getType(); });
  result.addTypes(resultTypes);

  Properties &props = result.getOrAddProperties<Properties>();
  props.setOperandSegmentSizes({static_cast<int32_t>(readonlyInits.size()),
                                static_cast<int32_t>(readwriteInits.size())});
  props.setSyncOnReturn(syncOnReturn);
  props.setNumLeadingArgs(0);
  props.setNumReadonlyRefs(readonlyInits.size());

  // Add empty initializer region.
  result.addRegion();

  // Add the main region.
  Region *region = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(region);
  Block &entryBlock = region->front();

  // Readonly sref args.
  for (Value init : readonlyInits) {
    ShapedType shapedType = cast<ShapedType>(init.getType());
    entryBlock.addArgument(
        ShapedRefType::get(b.getContext(), shapedType.getShape(),
                           shapedType.getElementType(), scope),
        result.location);
  }

  // Readwrite sref args.
  for (Value init : readwriteInits) {
    ShapedType shapedType = cast<ShapedType>(init.getType());
    entryBlock.addArgument(
        ShapedRefType::get(b.getContext(), shapedType.getShape(),
                           shapedType.getElementType(), scope),
        result.location);
  }

  // Threadgroup arg.
  entryBlock.addArgument(ThreadGroupType::get(b.getContext(), scope, {}),
                         result.location);
}

void SharedExecutorOp::getAsmBlockArgumentNames(Region &region,
                                                OpAsmSetValueNameFn setNameFn) {
  if (&region == &getInitializer()) {
    return;
  }

  assert(&region == &getRegion() && "Unexpected region");
  for (Value v : getReadonlyRefArgs()) {
    setNameFn(v, "ref");
  }
  for (Value v : getReadwriteRefArgs()) {
    setNameFn(v, "ref");
  }
  for (Value v : getLeadingArgs()) {
    setNameFn(v, "leading");
  }
  setNameFn(getThreadGroup(), "tg");
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

void WriteSliceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (isa<MemRefType>(getSourceType())) {
    // Source memrefs are read from.
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                         SideEffects::DefaultResource::get());
  }
  // The dest operand is written to.
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable(),
                       SideEffects::DefaultResource::get());
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
