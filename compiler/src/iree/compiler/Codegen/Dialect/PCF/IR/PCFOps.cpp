// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "llvm/ADT/DenseSet.h"
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

  int64_t numInits = llvm::count(isTied, true);
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
    SmallVector<OpAsmParser::Argument>::iterator it = regionRefArgs.begin();
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
  SmallVector<OpFoldResult> offsetValues =
      llvm::map_to_vector(offsets, llvm::StaticCastTo<OpFoldResult>);
  SmallVector<OpFoldResult> sizeValues =
      llvm::map_to_vector(sizes, llvm::StaticCastTo<OpFoldResult>);
  SmallVector<OpFoldResult> strideValues =
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
  SmallVector<OpFoldResult> offsetValues =
      llvm::map_to_vector(offsets, llvm::StaticCastTo<OpFoldResult>);
  SmallVector<OpFoldResult> sizeValues =
      llvm::map_to_vector(sizes, llvm::StaticCastTo<OpFoldResult>);
  SmallVector<OpFoldResult> strideValues =
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
  SmallVector<OpFoldResult> offsetValues =
      llvm::map_to_vector(offsets, llvm::StaticCastTo<OpFoldResult>);
  SmallVector<OpFoldResult> sizeValues =
      llvm::map_to_vector(sizes, llvm::StaticCastTo<OpFoldResult>);
  SmallVector<OpFoldResult> strideValues =
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
// BarrierOp
//===----------------------------------------------------------------------===//

void BarrierOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // A barrier synchronizes all workers at a given scope. Model as a write
  // to the default resource to prevent reordering, hoisting, or elimination.
  effects.emplace_back(MemoryEffects::Write::get());
}

//===----------------------------------------------------------------------===//
// FormGroupsOp
//===----------------------------------------------------------------------===//

void FormGroupsOp::build(OpBuilder &b, OperationState &result,
                         ScopeAttrInterface scope, ArrayRef<int64_t> sizes) {
  result.addAttribute(FormGroupsOp::getScopeAttrName(result.name), scope);
  result.getOrAddProperties<Properties>().setSizes(
      b.getDenseI64ArrayAttr(sizes));

  Region *body = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  Block *entryBlock = b.createBlock(body);

  // Create one group-typed block argument per size entry.
  for (auto [i, size] : llvm::enumerate(sizes)) {
    SmallVector<int64_t> ids = {static_cast<int64_t>(i)};
    GroupType groupType = GroupType::get(b.getContext(), ids, scope, {});
    entryBlock->addArgument(groupType, result.location);
  }
}

LogicalResult FormGroupsOp::verify() {
  ArrayRef<int64_t> sizes = getSizes();
  if (sizes.empty()) {
    return emitOpError("expected at least one group size");
  }

  for (auto [i, size] : llvm::enumerate(sizes)) {
    if (size <= 0) {
      return emitOpError("group size at index ")
             << i << " must be positive, got " << size;
    }
  }

  Block &entryBlock = getBody().front();
  if (entryBlock.getNumArguments() != static_cast<unsigned>(sizes.size())) {
    return emitOpError("expected ") << sizes.size() << " block arguments, got "
                                    << entryBlock.getNumArguments();
  }

  ScopeAttrInterface scope = getScope();
  for (auto [i, arg] : llvm::enumerate(entryBlock.getArguments())) {
    auto groupType = dyn_cast<GroupType>(arg.getType());
    if (!groupType) {
      return emitOpError("block argument ")
             << i << " must be a !pcf.group type";
    }
    if (groupType.getScope() != scope) {
      return emitOpError("block argument ")
             << i << " scope mismatch: expected " << scope << ", got "
             << groupType.getScope();
    }
    if (groupType.isGrouped()) {
      return emitOpError("block argument ")
             << i << " must be a single-ID group, got grouped";
    }
    if (groupType.getId() != static_cast<int64_t>(i)) {
      return emitOpError("block argument ")
             << i << " group ID mismatch: expected " << i << ", got "
             << groupType.getId();
    }
    if (groupType.hasStructElements()) {
      return emitOpError("block argument ")
             << i << " must not have struct elements";
    }
  }

  // The yield terminator must have zero operands.
  Operation *terminator = entryBlock.getTerminator();
  if (!terminator) {
    return emitOpError("body block must have a terminator");
  }
  if (auto yieldOp = dyn_cast<YieldOp>(terminator)) {
    if (yieldOp.getNumOperands() != 0) {
      return yieldOp.emitOpError("in form_groups body must have zero operands");
    }
  }

  return success();
}

ParseResult FormGroupsOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse scope attribute.
  Attribute scopeAttr;
  SMLoc scopeLoc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(scopeAttr))) {
    return parser.emitError(scopeLoc, "expected scope attribute");
  }

  auto scope = dyn_cast<ScopeAttrInterface>(scopeAttr);
  if (!scope) {
    return parser.emitError(
        scopeLoc, "expected attribute implementing ScopeAttrInterface");
  }
  result.addAttribute(FormGroupsOp::getScopeAttrName(result.name), scope);

  // Parse `sizes` keyword and sizes array.
  if (failed(parser.parseKeyword("sizes"))) {
    return failure();
  }

  SmallVector<int64_t> sizes;
  if (failed(parser.parseLSquare())) {
    return failure();
  }
  if (failed(parser.parseCommaSeparatedList([&]() -> ParseResult {
        int64_t size;
        if (failed(parser.parseInteger(size))) {
          return failure();
        }
        sizes.push_back(size);
        return success();
      }))) {
    return failure();
  }
  if (failed(parser.parseRSquare())) {
    return failure();
  }
  result.getOrAddProperties<Properties>().setSizes(
      parser.getBuilder().getDenseI64ArrayAttr(sizes));

  // Parse the region. Block arguments and their types are in the IR.
  Region *body = result.addRegion();
  if (failed(parser.parseRegion(*body))) {
    return failure();
  }

  // Parse optional attr-dict.
  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }

  return success();
}

void FormGroupsOp::print(OpAsmPrinter &p) {
  p << " " << getScope() << " sizes [";
  llvm::interleaveComma(getSizes(), p);
  p << "] ";
  p.printRegion(getBody());
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getScopeAttrName(), getSizesAttrName()});
}

//===----------------------------------------------------------------------===//
// ExecuteAsOp
//===----------------------------------------------------------------------===//

void ExecuteAsOp::build(OpBuilder &b, OperationState &result, Value group) {
  result.addOperands(group);

  // No result for side-effect-only variant.
  Region *body = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  Block *entryBlock = b.createBlock(body);

  // Unpack struct element types as block arguments.
  auto groupType = cast<GroupType>(group.getType());
  for (Type elemType : groupType.getStructTypes()) {
    entryBlock->addArgument(elemType, result.location);
  }
}

void ExecuteAsOp::build(OpBuilder &b, OperationState &result,
                        GroupType resultType, Value group) {
  result.addOperands(group);
  result.addTypes(resultType);

  Region *body = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  Block *entryBlock = b.createBlock(body);

  // Unpack struct element types as block arguments.
  auto groupType = cast<GroupType>(group.getType());
  for (Type elemType : groupType.getStructTypes()) {
    entryBlock->addArgument(elemType, result.location);
  }
}

GroupType ExecuteAsOp::getResultGroupType() {
  if (!hasResult()) {
    return {};
  }
  return cast<GroupType>(getResult().getType());
}

LogicalResult ExecuteAsOp::verify() {
  GroupType inputType = getGroupType();
  Block &entryBlock = getBody().front();

  // Block arguments must match input group struct element types.
  ArrayRef<Type> structTypes = inputType.getStructTypes();
  if (entryBlock.getNumArguments() != structTypes.size()) {
    return emitOpError("expected ")
           << structTypes.size()
           << " block arguments matching input group struct elements, got "
           << entryBlock.getNumArguments();
  }

  for (auto [i, pair] : llvm::enumerate(
           llvm::zip_equal(structTypes, entryBlock.getArguments()))) {
    auto [expectedType, arg] = pair;
    if (arg.getType() != expectedType) {
      return emitOpError("block argument ")
             << i << " type mismatch: expected " << expectedType << ", got "
             << arg.getType();
    }
  }

  // Check terminator consistency.
  Operation *terminator = entryBlock.getTerminator();
  if (!terminator) {
    return emitOpError("body block must have a terminator");
  }
  bool hasYield = isa<YieldOp>(terminator);
  bool hasReturn = isa<ReturnOp>(terminator);

  if (!hasYield && !hasReturn) {
    return emitOpError("body must be terminated by pcf.yield or pcf.return");
  }

  if (hasResult()) {
    // Must have yield.
    if (!hasYield) {
      return emitOpError("expected pcf.yield terminator when producing a "
                         "result, got pcf.return");
    }

    GroupType resultType = getResultGroupType();
    auto yieldOp = cast<YieldOp>(terminator);

    // Result group must share IDs and scope with input.
    if (resultType.getIds() != inputType.getIds()) {
      return emitOpError("result group IDs must match input group IDs");
    }
    if (resultType.getScope() != inputType.getScope()) {
      return emitOpError("result group scope must match input group scope");
    }

    // Yielded values must match result struct types.
    ArrayRef<Type> resultStructTypes = resultType.getStructTypes();
    if (yieldOp.getOperands().size() != resultStructTypes.size()) {
      return emitOpError("yield operand count ")
             << yieldOp.getOperands().size()
             << " does not match result group struct element count "
             << resultStructTypes.size();
    }

    for (auto [i, pair] : llvm::enumerate(
             llvm::zip_equal(resultStructTypes, yieldOp.getOperandTypes()))) {
      auto [expectedType, actualType] = pair;
      if (actualType != expectedType) {
        return emitOpError("yield operand ")
               << i << " type mismatch: expected " << expectedType << ", got "
               << actualType;
      }
    }
  } else {
    // No result: must have return.
    if (!hasReturn) {
      return emitOpError(
          "expected pcf.return terminator when producing no result");
    }
  }

  return success();
}

ParseResult ExecuteAsOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse `[` bundle_operand `]`.
  OpAsmParser::UnresolvedOperand groupOperand;
  if (failed(parser.parseLSquare()) ||
      failed(parser.parseOperand(groupOperand)) ||
      failed(parser.parseRSquare())) {
    return failure();
  }

  // Optionally parse `-> result_type`.
  if (succeeded(parser.parseOptionalArrow())) {
    Type resultType;
    if (failed(parser.parseType(resultType))) {
      return failure();
    }
    result.addTypes(resultType);
  }

  // Parse the region body.
  Region *body = result.addRegion();
  if (failed(parser.parseRegion(*body))) {
    return failure();
  }

  // Parse `: input_bundle_type` to resolve the group operand.
  Type groupType;
  if (failed(parser.parseColon()) || failed(parser.parseType(groupType))) {
    return failure();
  }
  if (failed(parser.resolveOperand(groupOperand, groupType, result.operands))) {
    return failure();
  }

  // Parse optional attr-dict.
  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }

  return success();
}

void ExecuteAsOp::print(OpAsmPrinter &p) {
  p << " [" << getGroup() << "]";
  if (hasResult()) {
    p << " -> " << getResult().getType();
  }
  p << " ";
  p.printRegion(getBody());
  p << " : " << getGroup().getType();
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// BindOp
//===----------------------------------------------------------------------===//

LogicalResult BindOp::verify() {
  GroupType inputType = cast<GroupType>(getGroup().getType());
  GroupType resultType = cast<GroupType>(getResult().getType());

  // Input must be a single-ID group.
  if (inputType.isGrouped()) {
    return emitOpError("input group must be a single-ID group, got grouped");
  }

  // Result must share ID and scope with input.
  if (resultType.getIds() != inputType.getIds()) {
    return emitOpError("result group IDs must match input group IDs");
  }
  if (resultType.getScope() != inputType.getScope()) {
    return emitOpError("result group scope must match input group scope");
  }

  // Values must match result struct element types.
  ArrayRef<Type> resultStructTypes = resultType.getStructTypes();
  if (getValues().size() != resultStructTypes.size()) {
    return emitOpError("expected ")
           << resultStructTypes.size()
           << " values matching result struct elements, got "
           << getValues().size();
  }

  for (auto [i, pair] : llvm::enumerate(
           llvm::zip_equal(resultStructTypes, getValues().getTypes()))) {
    auto [expectedType, actualType] = pair;
    if (actualType != expectedType) {
      return emitOpError("value ") << i << " type mismatch: expected "
                                   << expectedType << ", got " << actualType;
    }
  }

  // If input already has struct elements, replacement count must match.
  if (inputType.hasStructElements()) {
    if (inputType.getStructTypes().size() != getValues().size()) {
      return emitOpError("input group has ")
             << inputType.getStructTypes().size()
             << " struct elements but bind provides " << getValues().size()
             << " values; counts must match when replacing";
    }
  }

  return success();
}

ParseResult BindOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse: pcf.bind %group, %val0, %val1 : !pcf.group<...>
  OpAsmParser::UnresolvedOperand groupOperand;
  SmallVector<OpAsmParser::UnresolvedOperand> valueOperands;

  if (failed(parser.parseOperand(groupOperand))) {
    return failure();
  }

  // Parse optional comma-separated values.
  while (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand valueOperand;
    if (failed(parser.parseOperand(valueOperand))) {
      return failure();
    }
    valueOperands.push_back(valueOperand);
  }

  // Parse `: result_type`.
  GroupType resultType;
  if (failed(parser.parseColon()) || failed(parser.parseType(resultType))) {
    return failure();
  }
  result.addTypes(resultType);

  // The input group type is inferred as same IDs/scope but no struct elements.
  // For replacement (input already has struct elements), parse `from <type>`.
  // Parse optional `from <input_bundle_type>`.
  GroupType inputGroupType;
  if (succeeded(parser.parseOptionalKeyword("from"))) {
    if (failed(parser.parseType(inputGroupType))) {
      return failure();
    }
  } else {
    // Infer input type: same IDs and scope, no struct elements.
    inputGroupType = GroupType::get(parser.getContext(), resultType.getIds(),
                                    resultType.getScope(), {});
  }

  // Resolve group operand.
  if (failed(parser.resolveOperand(groupOperand, inputGroupType,
                                   result.operands))) {
    return failure();
  }

  // Resolve value operands from result struct types.
  ArrayRef<Type> structTypes = resultType.getStructTypes();
  if (valueOperands.size() != structTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(), "expected ")
           << structTypes.size() << " value operands, got "
           << valueOperands.size();
  }
  for (auto [operand, type] : llvm::zip_equal(valueOperands, structTypes)) {
    if (failed(parser.resolveOperand(operand, type, result.operands))) {
      return failure();
    }
  }

  // Parse optional attr-dict.
  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }

  return success();
}

void BindOp::print(OpAsmPrinter &p) {
  p << " " << getGroup();
  for (Value value : getValues()) {
    p << ", " << value;
  }

  GroupType resultType = cast<GroupType>(getResult().getType());
  GroupType inputType = cast<GroupType>(getGroup().getType());
  p << " : " << resultType;

  // Print `from <input_type>` if input has struct elements.
  if (inputType.hasStructElements()) {
    p << " from " << inputType;
  }

  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// JoinOp
//===----------------------------------------------------------------------===//

LogicalResult JoinOp::verify() {
  OperandRange groups = getGroups();
  if (groups.size() < 2) {
    return emitOpError("expected at least 2 input groups, got ")
           << groups.size();
  }

  GroupType firstType = cast<GroupType>(groups.front().getType());
  ScopeAttrInterface scope = firstType.getScope();

  // Collect all IDs and check constraints.
  SmallVector<int64_t> allIds;
  bool hasStruct = firstType.hasStructElements();
  ArrayRef<Type> structTypes =
      hasStruct ? firstType.getStructTypes() : ArrayRef<Type>{};

  for (auto [i, group] : llvm::enumerate(groups)) {
    GroupType groupType = cast<GroupType>(group.getType());

    // All inputs must be single-ID.
    if (groupType.isGrouped()) {
      return emitOpError("input group ")
             << i << " must be a single-ID group, got grouped";
    }

    // All inputs must have the same scope.
    if (groupType.getScope() != scope) {
      return emitOpError("input group ")
             << i << " scope mismatch: expected " << scope << ", got "
             << groupType.getScope();
    }

    // Struct element consistency.
    if (groupType.hasStructElements()) {
      if (!hasStruct) {
        return emitOpError("input group ")
               << i << " has struct elements but earlier groups do not";
      }
      if (groupType.getStructTypes() != structTypes) {
        return emitOpError("input group ")
               << i << " struct element types differ from first group";
      }
    } else if (hasStruct) {
      return emitOpError("input group ")
             << i << " lacks struct elements but earlier groups have them";
    }

    allIds.push_back(groupType.getId());
  }

  // Check for duplicate IDs across inputs.
  llvm::SmallDenseSet<int64_t> seenIds;
  for (int64_t id : allIds) {
    if (!seenIds.insert(id).second) {
      return emitOpError("duplicate group ID in join: ") << id;
    }
  }

  // Verify result type.
  GroupType resultType = cast<GroupType>(getResult().getType());
  if (resultType.getScope() != scope) {
    return emitOpError("result scope must match input scope");
  }
  if (resultType.getIds() != ArrayRef<int64_t>(allIds)) {
    return emitOpError("result IDs must be the concatenation of input IDs");
  }
  if (resultType.getStructTypes() != structTypes) {
    return emitOpError("result struct types must match input struct types");
  }

  return success();
}

ParseResult JoinOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse: pcf.join %b0, %b1, ... : !pcf.group<[0, 1, ...], #scope>
  SmallVector<OpAsmParser::UnresolvedOperand> groupOperands;
  if (failed(parser.parseOperandList(groupOperands))) {
    return failure();
  }

  // Parse `: result_type`.
  GroupType resultType;
  if (failed(parser.parseColon()) || failed(parser.parseType(resultType))) {
    return failure();
  }
  result.addTypes(resultType);

  // Infer input group types from result type. Each input is a single-ID
  // group with the corresponding ID from the result's ID list.
  ArrayRef<int64_t> resultIds = resultType.getIds();
  if (groupOperands.size() != resultIds.size()) {
    return parser.emitError(parser.getCurrentLocation(), "expected ")
           << resultIds.size() << " group operands matching result IDs, got "
           << groupOperands.size();
  }

  ScopeAttrInterface scope = resultType.getScope();
  ArrayRef<Type> structTypes = resultType.getStructTypes();
  for (auto [operand, id] : llvm::zip_equal(groupOperands, resultIds)) {
    GroupType inputType =
        GroupType::get(parser.getContext(), id, scope, structTypes);
    if (failed(parser.resolveOperand(operand, inputType, result.operands))) {
      return failure();
    }
  }

  // Parse optional attr-dict.
  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }

  return success();
}

void JoinOp::print(OpAsmPrinter &p) {
  p << " ";
  llvm::interleaveComma(getGroups(), p, [&](Value group) { p << group; });
  p << " : " << getResult().getType();
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

LogicalResult SplitOp::verify() {
  GroupType inputType = cast<GroupType>(getGroup().getType());
  ArrayRef<int64_t> sizes = getSizes();

  ArrayRef<int64_t> inputIds = inputType.getIds();

  // Sizes must be positive.
  for (auto [i, size] : llvm::enumerate(sizes)) {
    if (size <= 0) {
      return emitOpError("size at index ")
             << i << " must be positive, got " << size;
    }
  }

  // Sum of sizes must equal number of IDs.
  int64_t totalSize = llvm::sum_of(sizes);
  if (totalSize != static_cast<int64_t>(inputIds.size())) {
    return emitOpError("sum of sizes (")
           << totalSize << ") must equal input group ID count ("
           << inputIds.size() << ")";
  }

  // Number of results must match number of sizes.
  if (getNumResults() != sizes.size()) {
    return emitOpError("expected ")
           << sizes.size() << " results, got " << getNumResults();
  }

  // Verify each result type.
  ScopeAttrInterface scope = inputType.getScope();
  ArrayRef<Type> structTypes = inputType.getStructTypes();
  int64_t idOffset = 0;
  for (auto [i, size] : llvm::enumerate(sizes)) {
    GroupType resultType = cast<GroupType>(getResult(i).getType());

    // Scope must match.
    if (resultType.getScope() != scope) {
      return emitOpError("result ")
             << i << " scope mismatch: expected " << scope << ", got "
             << resultType.getScope();
    }

    // IDs must be the correct contiguous slice.
    ArrayRef<int64_t> expectedIds = inputIds.slice(idOffset, size);
    if (resultType.getIds() != expectedIds) {
      return emitOpError("result ") << i << " IDs do not match expected slice";
    }

    // Struct types must match.
    if (resultType.getStructTypes() != structTypes) {
      return emitOpError("result ")
             << i << " struct types must match input struct types";
    }

    idOffset += size;
  }

  return success();
}

ParseResult SplitOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse: pcf.split %grouped sizes [2, 2]
  //     : !pcf.group<[0, 1], #scope>, !pcf.group<[2, 3], #scope>
  OpAsmParser::UnresolvedOperand groupOperand;
  if (failed(parser.parseOperand(groupOperand))) {
    return failure();
  }

  // Parse `sizes` keyword and sizes array.
  if (failed(parser.parseKeyword("sizes"))) {
    return failure();
  }

  SmallVector<int64_t> sizes;
  if (failed(parser.parseLSquare())) {
    return failure();
  }
  if (failed(parser.parseCommaSeparatedList([&]() -> ParseResult {
        int64_t size;
        if (failed(parser.parseInteger(size))) {
          return failure();
        }
        sizes.push_back(size);
        return success();
      }))) {
    return failure();
  }
  if (failed(parser.parseRSquare())) {
    return failure();
  }
  result.getOrAddProperties<SplitOp::Properties>().setSizes(
      parser.getBuilder().getDenseI64ArrayAttr(sizes));

  // Parse `: result_type, result_type, ...`.
  if (failed(parser.parseColon())) {
    return failure();
  }

  SmallVector<Type> resultTypes;
  if (failed(parser.parseTypeList(resultTypes))) {
    return failure();
  }
  result.addTypes(resultTypes);

  // Infer input group type from result types: concatenate all IDs.
  SmallVector<int64_t> allIds;
  ScopeAttrInterface scope;
  ArrayRef<Type> structTypes;
  for (auto [i, type] : llvm::enumerate(resultTypes)) {
    GroupType groupType = cast<GroupType>(type);
    if (i == 0) {
      scope = groupType.getScope();
      structTypes = groupType.getStructTypes();
    }
    llvm::append_range(allIds, groupType.getIds());
  }

  GroupType inputType =
      GroupType::get(parser.getContext(), allIds, scope, structTypes);
  if (failed(parser.resolveOperand(groupOperand, inputType, result.operands))) {
    return failure();
  }

  // Parse optional attr-dict.
  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }

  return success();
}

void SplitOp::print(OpAsmPrinter &p) {
  p << " " << getGroup() << " sizes [";
  llvm::interleaveComma(getSizes(), p);
  p << "] : ";
  llvm::interleaveComma(getResultTypes(), p, [&](Type type) { p << type; });
  p.printOptionalAttrDict((*this)->getAttrs(), {getSizesAttrName()});
}

//===----------------------------------------------------------------------===//
// DefineWorkspaceOp
//===----------------------------------------------------------------------===//

LogicalResult DefineWorkspaceOp::verify() {
  GroupType inputType = cast<GroupType>(getGroup().getType());
  GroupType resultType = cast<GroupType>(getResult().getType());

  // Input must be single-ID.
  if (inputType.isGrouped()) {
    return emitOpError("input group must be a single-ID group, got grouped");
  }

  // Result must share ID and scope with input.
  if (resultType.getIds() != inputType.getIds()) {
    return emitOpError("result group IDs must match input group IDs");
  }
  if (resultType.getScope() != inputType.getScope()) {
    return emitOpError("result group scope must match input group scope");
  }

  // Validate mixed static/dynamic iteration space.
  int64_t iterSpaceRank =
      static_cast<int64_t>(getStaticIterationSpace().size());
  int64_t numDynamic = static_cast<int64_t>(getIterationSpace().size());
  int64_t numStaticDynamic =
      llvm::count(getStaticIterationSpace(), ShapedType::kDynamic);
  if (numDynamic != numStaticDynamic) {
    return emitOpError("expected ")
           << numStaticDynamic << " dynamic iteration_space values, got "
           << numDynamic;
  }

  // Validate indexing maps.
  ArrayAttr maps = getIndexingMapsAttr();
  if (maps.empty()) {
    return emitOpError("expected at least one indexing map");
  }

  int64_t numDistributedDims = 0;
  for (Attribute mapAttr : maps) {
    AffineMap map = cast<AffineMapAttr>(mapAttr).getValue();
    // Each map must produce iterSpaceRank results.
    if (static_cast<int64_t>(map.getNumResults()) != iterSpaceRank) {
      return emitOpError("indexing map has ")
             << map.getNumResults() << " results but iteration space has "
             << iterSpaceRank << " dimensions";
    }
    numDistributedDims += map.getNumDims();
  }

  // Result struct types = input struct types + 2*numDistributedDims index.
  ArrayRef<Type> inputStructTypes = inputType.getStructTypes();
  ArrayRef<Type> resultStructTypes = resultType.getStructTypes();
  int64_t expectedResultStructSize =
      static_cast<int64_t>(inputStructTypes.size()) + 2 * numDistributedDims;

  if (static_cast<int64_t>(resultStructTypes.size()) !=
      expectedResultStructSize) {
    return emitOpError("result group must have ")
           << expectedResultStructSize << " struct elements ("
           << inputStructTypes.size() << " from input + 2*"
           << numDistributedDims << " for distributed dims), got "
           << resultStructTypes.size();
  }

  // Input struct types must be preserved as prefix.
  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(inputStructTypes, resultStructTypes))) {
    auto [inputT, resultT] = pair;
    if (inputT != resultT) {
      return emitOpError("result struct element ")
             << i << " must match input struct element: expected " << inputT
             << ", got " << resultT;
    }
  }

  // Appended elements must all be index type.
  IndexType indexType = IndexType::get(getContext());
  for (int64_t i = static_cast<int64_t>(inputStructTypes.size());
       i < static_cast<int64_t>(resultStructTypes.size()); ++i) {
    if (resultStructTypes[i] != indexType) {
      return emitOpError("workspace struct element ")
             << i << " must be index type, got " << resultStructTypes[i];
    }
  }

  return success();
}

ParseResult DefineWorkspaceOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::UnresolvedOperand groupOperand;
  if (failed(parser.parseOperand(groupOperand))) {
    return failure();
  }

  // Parse `iteration_space(...)` with mixed static/dynamic values.
  if (failed(parser.parseKeyword("iteration_space"))) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand> iterSpaceOperands;
  SmallVector<int64_t> staticIterSpace;
  if (failed(parser.parseLParen())) {
    return failure();
  }

  // Parse comma-separated list of static ints or dynamic SSA values.
  if (failed(parser.parseOptionalRParen())) {
    do {
      // Try parsing as integer first.
      int64_t staticVal;
      OptionalParseResult intResult = parser.parseOptionalInteger(staticVal);
      if (intResult.has_value()) {
        if (failed(*intResult)) {
          return failure();
        }
        staticIterSpace.push_back(staticVal);
      } else {
        // Dynamic SSA value.
        OpAsmParser::UnresolvedOperand operand;
        if (failed(parser.parseOperand(operand))) {
          return failure();
        }
        iterSpaceOperands.push_back(operand);
        staticIterSpace.push_back(ShapedType::kDynamic);
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }

  result.addAttribute(
      "static_iteration_space",
      DenseI64ArrayAttr::get(parser.getContext(), staticIterSpace));

  // Parse `indexing_maps = [...]`.
  if (failed(parser.parseKeyword("indexing_maps")) ||
      failed(parser.parseEqual())) {
    return failure();
  }
  ArrayAttr indexingMaps;
  if (failed(parser.parseAttribute(indexingMaps))) {
    return failure();
  }
  result.addAttribute("indexing_maps", indexingMaps);

  // Parse `: result_type`.
  GroupType resultType;
  if (failed(parser.parseColon()) || failed(parser.parseType(resultType))) {
    return failure();
  }
  result.addTypes(resultType);

  // Infer input group type. The result appends 2*numDistDims index types
  // to the input's struct types. Count distributed dims from the maps.
  int64_t numDistDims = 0;
  for (Attribute mapAttr : indexingMaps) {
    AffineMap map = cast<AffineMapAttr>(mapAttr).getValue();
    numDistDims += map.getNumDims();
  }
  int64_t numAppended = 2 * numDistDims;

  ArrayRef<Type> resultStructTypes = resultType.getStructTypes();
  int64_t inputStructSize =
      static_cast<int64_t>(resultStructTypes.size()) - numAppended;

  GroupType inputGroupType;
  if (inputStructSize > 0) {
    inputGroupType = GroupType::get(
        parser.getContext(), resultType.getIds(), resultType.getScope(),
        resultStructTypes.take_front(inputStructSize));
  } else {
    inputGroupType = GroupType::get(parser.getContext(), resultType.getIds(),
                                    resultType.getScope(), {});
  }

  // Resolve group operand.
  if (failed(parser.resolveOperand(groupOperand, inputGroupType,
                                   result.operands))) {
    return failure();
  }

  // Resolve iteration space operands (all index type).
  IndexType indexType = IndexType::get(parser.getContext());
  for (OpAsmParser::UnresolvedOperand &operand : iterSpaceOperands) {
    if (failed(parser.resolveOperand(operand, indexType, result.operands))) {
      return failure();
    }
  }

  // Parse optional attr-dict.
  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }

  return success();
}

void DefineWorkspaceOp::print(OpAsmPrinter &p) {
  p << " " << getGroup() << " iteration_space(";
  int64_t dynamicIdx = 0;
  llvm::interleaveComma(getStaticIterationSpace(), p, [&](int64_t val) {
    if (ShapedType::isDynamic(val)) {
      p << getIterationSpace()[dynamicIdx++];
    } else {
      p << val;
    }
  });
  p << ") indexing_maps = " << getIndexingMapsAttr();
  p << " : " << getResult().getType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"static_iteration_space", "indexing_maps"});
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
