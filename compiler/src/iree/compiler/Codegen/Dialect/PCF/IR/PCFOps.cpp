// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include <numeric>
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
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

  // AllocOp must be inside an initializer-like region: either the symbol
  // region of a NamespaceOpInterface op, or the body of an InitSubscopeOp.
  Region *parentRegion = getOperation()->getParentRegion();
  if (!parentRegion) {
    return emitOpError("must be inside a region");
  }
  Operation *parentOp = parentRegion->getParentOp();
  if (isa<InitSubscopeOp>(parentOp)) {
    // InitSubscopeOp's body is a valid allocation context.
  } else if (auto nsOp =
                 dyn_cast_if_present<NamespaceOpInterface>(parentOp)) {
    if (parentRegion != &nsOp.getSymbolRegion()) {
      return emitOpError(
          "must be in the initializer region of the enclosing namespace op");
    }
  } else {
    return emitOpError("must be inside an initializer or init_subscope region");
  }

  return success();
}

SmallVector<OpFoldResult> AllocOp::getMixedSizes() {
  Builder b(getContext());
  return getMixedValues(getResultType().getShape(), getDynamicSizes(), b);
}

//===----------------------------------------------------------------------===//
// ToSrefOp
//===----------------------------------------------------------------------===//

LogicalResult ToSrefOp::verify() {
  ShapedType inputType = cast<ShapedType>(getInput().getType());
  ShapedRefType resultType = getResultType();

  if (inputType.getShape() != resultType.getShape()) {
    return emitOpError("result sref shape ")
           << resultType << " does not match input shape " << inputType;
  }
  if (inputType.getElementType() != resultType.getElementType()) {
    return emitOpError("result sref element type ")
           << resultType.getElementType()
           << " does not match input element type "
           << inputType.getElementType();
  }

  return success();
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
  int64_t numReadonly = op.getNumReadonlyRefs();
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
      numLeadingArgs + numReadonly + numResults + numIndexBodyArgs) {
    return op.emitOpError("expected region to have |numLeadingArgs| + "
                          "|numReadonlyRefs| + |numIndexArgs| + |numResults| "
                          "total arguments");
  }

  for (BlockArgument countArg : indexArgs) {
    if (!countArg.getType().isIndex()) {
      return op.emitOpError(
          "expected index type for thread count/id region arguments");
    }
  }

  PCF::ScopeAttrInterface scope = op.getScope();

  // Verify readonly sref args.
  for (BlockArgument refArg : op.getReadonlyRefArgs()) {
    auto srefType = dyn_cast<PCF::ShapedRefType>(refArg.getType());
    if (!srefType || srefType.getScope() != scope) {
      return op.emitOpError(
                 "expected readonly region argument to be of type !pcf.sref "
                 "with scope ")
             << scope;
    }
  }

  // Verify readwrite sref args.
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

/// Parses the execute body for pcf.generic and pcf.loop ops.
/// Handles both readonly (`<-`) and readwrite (`=`) bindings.
static ParseResult parseParallelExecutionBody(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &readonlyInits,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inits,
    SmallVectorImpl<Type> &initTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicSizes,
    SmallVectorImpl<Type> &resultTypes, SmallVectorImpl<bool> &isTied,
    Region &body, int64_t &numLeadingArgs, int64_t &numReadonlyRefs,
    bool parseOptionalLeadingArgs) {
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

  SmallVector<OpAsmParser::Argument> readonlyRefArgs;
  SmallVector<OpAsmParser::Argument> readwriteRefArgs;
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      OpAsmParser::Argument refArg;
      SMLoc argLoc = parser.getCurrentLocation();
      if (failed(parser.parseArgument(refArg, /*allowType=*/false,
                                      /*allowAttrs=*/true))) {
        return parser.emitError(argLoc, "failed to parse region ref argument");
      }

      // "<-" is parsed as two tokens: '<' then '-'.
      if (succeeded(parser.parseOptionalLess())) {
        if (parser.parseMinus()) {
          return failure();
        }
        readonlyRefArgs.push_back(refArg);
        readonlyInits.emplace_back();
        SMLoc initLoc = parser.getCurrentLocation();
        if (failed(parser.parseOperand(readonlyInits.back()))) {
          return parser.emitError(initLoc,
                                  "failed to parse readonly init operand");
        }
      } else if (succeeded(parser.parseOptionalEqual())) {
        // Readwrite tied init.
        readwriteRefArgs.push_back(refArg);
        inits.emplace_back();
        SMLoc initLoc = parser.getCurrentLocation();
        if (failed(parser.parseOperand(inits.back()))) {
          return parser.emitError(initLoc, "failed to parse tied init operand");
        }
        isTied.push_back(true);
      } else {
        // Readwrite untied.
        readwriteRefArgs.push_back(refArg);
        isTied.push_back(false);
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }

  numReadonlyRefs = readonlyRefArgs.size();

  SMLoc indexArgsLoc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::Argument> indexArgs;
  if (failed(parser.parseArgumentList(
          indexArgs, /*delimiter=*/OpAsmParser::Delimiter::Square,
          /*allowType=*/true, /*allowAttrs=*/true))) {
    return parser.emitError(indexArgsLoc,
                            "failed to parse index arguments list");
  }

  // Combine all ref args for type parsing.
  int64_t numRefs = readonlyRefArgs.size() + readwriteRefArgs.size();

  // If there is at least one region arg the arg types and op result types need
  // to be parsed.
  if (numRefs != 0) {
    if (failed(parser.parseColon())) {
      return failure();
    }

    // Parse "(<type list>)" directly into the type fields of ref args.
    SmallVector<Type> srefTypes;
    SMLoc refTypesLoc = parser.getCurrentLocation();
    if (failed(parser.parseCommaSeparatedList(
            OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
              Type ty;
              if (failed(parser.parseType(ty))) {
                return failure();
              }
              srefTypes.push_back(ty);
              return success();
            }))) {
      return parser.emitError(refTypesLoc,
                              "failed to parse region ref argument types");
    }
    if (srefTypes.size() != static_cast<size_t>(numRefs)) {
      return parser.emitError(refTypesLoc,
                              "sref type count does not match ref arg count");
    }

    // Assign sref types to readonly ref args.
    for (auto [i, arg] : llvm::enumerate(readonlyRefArgs)) {
      arg.type = srefTypes[i];
    }
    // Assign sref types to readwrite ref args.
    for (auto [i, arg] : llvm::enumerate(readwriteRefArgs)) {
      arg.type = srefTypes[readonlyRefArgs.size() + i];
    }

    // Parse optional result types "-> (types)".
    if (!readwriteRefArgs.empty()) {
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
  }

  // The stored argument order is:
  // (leading args) (readonly refs) (readwrite refs) (index args).
  SmallVector<OpAsmParser::Argument> args;
  args.append(regionLeadingArgs);
  args.append(readonlyRefArgs);
  args.append(readwriteRefArgs);
  args.append(indexArgs);
  return parser.parseRegion(body, args, /*enableNameShadowing=*/false);
}

/// Prints the execute body for pcf.generic and pcf.loop ops.
/// Handles both readonly (`<-`) and readwrite (`=`) bindings.
static void printParallelExecutionBody(
    OpAsmPrinter &p, Operation *op, OperandRange readonlyInits,
    OperandRange inits, TypeRange initTypes, OperandRange dynamicSizes,
    TypeRange resultTypes, ArrayRef<bool> isTied, Region &body,
    int64_t numLeadingArgs, int64_t numReadonlyRefs,
    bool printOptionalLeadingArgs) {
  if (printOptionalLeadingArgs && numLeadingArgs > 0) {
    p << " -> (";
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
  int64_t numIndexArgs =
      body.getNumArguments() - numResults - numReadonlyRefs - numLeadingArgs;
  MutableArrayRef<BlockArgument> threadCountArgRange =
      body.getArguments().take_back(numIndexArgs);
  MutableArrayRef<BlockArgument> readonlyRefArgRange =
      body.getArguments().slice(numLeadingArgs, numReadonlyRefs);
  MutableArrayRef<BlockArgument> readwriteRefArgRange =
      body.getArguments().slice(numLeadingArgs + numReadonlyRefs, numResults);

  int64_t numRefs = numReadonlyRefs + numResults;
  if (numRefs != 0) {
    p << "(";
    int64_t currReadonlyInit = 0;
    int64_t currReadwriteInit = 0;
    for (int64_t i = 0, e = numRefs; i < e; ++i) {
      if (i > 0) {
        p << ", ";
      }
      if (i < numReadonlyRefs) {
        p << readonlyRefArgRange[i] << " <- ";
        p << readonlyInits[currReadonlyInit];
        ++currReadonlyInit;
      } else {
        int64_t rwIdx = i - numReadonlyRefs;
        p << readwriteRefArgRange[rwIdx];
        if (isTied[rwIdx]) {
          p << " = ";
          p << inits[currReadwriteInit];
          ++currReadwriteInit;
        }
      }
    }
    p << ")";
  }
  p << "[";
  llvm::interleaveComma(threadCountArgRange, p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << "]";

  // Now print the function type.
  if (numRefs != 0) {
    p.printNewline();
    // Whitespace to line up parentheses.
    //   |--execute(
    //   |--_____: (
    p << "       : (";
    SmallVector<BlockArgument> allRefArgs;
    allRefArgs.append(readonlyRefArgRange.begin(), readonlyRefArgRange.end());
    allRefArgs.append(readwriteRefArgRange.begin(), readwriteRefArgRange.end());
    llvm::interleaveComma(allRefArgs, p,
                          [&](BlockArgument arg) { p << arg.getType(); });
    p << ")";

    if (numResults > 0) {
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
      // Readonly refs only, no results.
      p << " ";
    }
  } else {
    // Print a space before the region brace if there are no refs at all.
    p << " ";
  }
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

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
  for (Value v : getReadonlyRefArgs()) {
    setNameFn(v, "ref");
  }
  for (Value v : getRegionRefArgs()) {
    setNameFn(v, "ref");
  }
}

ParseResult GenericOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse optional "sync true".
  bool syncOnReturn = false;
  if (succeeded(parser.parseOptionalKeyword("sync"))) {
    if (parser.parseKeyword("true")) {
      return failure();
    }
    syncOnReturn = true;
  }

  // Parse "scope" "(" #attr ")".
  ScopeAttrInterface scope;
  if (parser.parseKeyword("scope") || parser.parseLParen() ||
      parser.parseAttribute(scope) || parser.parseRParen()) {
    return failure();
  }
  result.addAttribute(GenericOp::getScopeAttrName(result.name), scope);

  // Parse optional initializer region + leading args binding.
  Region *initializer = result.addRegion();
  if (succeeded(parser.parseOptionalKeyword("initialize"))) {
    if (parser.parseRegion(*initializer)) {
      return failure();
    }
  }

  // Parse the execute body.
  Region *body = result.addRegion();
  SmallVector<OpAsmParser::UnresolvedOperand> readonlyInits;
  SmallVector<OpAsmParser::UnresolvedOperand> inits;
  SmallVector<Type> initTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizes;
  SmallVector<Type> resultTypes;
  SmallVector<bool> isTied;
  int64_t numLeadingArgs = 0;
  int64_t numReadonlyRefs = 0;
  if (failed(parseParallelExecutionBody(parser, readonlyInits, inits, initTypes,
                                        dynamicSizes, resultTypes, isTied,
                                        *body, numLeadingArgs, numReadonlyRefs,
                                        /*parseOptionalLeadingArgs=*/true))) {
    return failure();
  }

  // Infer number of index args from the trailing index-typed block args.
  int64_t numIndexArgs = 0;
  for (BlockArgument bbArg :
       llvm::reverse(body->getArguments().drop_front(numLeadingArgs))) {
    if (!bbArg.getType().isIndex()) {
      break;
    }
    ++numIndexArgs;
  }

  // Parse optional prop-dict and attr-dict.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Resolve readonly init operands. Shape and element type are inferred from
  // sref types; the init may be either a tensor or a memref (after
  // bufferization), so try tensor first and fall back to memref. We use a
  // ScopedDiagnosticHandler to suppress the error from the tensor attempt.
  ArrayRef<BlockArgument> readonlyArgs =
      body->getArguments().slice(numLeadingArgs, numReadonlyRefs);
  for (int64_t i = 0, e = numReadonlyRefs; i < e; ++i) {
    ShapedRefType srefType = cast<ShapedRefType>(readonlyArgs[i].getType());
    Type tensorType =
        RankedTensorType::get(srefType.getShape(), srefType.getElementType());
    {
      // Suppress the diagnostic if tensor resolution fails.
      ScopedDiagnosticHandler diagHandler(parser.getContext(),
                                          [](Diagnostic &) { return success(); });
      if (succeeded(parser.resolveOperand(readonlyInits[i], tensorType,
                                          result.operands))) {
        continue;
      }
    }
    Type memrefType =
        MemRefType::get(srefType.getShape(), srefType.getElementType());
    if (parser.resolveOperand(readonlyInits[i], memrefType,
                              result.operands)) {
      return failure();
    }
  }

  // Resolve readwrite init operands.
  if (parser.resolveOperands(inits, initTypes, parser.getCurrentLocation(),
                             result.operands)) {
    return failure();
  }

  // Resolve dynamic size operands.
  SmallVector<Type> indexTypes(dynamicSizes.size(),
                               parser.getBuilder().getIndexType());
  if (parser.resolveOperands(dynamicSizes, indexTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  result.addTypes(resultTypes);

  // Set properties.
  Properties &props = result.getOrAddProperties<Properties>();
  props.setOperandSegmentSizes({static_cast<int32_t>(readonlyInits.size()),
                                static_cast<int32_t>(inits.size()),
                                static_cast<int32_t>(dynamicSizes.size())});
  props.setIsTied(isTied);
  props.setSyncOnReturn(syncOnReturn);
  props.setNumIndexArgs(numIndexArgs);
  props.setNumLeadingArgs(numLeadingArgs);
  props.setNumReadonlyRefs(numReadonlyRefs);

  return success();
}

void GenericOp::print(OpAsmPrinter &p) {
  if (getSyncOnReturn()) {
    p << " sync true";
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

  printParallelExecutionBody(p, getOperation(), getReadonlyInits(), getInits(),
                             getInits().getTypes(), getDynamicSizes(),
                             getResultTypes(), getIsTied(), getRegion(),
                             getNumLeadingArgs(), getNumReadonlyRefs(),
                             /*printOptionalLeadingArgs=*/true);

  SmallVector<StringRef> elidedAttrs = {getScopeAttrName().getValue(),
                                        getOperandSegmentSizesAttrName()};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
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
  // No readonly inits for this builder.
  result.addOperands(inits);
  result.addOperands(dynamicSizes);
  result.addTypes(resultTypes);

  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setOperandSegmentSizes(
      {/*readonlyInits=*/0, static_cast<int32_t>(inits.size()),
       static_cast<int32_t>(dynamicSizes.size())});
  inherentAttrs.setIsTied(isTied);
  inherentAttrs.setSyncOnReturn(syncOnReturn);
  inherentAttrs.setNumIndexArgs(2 * numIterators);
  inherentAttrs.setNumReadonlyRefs(0);

  // Add the initializer region.
  result.addRegion();

  // Add the main region.
  Region *region = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(region);
  Block &entryBlock = region->front();

  // Add block arguments.

  // Readwrite sref args.
  for (Type resultType : resultTypes) {
    ShapedType shapedType = cast<ShapedType>(resultType);
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

void GenericOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                      ScopeAttrInterface scope, ValueRange readonlyInits,
                      ValueRange inits, int64_t numIterators,
                      bool syncOnReturn) {
  SmallVector<bool> isTied(inits.size(), true);
  SmallVector<Type> resultTypes =
      llvm::map_to_vector(inits, [](Value v) -> Type { return v.getType(); });

  result.addAttribute(GenericOp::getScopeAttrName(result.name), scope);
  result.addOperands(readonlyInits);
  result.addOperands(inits);
  result.addTypes(resultTypes);

  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setOperandSegmentSizes(
      {static_cast<int32_t>(readonlyInits.size()),
       static_cast<int32_t>(inits.size()),
       /*dynamicSizes=*/0});
  inherentAttrs.setIsTied(isTied);
  inherentAttrs.setSyncOnReturn(syncOnReturn);
  inherentAttrs.setNumIndexArgs(2 * numIterators);
  inherentAttrs.setNumReadonlyRefs(readonlyInits.size());

  // Add the initializer region.
  result.addRegion();

  // Add the main region.
  Region *region = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(region);
  Block &entryBlock = region->front();

  // Readonly sref args — no sync scope needed since they are never written.
  for (Value init : readonlyInits) {
    ShapedType shapedType = cast<ShapedType>(init.getType());
    entryBlock.addArgument(
        PCF::ShapedRefType::get(b.getContext(), shapedType.getShape(),
                                shapedType.getElementType(), scope),
        result.location);
  }

  // Readwrite sref args (with SyncOnReturn semantics).
  Attribute syncScope = PCF::SyncOnReturnAttr::get(b.getContext());
  for (Type resultType : resultTypes) {
    ShapedType shapedType = cast<ShapedType>(resultType);
    entryBlock.addArgument(
        PCF::ShapedRefType::get(b.getContext(), shapedType.getShape(),
                                shapedType.getElementType(), scope, syncScope),
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
  int64_t rangeBegin = getNumLeadingArgs() + getNumReadonlyRefs();
  int64_t rangeEnd = rangeBegin + getNumResults();
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
  for (Value v : getReadonlyRefArgs()) {
    setNameFn(v, "ref");
  }
  for (Value v : getRegionRefArgs()) {
    setNameFn(v, "ref");
  }
}

ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse optional "sync true".
  bool syncOnReturn = false;
  if (succeeded(parser.parseOptionalKeyword("sync"))) {
    if (parser.parseKeyword("true")) {
      return failure();
    }
    syncOnReturn = true;
  }

  // Parse "scope" "(" #attr ")".
  ScopeAttrInterface scope;
  if (parser.parseKeyword("scope") || parser.parseLParen() ||
      parser.parseAttribute(scope) || parser.parseRParen()) {
    return failure();
  }
  result.addAttribute(LoopOp::getScopeAttrName(result.name), scope);

  // Parse "count" "(" operands ")".
  SmallVector<OpAsmParser::UnresolvedOperand> countOperands;
  if (parser.parseKeyword("count") || parser.parseLParen() ||
      parser.parseOperandList(countOperands) || parser.parseRParen()) {
    return failure();
  }

  // Resolve count operands as index types.
  SmallVector<Type> countTypes(countOperands.size(),
                               parser.getBuilder().getIndexType());
  if (parser.resolveOperands(countOperands, countTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  // Parse the execute body.
  Region *body = result.addRegion();
  SmallVector<OpAsmParser::UnresolvedOperand> readonlyInits;
  SmallVector<OpAsmParser::UnresolvedOperand> inits;
  SmallVector<Type> initTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizes;
  SmallVector<Type> resultTypes;
  SmallVector<bool> isTied;
  int64_t numLeadingArgs = 0;
  int64_t numReadonlyRefs = 0;
  if (failed(parseParallelExecutionBody(parser, readonlyInits, inits, initTypes,
                                        dynamicSizes, resultTypes, isTied,
                                        *body, numLeadingArgs, numReadonlyRefs,
                                        /*parseOptionalLeadingArgs=*/false))) {
    return failure();
  }

  // Parse optional prop-dict and attr-dict.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Resolve readonly init operands. Shape and element type are inferred from
  // sref types; the init may be either a tensor or a memref (after
  // bufferization), so try tensor first and fall back to memref.
  ArrayRef<BlockArgument> readonlyArgs =
      body->getArguments().take_front(numReadonlyRefs);
  for (int64_t i = 0, e = numReadonlyRefs; i < e; ++i) {
    ShapedRefType srefType = cast<ShapedRefType>(readonlyArgs[i].getType());
    Type tensorType =
        RankedTensorType::get(srefType.getShape(), srefType.getElementType());
    if (succeeded(parser.resolveOperand(readonlyInits[i], tensorType,
                                        result.operands))) {
      continue;
    }
    Type memrefType =
        MemRefType::get(srefType.getShape(), srefType.getElementType());
    if (parser.resolveOperand(readonlyInits[i], memrefType,
                              result.operands)) {
      return failure();
    }
  }

  // Resolve readwrite init operands.
  if (parser.resolveOperands(inits, initTypes, parser.getCurrentLocation(),
                             result.operands)) {
    return failure();
  }

  // Resolve dynamic size operands.
  SmallVector<Type> indexTypes(dynamicSizes.size(),
                               parser.getBuilder().getIndexType());
  if (parser.resolveOperands(dynamicSizes, indexTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  result.addTypes(resultTypes);

  // Set properties.
  Properties &props = result.getOrAddProperties<Properties>();
  props.setOperandSegmentSizes({static_cast<int32_t>(countOperands.size()),
                                static_cast<int32_t>(readonlyInits.size()),
                                static_cast<int32_t>(inits.size()),
                                static_cast<int32_t>(dynamicSizes.size())});
  props.setIsTied(isTied);
  props.setSyncOnReturn(syncOnReturn);
  props.setNumReadonlyRefs(numReadonlyRefs);

  return success();
}

void LoopOp::print(OpAsmPrinter &p) {
  if (getSyncOnReturn()) {
    p << " sync true";
  }
  p << " scope(";
  p.printAttribute(getScope());
  p << ") count(";
  llvm::interleaveComma(getCount(), p, [&](Value v) { p << v; });
  p << ")";

  printParallelExecutionBody(p, getOperation(), getReadonlyInits(), getInits(),
                             getInits().getTypes(), getDynamicSizes(),
                             getResultTypes(), getIsTied(), getRegion(),
                             /*numLeadingArgs=*/0, getNumReadonlyRefs(),
                             /*printOptionalLeadingArgs=*/false);

  SmallVector<StringRef> elidedAttrs = {getScopeAttrName().getValue(),
                                        getOperandSegmentSizesAttrName()};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
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
  // No readonly inits for this builder.
  result.addOperands(inits);
  result.addOperands(dynamicSizes);
  result.addTypes(resultTypes);

  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setOperandSegmentSizes(
      {static_cast<int32_t>(count.size()),
       /*readonlyInits=*/0, static_cast<int32_t>(inits.size()),
       static_cast<int32_t>(dynamicSizes.size())});
  inherentAttrs.setIsTied(isTied);
  inherentAttrs.setSyncOnReturn(syncOnReturn);
  inherentAttrs.setNumReadonlyRefs(0);

  Region *region = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(region);
  Block &entryBlock = region->front();

  // Add block arguments.

  // Readwrite sref args.
  for (Type resultType : resultTypes) {
    ShapedType shapedType = cast<ShapedType>(resultType);
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

void LoopOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                   ScopeAttrInterface scope, ValueRange count,
                   ValueRange readonlyInits, ValueRange inits,
                   bool syncOnReturn) {
  SmallVector<bool> isTied(inits.size(), true);
  SmallVector<Type> resultTypes =
      llvm::map_to_vector(inits, [](Value v) -> Type { return v.getType(); });

  result.addAttribute(LoopOp::getScopeAttrName(result.name), scope);
  result.addOperands(count);
  result.addOperands(readonlyInits);
  result.addOperands(inits);
  result.addTypes(resultTypes);

  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setOperandSegmentSizes(
      {static_cast<int32_t>(count.size()),
       static_cast<int32_t>(readonlyInits.size()),
       static_cast<int32_t>(inits.size()),
       /*dynamicSizes=*/0});
  inherentAttrs.setIsTied(isTied);
  inherentAttrs.setSyncOnReturn(syncOnReturn);
  inherentAttrs.setNumReadonlyRefs(readonlyInits.size());

  Region *region = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(region);
  Block &entryBlock = region->front();

  // Readonly sref args — no sync scope needed since they are never written.
  for (Value init : readonlyInits) {
    ShapedType shapedType = cast<ShapedType>(init.getType());
    entryBlock.addArgument(
        PCF::ShapedRefType::get(b.getContext(), shapedType.getShape(),
                                shapedType.getElementType(), scope),
        result.location);
  }

  // Readwrite sref args (with SyncOnReturn semantics).
  Attribute syncScope = PCF::SyncOnReturnAttr::get(b.getContext());
  for (Type resultType : resultTypes) {
    ShapedType shapedType = cast<ShapedType>(resultType);
    entryBlock.addArgument(
        PCF::ShapedRefType::get(b.getContext(), shapedType.getShape(),
                                shapedType.getElementType(), scope, syncScope),
        result.location);
  }

  // Thread count args.
  Type indexType = b.getIndexType();
  int64_t numCountArgs = count.empty() ? 1 : count.size();
  for (int64_t i = 0; i < numCountArgs; ++i) {
    entryBlock.addArgument(indexType, result.location);
  }
}

void LoopOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                   TypeRange resultTypes, ScopeAttrInterface scope,
                   ValueRange count, ValueRange readonlyInits, ValueRange inits,
                   ValueRange dynamicSizes, ArrayRef<bool> isTied,
                   bool syncOnReturn) {
  result.addAttribute(LoopOp::getScopeAttrName(result.name), scope);
  result.addOperands(count);
  result.addOperands(readonlyInits);
  result.addOperands(inits);
  result.addOperands(dynamicSizes);
  result.addTypes(resultTypes);

  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setOperandSegmentSizes(
      {static_cast<int32_t>(count.size()),
       static_cast<int32_t>(readonlyInits.size()),
       static_cast<int32_t>(inits.size()),
       static_cast<int32_t>(dynamicSizes.size())});
  inherentAttrs.setIsTied(isTied);
  inherentAttrs.setSyncOnReturn(syncOnReturn);
  inherentAttrs.setNumReadonlyRefs(readonlyInits.size());

  Region *region = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(region);
  Block &entryBlock = region->front();

  // Readonly sref args — no sync scope needed since they are never written.
  for (Value init : readonlyInits) {
    ShapedType shapedType = cast<ShapedType>(init.getType());
    entryBlock.addArgument(
        PCF::ShapedRefType::get(b.getContext(), shapedType.getShape(),
                                shapedType.getElementType(), scope),
        result.location);
  }

  // Readwrite sref args (with SyncOnReturn semantics).
  Attribute syncScope = PCF::SyncOnReturnAttr::get(b.getContext());
  for (Type resultType : resultTypes) {
    ShapedType shapedType = cast<ShapedType>(resultType);
    entryBlock.addArgument(
        PCF::ShapedRefType::get(b.getContext(), shapedType.getShape(),
                                shapedType.getElementType(), scope, syncScope),
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

  // Resolve readonly init operands. Shape and element type are inferred from
  // sref types; the init may be either a tensor or a memref (after
  // bufferization), so try tensor first and fall back to memref.
  for (int64_t i = 0, e = readonlyRefArgs.size(); i < e; ++i) {
    ShapedRefType srefType = cast<ShapedRefType>(srefTypes[i]);
    Type tensorType =
        RankedTensorType::get(srefType.getShape(), srefType.getElementType());
    if (succeeded(parser.resolveOperand(readonlyInits[i], tensorType,
                                        result.operands))) {
      continue;
    }
    Type memrefType =
        MemRefType::get(srefType.getShape(), srefType.getElementType());
    if (parser.resolveOperand(readonlyInits[i], memrefType,
                              result.operands)) {
      return failure();
    }
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
  props.setNumLeadingArgs(leadingArgs.size());
  props.setNumReadonlyRefs(readonlyRefArgs.size());

  return success();
}

void SharedExecutorOp::print(OpAsmPrinter &p) {
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

  return success();
}

void SharedExecutorOp::build(OpBuilder &b, OperationState &result,
                             ScopeAttrInterface scope,
                             ValueRange readwriteInits) {
  SharedExecutorOp::build(b, result, scope, /*readonlyInits=*/ValueRange(),
                          readwriteInits);
}

void SharedExecutorOp::build(OpBuilder &b, OperationState &result,
                             ScopeAttrInterface scope, ValueRange readonlyInits,
                             ValueRange readwriteInits) {
  result.addAttribute(SharedExecutorOp::getScopeAttrName(result.name), scope);
  result.addOperands(readonlyInits);
  result.addOperands(readwriteInits);

  SmallVector<Type> resultTypes = llvm::map_to_vector(
      readwriteInits, [](Value v) -> Type { return v.getType(); });
  result.addTypes(resultTypes);

  Properties &props = result.getOrAddProperties<Properties>();
  props.setOperandSegmentSizes({static_cast<int32_t>(readonlyInits.size()),
                                static_cast<int32_t>(readwriteInits.size())});
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
  setNameFn(getThreadGroup(), "tg");
}

//===----------------------------------------------------------------------===//
// TileGroupOp
//===----------------------------------------------------------------------===//

ScopeAttrInterface TileGroupOp::getScope() {
  Type sourceType = getSource().getType();
  if (auto tg = dyn_cast<ThreadGroupType>(sourceType)) {
    return tg.getScope();
  }
  return cast<ClusterType>(sourceType).getScope();
}

SmallVector<SmallVector<Value>> TileGroupOp::getSplitPointsPerDim() {
  ArrayRef<int64_t> numSplits = getNumSplitsPerDim();
  OperandRange allSplits = getSplitPoints();
  SmallVector<SmallVector<Value>> result;
  int64_t offset = 0;
  for (int64_t n : numSplits) {
    SmallVector<Value> dimSplits;
    llvm::append_range(dimSplits, allSplits.slice(offset, n));
    result.push_back(std::move(dimSplits));
    offset += n;
  }
  return result;
}

std::optional<StringAttr> TileGroupOp::getNamespaceName() {
  if (StringAttr name = getNsNameAttr()) {
    return name;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// GenericOp - NamespaceOpInterface
//===----------------------------------------------------------------------===//

std::optional<StringAttr> GenericOp::getNamespaceName() {
  return std::nullopt; // Anonymous namespace.
}

Region &GenericOp::getSymbolRegion() { return getInitializer(); }

SmallVector<std::pair<Attribute, OpFoldResult>> GenericOp::getDefinedSymbols() {
  SmallVector<std::pair<Attribute, OpFoldResult>> symbols;
  if (getInitializer().empty()) {
    return symbols;
  }
  for (Operation &op : getInitializer().front()) {
    if (auto symOp = dyn_cast<NamespaceSymbolOpInterface>(&op)) {
      StringAttr name = symOp.getSymbolName();
      NamespacedSymbolAttr nsSym =
          NamespacedSymbolAttr::get(getContext(), ArrayRef<StringAttr>{name});
      symbols.push_back({nsSym, symOp.getSymbolDefinition()});
    }
  }
  return symbols;
}

//===----------------------------------------------------------------------===//
// SharedExecutorOp - NamespaceOpInterface
//===----------------------------------------------------------------------===//

std::optional<StringAttr> SharedExecutorOp::getNamespaceName() {
  return std::nullopt; // Anonymous namespace.
}

Region &SharedExecutorOp::getSymbolRegion() { return getInitializer(); }

SmallVector<std::pair<Attribute, OpFoldResult>>
SharedExecutorOp::getDefinedSymbols() {
  SmallVector<std::pair<Attribute, OpFoldResult>> symbols;
  if (getInitializer().empty()) {
    return symbols;
  }
  for (Operation &op : getInitializer().front()) {
    if (auto symOp = dyn_cast<NamespaceSymbolOpInterface>(&op)) {
      StringAttr name = symOp.getSymbolName();
      NamespacedSymbolAttr nsSym =
          NamespacedSymbolAttr::get(getContext(), ArrayRef<StringAttr>{name});
      symbols.push_back({nsSym, symOp.getSymbolDefinition()});
    }
  }
  return symbols;
}

//===----------------------------------------------------------------------===//
// TileGroupOp - NamespaceOpInterface
//===----------------------------------------------------------------------===//

Region &TileGroupOp::getSymbolRegion() { return getBody(); }

SmallVector<std::pair<Attribute, OpFoldResult>>
TileGroupOp::getDefinedSymbols() {
  SmallVector<std::pair<Attribute, OpFoldResult>> symbols;
  for (BlockArgument arg : getBody().getArguments()) {
    ClusterType ct = cast<ClusterType>(arg.getType());
    NamespacedSymbolAttr id = ct.getId();
    OpFoldResult def = TypeAttr::get(ct);
    symbols.push_back({id, def});
  }
  return symbols;
}

void TileGroupOp::build(OpBuilder &builder, OperationState &result,
                        Value source,
                        ArrayRef<SmallVector<Value>> splitPointsPerDim,
                        ArrayRef<ClusterType> resultClusterTypes,
                        StringAttr nsName) {
  result.addOperands(source);
  SmallVector<Value> flatSplits;
  SmallVector<int64_t> numSplits;
  for (const SmallVector<Value> &dimSplits : splitPointsPerDim) {
    numSplits.push_back(dimSplits.size());
    llvm::append_range(flatSplits, dimSplits);
  }
  result.addOperands(flatSplits);
  result.addAttribute("numSplitsPerDim",
                      builder.getDenseI64ArrayAttr(numSplits));
  if (nsName) {
    result.addAttribute("nsName", nsName);
  }
  Region *body = result.addRegion();
  Block *block = new Block();
  body->push_back(block);
  for (ClusterType ct : resultClusterTypes) {
    block->addArgument(ct, result.location);
  }
}

// Parse format:
//   pcf.shared_executor.tile_group %source split [[%v0, %v1], [%v2], []]
//       (%arg0: !pcf.cluster<...>, %arg1: !pcf.cluster<...>, ...) {
//     ...
//   } : source_type
ParseResult TileGroupOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  if (parser.parseOperand(source)) {
    return failure();
  }

  // Parse optional namespace: ns(name).
  StringAttr nsNameAttr;
  if (succeeded(parser.parseOptionalKeyword("ns"))) {
    if (parser.parseLParen()) {
      return failure();
    }
    StringRef nsNameStr;
    if (failed(parser.parseKeyword(&nsNameStr))) {
      return failure();
    }
    nsNameAttr = StringAttr::get(parser.getContext(), nsNameStr);
    if (parser.parseRParen()) {
      return failure();
    }
  }

  // Parse "split".
  if (parser.parseKeyword("split")) {
    return failure();
  }

  // Parse nested split point list: [[%v0, %v1], [%v2], []].
  SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> splitPointsPerDim;
  SmallVector<int64_t> numSplitsPerDim;

  if (parser.parseLSquare()) {
    return failure();
  }

  // Parse comma-separated inner brackets.
  bool first = true;
  while (true) {
    if (!first) {
      if (failed(parser.parseOptionalComma())) {
        break;
      }
    }
    first = false;

    if (parser.parseLSquare()) {
      return failure();
    }
    SmallVector<OpAsmParser::UnresolvedOperand> dimSplits;
    // Parse optional comma-separated operand list inside brackets.
    if (failed(parser.parseOptionalRSquare())) {
      if (parser.parseOperandList(dimSplits) || parser.parseRSquare()) {
        return failure();
      }
    }
    numSplitsPerDim.push_back(dimSplits.size());
    splitPointsPerDim.push_back(std::move(dimSplits));
  }

  if (parser.parseRSquare()) {
    return failure();
  }

  // Parse block argument list with types: (%name: type, ...).
  SmallVector<OpAsmParser::Argument> blockArgs;
  if (parser.parseArgumentList(blockArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true)) {
    return failure();
  }

  // Parse the region body.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs)) {
    return failure();
  }

  // Parse trailing ": source_type".
  Type sourceType;
  if (parser.parseColon() || parser.parseType(sourceType)) {
    return failure();
  }

  // Resolve the source operand.
  if (parser.resolveOperand(source, sourceType, result.operands)) {
    return failure();
  }

  // Flatten and resolve split point operands.
  Type indexType = parser.getBuilder().getIndexType();
  for (SmallVector<OpAsmParser::UnresolvedOperand> &dimSplits :
       splitPointsPerDim) {
    if (parser.resolveOperands(dimSplits, indexType, result.operands)) {
      return failure();
    }
  }

  // Set the numSplitsPerDim attribute.
  result.addAttribute(
      "numSplitsPerDim",
      parser.getBuilder().getDenseI64ArrayAttr(numSplitsPerDim));

  // Store the namespace name if present.
  if (nsNameAttr) {
    result.addAttribute("nsName", nsNameAttr);
  }

  // Parse optional attr-dict.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void TileGroupOp::print(OpAsmPrinter &p) {
  p << " " << getSource();
  if (StringAttr nsAttr = getNsNameAttr()) {
    p << " ns(" << nsAttr.getValue() << ")";
  }
  p << " split [";

  SmallVector<SmallVector<Value>> splitsPerDim = getSplitPointsPerDim();
  for (int64_t i = 0, e = splitsPerDim.size(); i < e; ++i) {
    if (i > 0) {
      p << ", ";
    }
    p << "[";
    llvm::interleaveComma(splitsPerDim[i], p, [&](Value v) { p << v; });
    p << "]";
  }
  p << "]";

  // Print block arguments with types.
  p.printNewline();
  p << "    (";
  llvm::interleaveComma(getBody().getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ") ";

  // Print the region body without entry block args (already printed above).
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  // Print trailing source type.
  p << " : " << getSource().getType();

  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"numSplitsPerDim", "nsName"});
}

LogicalResult TileGroupOp::verify() {
  Type sourceType = getSource().getType();

  // Source must be threadgroup or cluster.
  ThreadGroupType tgType = dyn_cast<ThreadGroupType>(sourceType);
  ClusterType clusterType = dyn_cast<ClusterType>(sourceType);
  if (!tgType && !clusterType) {
    return emitOpError("source must be !pcf.threadgroup or !pcf.cluster");
  }

  // Source must not have struct elements.
  bool hasStruct =
      tgType ? tgType.hasStructElements() : clusterType.hasStructElements();
  if (hasStruct) {
    return emitOpError("source must not have struct elements");
  }

  // Determine source rank.
  int64_t sourceRank = tgType ? tgType.getScope().getNativeNumProcessorIds()
                              : clusterType.getRank();

  // Number of split dimension lists must match source rank.
  ArrayRef<int64_t> numSplits = getNumSplitsPerDim();
  if (static_cast<int64_t>(numSplits.size()) != sourceRank) {
    return emitOpError("expected ")
           << sourceRank << " split dimension lists but got "
           << numSplits.size();
  }

  // Verify total split point count matches the flat variadic.
  int64_t totalSplits = 0;
  for (int64_t n : numSplits) {
    totalSplits += n;
  }
  if (static_cast<int64_t>(getSplitPoints().size()) != totalSplits) {
    return emitOpError("expected ")
           << totalSplits << " total split points but got "
           << getSplitPoints().size();
  }

  // Expected number of block args: product(numSplits[i] + 1).
  int64_t expectedArgs = 1;
  for (int64_t n : numSplits) {
    expectedArgs *= (n + 1);
  }
  int64_t actualArgs = getBody().getNumArguments();
  if (actualArgs != expectedArgs) {
    return emitOpError("expected ")
           << expectedArgs << " cluster block arguments but got " << actualArgs;
  }

  // All block args must be ClusterType with matching scope.
  ScopeAttrInterface scope = getScope();
  for (BlockArgument arg : getBody().getArguments()) {
    ClusterType ct = dyn_cast<ClusterType>(arg.getType());
    if (!ct) {
      return emitOpError("block argument must be !pcf.cluster");
    }
    if (ct.getScope() != scope) {
      return emitOpError("cluster scope mismatch");
    }
    if (ct.hasStructElements()) {
      return emitOpError("result clusters must not have struct elements");
    }
  }

  // Verify namespace constraints on cluster IDs.
  std::optional<StringAttr> nsName = getNamespaceName();
  llvm::SmallDenseSet<StringAttr> leafNames;

  for (BlockArgument arg : getBody().getArguments()) {
    ClusterType ct = cast<ClusterType>(arg.getType());
    NamespacedSymbolAttr id = ct.getId();

    // Check leaf uniqueness.
    if (!leafNames.insert(id.getLeaf()).second) {
      return emitOpError("duplicate leaf symbol name '")
             << id.getLeaf().getValue() << "' in namespace";
    }

    if (nsName) {
      // Named namespace: ID must be qualified with the namespace name.
      if (id.isLeafOnly()) {
        return emitOpError("cluster ID '")
               << id.getLeaf().getValue()
               << "' must be qualified with namespace name '"
               << nsName->getValue() << "'";
      }
      // First segment must match namespace name.
      if (id.getPath().front() != *nsName) {
        return emitOpError("cluster ID first segment '")
               << id.getPath().front().getValue()
               << "' does not match namespace name '" << nsName->getValue()
               << "'";
      }
    } else {
      // Anonymous namespace: ID must be leaf-only.
      if (!id.isLeafOnly()) {
        std::string fullPath =
            llvm::join(llvm::map_range(id.getPath(),
                                       [](StringAttr s) -> StringRef {
                                         return s.getValue();
                                       }),
                       ".");
        return emitOpError("cluster ID '")
               << fullPath << "' must be leaf-only in anonymous namespace";
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ClusterYieldOp
//===----------------------------------------------------------------------===//

// Parse format:
//   pcf.cluster_yield
//   pcf.cluster_yield %v1, %v2 : type1, type2
ParseResult ClusterYieldOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> valueOperands;
  SmallVector<Type> valueTypes;

  // Try parsing operands: %v0, %v1 : type0, type1.
  OpAsmParser::UnresolvedOperand firstOperand;
  OptionalParseResult optResult = parser.parseOptionalOperand(firstOperand);
  if (optResult.has_value() && succeeded(*optResult)) {
    valueOperands.push_back(firstOperand);
    while (succeeded(parser.parseOptionalComma())) {
      if (parser.parseOperand(valueOperands.emplace_back())) {
        return failure();
      }
    }
    if (parser.parseColonTypeList(valueTypes)) {
      return failure();
    }
  }

  // Resolve operands.
  if (parser.resolveOperands(valueOperands, valueTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  return success();
}

void ClusterYieldOp::print(OpAsmPrinter &p) {
  if (!getValues().empty()) {
    p << " ";
    llvm::interleaveComma(getValues(), p, [&](Value v) { p << v; });
    p << " : ";
    llvm::interleaveComma(getValues().getTypes(), p);
  }
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// RunClusterOp / RunThreadOp
//===----------------------------------------------------------------------===//

/// Shared verifier for RunClusterOp and RunThreadOp.
/// `isClusterMode` is true for RunClusterOp, false for RunThreadOp.
static LogicalResult verifyRunOp(Operation *op, ValueRange sources,
                                 ValueRange rangeValues, Region &body,
                                 Value result, bool isClusterMode) {
  if (sources.empty()) {
    return op->emitOpError("expected at least one source cluster");
  }

  // All sources must be ClusterType.
  SmallVector<ClusterType> sourceTypes;
  for (Value src : sources) {
    ClusterType ct = dyn_cast<ClusterType>(src.getType());
    if (!ct) {
      return op->emitOpError("source must be !pcf.cluster");
    }
    sourceTypes.push_back(ct);
  }

  // All sources must have identical scope and boundsMap.
  ScopeAttrInterface scope = sourceTypes[0].getScope();
  AffineMap boundsMap = sourceTypes[0].getBoundsMap();
  for (int64_t i = 1, e = static_cast<int64_t>(sourceTypes.size()); i < e;
       ++i) {
    if (sourceTypes[i].getScope() != scope) {
      return op->emitOpError("all source clusters must have the same scope");
    }
    if (sourceTypes[i].getBoundsMap() != boundsMap) {
      return op->emitOpError(
          "all source clusters must have the same boundsMap");
    }
  }

  // All source clusters must have the same ID.
  NamespacedSymbolAttr sourceId = sourceTypes[0].getId();
  for (int64_t i = 1, e = static_cast<int64_t>(sourceTypes.size()); i < e;
       ++i) {
    if (sourceTypes[i].getId() != sourceId) {
      return op->emitOpError("all source clusters must have the same ID");
    }
  }

  // rangeValues count must match boundsMap dependent values (dims only).
  // Symbols are implicit scope grid sizes, not SSA operands.
  int64_t expectedRangeValues = boundsMap.getNumDims();
  if (static_cast<int64_t>(rangeValues.size()) != expectedRangeValues) {
    return op->emitOpError("expected ")
           << expectedRangeValues << " range values but got "
           << rangeValues.size();
  }

  // Build expected block arg types.
  SmallVector<Type> expectedArgTypes;
  for (ClusterType ct : sourceTypes) {
    if (isClusterMode) {
      llvm::append_range(expectedArgTypes, ct.getSharedTypes());
    } else {
      llvm::append_range(expectedArgTypes, ct.getPrivateTypes());
    }
  }

  // For RunThreadOp, add thread ID args.
  int64_t numThreadIds = 0;
  if (!isClusterMode) {
    numThreadIds = scope.getNativeNumProcessorIds();
    for (int64_t i = 0; i < numThreadIds; ++i) {
      expectedArgTypes.push_back(IndexType::get(op->getContext()));
    }
  }

  // Verify block arg count and types.
  Block &block = body.front();
  if (block.getNumArguments() != expectedArgTypes.size()) {
    return op->emitOpError("expected ")
           << expectedArgTypes.size() << " block arguments but got "
           << block.getNumArguments();
  }
  for (int64_t i = 0, e = static_cast<int64_t>(expectedArgTypes.size()); i < e;
       ++i) {
    if (block.getArgument(i).getType() != expectedArgTypes[i]) {
      return op->emitOpError("block argument ")
             << i << " type mismatch: expected " << expectedArgTypes[i]
             << " but got " << block.getArgument(i).getType();
    }
  }

  // Verify yield against result.
  ClusterYieldOp yield = cast<ClusterYieldOp>(body.front().getTerminator());
  if (!result) {
    // No result cluster — yield must be empty.
    if (!yield.getValues().empty()) {
      return op->emitOpError(
          "cluster_yield must have no operands when parent has no result");
    }
    return success();
  }

  // Verify result cluster constraints.
  ClusterType resultType = cast<ClusterType>(result.getType());
  if (resultType.getScope() != scope) {
    return op->emitOpError("result cluster scope mismatch");
  }
  if (resultType.getBoundsMap() != boundsMap) {
    return op->emitOpError("result cluster boundsMap mismatch");
  }
  // Result cluster ID must match source cluster ID.
  if (resultType.getId() != sourceId) {
    return op->emitOpError(
        "result cluster ID does not match source cluster ID");
  }
  if (isClusterMode && !resultType.getPrivateTypes().empty()) {
    return op->emitOpError("run_cluster result must not have private types");
  }
  if (!isClusterMode && !resultType.getSharedTypes().empty()) {
    return op->emitOpError("run_thread result must not have shared types");
  }

  // Verify yield types match result.
  ArrayRef<Type> expectedValueTypes = isClusterMode
                                          ? resultType.getSharedTypes()
                                          : resultType.getPrivateTypes();
  if (yield.getValues().getTypes() != expectedValueTypes) {
    return op->emitOpError("yield value types do not match result");
  }

  return success();
}

LogicalResult RunClusterOp::verify() {
  return verifyRunOp(*this, getSources(), getRangeValues(), getBody(),
                     getResult(), /*isClusterMode=*/true);
}

void RunClusterOp::build(OpBuilder &builder, OperationState &result,
                         ValueRange sources, ValueRange rangeValues,
                         ArrayRef<Type> bodyArgTypes, ClusterType resultType) {
  result.addOperands(sources);
  result.addOperands(rangeValues);
  result.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(sources.size()),
                                    static_cast<int32_t>(rangeValues.size())}));
  if (resultType) {
    result.addTypes(resultType);
  }
  Region *body = result.addRegion();
  Block *block = new Block();
  body->push_back(block);
  for (Type t : bodyArgTypes) {
    block->addArgument(t, result.location);
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(block);
  ClusterYieldOp::create(builder, result.location, ValueRange{});
}

LogicalResult RunThreadOp::verify() {
  return verifyRunOp(*this, getSources(), getRangeValues(), getBody(),
                     getResult(), /*isClusterMode=*/false);
}

void RunThreadOp::build(OpBuilder &builder, OperationState &result,
                        ValueRange sources, ValueRange rangeValues,
                        ArrayRef<Type> bodyArgTypes, int64_t numThreadIds,
                        ClusterType resultType) {
  result.addOperands(sources);
  result.addOperands(rangeValues);
  result.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(sources.size()),
                                    static_cast<int32_t>(rangeValues.size())}));
  if (resultType) {
    result.addTypes(resultType);
  }
  Region *body = result.addRegion();
  Block *block = new Block();
  body->push_back(block);
  for (Type t : bodyArgTypes) {
    block->addArgument(t, result.location);
  }
  Type indexType = builder.getIndexType();
  for (int64_t i = 0; i < numThreadIds; ++i) {
    block->addArgument(indexType, result.location);
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(block);
  ClusterYieldOp::create(builder, result.location, ValueRange{});
}

/// Shared parse helper for RunClusterOp and RunThreadOp.
static ParseResult parseRunOp(OpAsmParser &parser, OperationState &result,
                              bool isThreadMode) {
  // Parse "(" source operands ")".
  SmallVector<OpAsmParser::UnresolvedOperand> sourceOperands;
  if (parser.parseLParen() || parser.parseOperandList(sourceOperands) ||
      parser.parseRParen()) {
    return failure();
  }

  // Parse "[" range value operands "]".
  SmallVector<OpAsmParser::UnresolvedOperand> rangeOperands;
  if (parser.parseLSquare() || parser.parseOperandList(rangeOperands) ||
      parser.parseRSquare()) {
    return failure();
  }

  // Parse struct block args: "(" %name: type, ... ")".
  SmallVector<OpAsmParser::Argument> structArgs;
  if (parser.parseArgumentList(structArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true)) {
    return failure();
  }

  // For RunThreadOp, parse thread ID block args: "[" %name: type, ... "]".
  SmallVector<OpAsmParser::Argument> threadIdArgs;
  if (isThreadMode) {
    if (parser.parseArgumentList(threadIdArgs, OpAsmParser::Delimiter::Square,
                                 /*allowType=*/true)) {
      return failure();
    }
  }

  // Combine all block args and parse the region.
  SmallVector<OpAsmParser::Argument> allArgs;
  llvm::append_range(allArgs, structArgs);
  llvm::append_range(allArgs, threadIdArgs);
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, allArgs)) {
    return failure();
  }

  // Parse ": " followed by functional type annotation.
  // "(" source_types ")" optional("-> " result_type).
  if (parser.parseColon()) {
    return failure();
  }

  SmallVector<Type> sourceTypes;
  if (parser.parseLParen() || parser.parseTypeList(sourceTypes) ||
      parser.parseRParen()) {
    return failure();
  }

  // Parse optional result type.
  if (succeeded(parser.parseOptionalArrow())) {
    Type resultType;
    if (parser.parseType(resultType)) {
      return failure();
    }
    result.addTypes(resultType);
  }

  // Resolve source operands.
  if (parser.resolveOperands(sourceOperands, sourceTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  // Resolve range value operands as index.
  Type indexType = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(rangeOperands, indexType,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  // Set operand segment sizes.
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(sourceOperands.size()),
                           static_cast<int32_t>(rangeOperands.size())}));

  // Parse optional attr-dict.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

/// Shared print helper for RunClusterOp and RunThreadOp.
static void printRunOp(OpAsmPrinter &p, Operation *op, ValueRange sources,
                       ValueRange rangeValues, Region &body, Value result,
                       bool isThreadMode) {
  // Print sources.
  p << "(";
  llvm::interleaveComma(sources, p, [&](Value v) { p << v; });
  p << ")";

  // Print range values.
  p << "[";
  llvm::interleaveComma(rangeValues, p, [&](Value v) { p << v; });
  p << "]";

  // Print struct block args.
  Block &block = body.front();
  int64_t numStructArgs = block.getNumArguments();
  if (isThreadMode) {
    ScopeAttrInterface scope =
        cast<ClusterType>(sources.front().getType()).getScope();
    numStructArgs -= scope.getNativeNumProcessorIds();
  }

  p.printNewline();
  p << "    (";
  for (int64_t i = 0; i < numStructArgs; ++i) {
    if (i > 0) {
      p << ", ";
    }
    p.printRegionArgument(block.getArgument(i));
  }
  p << ")";

  // Print thread IDs if RunThreadOp.
  if (isThreadMode) {
    p << "[";
    for (int64_t i = numStructArgs,
                 e = static_cast<int64_t>(block.getNumArguments());
         i < e; ++i) {
      if (i > numStructArgs) {
        p << ", ";
      }
      p.printRegionArgument(block.getArgument(i));
    }
    p << "]";
  }

  p << " ";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  // Print functional type: (source_types) -> result_type.
  p << " : (";
  llvm::interleaveComma(sources.getTypes(), p);
  p << ")";
  if (result) {
    p << " -> " << result.getType();
  }

  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult RunClusterOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseRunOp(parser, result, /*isThreadMode=*/false);
}

void RunClusterOp::print(OpAsmPrinter &p) {
  printRunOp(p, *this, getSources(), getRangeValues(), getBody(), getResult(),
             /*isThreadMode=*/false);
}

ParseResult RunThreadOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseRunOp(parser, result, /*isThreadMode=*/true);
}

void RunThreadOp::print(OpAsmPrinter &p) {
  printRunOp(p, *this, getSources(), getRangeValues(), getBody(), getResult(),
             /*isThreadMode=*/true);
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

void ReadSliceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Reading from an sref is a memory read.
  if (!isa<RankedTensorType>(getSource().getType())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                         SideEffects::DefaultResource::get());
  }
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
// ExpandShapeOp
//===----------------------------------------------------------------------===//

LogicalResult ExpandShapeOp::verify() {
  ShapedRefType srcType = cast<ShapedRefType>(getSrc().getType());
  ShapedRefType resultType = cast<ShapedRefType>(getResult().getType());

  // Scope and element type must match.
  if (srcType.getScope() != resultType.getScope()) {
    return emitOpError("source and result must have the same scope");
  }
  if (srcType.getElementType() != resultType.getElementType()) {
    return emitOpError("source and result must have the same element type");
  }
  if (srcType.getSyncScope() != resultType.getSyncScope()) {
    return emitOpError("source and result must have the same sync scope");
  }

  // Validate reassociation dimensions.
  ArrayAttr reassociation = getReassociation();
  if (static_cast<int64_t>(reassociation.size()) != srcType.getRank()) {
    return emitOpError("reassociation map count (")
           << reassociation.size() << ") must match source rank ("
           << srcType.getRank() << ")";
  }

  // Count total expanded dims.
  int64_t totalExpandedDims = 0;
  for (Attribute group : reassociation) {
    totalExpandedDims += cast<ArrayAttr>(group).size();
  }
  if (totalExpandedDims != resultType.getRank()) {
    return emitOpError("total expanded dimensions (")
           << totalExpandedDims << ") must match result rank ("
           << resultType.getRank() << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

void SubviewOp::build(OpBuilder &b, OperationState &result, Type resultType,
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

void SubviewOp::build(OpBuilder &b, OperationState &result, Type resultType,
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
  bool foldedOffsets =
      succeeded(foldDynamicIndexList(mixedOffsets, /*onlyNonNegative=*/true));
  bool foldedStrides = succeeded(foldDynamicIndexList(mixedStrides));
  if (!foldedOffsets && !foldedStrides) {
    return {};
  }

  OpBuilder builder(getContext());

  // Dispatch back to static/dynamic.
  SmallVector<int64_t> staticOffsets, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicStrides;
  dispatchIndexOpFoldResults(mixedOffsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

  // Only update if something actually changed to avoid infinite notification
  // loops in the greedy pattern driver.
  bool changed = false;
  if (foldedOffsets) {
    SmallVector<int64_t> oldOffsets(getStaticOffsets());
    if (staticOffsets != oldOffsets) {
      setStaticOffsetsAttr(builder.getDenseI64ArrayAttr(staticOffsets));
      getOffsetsMutable().assign(dynamicOffsets);
      changed = true;
    }
  }
  if (foldedStrides) {
    SmallVector<int64_t> oldStrides(getStaticStrides());
    if (staticStrides != oldStrides) {
      setStaticStridesAttr(builder.getDenseI64ArrayAttr(staticStrides));
      getStridesMutable().assign(dynamicStrides);
      changed = true;
    }
  }

  (void)changed;
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

OpFoldResult SubviewOp::fold(FoldAdaptor adaptor) {
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
// InitSubscopeOp
//===----------------------------------------------------------------------===//

ParseResult InitSubscopeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse source operand.
  OpAsmParser::UnresolvedOperand sourceOperand;
  if (parser.parseOperand(sourceOperand)) {
    return failure();
  }

  // Parse body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body)) {
    return failure();
  }

  // Ensure the body has a terminator.
  InitSubscopeOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  // Parse "-> result_type".
  Type resultType;
  if (parser.parseArrow() || parser.parseType(resultType)) {
    return failure();
  }

  // Derive input type from result type: same scope, no struct fields.
  ThreadGroupType resultTgType = dyn_cast<ThreadGroupType>(resultType);
  if (!resultTgType) {
    return parser.emitError(parser.getCurrentLocation(),
                            "result type must be !pcf.threadgroup");
  }
  ThreadGroupType sourceType =
      ThreadGroupType::get(parser.getContext(), resultTgType.getScope(), {});

  // Resolve source operand.
  if (parser.resolveOperand(sourceOperand, sourceType, result.operands)) {
    return failure();
  }

  result.addTypes(resultType);
  return success();
}

void InitSubscopeOp::print(OpAsmPrinter &p) {
  p << " " << getSource() << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p << " -> " << getResult().getType();
}

LogicalResult InitSubscopeOp::verify() {
  ThreadGroupType sourceType = cast<ThreadGroupType>(getSource().getType());
  ThreadGroupType resultType = cast<ThreadGroupType>(getResult().getType());

  // Input threadgroup must have no struct fields.
  if (!sourceType.getStructTypes().empty()) {
    return emitOpError("input threadgroup must have no struct fields");
  }

  // Scopes must match.
  if (sourceType.getScope() != resultType.getScope()) {
    return emitOpError("result scope must match input scope");
  }

  // Yielded types must match result struct field types.
  YieldOp yieldOp = cast<YieldOp>(getBody().front().getTerminator());
  TypeRange yieldedTypes = yieldOp.getOperandTypes();
  ArrayRef<Type> structTypes = resultType.getStructTypes();

  if (yieldedTypes.size() != structTypes.size()) {
    return emitOpError("yielded value count (")
           << yieldedTypes.size()
           << ") does not match result struct field count ("
           << structTypes.size() << ")";
  }

  for (auto [i, pair] : llvm::enumerate(llvm::zip(yieldedTypes, structTypes))) {
    if (std::get<0>(pair) != std::get<1>(pair)) {
      return emitOpError("yielded type ")
             << std::get<0>(pair) << " at index " << i
             << " does not match result struct field type "
             << std::get<1>(pair);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TelescopeOp
//===----------------------------------------------------------------------===//

ParseResult TelescopeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse source operand.
  OpAsmParser::UnresolvedOperand sourceOperand;
  if (parser.parseOperand(sourceOperand)) {
    return failure();
  }

  // Parse "[" thread ID "]".
  OpAsmParser::UnresolvedOperand threadIdOperand;
  if (parser.parseLSquare() || parser.parseOperand(threadIdOperand) ||
      parser.parseRSquare()) {
    return failure();
  }

  // Parse ":" source type.
  Type sourceType;
  if (parser.parseColon() || parser.parseType(sourceType)) {
    return failure();
  }

  // Parse "->" result types.
  SmallVector<Type> resultTypes;
  if (parser.parseArrow()) {
    return failure();
  }

  // Handle parenthesized (multiple) or bare (single) result types.
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseTypeList(resultTypes) || parser.parseRParen()) {
      return failure();
    }
  } else {
    Type singleType;
    if (parser.parseType(singleType)) {
      return failure();
    }
    resultTypes.push_back(singleType);
  }

  // Resolve operands.
  if (parser.resolveOperand(sourceOperand, sourceType, result.operands) ||
      parser.resolveOperand(threadIdOperand,
                            IndexType::get(parser.getContext()),
                            result.operands)) {
    return failure();
  }

  result.addTypes(resultTypes);
  return success();
}

void TelescopeOp::print(OpAsmPrinter &p) {
  p << " " << getSource() << "[" << getThreadId() << "]";
  p << " : " << getSource().getType() << " -> ";
  if (getNumResults() > 1) {
    p << "(";
    llvm::interleaveComma(getResultTypes(), p, [&](Type type) { p << type; });
    p << ")";
  } else {
    p << getResultTypes().front();
  }
}

LogicalResult TelescopeOp::verify() {
  // Must have at least one result.
  if (getNumResults() == 0) {
    return emitOpError("must have at least one result");
  }

  // First result must be a ThreadGroupType with no struct fields.
  ThreadGroupType resultTgType =
      dyn_cast<ThreadGroupType>(getResults().front().getType());
  if (!resultTgType) {
    return emitOpError("first result must be !pcf.threadgroup, got ")
           << getResults().front().getType();
  }
  if (!resultTgType.getStructTypes().empty()) {
    return emitOpError("result threadgroup must have no struct fields");
  }

  ThreadGroupType sourceType = cast<ThreadGroupType>(getSource().getType());
  ArrayRef<Type> structTypes = sourceType.getStructTypes();

  if (structTypes.empty()) {
    // No struct fields: must have exactly one result.
    if (getNumResults() != 1) {
      return emitOpError("source has no struct fields, expected exactly one "
                         "result but got ")
             << getNumResults();
    }
  } else {
    // Has struct fields: remaining results must match.
    int64_t expectedResults = 1 + static_cast<int64_t>(structTypes.size());
    if (getNumResults() != expectedResults) {
      return emitOpError("expected ")
             << expectedResults << " results (1 threadgroup + "
             << structTypes.size() << " struct fields) but got "
             << getNumResults();
    }
    for (int64_t i = 0, e = static_cast<int64_t>(structTypes.size()); i < e;
         ++i) {
      Type resultFieldType = getResults()[i + 1].getType();
      if (resultFieldType != structTypes[i]) {
        return emitOpError("result type ")
               << resultFieldType << " at index " << (i + 1)
               << " does not match source struct field type " << structTypes[i];
      }
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
