// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateOps.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::iree_compiler::IREE::Template {

//===----------------------------------------------------------------------===//
// Custom parsing/printing for FuncOp signature
//===----------------------------------------------------------------------===//

static ParseResult parseTemplateFuncSignature(OpAsmParser &parser,
                                              TypeAttr &functionType,
                                              Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> argTypes;
  SmallVector<Type> resultTypes;

  // Parse optional argument list: (%arg0: type0, %arg1: type1).
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseArgumentList(args, OpAsmParser::Delimiter::None,
                                        /*allowType=*/true))) {
      return failure();
    }
    if (failed(parser.parseRParen())) {
      return failure();
    }
    for (const auto &arg : args) {
      argTypes.push_back(arg.type);
    }
  }

  // Parse optional result types: -> (type0, type1) or -> type0.
  if (succeeded(parser.parseOptionalArrow())) {
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parser.parseTypeList(resultTypes))) {
        return failure();
      }
      if (failed(parser.parseRParen())) {
        return failure();
      }
    } else {
      Type singleResult;
      if (failed(parser.parseType(singleResult))) {
        return failure();
      }
      resultTypes.push_back(singleResult);
    }
  }

  // Build function type.
  FunctionType fnType =
      FunctionType::get(parser.getContext(), argTypes, resultTypes);
  functionType = TypeAttr::get(fnType);

  // Parse the region.
  if (failed(parser.parseRegion(region, args))) {
    return failure();
  }

  return success();
}

static void printTemplateFuncSignature(OpAsmPrinter &printer, FuncOp op,
                                       TypeAttr functionType, Region &region) {
  FunctionType fnType = cast<FunctionType>(functionType.getValue());

  // Print arguments.
  if (!region.empty() && !region.front().getArguments().empty()) {
    printer << "(";
    llvm::interleaveComma(
        region.front().getArguments(), printer,
        [&](BlockArgument arg) { printer.printRegionArgument(arg); });
    printer << ")";
  }

  // Print result types.
  ArrayRef<Type> results = fnType.getResults();
  if (!results.empty()) {
    printer << " -> ";
    if (results.size() > 1) {
      printer << "(";
    }
    llvm::interleaveComma(results, printer, [&](Type t) { printer << t; });
    if (results.size() > 1) {
      printer << ")";
    }
  }

  // Print space before region.
  printer << " ";
  printer.printRegion(region, /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verify() {
  // Verify each implementation block ends with template.return.
  for (Block &block : getImplementations()) {
    if (block.empty() || !isa<ReturnOp>(block.getTerminator())) {
      return emitOpError(
          "implementation blocks must be terminated with template.return");
    }
  }

  // Main region single-block constraint enforced by SizedRegion<1> in tablegen.

  // Verify main region ends with template.return.
  Block &mainBlock = getMain().front();
  if (mainBlock.empty() || !isa<ReturnOp>(mainBlock.getTerminator())) {
    return emitOpError("main region must be terminated with template.return");
  }

  // Verify return operand types match instance result types.
  auto returnOp = cast<ReturnOp>(mainBlock.getTerminator());
  TypeRange returnTypes = returnOp.getOperands().getTypes();
  TypeRange resultTypes = getResults().getTypes();
  if (returnTypes != resultTypes) {
    return emitOpError("return operand types ")
           << returnTypes << " must match instance result types "
           << resultTypes;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

LogicalResult FuncOp::verify() {
  // Verify each implementation block ends with a valid terminator.
  for (Block &block : getImplementations()) {
    if (block.empty()) {
      return emitOpError("implementation blocks must have a terminator");
    }
    Operation *terminator = block.getTerminator();
    if (!isa<ReturnOp, UnimplementedOp>(terminator)) {
      return emitOpError("implementation blocks must be terminated with "
                         "template.return or template.unimplemented");
    }
  }

  // Main region single-block constraint enforced by SizedRegion<1> in tablegen.

  // Verify main region ends with template.return.
  Block &mainBlock = getMain().front();
  if (mainBlock.empty() || !isa<ReturnOp>(mainBlock.getTerminator())) {
    return emitOpError("main region must be terminated with template.return");
  }

  // Verify main block argument types match function input types.
  FunctionType fnType = getFunctionType();
  if (mainBlock.getArgumentTypes() != fnType.getInputs()) {
    return emitOpError("main block argument types ")
           << mainBlock.getArgumentTypes()
           << " must match function input types " << fnType.getInputs();
  }

  // Verify return operand types match function result types.
  auto returnOp = cast<ReturnOp>(mainBlock.getTerminator());
  if (returnOp.getOperands().getTypes() != fnType.getResults()) {
    return emitOpError("return operand types ")
           << returnOp.getOperands().getTypes()
           << " must match function result types " << fnType.getResults();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

LogicalResult BranchOp::verify() {
  // Find the enclosing InstanceOp or FuncOp.
  Region *implRegion = nullptr;

  if (auto instanceOp = (*this)->getParentOfType<InstanceOp>()) {
    implRegion = &instanceOp.getImplementations();
  } else if (auto funcOp = (*this)->getParentOfType<FuncOp>()) {
    implRegion = &funcOp.getImplementations();
  } else {
    return emitOpError(
        "must be nested within template.instance or template.func");
  }

  // Verify the implementations region is not empty.
  if (implRegion->empty()) {
    return emitOpError(
        "enclosing op has no implementation blocks to branch to");
  }

  // Verify block index is within bounds.
  int64_t blockIndex = getBlockIndex();
  int64_t numBlocks = std::distance(implRegion->begin(), implRegion->end());
  if (blockIndex < 0 || blockIndex >= numBlocks) {
    return emitOpError("block index ")
           << blockIndex << " out of bounds for implementations region with "
           << numBlocks << " blocks";
  }

  // Get the target block.
  Block &targetBlock = *std::next(implRegion->begin(), blockIndex);

  // Verify argument types match the target block's argument types.
  TypeRange branchArgTypes = getArguments().getTypes();
  TypeRange blockArgTypes = targetBlock.getArgumentTypes();
  if (branchArgTypes != blockArgTypes) {
    return emitOpError("branch argument types ")
           << branchArgTypes << " must match target block argument types "
           << blockArgTypes;
  }

  // Verify result types match the target block's terminator return types.
  Operation *terminator = targetBlock.getTerminator();
  TypeRange branchResultTypes = getResults().getTypes();
  if (auto returnOp = dyn_cast<ReturnOp>(terminator)) {
    TypeRange returnTypes = returnOp.getOperands().getTypes();
    if (branchResultTypes != returnTypes) {
      return emitOpError("branch result types ")
             << branchResultTypes << " must match target block return types "
             << returnTypes;
    }
  } else if (auto unimplOp = dyn_cast<UnimplementedOp>(terminator)) {
    TypeRange unimplTypes = unimplOp.getResults().getTypes();
    if (branchResultTypes != unimplTypes) {
      return emitOpError("branch result types ")
             << branchResultTypes
             << " must match target block unimplemented types " << unimplTypes;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp parsing
//===----------------------------------------------------------------------===//

/// Parse type bindings: <type, [type, type], [], type>
static ParseResult
parseTypeBindings(OpAsmParser &parser,
                  SmallVectorImpl<SmallVector<Type>> &typeBindings) {
  if (failed(parser.parseLess())) {
    return failure();
  }

  // Handle empty bindings case: <>
  if (succeeded(parser.parseOptionalGreater())) {
    return success();
  }

  auto parseBinding = [&]() -> ParseResult {
    // Check for list syntax: [type, type, ...]
    if (succeeded(parser.parseOptionalLSquare())) {
      SmallVector<Type> types;
      // Handle empty list: []
      if (succeeded(parser.parseOptionalRSquare())) {
        typeBindings.push_back(std::move(types));
        return success();
      }
      // Parse non-empty list
      if (failed(parser.parseTypeList(types))) {
        return failure();
      }
      if (failed(parser.parseRSquare())) {
        return failure();
      }
      typeBindings.push_back(std::move(types));
      return success();
    }
    // Single type (no brackets)
    Type singleType;
    if (failed(parser.parseType(singleType))) {
      return failure();
    }
    typeBindings.push_back({singleType});
    return success();
  };

  if (failed(parseBinding())) {
    return failure();
  }
  while (succeeded(parser.parseOptionalComma())) {
    if (failed(parseBinding())) {
      return failure();
    }
  }

  if (failed(parser.parseGreater())) {
    return failure();
  }
  return success();
}

ParseResult CallOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse callee symbol
  FlatSymbolRefAttr calleeAttr;
  if (failed(parser.parseCustomAttributeWithFallback(calleeAttr))) {
    return failure();
  }
  result.addAttribute("callee", calleeAttr);

  // Parse type bindings: <...>
  SmallVector<SmallVector<Type>> typeBindings;
  if (failed(parseTypeBindings(parser, typeBindings))) {
    return failure();
  }

  // Set the type bindings property.
  CallOp::Properties &props = result.getOrAddProperties<CallOp::Properties>();
  props.type_bindings.clear();
  for (auto &binding : typeBindings) {
    props.type_bindings.push_back(std::move(binding));
  }

  // Parse operands: (%arg0, %arg1 : type0, type1)
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandTypes;
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseOperandList(operands)) ||
        failed(parser.parseColonTypeList(operandTypes)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  }

  // Parse result types: -> (type0, type1) or -> type0
  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parser.parseTypeList(resultTypes)) ||
          failed(parser.parseRParen())) {
        return failure();
      }
    } else {
      Type singleResult;
      if (failed(parser.parseType(singleResult))) {
        return failure();
      }
      resultTypes.push_back(singleResult);
    }
  }

  // Parse optional implementations region.
  Region *implRegion = result.addRegion();
  OptionalParseResult regionResult = parser.parseOptionalRegion(*implRegion);
  if (regionResult.has_value() && failed(*regionResult)) {
    return failure();
  }

  // Parse optional attributes.
  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }

  // Resolve operands.
  if (failed(parser.resolveOperands(operands, operandTypes,
                                    parser.getCurrentLocation(),
                                    result.operands))) {
    return failure();
  }

  result.addTypes(resultTypes);
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp printing
//===----------------------------------------------------------------------===//

void CallOp::print(OpAsmPrinter &printer) {
  // Print callee.
  printer << " ";
  printer.printSymbolName(getCallee());

  // Print type bindings: <type, [type, type], [], type>.
  printer << "<";
  const llvm::SmallVector<llvm::SmallVector<Type>> &bindings =
      getProperties().type_bindings;
  llvm::interleaveComma(
      bindings, printer, [&](const llvm::SmallVector<Type> &binding) {
        if (binding.size() == 1) {
          // Single type without brackets.
          printer << binding[0];
        } else {
          // Multiple types (or empty) with brackets.
          printer << "[";
          llvm::interleaveComma(binding, printer,
                                [&](Type t) { printer << t; });
          printer << "]";
        }
      });
  printer << ">";

  // Print operands.
  if (!getOperands().empty()) {
    printer << "(";
    llvm::interleaveComma(getOperands(), printer,
                          [&](Value v) { printer << v; });
    printer << " : ";
    llvm::interleaveComma(getOperands().getTypes(), printer);
    printer << ")";
  }

  // Print result types.
  if (!getResults().empty()) {
    printer << " -> ";
    if (getResults().size() > 1) {
      printer << "(";
    }
    llvm::interleaveComma(getResults().getTypes(), printer);
    if (getResults().size() > 1) {
      printer << ")";
    }
  }

  // Print implementations region.
  if (!getImplementations().empty()) {
    printer << " ";
    printer.printRegion(getImplementations(), /*printEntryBlockArgs=*/true,
                        /*printBlockTerminators=*/true);
  }

  // Print attributes (excluding callee which is printed specially).
  printer.printOptionalAttrDict((*this)->getAttrs(), {"callee"});
}

//===----------------------------------------------------------------------===//
// CallOp verification
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verify() {
  // Verify implementation blocks are terminated with template.return.
  for (Block &block : getImplementations()) {
    if (block.empty()) {
      return emitOpError("implementation blocks must have a terminator");
    }
    if (!isa<ReturnOp>(block.getTerminator())) {
      return emitOpError(
          "implementation blocks must be terminated with template.return");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// UnimplementedOp
//===----------------------------------------------------------------------===//

LogicalResult UnimplementedOp::verify() {
  // Verify this op has no predecessors (is the only op in its block).
  Block *block = (*this)->getBlock();
  if (block && &block->front() != getOperation()) {
    return emitOpError("must be the only operation in its block");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void TemplateDialect::registerOperations() {
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::Template

//===----------------------------------------------------------------------===//
// TableGen definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateOps.cpp.inc" // IWYU pragma: keep
