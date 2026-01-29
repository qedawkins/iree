// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateOps.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::iree_compiler::IREE::Template {

//===----------------------------------------------------------------------===//
// Custom parsing/printing for FuncOp main region
//===----------------------------------------------------------------------===//

static ParseResult parseTemplateFuncMain(OpAsmParser &parser, Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseArgumentList(args, OpAsmParser::Delimiter::None,
                                        /*allowType=*/true))) {
      return failure();
    }
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }

  if (failed(parser.parseRegion(region, args))) {
    return failure();
  }

  return success();
}

static void printTemplateFuncMain(OpAsmPrinter &printer, FuncOp op,
                                  Region &region) {
  bool hasArgs = !region.empty() && !region.front().getArguments().empty();
  if (hasArgs) {
    // Print args without leading space (format already adds space after
    // sym_name). This removes it with a backspace approach not possible here,
    // so we accept the space before '(' for now.
    printer << "(";
    llvm::interleaveComma(region.front().getArguments(), printer,
                          [&](BlockArgument arg) {
                            printer.printRegionArgument(arg);
                          });
    printer << ") ";
  }
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
           << returnTypes << " must match instance result types " << resultTypes;
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
    return emitOpError("enclosing op has no implementation blocks to branch to");
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
             << branchResultTypes
             << " must match target block return types " << returnTypes;
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
