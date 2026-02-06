// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Template/IR/Template.h"
#include "iree/compiler/Codegen/Dialect/Template/IR/TemplateInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Template/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-template-concretize-calls"

namespace mlir::iree_compiler::IREE::Template {

#define GEN_PASS_DEF_CONCRETIZETEMPLATECALLSPASS
#include "iree/compiler/Codegen/Dialect/Template/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type Conversion Patterns for Template Ops
//===----------------------------------------------------------------------===//

/// Pattern to convert template.instance result and region types.
struct ConvertTemplateInstance : public OpConversionPattern<InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Always convert region types - there may be template types in the regions
    // even if result types are already concrete.
    if (failed(rewriter.convertRegionTypes(&op.getMain(), *typeConverter))) {
      return failure();
    }
    if (failed(rewriter.convertRegionTypes(&op.getImplementations(),
                                           *typeConverter))) {
      return failure();
    }

    // Convert result types.
    SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes))) {
      return failure();
    }

    // Only recreate the op if result types changed.
    if (newResultTypes == SmallVector<Type>(op.getResultTypes())) {
      return failure();
    }

    // Create new instance with converted types.
    OperationState state(op.getLoc(), InstanceOp::getOperationName());
    state.addOperands(adaptor.getInputs());
    state.addTypes(newResultTypes);
    state.addRegion();
    state.addRegion();
    InstanceOp newOp = cast<InstanceOp>(rewriter.create(state));

    // Move regions.
    rewriter.inlineRegionBefore(op.getMain(), newOp.getMain(),
                                newOp.getMain().end());
    rewriter.inlineRegionBefore(op.getImplementations(),
                                newOp.getImplementations(),
                                newOp.getImplementations().end());

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

/// Pattern to convert template.return operand types.
struct ConvertTemplateReturn : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Simply use the converted operands from the adaptor.
    // 1:N expansion is handled during the cloning phase before conversion.
    ReturnOp::create(rewriter, op.getLoc(), adaptor.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to convert template.branch result types.
struct ConvertTemplateBranch : public OpConversionPattern<BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert result types.
    SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes))) {
      return failure();
    }

    BranchOp::create(rewriter, op.getLoc(), newResultTypes, op.getBlockIndex(),
                     adaptor.getArguments());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Declaration
//===----------------------------------------------------------------------===//

class ConcretizeTemplateCallsPass
    : public impl::ConcretizeTemplateCallsPassBase<
          ConcretizeTemplateCallsPass> {
public:
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Recursively check if a template.func leads to recursion.
static LogicalResult checkForRecursion(FuncOp funcOp, SymbolTable &symbolTable,
                                       llvm::SetVector<StringRef> &callStack) {
  StringRef name = funcOp.getSymName();

  // Check if we've seen this function before.
  if (callStack.contains(name)) {
    return failure();
  }

  // Add to call stack.
  callStack.insert(name);

  // Check all template.call ops in this function.
  LogicalResult result = success();
  funcOp.walk([&](CallOp nestedCall) {
    if (failed(result)) {
      return;
    }
    auto nestedFunc = symbolTable.lookup<FuncOp>(nestedCall.getCallee());
    if (nestedFunc) {
      result = checkForRecursion(nestedFunc, symbolTable, callStack);
    }
  });

  // Remove from call stack.
  callStack.remove(name);

  return result;
}

/// Convert types in an operation and its nested regions.
static void convertOpTypes(Operation *op, TypeConverter &typeConverter) {
  // Update result types.
  for (OpResult result : op->getResults()) {
    SmallVector<Type> convertedTypes;
    if (succeeded(
            typeConverter.convertType(result.getType(), convertedTypes)) &&
        convertedTypes.size() == 1) {
      result.setType(convertedTypes[0]);
    }
  }
  // Update block argument types in nested regions.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument arg : block.getArguments()) {
        SmallVector<Type> convertedTypes;
        if (succeeded(
                typeConverter.convertType(arg.getType(), convertedTypes)) &&
            convertedTypes.size() == 1) {
          arg.setType(convertedTypes[0]);
        }
      }
    }
  }
  // If this is a nested template.call, convert its type_bindings.
  if (isa<CallOp>(op)) {
    auto nestedCall = cast<CallOp>(op);
    auto bindings = nestedCall.getTypeBindings();
    SmallVector<SmallVector<Type>> newBindings;
    for (const auto &binding : bindings) {
      SmallVector<Type> newBinding;
      for (Type t : binding) {
        SmallVector<Type> converted;
        if (succeeded(typeConverter.convertType(t, converted))) {
          for (Type ct : converted) {
            newBinding.push_back(ct);
          }
        } else {
          newBinding.push_back(t);
        }
      }
      newBindings.push_back(newBinding);
    }
    nestedCall.getProperties().type_bindings = newBindings;
  }
}

/// Fix up template.return ops for 1:N type expansion by unpacking
/// UnrealizedConversionCast ops that pack multiple values into one.
static void fixupReturnOpsFor1NExpansion(InstanceOp instanceOp) {
  instanceOp.walk([&](ReturnOp returnOp) {
    SmallVector<Value> newOperands;
    bool needsUpdate = false;
    for (Value operand : returnOp.getOperands()) {
      if (auto castOp = operand.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() > 1 && castOp.getOutputs().size() == 1) {
          // This is a pack cast - use the unpacked values instead.
          for (Value v : castOp.getInputs()) {
            newOperands.push_back(v);
          }
          needsUpdate = true;
          continue;
        }
      }
      newOperands.push_back(operand);
    }
    if (needsUpdate) {
      OpBuilder returnBuilder(returnOp);
      ReturnOp::create(returnBuilder, returnOp.getLoc(), newOperands);
      returnOp.erase();
    }
  });
}

/// Convert a TemplateCallOpInterface to template.instance.
static LogicalResult
convertTemplateCallInterface(TemplateCallOpInterface callInterface,
                             FuncOp funcOp, SymbolTable &symbolTable) {
  Operation *callOp = callInterface.getOperation();
  MLIRContext *context = callOp->getContext();
  Location loc = callOp->getLoc();
  OpBuilder builder(callOp);

  FlatSymbolRefAttr calleeAttr = callInterface.getCalledSymbol();
  SmallVector<SmallVector<Type>> typeBindings =
      callInterface.getTemplateTypes();

  LLVM_DEBUG(llvm::dbgs() << "Converting call to @" << calleeAttr.getValue()
                          << "\n");

  // a) Setup TypeConverter for template types.
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion(
      [&typeBindings](TypeType templateType, SmallVectorImpl<Type> &results)
          -> std::optional<LogicalResult> {
        int64_t id = templateType.getId();
        if (id < 0 || static_cast<size_t>(id) >= typeBindings.size()) {
          return success(); // Empty expansion.
        }
        ArrayRef<Type> expansion = typeBindings[id];
        for (Type t : expansion) {
          results.push_back(t);
        }
        return success();
      });

  // Add materializations for type conversion.
  typeConverter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });

  // b) Calculate expected types for function inputs.
  FunctionType funcType = funcOp.getFunctionType();
  SmallVector<Type> convertedInputTypes;
  for (Type inputType : funcType.getInputs()) {
    SmallVector<Type> converted;
    if (failed(typeConverter.convertType(inputType, converted))) {
      return callOp->emitOpError("failed to convert input type ") << inputType;
    }
    for (Type t : converted) {
      convertedInputTypes.push_back(t);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Converted input types: "
                          << convertedInputTypes.size() << "\n");

  // Get operands to pass to instance via the interface.
  // This allows ops like process_inner_tile to provide only the operands
  // that correspond to template.func arguments, not metadata like bounds.
  SmallVector<Value> instanceOperands = callInterface.getCallOperands();
  if (instanceOperands.size() != convertedInputTypes.size()) {
    return callOp->emitOpError("template.func expects ")
           << convertedInputTypes.size() << " operands but interface provides "
           << instanceOperands.size();
  }

  // c) Calculate expected result types.
  SmallVector<Type> convertedResultTypes;
  for (Type resultType : funcType.getResults()) {
    SmallVector<Type> converted;
    if (failed(typeConverter.convertType(resultType, converted))) {
      return callOp->emitOpError("failed to convert result type ")
             << resultType;
    }
    for (Type t : converted) {
      convertedResultTypes.push_back(t);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Converted result types: "
                          << convertedResultTypes.size() << "\n");

  // d) Create template.instance with converted types.
  OperationState instanceState(loc, InstanceOp::getOperationName());
  instanceState.addOperands(instanceOperands);
  instanceState.addTypes(convertedResultTypes);
  instanceState.addRegion(); // main
  instanceState.addRegion(); // implementations

  Operation *instanceOperation = builder.create(instanceState);
  InstanceOp instanceOp = cast<InstanceOp>(instanceOperation);

  LLVM_DEBUG(llvm::dbgs() << "  Created instance op\n");

  // e) Create main block and clone funcOp's main region.
  Block *instanceMainBlock = builder.createBlock(&instanceOp.getMain());
  Block &funcMainBlock = funcOp.getMain().front();

  LLVM_DEBUG(
      llvm::dbgs() << "  Func main block has "
                   << funcMainBlock.getNumArguments() << " args and "
                   << std::distance(funcMainBlock.begin(), funcMainBlock.end())
                   << " ops\n");

  // Set up value mapping from func args to call operands.
  IRMapping mapping;
  size_t operandIdx = 0;
  for (auto [i, inputType] : llvm::enumerate(funcType.getInputs())) {
    SmallVector<Type> converted;
    (void)typeConverter.convertType(inputType, converted);
    size_t numConverted = converted.size();

    BlockArgument funcArg = funcMainBlock.getArgument(i);

    if (numConverted == 1) {
      // 1:1 mapping.
      mapping.map(funcArg, instanceOperands[operandIdx]);
    } else if (numConverted > 1) {
      // 1:N mapping - create cast back to original type.
      builder.setInsertionPointToStart(instanceMainBlock);
      ValueRange segment(
          ArrayRef<Value>(instanceOperands).slice(operandIdx, numConverted));
      auto castOp =
          UnrealizedConversionCastOp::create(builder, loc, inputType, segment);
      mapping.map(funcArg, castOp.getResult(0));
    }
    operandIdx += numConverted;
  }

  // Clone operations into instance main block.
  builder.setInsertionPointToEnd(instanceMainBlock);
  for (Operation &op : funcMainBlock.getOperations()) {
    Operation *clonedOp = builder.clone(op, mapping);
    // Update types for template type substitution.
    convertOpTypes(clonedOp, typeConverter);
    // Recursively convert nested ops.
    clonedOp->walk([&](Operation *nested) {
      if (nested != clonedOp) {
        convertOpTypes(nested, typeConverter);
      }
    });
  }

  LLVM_DEBUG(llvm::dbgs() << "  Cloned main block ops\n");

  // f) Clone funcOp's implementations region into instance.
  // Note: We only do 1:1 type conversion here. The dialect conversion in
  // step (i) handles 1:N expansion for template.branch ops and block args.
  // inlineImplementationBlocks() gets expanded types from getTemplateTypes().
  SmallVector<Block *> clonedImplBlocks;
  for (Block &funcImplBlock : funcOp.getImplementations()) {
    // Convert block argument types.
    SmallVector<Type> argTypes;
    for (Type argType : funcImplBlock.getArgumentTypes()) {
      SmallVector<Type> converted;
      if (succeeded(typeConverter.convertType(argType, converted)) &&
          converted.size() == 1) {
        argTypes.push_back(converted[0]);
      } else {
        argTypes.push_back(argType);
      }
    }

    Block *newBlock = builder.createBlock(
        &instanceOp.getImplementations(), instanceOp.getImplementations().end(),
        argTypes, SmallVector<Location>(funcImplBlock.getNumArguments(), loc));
    clonedImplBlocks.push_back(newBlock);

    IRMapping implMapping;
    for (auto [oldArg, newArg] :
         llvm::zip(funcImplBlock.getArguments(), newBlock->getArguments())) {
      implMapping.map(oldArg, newArg);
    }

    builder.setInsertionPointToStart(newBlock);
    for (Operation &op : funcImplBlock.getOperations()) {
      Operation *clonedOp = builder.clone(op, implMapping);
      // Update types for template type substitution.
      convertOpTypes(clonedOp, typeConverter);
      // Recursively convert nested ops.
      clonedOp->walk([&](Operation *nested) {
        if (nested != clonedOp) {
          convertOpTypes(nested, typeConverter);
        }
      });
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Cloned " << clonedImplBlocks.size()
                          << " impl blocks\n");

  // g) Populate unimplemented blocks using the interface.
  SmallVector<Block *> blocksToPopulate;
  for (Block *implBlock : clonedImplBlocks) {
    // Skip empty blocks.
    if (implBlock->empty()) {
      LLVM_DEBUG(llvm::dbgs() << "    Skipping empty impl block\n");
      continue;
    }

    auto unimplOp = dyn_cast<UnimplementedOp>(implBlock->getTerminator());
    if (!unimplOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Impl block has non-unimplemented terminator\n");
      continue; // Already implemented.
    }

    blocksToPopulate.push_back(implBlock);
  }

  // Call the interface to populate the blocks.
  if (!blocksToPopulate.empty()) {
    if (failed(callInterface.inlineImplementationBlocks(builder,
                                                        blocksToPopulate))) {
      return callOp->emitOpError("failed to inline implementation blocks");
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Populated unimplemented blocks\n");

  // h) Fix up template.return ops for 1:N type expansion.
  fixupReturnOpsFor1NExpansion(instanceOp);

  LLVM_DEBUG(llvm::dbgs() << "  Fixed up template.return ops\n");

  // i) Run dialect conversion on the instance to convert SCF ops with
  //    template types and nested template ops.
  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                         tensor::TensorDialect>();
  target.addDynamicallyLegalOp<InstanceOp>([&](InstanceOp op) {
    return typeConverter.isLegal(op.getResultTypes());
  });
  target.addDynamicallyLegalOp<ReturnOp>(
      [&](ReturnOp op) { return typeConverter.isLegal(op.getOperandTypes()); });
  target.addDynamicallyLegalOp<BranchOp>(
      [&](BranchOp op) { return typeConverter.isLegal(op.getResultTypes()); });

  RewritePatternSet patterns(context);
  patterns.add<ConvertTemplateInstance, ConvertTemplateReturn,
               ConvertTemplateBranch>(typeConverter, context);
  // Add SCF type conversion patterns and legality.
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);

  if (failed(applyPartialConversion(instanceOp, target, std::move(patterns)))) {
    return instanceOp.emitOpError("failed to convert template types");
  }

  LLVM_DEBUG(llvm::dbgs() << "  Applied type conversion\n");

  // j) Replace uses and erase original call.
  callOp->replaceAllUsesWith(instanceOp.getResults());
  callOp->erase();

  LLVM_DEBUG(llvm::dbgs() << "  Conversion complete\n");

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void ConcretizeTemplateCallsPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  SymbolTable symbolTable(moduleOp);

  LLVM_DEBUG(llvm::dbgs() << "=== ConcretizeTemplateCallsPass ===\n");

  // Process TemplateCallOpInterface ops iteratively until none remain.
  bool changed = true;
  while (changed) {
    changed = false;

    // Collect all ops implementing TemplateCallOpInterface not inside
    // template.func.
    SmallVector<TemplateCallOpInterface> callsToProcess;
    moduleOp.walk([&](Operation *op) {
      if (op->getParentOfType<FuncOp>()) {
        return;
      }
      if (auto callInterface = dyn_cast<TemplateCallOpInterface>(op)) {
        callsToProcess.push_back(callInterface);
      }
    });

    LLVM_DEBUG(llvm::dbgs()
               << "Found " << callsToProcess.size() << " calls to process\n");

    if (callsToProcess.empty()) {
      break;
    }

    for (TemplateCallOpInterface callInterface : callsToProcess) {
      Operation *op = callInterface.getOperation();
      // Skip if already erased by a previous conversion.
      if (!op->getParentOp()) {
        continue;
      }

      // Look up the callee.
      FlatSymbolRefAttr calleeAttr = callInterface.getCalledSymbol();
      auto funcOp = symbolTable.lookup<FuncOp>(calleeAttr.getValue());
      if (!funcOp) {
        op->emitOpError("cannot find template function '")
            << calleeAttr.getValue() << "'";
        return signalPassFailure();
      }

      // Check for recursion using call stack detection.
      llvm::SetVector<StringRef> callStack;
      if (failed(checkForRecursion(funcOp, symbolTable, callStack))) {
        op->emitOpError("recursive template call detected");
        return signalPassFailure();
      }

      // Convert the call.
      if (failed(convertTemplateCallInterface(callInterface, funcOp,
                                              symbolTable))) {
        return signalPassFailure();
      }

      changed = true;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "=== Pass complete ===\n");
}

} // namespace

} // namespace mlir::iree_compiler::IREE::Template
