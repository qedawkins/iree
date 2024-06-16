// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.cpp.inc"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/LoweringConfigEnums.cpp.inc"

static const char kConfigAttrName[] = "lowering_config";
static const char kTranslationInfoAttrName[] = "translation_info";
static const char kCompilationInfoAttrName[] = "compilation_info";

namespace mlir::iree_compiler {

/// Returns an `ArrayAttr` where each element is an `IntegerAttr` of 64-bit
/// integer type whose values is obtained from `values`.
static ArrayAttr getIndexArrayAttr(MLIRContext *context,
                                   ArrayRef<int64_t> values) {
  return ArrayAttr::get(
      context, llvm::map_to_vector(values, [&](int64_t value) -> Attribute {
        return IntegerAttr::get(IndexType::get(context), APInt(64, value));
      }));
}

} // namespace mlir::iree_compiler

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// iree_codegen.export_config
//===----------------------------------------------------------------------===//

LogicalResult
ExportConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<int64_t> workgroupSize) {
  if (workgroupSize.size() > 3) {
    return emitError() << "expected workgroup size to have atmost 3 entries";
  }
  return success();
}

ArrayAttr ExportConfigAttr::getWorkgroupSizeIndexArray() {
  return getIndexArrayAttr(getContext(), getWorkgroupSize());
}

//===----------------------------------------------------------------------===//
// iree_codegen.translation_info
//===----------------------------------------------------------------------===//

TranslationInfoAttr TranslationInfoAttr::get(
    MLIRContext *context, DispatchLoweringPassPipeline passPipeline,
    SymbolRefAttr codegenSpec, ArrayRef<int64_t> workgroupSize,
    std::optional<int64_t> subgroupSize, DictionaryAttr configuration) {
  auto pipelineAttr =
      DispatchLoweringPassPipelineAttr::get(context, passPipeline);
  return get(context, pipelineAttr, codegenSpec, workgroupSize,
             subgroupSize.value_or(int64_t()), configuration);
}

TranslationInfoAttr TranslationInfoAttr::get(
    MLIRContext *context, DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workgroupSize, std::optional<int64_t> subgroupSize,
    DictionaryAttr configuration) {
  auto pipelineAttr =
      DispatchLoweringPassPipelineAttr::get(context, passPipeline);
  return get(context, pipelineAttr, /*codegenSpec=*/SymbolRefAttr(),
             workgroupSize, subgroupSize.value_or(int64_t()), configuration);
}

DispatchLoweringPassPipeline
TranslationInfoAttr::getDispatchLoweringPassPipeline() {
  return getPassPipeline().getValue();
}

LogicalResult TranslationInfoAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    IREE::Codegen::DispatchLoweringPassPipelineAttr passPipeline,
    SymbolRefAttr codegenSpec, ArrayRef<int64_t> workgroupSize,
    int64_t subgroupSize, DictionaryAttr configuration) {
  if (!passPipeline) {
    return emitError() << "missing pass pipeline specification";
  }
  auto passPipelineValue = passPipeline.getValue();
  if (passPipelineValue > IREE::Codegen::DispatchLoweringPassPipeline::None) {
    return emitError() << "invalid pass pipeline value : "
                       << stringifyEnum(passPipeline.getValue());
  }
  auto tdPassPipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::TransformDialectCodegen;
  if (codegenSpec && passPipelineValue != tdPassPipeline) {
    return emitError()
           << "transform dialect codegen spec requires pass pipeline : "
           << stringifyEnum(tdPassPipeline);
  }
  if (workgroupSize.size() > 3) {
    return emitError() << "workgroup size cannot have more than 3 entries";
  }
  if (llvm::any_of(workgroupSize, [](int64_t value) { return value <= 0; })) {
    return emitError() << "workgroup size value has to be greater than zero";
  }
  if (subgroupSize < 0) {
    return emitError() << "subgroup size value cannot be negative";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// iree_codegen.lowering_config_level
//===----------------------------------------------------------------------===//

void LoweringConfigTilingLevelAttr::print(mlir::AsmPrinter &printer) const {
  auto tileInterchange = getInterchange();
  auto printTileSizes = [&] {
    printer << '[';
    if (getScalableFlags().empty()) {
      printer.printStrippedAttrOrType(getSizes());
    } else {
      llvm::interleaveComma(llvm::zip(getSizes(), getScalableFlags()), printer,
                            [&](auto pair) {
                              auto [tileSize, isScalable] = pair;
                              // Wrap scalable sizes in square brackets.
                              if (isScalable)
                                printer << '[';
                              printer << tileSize;
                              if (isScalable)
                                printer << ']';
                            });
    }
    printer << ']';
  };
  if (tileInterchange.empty()) {
    printTileSizes();
  } else {
    printer << "{sizes = ";
    printTileSizes();
    printer << ", interchange = [";
    printer.printStrippedAttrOrType(tileInterchange);
    printer << "]}";
  }
}

Attribute LoweringConfigTilingLevelAttr::parse(mlir::AsmParser &parser,
                                               mlir::Type) {
  auto loc = parser.getCurrentLocation();
  auto parseListOfSizes = [&](SmallVector<bool> *scalableFlags = nullptr,
                              bool prefixChecked =
                                  false) -> FailureOr<SmallVector<int64_t>> {
    if (!prefixChecked && parser.parseLSquare())
      return failure();
    if (parser.parseOptionalRSquare().succeeded()) {
      // Empty list.
      return SmallVector<int64_t>();
    }
    SmallVector<int64_t> sizes;
    bool expectScalableSizes = scalableFlags != nullptr;
    auto listParse =
        parser.parseCommaSeparatedList(AsmParser::Delimiter::None, [&] {
          bool isScalable =
              expectScalableSizes && parser.parseOptionalLSquare().succeeded();
          int64_t size = 0;
          if (parser.parseInteger(size) ||
              (isScalable && parser.parseRSquare()))
            return failure();
          sizes.push_back(size);
          if (scalableFlags)
            scalableFlags->push_back(isScalable);
          return success();
        });
    if (failed(listParse) || parser.parseRSquare())
      return failure();
    return sizes;
  };
  SmallVector<bool> scalableFlags;
  if (parser.parseOptionalLSquare().succeeded()) {
    // Case 1: Simple list of tile sizes, e.g.:
    // [0, [32], 16]
    auto tileSizes = parseListOfSizes(&scalableFlags, /*prefixChecked=*/true);
    if (failed(tileSizes))
      return {};
    return parser.getChecked<LoweringConfigTilingLevelAttr>(
        loc, parser.getContext(), *tileSizes, ArrayRef<int64_t>{},
        scalableFlags);
  }
  // Case 2: sizes and interchange, e.g.:
  // {sizes = [0, [32], 16], interchange = [0, 1, 2]}
  if (parser.parseLBrace() || parser.parseKeyword("sizes") ||
      parser.parseEqual())
    return {};
  auto tileSizes = parseListOfSizes(&scalableFlags);
  if (failed(tileSizes) || parser.parseComma() ||
      parser.parseKeyword("interchange") || parser.parseEqual())
    return {};
  auto tileInterchange = parseListOfSizes();
  if (failed(tileInterchange) || parser.parseRBrace())
    return {};
  return parser.getChecked<LoweringConfigTilingLevelAttr>(
      loc, parser.getContext(), *tileSizes, *tileInterchange, scalableFlags);
}

LogicalResult LoweringConfigTilingLevelAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> tileInterchange, ArrayRef<bool> scalableFlags) {
  if (!scalableFlags.empty() && scalableFlags.size() != tileSizes.size())
    return emitError() << "scalable flags length does not match tile sizes";
  return success();
}

//===----------------------------------------------------------------------===//
// iree_codegen.lowering_config
//===----------------------------------------------------------------------===//

LoweringConfigAttr
LoweringConfigAttr::get(MLIRContext *context, TileSizesListTypeRef tileSizes,
                        ScalableTileFlagsListTypeRef scalableTileFlags,
                        TileSizesListTypeRef tileInterchange,
                        ArrayRef<int64_t> nativeVectorSize) {
  SmallVector<LoweringConfigTilingLevelAttr> tilinglevels;
  for (auto [level, sizes] : llvm::enumerate(tileSizes)) {
    ArrayRef<int64_t> interchange = level < tileInterchange.size()
                                        ? tileInterchange[level]
                                        : ArrayRef<int64_t>{};
    ArrayRef<bool> scalableFlags = level < scalableTileFlags.size()
                                       ? scalableTileFlags[level]
                                       : ArrayRef<bool>{};
    tilinglevels.push_back(LoweringConfigTilingLevelAttr::get(
        context, sizes, interchange, scalableFlags));
  }
  return get(context,
             LoweringConfigTilingLevelsAttr::get(context, tilinglevels),
             nativeVectorSize);
}

LoweringConfigAttr LoweringConfigAttr::get(MLIRContext *context,
                                           TileSizesListTypeRef tileSizes,
                                           TileSizesListTypeRef tileInterchange,
                                           ArrayRef<int64_t> nativeVectorSize) {

  return get(context, tileSizes, {}, tileInterchange, nativeVectorSize);
}

TileSizesListType LoweringConfigAttr::getTileSizeVals() const {
  TileSizesListType tileSizes;
  for (auto &level : getTilingLevels())
    tileSizes.push_back(SmallVector<int64_t>(level.getSizes()));
  return tileSizes;
}

SmallVector<int64_t> LoweringConfigAttr::getTileSizeVals(unsigned level) const {
  auto levels = getTilingLevels();
  if (level >= levels.size())
    return {};
  return SmallVector<int64_t>(levels[level].getSizes());
}

ScalableTileFlagsListType LoweringConfigAttr::getScalableTileFlagVals() {
  ScalableTileFlagsListType scalableFlags;
  for (auto &level : getTilingLevels())
    scalableFlags.push_back(SmallVector<bool>(level.getScalableFlags()));
  return scalableFlags;
}

SmallVector<bool> LoweringConfigAttr::getScalableTileFlagVals(unsigned level) {
  auto levels = getTilingLevels();
  if (level >= levels.size())
    return {};
  SmallVector<bool> scalableFlags(levels[level].getScalableFlags());
  // Extend the scalable flags with `false` to match the length of the sizes.
  scalableFlags.resize(levels[level].getSizes().size());
  return scalableFlags;
}

SmallVector<int64_t>
LoweringConfigAttr::getTileInterchangeVals(unsigned level) const {
  auto levels = getTilingLevels();
  if (level >= levels.size())
    return {};
  return SmallVector<int64_t>(levels[level].getInterchange());
}

bool LoweringConfigAttr::isInterchangeEmpty() {
  return llvm::none_of(getTilingLevels(), [](auto level) {
    return !level.getInterchange().empty();
  });
}

SmallVector<int64_t> LoweringConfigAttr::getWorkgroupTileSizes() const {
  return getTileSizeVals(0);
}

SmallVector<int64_t> LoweringConfigAttr::getWorkgroupInterchange() const {
  return getTileInterchangeVals(0);
}

SmallVector<int64_t>
LoweringConfigAttr::getStaticTilingLevelSizes(unsigned level,
                                              Operation *) const {
  return getTileSizeVals(level);
}

SmallVector<OpFoldResult>
LoweringConfigAttr::getTilingLevelSizes(OpBuilder &builder, unsigned level,
                                        Operation *op) const {
  return llvm::map_to_vector(
      getStaticTilingLevelSizes(level, op),
      [&](int64_t t) -> OpFoldResult { return builder.getIndexAttr(t); });
}

bool LoweringConfigAttr::hasTilingLevel(unsigned level) const {
  return !getTileSizeVals(level).empty();
}

LogicalResult
LoweringConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           LoweringConfigTilingLevelsAttr levels,
                           ArrayRef<int64_t> nativeVectorSizes) {
  (void)nativeVectorSizes;
  if (!levels)
    return emitError() << "missing lowering config levels";
  return success();
}

//===----------------------------------------------------------------------===//
// iree.compilation_info
//===----------------------------------------------------------------------===//

LogicalResult
CompilationInfoAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            LoweringConfigAttr loweringConfig,
                            TranslationInfoAttr translationInfo) {
  if (!loweringConfig) {
    return emitError() << "missing lowering config";
  }
  if (failed(LoweringConfigAttr::verify(
          emitError, loweringConfig.getTilingLevels(),
          loweringConfig.getNativeVectorSize()))) {
    return failure();
  }
  if (!translationInfo) {
    return emitError() << "missing translation info";
  }
  if (failed(TranslationInfoAttr::verify(
          emitError, translationInfo.getPassPipeline(),
          translationInfo.getCodegenSpec(), translationInfo.getWorkgroupSize(),
          translationInfo.getSubgroupSize(),
          translationInfo.getConfiguration()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Initialize attributes
//===----------------------------------------------------------------------===//

void IREECodegenDialect::initializeCodegenAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.cpp.inc" // IWYU pragma: keeep
      >();
}

} // namespace mlir::iree_compiler::IREE::Codegen

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Helpers for getting/setting iree_codegen.translation_info attribute on the
// `hal.executable.export`
// ===----------------------------------------------------------------------===//

IREE::Codegen::TranslationInfoAttr
getTranslationInfo(FunctionOpInterface funcOp) {
  return funcOp->getAttrOfType<IREE::Codegen::TranslationInfoAttr>(
      kTranslationInfoAttrName);
}

std::optional<SmallVector<int64_t>>
getWorkgroupSize(FunctionOpInterface funcOp) {
  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo) {
    return std::nullopt;
  }
  return llvm::to_vector(translationInfo.getWorkgroupSize());
}

std::optional<int64_t> getSubgroupSize(FunctionOpInterface funcOp) {
  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo) {
    return std::nullopt;
  }
  // The underlying storage sets 0 to optional scalar integer value. So if set
  // to 0, return as not set.
  if (translationInfo.getSubgroupSize() == int64_t()) {
    return std::nullopt;
  }
  return translationInfo.getSubgroupSize();
}

LogicalResult
setTranslationInfo(mlir::FunctionOpInterface entryPoint,
                   IREE::Codegen::TranslationInfoAttr translationInfo) {
  entryPoint->setAttr(kTranslationInfoAttrName, translationInfo);
  return success();
}

void eraseTranslationInfo(FunctionOpInterface funcOp) {
  funcOp->removeAttr(kTranslationInfoAttrName);
}

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.lowering_config` attribute on root
// operations.
// ===----------------------------------------------------------------------===//

SmallVector<int64_t> getTileSizes(Operation *op, unsigned level) {
  IREE::Codegen::LoweringConfigAttrInterface configAttr = getLoweringConfig(op);
  if (!configAttr)
    return {};
  return configAttr.getStaticTilingLevelSizes(level, op);
}
SmallVector<Value> getTileSizes(OpBuilder &b, Operation *op, unsigned level) {
  IREE::Codegen::LoweringConfigAttrInterface configAttr = getLoweringConfig(op);
  if (!configAttr)
    return {};
  return llvm::map_to_vector(configAttr.getTilingLevelSizes(b, level, op),
                             [&](OpFoldResult s) -> Value {
                               return getValueOrCreateConstantIndexOp(
                                   b, op->getLoc(), s);
                             });
}

void setLoweringConfig(Operation *op, Attribute config) {
  op->setAttr(kConfigAttrName, config);
}

void eraseLoweringConfig(Operation *op) { op->removeAttr(kConfigAttrName); }

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.compilation_info` attribute on root
// operations to override IREEs default compilation.
// ===----------------------------------------------------------------------===//

IREE::Codegen::CompilationInfoAttr getCompilationInfo(Operation *op) {
  return op->getAttrOfType<IREE::Codegen::CompilationInfoAttr>(
      kCompilationInfoAttrName);
}

void setCompilationInfo(Operation *op,
                        IREE::Codegen::CompilationInfoAttr config) {
  op->setAttr(kCompilationInfoAttrName, config);
}

void eraseCompilationInfo(Operation *op) {
  op->removeAttr(kCompilationInfoAttrName);
}

} // namespace mlir::iree_compiler
