// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PLUGINS_TARGET_ROCM_ROCDLPIPELINEOPTIONS_H_
#define IREE_PLUGINS_TARGET_ROCM_ROCDLPIPELINEOPTIONS_H_

#include <optional>
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::IREE::ROCM {

/// Phases of the ROCDL translation pipeline. These define split points
/// for compile-from/compile-to functionality.
///
/// The pipeline has two phases:
///   1. ConfigurationControlledTranslation: The initial phase covering
///      executable lowering, transform dialect, and strategy selection.
///   2. LLVMTranslation: Everything from linalg-to-loops through
///      LLVM+ROCDL conversion and kernel annotation.
enum class TranslationPhase : uint8_t {
  ConfigurationControlledTranslation = 0,
  LLVMTranslation = 1,
};

/// Returns the string name of a TranslationPhase.
inline llvm::StringRef stringifyTranslationPhase(TranslationPhase phase) {
  switch (phase) {
  case TranslationPhase::ConfigurationControlledTranslation:
    return "configuration_controlled_translation";
  case TranslationPhase::LLVMTranslation:
    return "llvm_translation";
  }
  llvm_unreachable("unknown TranslationPhase");
}

/// Parses a TranslationPhase from a string.
inline std::optional<TranslationPhase>
symbolizeTranslationPhase(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<TranslationPhase>>(str)
      .Case("configuration_controlled_translation",
            TranslationPhase::ConfigurationControlledTranslation)
      .Case("llvm_translation", TranslationPhase::LLVMTranslation)
      .Default(std::nullopt);
}

/// Options controlling which phases of the ROCDL translation pipeline
/// execute. Acts as compile-from/compile-to for the translation pass
/// pipeline.
///
/// By default, all phases run (compileFrom = first, compileTo = last).
struct ROCDLPipelineOptions {
  /// The phase to start compilation from (inclusive).
  TranslationPhase compileFrom =
      TranslationPhase::ConfigurationControlledTranslation;

  /// The phase to compile up to (inclusive).
  TranslationPhase compileTo = TranslationPhase::LLVMTranslation;

  /// When true, use the experimental staged pass pipeline instead of the
  /// default ConfigurationControlledTranslation path. The staged pipeline
  /// splits codegen into workgroup distribution, ukernels, subgroup/thread
  /// distribution, bufferization, and post-bufferization phases with
  /// module-level break points between them.
  bool experimentalStagedPipeline = false;

  /// Returns true if the given phase should execute.
  bool shouldRunPhase(TranslationPhase phase) const {
    return static_cast<uint8_t>(phase) >= static_cast<uint8_t>(compileFrom) &&
           static_cast<uint8_t>(phase) <= static_cast<uint8_t>(compileTo);
  }

  bool operator==(const ROCDLPipelineOptions &rhs) const {
    return compileFrom == rhs.compileFrom && compileTo == rhs.compileTo &&
           experimentalStagedPipeline == rhs.experimentalStagedPipeline;
  }

  bool operator!=(const ROCDLPipelineOptions &rhs) const {
    return !(*this == rhs);
  }
};

inline llvm::hash_code hash_value(const ROCDLPipelineOptions &opts) {
  return llvm::hash_combine(static_cast<uint8_t>(opts.compileFrom),
                            static_cast<uint8_t>(opts.compileTo),
                            opts.experimentalStagedPipeline);
}

} // namespace mlir::iree_compiler::IREE::ROCM

#endif // IREE_PLUGINS_TARGET_ROCM_ROCDLPIPELINEOPTIONS_H_
