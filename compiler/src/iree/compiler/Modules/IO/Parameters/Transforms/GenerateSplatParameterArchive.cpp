// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/tooling/parameter_util.h"

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/ArchiveUtils.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

#define GEN_PASS_DEF_GENERATESPLATPARAMETERARCHIVEPASS
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h.inc"

static llvm::cl::opt<bool> clUseFullSplatParameters(
    "iree-io-use-full-splat-parameters",
    llvm::cl::desc("Includes the full splatted constant in the archive"),
    llvm::cl::init(false));

namespace {

struct GenerateSplatParameterArchivePass
    : public IREE::IO::Parameters::impl::GenerateSplatParameterArchivePassBase<
          GenerateSplatParameterArchivePass> {
  using IREE::IO::Parameters::impl::GenerateSplatParameterArchivePassBase<
      GenerateSplatParameterArchivePass>::GenerateSplatParameterArchivePassBase;

  void runOnOperation() override {
    // Nothing to do if no path specified.
    if (archivePath.empty()) {
      return;
    }

    ModuleOp moduleOp = getOperation();

    iree_allocator_t host_allocator = iree_allocator_system();

    // Create the parameter archive builder.
    iree_io_parameter_archive_builder_t builder;
    iree_io_parameter_archive_builder_initialize(host_allocator, &builder);

    auto deinitializeExit = llvm::make_scope_exit([&]() {
      return iree_io_parameter_archive_builder_deinitialize(&builder);
    });

    bool hasParameter = false;
    SmallVector<IREE::Util::GlobalOp> fullGlobals;
    // Walk the globals in the module.
    for (auto global : moduleOp.getOps<IREE::Util::GlobalOp>()) {
      // Look for globals backed by parameters.
      auto initialValueAttr = global.getInitialValueAttr();
      if (!initialValueAttr) {
        continue;
      }
      auto parameterAttr =
          dyn_cast<IREE::Stream::NamedParameterAttr>(initialValueAttr);
      if (!parameterAttr) {
        continue;
      }

      // Note that the scope is not a part of the parameter archive. If the
      // module includes multiple scopes, multiple copies of the splat archive
      // would need to be passed in with all possible scopes.
      std::string parameterName = parameterAttr.getKey().str();
      iree_io_physical_size_t storageSize = parameterAttr.getStorageSize();

      // Add a zero-splat entry to the builder for this global.
      if (clUseFullSplatParameters) {
        // Add a data entry to the builder for this global instead.
        iree_status_t status = iree_io_parameter_archive_builder_add_data_entry(
            &builder,
            iree_string_view_t{
                parameterName.data(),
                static_cast<iree_host_size_t>(parameterName.size())},
            /*metadata=*/iree_const_byte_span_empty(),
            /*alignment=*/IREE_IO_PARAMETER_ARCHIVE_DEFAULT_DATA_ALIGNMENT,
            storageSize);
        if (failed(handleRuntimeError(moduleOp, status,
                                      "Failed to add data entry for global"))) {
          return signalPassFailure();
        }
        fullGlobals.push_back(global);
      } else {
        char c0 = 0;
        iree_status_t status =
            iree_io_parameter_archive_builder_add_splat_entry(
                &builder,
                iree_string_view_t{
                    parameterName.data(),
                    static_cast<iree_host_size_t>(parameterName.size())},
                /*metadata=*/iree_const_byte_span_empty(),
                /*pattern=*/&c0, /*pattern_length=*/1,
                /*data_length=*/storageSize);
        if (failed(handleRuntimeError(
                moduleOp, status, "Failed to add splat entry for global"))) {
          return signalPassFailure();
        }
      }
      hasParameter = true;
    }

    // Early exit if no parameter backed globals present.
    if (!hasParameter) {
      return;
    }

    // Open a file of sufficient size (now that we know it) for writing.
    iree_io_physical_size_t archive_length =
        iree_io_parameter_archive_builder_total_size(&builder);

    auto FileOrErr =
        llvm::FileOutputBuffer::create(archivePath, archive_length);
    if (!FileOrErr) {
      moduleOp.emitError()
          << "Failed to create file output buffer at " << archivePath
          << " with error: "
          << llvm::errorToErrorCode(FileOrErr.takeError()).message();
      return signalPassFailure();
    }
    std::unique_ptr<llvm::FileOutputBuffer> fileBuffer = std::move(*FileOrErr);

    iree_io_file_handle_t *target_file_handle = NULL;
    iree_io_stream_t *target_stream = NULL;
    iree_io_parameter_index_t *built_index = NULL;
    if (failed(writeParameterIndex(moduleOp, host_allocator, builder,
                                   fileBuffer, &target_file_handle,
                                   &target_stream, &built_index))) {
      return signalPassFailure();
    }

    auto releaseFileExit = llvm::make_scope_exit([&]() -> void {
      iree_io_stream_release(target_stream);
      iree_io_parameter_index_release(built_index);
      iree_io_file_handle_release(target_file_handle);
    });

    if (clUseFullSplatParameters) {
      iree_io_stream_ostream llvm_stream(target_stream);

      // Write all of the global contents to the appropriate data storage
      // segments.
      for (auto constantGlobal : fullGlobals) {
        auto initialValueAttr = constantGlobal.getInitialValueAttr();
        if (!initialValueAttr) {
          continue;
        }
        auto parameterAttr =
            dyn_cast<IREE::Stream::NamedParameterAttr>(initialValueAttr);
        if (!parameterAttr) {
          continue;
        }

        // Note that the scope is not a part of the parameter archive. If the
        // module includes multiple scopes, multiple copies of the splat archive
        // would need to be passed in with all possible scopes.
        std::string parameterName = parameterAttr.getKey().str();

        const iree_io_parameter_index_entry_t *target_entry = NULL;
        iree_status_t status = iree_io_parameter_index_lookup(
            built_index,
            iree_string_view_t{
                parameterName.data(),
                static_cast<iree_host_size_t>(parameterName.size())},
            &target_entry);
        if (failed(handleRuntimeError(
                moduleOp, status,
                "Failed to write parameter index header to output file"))) {
          return signalPassFailure();
        }
        status = iree_io_stream_seek(target_stream, IREE_IO_STREAM_SEEK_SET,
                                     target_entry->storage.file.offset);
        if (failed(handleRuntimeError(
                moduleOp, status,
                "Failed to seek to location of global in index"))) {
          return signalPassFailure();
        }

        Util::BytePatternAttr serializableAttr = Util::BytePatternAttr::get(
            constantGlobal->getContext(), parameterAttr.getType(), 0);

        if (failed(serializableAttr.serializeToStream(constantGlobal.getLoc(),
                                                      llvm::endianness::native,
                                                      llvm_stream))) {
          moduleOp.emitError()
              << "Failed to serialize global " << constantGlobal;
          return signalPassFailure();
        }
        llvm_stream.flush();
      }
    }

    // Commit the written file.
    llvm::Error maybeCommit = fileBuffer->commit();
    if (maybeCommit) {
      InFlightDiagnostic errorStream =
          moduleOp.emitError() << "Failed to commit archive with error: ";
      llvm::handleAllErrors(std::move(maybeCommit),
                            [&](const llvm::ErrorInfoBase &PE) {
                              errorStream << PE.message() << "\n";
                            });
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::IO::Parameters
