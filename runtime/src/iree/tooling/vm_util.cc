// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/vm_util.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/numpy_io.h"
#include "iree/vm/ref_cc.h"

// TODO(benvanik): drop use of stdio and make an iree_io_stream_t.
#if defined(IREE_PLATFORM_WINDOWS)
static uint64_t iree_file_query_length(FILE* file) {
  _fseeki64(file, 0, SEEK_END);
  uint64_t file_length = _ftelli64(file);
  _fseeki64(file, 0, SEEK_SET);
  return file_length;
}
static bool iree_file_is_eof(FILE* file, uint64_t file_length) {
  return _ftelli64(file) == file_length;
}
#else
static uint64_t iree_file_query_length(FILE* file) {
  fseeko(file, 0, SEEK_END);
  uint64_t file_length = ftello(file);
  fseeko(file, 0, SEEK_SET);
  return file_length;
}
static bool iree_file_is_eof(FILE* file, uint64_t file_length) {
  return ftello(file) == file_length;
}
#endif  // IREE_PLATFORM_*

using namespace iree;

namespace iree {

static iree_status_t iree_tooling_load_ndarrays_from_file(
    iree_string_view_t file_path, iree_hal_allocator_t* device_allocator,
    iree_vm_list_t* variant_list) {
  // Open the file for reading.
  std::string file_path_str(file_path.data, file_path.size);
  FILE* file = fopen(file_path_str.c_str(), "rb");
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%.*s'", (int)file_path.size,
                            file_path.data);
  }

  uint64_t file_length = iree_file_query_length(file);

  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && !iree_file_is_eof(file, file_length)) {
    iree_hal_buffer_view_t* buffer_view = NULL;
    status = iree_numpy_npy_load_ndarray(
        file, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params,
        device_allocator, &buffer_view);
    if (iree_status_is_ok(status)) {
      auto buffer_view_ref = iree_hal_buffer_view_retain_ref(buffer_view);
      status = iree_vm_list_push_ref_move(variant_list, &buffer_view_ref);
    }
    iree_hal_buffer_view_release(buffer_view);
  }

  fclose(file);
  return status;
}

// Creates a HAL buffer view with the given |metadata| and reads the contents
// from the file at |file_path|.
//
// The file contents are directly read in to memory with no processing.
static iree_status_t CreateBufferViewFromFile(
    iree_string_view_t metadata, iree_string_view_t file_path,
    iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  *out_buffer_view = NULL;

  // Parse shape and element type used to allocate the buffer view.
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  iree_host_size_t shape_rank = 0;
  iree_status_t shape_result = iree_hal_parse_shape_and_element_type(
      metadata, 0, &shape_rank, NULL, &element_type);
  if (!iree_status_is_ok(shape_result) &&
      !iree_status_is_out_of_range(shape_result)) {
    return shape_result;
  } else if (shape_rank > 128) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "a shape rank of %zu is just a little bit excessive, eh?", shape_rank);
  }
  iree_status_ignore(shape_result);
  iree_hal_dim_t* shape =
      (iree_hal_dim_t*)iree_alloca(shape_rank * sizeof(iree_hal_dim_t));
  IREE_RETURN_IF_ERROR(iree_hal_parse_shape_and_element_type(
      metadata, shape_rank, &shape_rank, shape, &element_type));

  // TODO(benvanik): allow specifying the encoding.
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;

  // Open the file for reading.
  std::string file_path_str(file_path.data, file_path.size);
  FILE* file = fopen(file_path_str.c_str(), "rb");
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%.*s'", (int)file_path.size,
                            file_path.data);
  }

  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  struct read_params_t {
    FILE* file;
  } read_params = {
      file,
  };
  iree_status_t status = iree_hal_buffer_view_generate_buffer(
      device_allocator, shape_rank, shape, element_type, encoding_type,
      buffer_params,
      +[](iree_hal_buffer_mapping_t* mapping, void* user_data) {
        auto* read_params = reinterpret_cast<read_params_t*>(user_data);
        size_t bytes_read =
            fread(mapping->contents.data, 1, mapping->contents.data_length,
                  read_params->file);
        if (bytes_read != mapping->contents.data_length) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "file contents truncated; expected %zu bytes "
                                  "based on buffer view size",
                                  mapping->contents.data_length);
        }
        return iree_ok_status();
      },
      &read_params, out_buffer_view);

  fclose(file);

  return status;
}

Status ParseToVariantList(iree_hal_allocator_t* device_allocator,
                          iree::span<const std::string> input_strings,
                          iree_allocator_t host_allocator,
                          iree_vm_list_t** out_list) {
  IREE_TRACE_SCOPE();

  *out_list = NULL;
  vm::ref<iree_vm_list_t> variant_list;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
      /*element_type=*/nullptr, input_strings.size(), host_allocator,
      &variant_list));
  for (size_t i = 0; i < input_strings.size(); ++i) {
    iree_string_view_t input_view = iree_string_view_trim(iree_make_string_view(
        input_strings[i].data(), input_strings[i].size()));
    if (iree_string_view_consume_prefix(&input_view, IREE_SV("@"))) {
      IREE_RETURN_IF_ERROR(iree_tooling_load_ndarrays_from_file(
          input_view, device_allocator, variant_list.get()));
      continue;
    } else if (iree_string_view_equal(input_view, IREE_SV("(null)")) ||
               iree_string_view_equal(input_view, IREE_SV("(ignored)"))) {
      iree_vm_ref_t null_ref = iree_vm_ref_null();
      IREE_RETURN_IF_ERROR(
          iree_vm_list_push_ref_retain(variant_list.get(), &null_ref));
      continue;
    }
    bool has_equal =
        iree_string_view_find_char(input_view, '=', 0) != IREE_STRING_VIEW_NPOS;
    bool has_x =
        iree_string_view_find_char(input_view, 'x', 0) != IREE_STRING_VIEW_NPOS;
    if (has_equal || has_x) {
      // Buffer view (either just a shape or a shape=value) or buffer.
      bool is_storage_reference = iree_string_view_consume_prefix(
          &input_view, iree_make_cstring_view("&"));
      iree_hal_buffer_view_t* buffer_view = nullptr;
      bool has_at = iree_string_view_find_char(input_view, '@', 0) !=
                    IREE_STRING_VIEW_NPOS;
      if (has_at) {
        // Referencing an external file; split into the portion used to
        // initialize the buffer view and the file contents.
        iree_string_view_t metadata, file_path;
        iree_string_view_split(input_view, '@', &metadata, &file_path);
        iree_string_view_consume_suffix(&metadata, iree_make_cstring_view("="));
        IREE_RETURN_IF_ERROR(CreateBufferViewFromFile(
            metadata, file_path, device_allocator, &buffer_view));
      } else {
        IREE_RETURN_IF_ERROR(iree_hal_buffer_view_parse(
                                 input_view, device_allocator, &buffer_view),
                             "parsing value '%.*s'", (int)input_view.size,
                             input_view.data);
      }
      if (is_storage_reference) {
        // Storage buffer reference; just take the storage for the buffer view -
        // it'll still have whatever contents were specified (or 0) but we'll
        // discard the metadata.
        auto buffer_ref = iree_hal_buffer_retain_ref(
            iree_hal_buffer_view_buffer(buffer_view));
        iree_hal_buffer_view_release(buffer_view);
        IREE_RETURN_IF_ERROR(
            iree_vm_list_push_ref_move(variant_list.get(), &buffer_ref));
      } else {
        auto buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
        IREE_RETURN_IF_ERROR(
            iree_vm_list_push_ref_move(variant_list.get(), &buffer_view_ref));
      }
    } else {
      // Scalar.
      bool has_dot = iree_string_view_find_char(input_view, '.', 0) !=
                     IREE_STRING_VIEW_NPOS;
      iree_vm_value_t val;
      if (has_dot) {
        // Float.
        val = iree_vm_value_make_f32(0.0f);
        if (!iree_string_view_atof(input_view, &val.f32)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value '%.*s' as f32",
                                  (int)input_view.size, input_view.data);
        }
      } else {
        // Integer.
        val = iree_vm_value_make_i32(0);
        if (!iree_string_view_atoi_int32(input_view, &val.i32)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value '%.*s' as i32",
                                  (int)input_view.size, input_view.data);
        }
      }
      IREE_RETURN_IF_ERROR(iree_vm_list_push_value(variant_list.get(), &val));
    }
  }
  *out_list = variant_list.release();
  return OkStatus();
}

// Prints a buffer view with contents without a trailing newline.
static iree_status_t PrintBufferView(iree_hal_buffer_view_t* buffer_view,
                                     iree_host_size_t max_element_count,
                                     iree_string_builder_t* builder) {
  std::string result_str(4096, '\0');
  iree_status_t status;
  do {
    iree_host_size_t actual_length = 0;
    status = iree_hal_buffer_view_format(buffer_view, max_element_count,
                                         result_str.size() + 1, &result_str[0],
                                         &actual_length);
    result_str.resize(actual_length);
  } while (iree_status_is_out_of_range(status));
  IREE_RETURN_IF_ERROR(status);
  iree_string_builder_append_string(
      builder, iree_make_string_view(result_str.data(), result_str.size()));
  return iree_ok_status();
}

#define IREE_PRINTVARIANT_CASE_I(SIZE, B, V, STATUS)     \
  case IREE_VM_VALUE_TYPE_I##SIZE:                       \
    STATUS = iree_string_builder_append_format(          \
        B, "i" #SIZE "=%" PRIi##SIZE "\n", (V).i##SIZE); \
    break;

#define IREE_PRINTVARIANT_CASE_F(SIZE, B, V, STATUS)                          \
  case IREE_VM_VALUE_TYPE_F##SIZE:                                            \
    STATUS =                                                                  \
        iree_string_builder_append_format(B, "f" #SIZE "=%g\n", (V).f##SIZE); \
    break;

// Prints variant description including a trailing newline.
static Status PrintVariant(iree_vm_variant_t variant, size_t max_element_count,
                           iree_string_builder_t* builder) {
  if (iree_vm_variant_is_empty(variant)) {
    iree_string_builder_append_string(builder, IREE_SV("(null)\n"));
  } else if (iree_vm_variant_is_value(variant)) {
    iree_status_t status = iree_ok_status();
    switch (variant.type.value_type) {
      IREE_PRINTVARIANT_CASE_I(8, builder, variant, status)
      IREE_PRINTVARIANT_CASE_I(16, builder, variant, status)
      IREE_PRINTVARIANT_CASE_I(32, builder, variant, status)
      IREE_PRINTVARIANT_CASE_I(64, builder, variant, status)
      IREE_PRINTVARIANT_CASE_F(32, builder, variant, status)
      IREE_PRINTVARIANT_CASE_F(64, builder, variant, status)
      default:
        status = iree_string_builder_append_string(builder, IREE_SV("?\n"));
        break;
    }
    IREE_RETURN_IF_ERROR(status);
  } else if (iree_vm_variant_is_ref(variant)) {
    iree_string_view_t type_name = iree_vm_ref_type_name(variant.type.ref_type);
    iree_string_builder_append_string(builder, type_name);
    iree_string_builder_append_string(builder, IREE_SV("\n"));
    if (iree_hal_buffer_view_isa(variant.ref)) {
      auto* buffer_view = iree_hal_buffer_view_deref(variant.ref);
      IREE_RETURN_IF_ERROR(
          PrintBufferView(buffer_view, max_element_count, builder));
      iree_string_builder_append_string(builder, IREE_SV("\n"));
    } else {
      // TODO(benvanik): a way for ref types to describe themselves.
      iree_string_builder_append_string(builder, IREE_SV("(no printer)\n"));
    }
  } else {
    iree_string_builder_append_string(builder, IREE_SV("(null)\n"));
  }
  return OkStatus();
}

Status PrintVariantList(iree_vm_list_t* variant_list, size_t max_element_count,
                        iree_string_builder_t* builder) {
  IREE_TRACE_SCOPE();
  for (iree_host_size_t i = 0; i < iree_vm_list_size(variant_list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant(variant_list, i, &variant),
                         "variant %zu not present", i);
    iree_string_builder_append_format(builder, "result[%zu]: ", i);
    IREE_RETURN_IF_ERROR(PrintVariant(variant, max_element_count, builder));
  }
  return OkStatus();
}

Status PrintVariantList(iree_vm_list_t* variant_list, size_t max_element_count,
                        std::string* out_string) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_RETURN_IF_ERROR(
      PrintVariantList(variant_list, max_element_count, &builder));
  out_string->assign(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return iree_ok_status();
}

Status PrintVariantList(iree_vm_list_t* variant_list,
                        size_t max_element_count) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_RETURN_IF_ERROR(
      PrintVariantList(variant_list, max_element_count, &builder));
  printf("%.*s", (int)iree_string_builder_size(&builder),
         iree_string_builder_buffer(&builder));
  iree_string_builder_deinitialize(&builder);
  return iree_ok_status();
}

}  // namespace iree
