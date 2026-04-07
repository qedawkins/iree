// RUN: iree-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @promote_basic
func.func @promote_basic(
    %src: !pcf.sref<64x128xf16, #pcf.test_scope>) -> !pcf.sref<64x128xf16, #pcf.sequential> {
  %promoted = iree_gpu.promote_operand
      #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>
      %src ["dim0", "dim1"]
      : !pcf.sref<64x128xf16, #pcf.test_scope>
      -> !pcf.sref<64x128xf16, #pcf.sequential>
  // CHECK: iree_gpu.promote_operand
  // CHECK-SAME: ["dim0", "dim1"] : !pcf.sref<64x128xf16, #pcf.test_scope> -> !pcf.sref<64x128xf16, #pcf.sequential>
  return %promoted : !pcf.sref<64x128xf16, #pcf.sequential>
}

// -----

func.func @promote_wrong_symbols(
    %src: !pcf.sref<64x128xf16, #pcf.test_scope>) -> !pcf.sref<64x128xf16, #pcf.sequential> {
  // expected-error @+1 {{number of symbols (1) must match source rank (2)}}
  %promoted = iree_gpu.promote_operand
      #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>
      %src ["dim0"]
      : !pcf.sref<64x128xf16, #pcf.test_scope>
      -> !pcf.sref<64x128xf16, #pcf.sequential>
  return %promoted : !pcf.sref<64x128xf16, #pcf.sequential>
}

// -----

// Dynamic-shaped sref: promote an operand with unknown dimensions.
// CHECK-LABEL: @promote_dynamic
func.func @promote_dynamic(
    %src: !pcf.sref<?x?xf16, #pcf.test_scope>) -> !pcf.sref<?x?xf16, #pcf.sequential> {
  %promoted = iree_gpu.promote_operand
      #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>
      %src ["dim0", "dim1"]
      : !pcf.sref<?x?xf16, #pcf.test_scope>
      -> !pcf.sref<?x?xf16, #pcf.sequential>
  // CHECK: iree_gpu.promote_operand
  // CHECK-SAME: ["dim0", "dim1"] : !pcf.sref<?x?xf16, #pcf.test_scope> -> !pcf.sref<?x?xf16, #pcf.sequential>
  return %promoted : !pcf.sref<?x?xf16, #pcf.sequential>
}

// -----

// Rank-1 operand: promote a 1D sref.
// CHECK-LABEL: @promote_rank1
func.func @promote_rank1(
    %src: !pcf.sref<128xf16, #pcf.test_scope>) -> !pcf.sref<128xf16, #pcf.sequential> {
  %promoted = iree_gpu.promote_operand
      #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>
      %src ["dim0"]
      : !pcf.sref<128xf16, #pcf.test_scope>
      -> !pcf.sref<128xf16, #pcf.sequential>
  // CHECK: iree_gpu.promote_operand
  // CHECK-SAME: ["dim0"] : !pcf.sref<128xf16, #pcf.test_scope> -> !pcf.sref<128xf16, #pcf.sequential>
  return %promoted : !pcf.sref<128xf16, #pcf.sequential>
}

// -----

func.func @promote_non_sref_source(
    %src: tensor<64x128xf16>) -> !pcf.sref<64x128xf16, #pcf.sequential> {
  // expected-error @+1 {{source must be a pcf.sref type}}
  %promoted = iree_gpu.promote_operand
      #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>
      %src ["dim0", "dim1"]
      : tensor<64x128xf16>
      -> !pcf.sref<64x128xf16, #pcf.sequential>
  return %promoted : !pcf.sref<64x128xf16, #pcf.sequential>
}

// -----

func.func @promote_non_sref_result(
    %src: !pcf.sref<64x128xf16, #pcf.test_scope>) -> tensor<64x128xf16> {
  // expected-error @+1 {{result must be a pcf.sref type}}
  %promoted = iree_gpu.promote_operand
      #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>
      %src ["dim0", "dim1"]
      : !pcf.sref<64x128xf16, #pcf.test_scope>
      -> tensor<64x128xf16>
  return %promoted : tensor<64x128xf16>
}

// -----

func.func @promote_non_string_symbols(
    %src: !pcf.sref<64x128xf16, #pcf.test_scope>) -> !pcf.sref<64x128xf16, #pcf.sequential> {
  // expected-error @+1 {{all symbols must be string attributes}}
  %promoted = iree_gpu.promote_operand
      #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>
      %src ["dim0", 0]
      : !pcf.sref<64x128xf16, #pcf.test_scope>
      -> !pcf.sref<64x128xf16, #pcf.sequential>
  return %promoted : !pcf.sref<64x128xf16, #pcf.sequential>
}

// -----

func.func @promote_element_type_mismatch(
    %src: !pcf.sref<64x128xf16, #pcf.test_scope>) -> !pcf.sref<64x128xf32, #pcf.sequential> {
  // expected-error @+1 {{source and result element types must match}}
  %promoted = iree_gpu.promote_operand
      #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>
      %src ["dim0", "dim1"]
      : !pcf.sref<64x128xf16, #pcf.test_scope>
      -> !pcf.sref<64x128xf32, #pcf.sequential>
  return %promoted : !pcf.sref<64x128xf32, #pcf.sequential>
}

// -----

func.func @promote_shape_mismatch(
    %src: !pcf.sref<64x128xf16, #pcf.test_scope>) -> !pcf.sref<64x64xf16, #pcf.sequential> {
  // expected-error @+1 {{source and result shapes must match}}
  %promoted = iree_gpu.promote_operand
      #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>
      %src ["dim0", "dim1"]
      : !pcf.sref<64x128xf16, #pcf.test_scope>
      -> !pcf.sref<64x64xf16, #pcf.sequential>
  return %promoted : !pcf.sref<64x64xf16, #pcf.sequential>
}
