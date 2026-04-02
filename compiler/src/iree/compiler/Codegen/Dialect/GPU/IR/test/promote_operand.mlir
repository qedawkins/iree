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
