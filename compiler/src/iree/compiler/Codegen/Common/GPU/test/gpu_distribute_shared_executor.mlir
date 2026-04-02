// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-gpu-distribute-shared-executor))" \
// RUN:   --allow-unregistered-dialect --split-input-file | FileCheck %s

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [1, 1],
  element_tile     = [16, 16],
  subgroup_strides = [1, 1],
  thread_strides   = [1, 1]
>

// Test: The pass reads workgroup_size and subgroup_size from translation_info
// and uses them for vector distribution within tile_group/run_cluster ops.
// With trivial distribution (1 subgroup, 1 thread), to_simt/to_simd pairs
// fold to identity, leaving the original SIMD-shaped constant.
//
// CHECK-LABEL: func.func @distribute_from_translation_info
// CHECK: arith.constant dense<0.000000e+00> : vector<16x16xf32>
// CHECK: scf.index_switch
// CHECK-NOT: tile_group
func.func @distribute_from_translation_info(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index)
    attributes {
      translation_info = #iree_codegen.translation_info<
        pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>
    } {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    pcf.shared_executor.run_cluster(%left)[%k]
        () {
      %cst = arith.constant dense<0.0> : vector<16x16xf32>
      %anchored = iree_vector_ext.to_layout %cst to layout(#nested) : vector<16x16xf32>
      %result = arith.addf %anchored, %anchored : vector<16x16xf32>
      "test.use"(%result) : (vector<16x16xf32>) -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  return
}

// -----

// Test: Without translation_info, the pass performs structural lowering only
// (no vector distribution). The shared_executor is lowered to pcf.generic.
//
// CHECK-LABEL: func.func @structural_lowering_only
// CHECK: pcf.generic
// CHECK-NOT: shared_executor
func.func @structural_lowering_only(
    %init: tensor<16xf32>) -> tensor<16xf32> {
  %result = pcf.shared_executor scope(#pcf.test_scope)
    execute(%ref = %init)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<16xf32, #pcf.test_scope>)
        -> (tensor<16xf32>) {
    "test.use"(%ref) : (!pcf.sref<16xf32, #pcf.test_scope>) -> ()
    pcf.return
  }
  return %result : tensor<16xf32>
}
