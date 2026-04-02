// The distribution factory is registered globally by registerCodegenCommonGPUPasses().
// The error path in DistributeAndLowerPass ("no factory registered") is only
// reachable if Common/GPU passes are not loaded into iree-opt.

// RUN: iree-opt --iree-pcf-distribute-and-lower="vector-distribution=true subgroup-size=64 workgroup-size=64" \
// RUN:   --allow-unregistered-dialect --split-input-file %s | FileCheck %s

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [1, 1],
  element_tile     = [16, 16],
  subgroup_strides = [1, 1],
  thread_strides   = [1, 1]
>

// Test: Vector ops inside run_cluster get distributed via VectorDistribute.
// With trivial distribution (1 subgroup, 1 thread), the to_simt/to_simd
// pairs fold to identity, so the constant stays in SIMD shape. The
// structural lowering (tile_group -> scf.index_switch) still applies.
//
// CHECK-LABEL: util.func private @vector_distribute_in_cluster
// CHECK: arith.constant dense<0.000000e+00> : vector<16x16xf32>
// CHECK: scf.index_switch
// CHECK-NOT: tile_group
util.func private @vector_distribute_in_cluster(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
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
  util.return
}

// -----

#nested2 = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [1, 1],
  element_tile     = [16, 16],
  subgroup_strides = [1, 1],
  thread_strides   = [1, 1]
>

// Test: Two run_cluster ops targeting different clusters, each with vector
// ops and layout anchors. Both get lowered independently. With trivial
// distribution, constants stay in SIMD shape. Structural lowering still
// converts tile_group to scf.index_switch.
//
// CHECK-LABEL: util.func private @two_clusters_distributed
// CHECK: scf.index_switch
// CHECK-NOT: tile_group
util.func private @two_clusters_distributed(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    pcf.shared_executor.run_cluster(%left)[%k]
        () {
      %cst1 = arith.constant dense<1.0> : vector<16x16xf32>
      %a1 = iree_vector_ext.to_layout %cst1 to layout(#nested2) : vector<16x16xf32>
      %r1 = arith.mulf %a1, %a1 : vector<16x16xf32>
      "test.use_left"(%r1) : (vector<16x16xf32>) -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
    pcf.shared_executor.run_cluster(%right)[%k]
        () {
      %cst2 = arith.constant dense<2.0> : vector<16x16xf32>
      %a2 = iree_vector_ext.to_layout %cst2 to layout(#nested2) : vector<16x16xf32>
      %r2 = arith.addf %a2, %a2 : vector<16x16xf32>
      "test.use_right"(%r2) : (vector<16x16xf32>) -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (d0 -> s0), right>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}
