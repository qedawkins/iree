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
// The constant is distributed to vector<1x1x1x1x16x16xf32> and
// to_simd converts back at the boundary.
//
// CHECK-LABEL: util.func private @vector_distribute_in_cluster
// CHECK: %[[DIST_CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x1x1x16x16xf32>
// CHECK: scf.index_switch
// CHECK: case 0 {
// CHECK:   %[[SIMD:.+]] = iree_vector_ext.to_simd %[[DIST_CST]] : vector<1x1x1x1x16x16xf32> -> vector<16x16xf32>
// CHECK:   "test.use"(%[[SIMD]]) : (vector<16x16xf32>) -> ()
// CHECK:   scf.yield
// CHECK: }
// CHECK: default {
// CHECK-NOT: "test.use"
// CHECK: }
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
// ops and layout anchors. Both get distributed independently.
//
// CHECK-LABEL: util.func private @two_clusters_distributed
// CHECK-DAG: %[[RIGHT_CST:.+]] = arith.constant dense<2.000000e+00> : vector<16x16xf32>
// CHECK-DAG: %[[LEFT_DIST_CST:.+]] = arith.constant dense<1.000000e+00> : vector<1x1x1x1x16x16xf32>
// CHECK: scf.index_switch
// CHECK: case 0 {
// CHECK:   %[[LEFT_SIMD:.+]] = iree_vector_ext.to_simd %[[LEFT_DIST_CST]] : vector<1x1x1x1x16x16xf32> -> vector<16x16xf32>
// CHECK:   "test.use_left"(%[[LEFT_SIMD]]) : (vector<16x16xf32>) -> ()
// CHECK-NOT: "test.use_right"
// CHECK:   scf.yield
// CHECK: }
// CHECK: default {
// CHECK-NOT: "test.use_left"
// CHECK:   %[[RIGHT_DIST0:.+]] = iree_vector_ext.to_simt %[[RIGHT_CST]] : vector<16x16xf32> -> vector<1x1x1x1x16x16xf32>
// CHECK:   %[[RIGHT_DIST1:.+]] = iree_vector_ext.to_simt %[[RIGHT_CST]] : vector<16x16xf32> -> vector<1x1x1x1x16x16xf32>
// CHECK:   %[[RIGHT_SUM:.+]] = arith.addf %[[RIGHT_DIST0]], %[[RIGHT_DIST1]] : vector<1x1x1x1x16x16xf32>
// CHECK:   %[[RIGHT_SIMD:.+]] = iree_vector_ext.to_simd %[[RIGHT_SUM]] : vector<1x1x1x1x16x16xf32> -> vector<16x16xf32>
// CHECK:   "test.use_right"(%[[RIGHT_SIMD]]) : (vector<16x16xf32>) -> ()
// CHECK: }
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
