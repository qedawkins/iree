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

// DO NOT SUBMIT: Both subtests below are disabled because
// FoldToSIMT/SIMDSplatConstant canonicalization (059787256c) folds splat
// constants through to_simt/to_simd, undoing the distribution.

// Test: Vector ops inside run_cluster get distributed via VectorDistribute.
// The to_layout anchors seed the layout analysis, and the distribution
// converts the vector ops to their distributed (SIMT) forms.
// The constant and addf now operate on vector<1x1x1x1x16x16xf32> (distributed).
//
// CHECK-LABEL: util.func private @vector_distribute_in_cluster
// NOCHECK: arith.constant dense<0.000000e+00> : vector<1x1x1x1x16x16xf32>
// NOCHECK: scf.index_switch
// NOCHECK: iree_vector_ext.to_simd {{.*}} : vector<1x1x1x1x16x16xf32> -> vector<16x16xf32>
// NOCHECK-NOT: tile_group
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
// ops and layout anchors. Both get distributed independently. The left
// cluster ops appear only in case 0, the right cluster ops only in default.
//
// NOCHECK-LABEL: util.func private @two_clusters_distributed
// NOCHECK-DAG: arith.constant dense<1.000000e+00> : vector<1x1x1x1x16x16xf32>
// NOCHECK-DAG: arith.constant dense<2.000000e+00> : vector<16x16xf32>
// NOCHECK: scf.index_switch
// NOCHECK-NEXT: case 0 {
// NOCHECK:   iree_vector_ext.to_simd {{.*}} : vector<1x1x1x1x16x16xf32> -> vector<16x16xf32>
// NOCHECK:   scf.yield
// NOCHECK: }
// NOCHECK-NEXT: default {
// NOCHECK: }
// NOCHECK-NOT: tile_group
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
