// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

util.func private @shaped_ref_with_no_sync(!pcf.sref<1x?x3x?xi32, #pcf.test_scope>)
// CHECK: @shaped_ref_with_no_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, #pcf.test_scope>

util.func private @shaped_ref_with_type_sync(!pcf.sref<1x?x3x?xi32, #pcf.test_scope, i32>)
// CHECK: @shaped_ref_with_type_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, #pcf.test_scope, i32>

util.func private @shaped_ref_with_attr_sync(!pcf.sref<1x?x3x?xi32, #pcf.test_scope, 42>)
// CHECK: @shaped_ref_with_attr_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, #pcf.test_scope, 42 : i64>

util.func private @shaped_ref_with_parent_sync(!pcf.sref<1x?x3x?xi32, sync(#pcf.test_scope)>)
// CHECK: @shaped_ref_with_parent_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, sync(#pcf.test_scope)>

// -----

util.func private @threadgroup_no_struct(!pcf.threadgroup<#pcf.test_scope>)
// CHECK: @threadgroup_no_struct
// CHECK-SAME: !pcf.threadgroup<#pcf.test_scope>

// -----

util.func private @threadgroup_with_struct(!pcf.threadgroup<#pcf.test_scope, {index, tensor<4x8xf32>}>)
// CHECK: @threadgroup_with_struct
// CHECK-SAME: !pcf.threadgroup<#pcf.test_scope, {index, tensor<4x8xf32>}>

// -----

util.func private @threadgroup_sequential(!pcf.threadgroup<#pcf.sequential>)
// CHECK: @threadgroup_sequential
// CHECK-SAME: !pcf.threadgroup<#pcf.sequential>

// -----

// Cluster with one dim, one dependent value.
util.func private @cluster_1d_one_dep(!pcf.cluster<#pcf.test_scope, (0 -> d0), c0>)
// CHECK: @cluster_1d_one_dep
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0), c0>

// -----

// Cluster with two dims, one shared dependent value.
util.func private @cluster_2d_shared_dep(!pcf.cluster<#pcf.test_scope, (0 -> d0) x (d0 -> s1), c1>)
// CHECK: @cluster_2d_shared_dep
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0) x (d0 -> s1), c1>

// -----

// Cluster with two dims, two dependent values.
util.func private @cluster_2d_two_deps(!pcf.cluster<#pcf.test_scope, (0 -> d0) x (0 -> d1), c2>)
// CHECK: @cluster_2d_two_deps
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0) x (0 -> d1), c2>

// -----

// Cluster with full grid (no dependent values).
util.func private @cluster_full_grid(!pcf.cluster<#pcf.test_scope, (0 -> s0) x (0 -> s1), full>)
// CHECK: @cluster_full_grid
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> s0) x (0 -> s1), full>

// -----

// Cluster with private struct elements.
util.func private @cluster_private(!pcf.cluster<#pcf.test_scope, (0 -> s0), private: {f32}, c3>)
// CHECK: @cluster_private
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> s0), private: {f32}, c3>

// -----

// Cluster with shared struct elements.
util.func private @cluster_shared(!pcf.cluster<#pcf.test_scope, (0 -> s0), shared: {tensor<64xf32>}, c4>)
// CHECK: @cluster_shared
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> s0), shared: {tensor<64xf32>}, c4>

// -----

// Cluster with both struct kinds.
util.func private @cluster_both_kinds(!pcf.cluster<#pcf.test_scope, (0 -> d0), private: {f32}, shared: {tensor<64xf32>}, c6>)
// CHECK: @cluster_both_kinds
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0), private: {f32}, shared: {tensor<64xf32>}, c6>

// -----

// Cluster with single-segment (leaf-only) ID.
util.func private @cluster_leaf_id(!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
// CHECK: @cluster_leaf_id
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>

// -----

// Cluster with multi-segment (qualified) ID.
util.func private @cluster_qualified_id(!pcf.cluster<#pcf.test_scope, (0 -> d0), tg.left>)
// CHECK: @cluster_qualified_id
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0), tg.left>

// -----

// Cluster with deeply nested ID.
util.func private @cluster_deep_id(!pcf.cluster<#pcf.test_scope, (0 -> d0), outer.inner.leaf>)
// CHECK: @cluster_deep_id
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.inner.leaf>

// -----

// Cluster with struct groups and ID.
util.func private @cluster_struct_and_id(!pcf.cluster<#pcf.test_scope, (0 -> d0), shared: {f32}, tg.left>)
// CHECK: @cluster_struct_and_id
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0), shared: {f32}, tg.left>

// -----

// Standalone NamespacedSymbolAttr roundtrip (leaf-only).
util.func private @ns_sym_leaf(!pcf.cluster<#pcf.test_scope, (0 -> d0), leaf>)
// CHECK: @ns_sym_leaf
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0), leaf>

// -----

// Standalone NamespacedSymbolAttr roundtrip (qualified).
util.func private @ns_sym_qualified(!pcf.cluster<#pcf.test_scope, (0 -> d0), ns.child>)
// CHECK: @ns_sym_qualified
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0), ns.child>

// -----

// Standalone NamespacedSymbolAttr roundtrip (deeply nested).
util.func private @ns_sym_deep(!pcf.cluster<#pcf.test_scope, (0 -> d0), a.b.c.d>)
// CHECK: @ns_sym_deep
// CHECK-SAME: !pcf.cluster<#pcf.test_scope, (0 -> d0), a.b.c.d>
