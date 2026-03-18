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
