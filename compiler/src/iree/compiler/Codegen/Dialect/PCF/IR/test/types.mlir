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

// CHECK-LABEL: func @bundle_single_id
// CHECK-SAME: !pcf.group<0, #pcf.test_scope>
func.func @bundle_single_id(%b: !pcf.group<0, #pcf.test_scope>) {
  return
}

// -----

// CHECK-LABEL: func @bundle_with_struct
// CHECK-SAME: !pcf.group<1, #pcf.test_scope, {index, tensor<4xf32>}>
func.func @bundle_with_struct(%b: !pcf.group<1, #pcf.test_scope, {index, tensor<4xf32>}>) {
  return
}

// -----

// CHECK-LABEL: func @bundle_single_struct
// CHECK-SAME: !pcf.group<0, #pcf.test_scope, {index}>
func.func @bundle_single_struct(%b: !pcf.group<0, #pcf.test_scope, {index}>) {
  return
}

// -----

// CHECK-LABEL: func @bundle_grouped
// CHECK-SAME: !pcf.group<[0, 1], #pcf.test_scope>
func.func @bundle_grouped(%b: !pcf.group<[0, 1], #pcf.test_scope>) {
  return
}

// -----

// CHECK-LABEL: func @bundle_grouped_with_struct
// CHECK-SAME: !pcf.group<[0, 1, 2], #pcf.test_scope, {index, index, tensor<8xf16>}>
func.func @bundle_grouped_with_struct(
    %b: !pcf.group<[0, 1, 2], #pcf.test_scope, {index, index, tensor<8xf16>}>) {
  return
}
