// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 2],
  outer_tile    = [1, 1],
  thread_tile   = [8, 1],
  element_tile  = [1, 8],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 1]
>

// CHECK-LABEL: @distribute_ext_transfer_read
func.func @distribute_ext_transfer_read(%arg0: memref<32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = iree_vector_ext.transfer_read %arg0[%c0, %c0], %cst
      {in_bounds = [true, true]} : memref<32x32xf16>, vector<16x16xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16x16xf16>
  func.return %rootl : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// Verify the distributed reads produce the correct element tile shape.
// CHECK: gpu.thread_id x
// CHECK: iree_vector_ext.transfer_read {{.*}} : memref<32x32xf16>, vector<1x8xf16>
// CHECK: vector.insert_strided_slice {{.*}} {offsets = [0, 0, 0, 0, 0, 0]
// CHECK: iree_vector_ext.transfer_read {{.*}} : memref<32x32xf16>, vector<1x8xf16>
// CHECK: iree_vector_ext.transfer_read {{.*}} : memref<32x32xf16>, vector<1x8xf16>
// CHECK: iree_vector_ext.transfer_read {{.*}} : memref<32x32xf16>, vector<1x8xf16>
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<2x2x1x1x1x8xf16> -> vector<16x16xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 2],
  outer_tile    = [1, 1],
  thread_tile   = [8, 1],
  element_tile  = [1, 8],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 1]
>

// CHECK-LABEL: @distribute_ext_transfer_write
func.func @distribute_ext_transfer_write(%val: vector<16x16xf16>, %arg0: memref<32x32xf16>) {
  %c0 = arith.constant 0 : index
  %rootl = iree_vector_ext.to_layout %val to layout(#layout) : vector<16x16xf16>
  iree_vector_ext.transfer_write %rootl, %arg0[%c0, %c0]
      {in_bounds = [true, true]} : vector<16x16xf16>, memref<32x32xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// Verify the distributed writes produce the correct element tile shape.
// CHECK: gpu.thread_id x
// CHECK: iree_vector_ext.transfer_write {{.*}} : vector<1x8xf16>, memref<32x32xf16>
// CHECK: iree_vector_ext.transfer_write {{.*}} : vector<1x8xf16>, memref<32x32xf16>
// CHECK: iree_vector_ext.transfer_write {{.*}} : vector<1x8xf16>, memref<32x32xf16>
// CHECK: iree_vector_ext.transfer_write {{.*}} : vector<1x8xf16>, memref<32x32xf16>
