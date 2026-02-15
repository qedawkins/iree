// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// Test pcf.constrain_layout with a VectorExt NestedLayoutAttr.

// CHECK-LABEL: @constrain_layout_nested
// CHECK-SAME: %[[INPUT:.+]]: tensor<128x128xf32>
func.func @constrain_layout_nested(%input: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // CHECK: pcf.constrain_layout %[[INPUT]]
  // CHECK-SAME: layout(#iree_vector_ext.nested_layout<
  // CHECK-SAME:   subgroup_tile = [2, 2]
  // CHECK-SAME:   batch_tile = [1, 1]
  // CHECK-SAME:   outer_tile = [1, 1]
  // CHECK-SAME:   thread_tile = [4, 16]
  // CHECK-SAME:   element_tile = [1, 4]
  // CHECK-SAME:   subgroup_strides = [1, 2]
  // CHECK-SAME:   thread_strides = [16, 1]
  // CHECK-SAME: : tensor<128x128xf32>
  %0 = pcf.constrain_layout %input
      layout(#iree_vector_ext.nested_layout<
        subgroup_tile = [2, 2],
        batch_tile    = [1, 1],
        outer_tile    = [1, 1],
        thread_tile   = [4, 16],
        element_tile  = [1, 4],
        subgroup_strides = [1, 2],
        thread_strides   = [16, 1]
      >)
      : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

// Test pcf.constrain_mma with an MMA layout attribute.

// CHECK-LABEL: @constrain_mma_wmma
func.func @constrain_mma_wmma(
    %lhs: tensor<16x16xf16>,
    %rhs: tensor<16x16xf16>,
    %acc: tensor<16x16xf32>) -> (tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf32>) {
  // CHECK: pcf.constrain_mma
  // CHECK-SAME: kind(#iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>)
  // CHECK-SAME: lhs(%{{.+}} : tensor<16x16xf16>)
  // CHECK-SAME: rhs(%{{.+}} : tensor<16x16xf16>)
  // CHECK-SAME: acc(%{{.+}} : tensor<16x16xf32>)
  %lhs_c, %rhs_c, %acc_c = pcf.constrain_mma
      kind(#iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>)
      lhs(%lhs : tensor<16x16xf16>)
      rhs(%rhs : tensor<16x16xf16>)
      acc(%acc : tensor<16x16xf32>)
      : tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf32>
  return %lhs_c, %rhs_c, %acc_c
      : tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf32>
}

// -----

// Test pcf.redistribute between two NestedLayoutAttr instances.

// CHECK-LABEL: @redistribute_nested
// CHECK-SAME: %[[INPUT:.+]]: tensor<128x128xf32>
func.func @redistribute_nested(%input: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // CHECK: pcf.redistribute %[[INPUT]]
  // CHECK-SAME: from layout(
  // CHECK-SAME: to layout(
  // CHECK-SAME: via shared_memory
  // CHECK-SAME: : tensor<128x128xf32>
  %0 = pcf.redistribute %input
      from layout(#iree_vector_ext.nested_layout<
        subgroup_tile = [2, 2],
        batch_tile    = [2, 2],
        outer_tile    = [1, 1],
        thread_tile   = [4, 4],
        element_tile  = [1, 4],
        subgroup_strides = [2, 1],
        thread_strides   = [4, 1]
      >)
      to layout(#iree_vector_ext.nested_layout<
        subgroup_tile = [4, 1],
        batch_tile    = [1, 1],
        outer_tile    = [1, 1],
        thread_tile   = [2, 64],
        element_tile  = [1, 1],
        subgroup_strides = [1, 0],
        thread_strides   = [64, 1]
      >)
      via shared_memory
      : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

// Test pcf.redistribute with shuffle method.

// CHECK-LABEL: @redistribute_shuffle
func.func @redistribute_shuffle(%input: tensor<64x64xf16>) -> tensor<64x64xf16> {
  // CHECK: pcf.redistribute %{{.+}}
  // CHECK-SAME: via shuffle
  %0 = pcf.redistribute %input
      from layout(#iree_vector_ext.nested_layout<
        subgroup_tile = [1, 1],
        batch_tile    = [1, 1],
        outer_tile    = [1, 1],
        thread_tile   = [16, 4],
        element_tile  = [4, 16],
        subgroup_strides = [0, 0],
        thread_strides   = [4, 1]
      >)
      to layout(#iree_vector_ext.nested_layout<
        subgroup_tile = [1, 1],
        batch_tile    = [1, 1],
        outer_tile    = [1, 1],
        thread_tile   = [4, 16],
        element_tile  = [16, 4],
        subgroup_strides = [0, 0],
        thread_strides   = [16, 1]
      >)
      via shuffle
      : tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// Test pcf.constrain_layout with 1D tensor.

// CHECK-LABEL: @constrain_layout_1d
func.func @constrain_layout_1d(%input: tensor<256xf32>) -> tensor<256xf32> {
  // CHECK: pcf.constrain_layout %{{.+}}
  // CHECK-SAME: layout(#iree_vector_ext.nested_layout<
  // CHECK-SAME:   subgroup_tile = [1]
  // CHECK-SAME:   thread_tile = [64]
  // CHECK-SAME:   element_tile = [4]
  %0 = pcf.constrain_layout %input
      layout(#iree_vector_ext.nested_layout<
        subgroup_tile = [1],
        batch_tile    = [1],
        outer_tile    = [1],
        thread_tile   = [64],
        element_tile  = [4],
        subgroup_strides = [0],
        thread_strides   = [1]
      >)
      : tensor<256xf32>
  return %0 : tensor<256xf32>
}
