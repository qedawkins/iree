// RUN: iree-opt %s --iree-llvmgpu-create-template-for-mma-ops --split-input-file | FileCheck %s

// Test: Create template.func for linalg.matmul with mma_kind lowering config.
// The pass should create a template.func and update the lowering config with template_call.

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

#config = #iree_gpu.lowering_config<{
  workgroup = [128, 128, 0],
  reduction = [0, 0, 32],
  subgroup = [4, 4, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1]
}>

module {
  func.func @matmul_with_mma_config(%lhs: tensor<128x64xf16>,
                                     %rhs: tensor<64x128xf16>,
                                     %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %result = linalg.matmul {lowering_config = #config}
        ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x128xf16>)
        outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}

// CHECK-LABEL: module
// A template.func should be created at module level.
// CHECK: template.func @__mma_template_0
// CHECK: template.branch 0
// CHECK: template.branch 1
// CHECK: template.branch 2
// CHECK: template.branch 3
// CHECK: template.branch 4
// CHECK: template.return

// The implementation blocks should have unimplemented terminators.
// First block has no label (entry block of implementations region).
// CHECK: template.unimplemented -> !template.type<3>
// CHECK: ^bb1(%{{.*}}: index, %{{.*}}: index, %{{.*}}: !template.type<1>):
// CHECK: template.unimplemented
// CHECK: ^bb2(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: !template.type<3>):
// CHECK: template.unimplemented
// CHECK: ^bb3(%{{.*}}: index, %{{.*}}: index, %{{.*}}: !template.type<2>, %{{.*}}: !template.type<3>):
// CHECK: template.unimplemented
// CHECK: ^bb4(%{{.*}}: index, %{{.*}}: index, %{{.*}}: !template.type<2>, %{{.*}}: !template.type<1>):
// CHECK: template.unimplemented

// The lowering config should be updated with template_call.
// CHECK-LABEL: func.func @matmul_with_mma_config
// CHECK: linalg.matmul
// CHECK-SAME: lowering_config = #iree_gpu.lowering_config<{
// CHECK-SAME: mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// CHECK-SAME: promote_operands = [0, 1]
// CHECK-SAME: template_call = @__mma_template_0

// -----

// Test: Skip ops that already have template_call set.

#config_with_template = #iree_gpu.lowering_config<{
  workgroup = [128, 128, 0],
  reduction = [0, 0, 32],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1],
  template_call = @existing_template
}>

module {
  template.func @existing_template(%out: !template.type<0>) -> !template.type<0> {
    template.return %out : !template.type<0>
  } {
  ^bb0:
    template.unimplemented
  }

  func.func @matmul_already_has_template(%lhs: tensor<128x64xf16>,
                                          %rhs: tensor<64x128xf16>,
                                          %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %result = linalg.matmul {lowering_config = #config_with_template}
        ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x128xf16>)
        outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}

// The existing template should NOT be duplicated.
// CHECK-LABEL: module
// CHECK: template.func @existing_template
// CHECK-NOT: template.func @__mma_template

// -----

// Test: Skip ops without promote_operands = [0, 1].

#config_no_promote = #iree_gpu.lowering_config<{
  workgroup = [128, 128, 0],
  reduction = [0, 0, 32],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
}>

module {
  func.func @matmul_no_promote(%lhs: tensor<128x64xf16>,
                                %rhs: tensor<64x128xf16>,
                                %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %result = linalg.matmul {lowering_config = #config_no_promote}
        ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x128xf16>)
        outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}

// No template should be created.
// CHECK-LABEL: module
// CHECK-NOT: template.func
// CHECK: func.func @matmul_no_promote
