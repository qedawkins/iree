// RUN: iree-opt %s --pass-pipeline='builtin.module(any(iree-llvmgpu-convert-mma-to-process-inner-tile))' --split-input-file | FileCheck %s

// Test: Convert linalg.matmul with mma_kind and template_call to process_inner_tile.

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

#config = #iree_gpu.lowering_config<{
  workgroup = [128, 128, 0],
  reduction = [0, 0, 32],
  subgroup = [4, 4, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1],
  template_call = @matmul_template
}>

module {
  template.func @matmul_template(%out: !template.type<0>) -> !template.type<0> {
    %c0 = arith.constant 0 : index
    %out_ref = builtin.unrealized_conversion_cast %out : !template.type<0> to !template.type<1>
    %allocs = template.branch 0() : () -> !template.type<3>
    %acc_init = template.branch 1(%c0, %c0, %out_ref) : (index, index, !template.type<1>) -> !template.type<2>
    template.branch 2(%c0, %c0, %c0, %allocs) : (index, index, index, !template.type<3>) -> ()
    %result_acc = template.branch 3(%c0, %c0, %acc_init, %allocs) : (index, index, !template.type<2>, !template.type<3>) -> !template.type<2>
    template.branch 4(%c0, %c0, %result_acc, %out_ref) : (index, index, !template.type<2>, !template.type<1>) -> ()
    template.return %out : !template.type<0>
  } {
  ^bb0:
    %0 = template.unimplemented -> !template.type<3>
  ^bb1(%sg1: index, %lane1: index, %dest1: !template.type<1>):
    %1 = template.unimplemented -> !template.type<2>
  ^bb2(%sg2: index, %lane2: index, %k2: index, %allocs2: !template.type<3>):
    template.unimplemented
  ^bb3(%sg3: index, %lane3: index, %acc3: !template.type<2>, %allocs3: !template.type<3>):
    %2 = template.unimplemented -> !template.type<2>
  ^bb4(%sg4: index, %lane4: index, %res4: !template.type<2>, %dest4: !template.type<1>):
    template.unimplemented
  }

  func.func @matmul_with_template(%lhs: tensor<128x64xf16>,
                                   %rhs: tensor<64x128xf16>,
                                   %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %result = linalg.matmul {lowering_config = #config}
        ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x128xf16>)
        outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}

// The linalg.matmul should be converted to iree_gpu.process_inner_tile.
// CHECK-LABEL: func.func @matmul_with_template
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: tensor<128x64xf16>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9_]+]]: tensor<64x128xf16>
// CHECK-SAME: %[[INIT:[a-zA-Z0-9_]+]]: tensor<128x128xf32>
// CHECK: %[[RESULT:.*]] = iree_gpu.process_inner_tile
// CHECK-SAME: bounds(
// CHECK-SAME: kind(#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>)
// CHECK-SAME: indexing_maps = [
// CHECK-SAME: iterator_types = [
// CHECK-SAME: outer_dim_distribution = [4, 4]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<128x64xf16>, tensor<64x128xf16>)
// CHECK-SAME: outs(%[[INIT]] : tensor<128x128xf32>)
// CHECK-SAME: @matmul_template -> tensor<128x128xf32>
// CHECK: return %[[RESULT]]

// -----

// Test: Skip ops without template_call.

#config_no_template = #iree_gpu.lowering_config<{
  workgroup = [128, 128, 0],
  reduction = [0, 0, 32],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
  promote_operands = [0, 1]
}>

module {
  func.func @matmul_no_template(%lhs: tensor<128x64xf16>,
                                 %rhs: tensor<64x128xf16>,
                                 %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %result = linalg.matmul {lowering_config = #config_no_template}
        ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x128xf16>)
        outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}

// Should NOT be converted - no template_call.
// CHECK-LABEL: func.func @matmul_no_template
// CHECK: linalg.matmul
// CHECK-NOT: iree_gpu.process_inner_tile
