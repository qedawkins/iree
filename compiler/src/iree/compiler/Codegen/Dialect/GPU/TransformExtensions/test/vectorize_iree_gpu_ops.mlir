// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @tensor_multi_mma(%lhs: tensor<2x3x4xf16>, %rhs: tensor<3x5x4xf16>, %acc: tensor<2x5x4xf32>) -> tensor<2x5x4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : tensor<2x3x4xf16>, tensor<3x5x4xf16> into tensor<2x5x4xf32>
  return %0 : tensor<2x5x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.vectorize_iree_gpu
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @tensor_multi_mma

//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG:   %[[CSTF32:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[LHS:.+]] = vector.transfer_read %arg0[%c0, %c0, %c0], %[[CST]] {{.*}} : tensor<2x3x4xf16>, vector<2x3x4xf16>
//   CHECK-DAG:   %[[RHS:.+]] = vector.transfer_read %arg1[%c0, %c0, %c0], %[[CST]] {{.*}} : tensor<3x5x4xf16>, vector<3x5x4xf16>
//   CHECK-DAG:   %[[ACC:.+]] = vector.transfer_read %arg2[%c0, %c0, %c0], %[[CSTF32]] {{.*}} : tensor<2x5x4xf32>, vector<2x5x4xf32>
//       CHECK:   %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
//  CHECK-SAME:     : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
//       CHECK:   vector.transfer_write %[[MMA]], %arg2[%c0, %c0, %c0] {{.*}} : vector<2x5x4xf32>, tensor<2x5x4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @tensor_single_multi_mma(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>, %acc: tensor<4xf32>) -> tensor<4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : tensor<4xf16>, tensor<4xf16> into tensor<4xf32>
  return %0 : tensor<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.vectorize_iree_gpu
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @tensor_single_multi_mma

//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG:   %[[CSTF32:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[LHS:.+]] = vector.transfer_read %arg0[%c0], %[[CST]] {in_bounds = [true]} : tensor<4xf16>, vector<4xf16>
//   CHECK-DAG:   %[[RHS:.+]] = vector.transfer_read %arg1[%c0], %[[CST]] {in_bounds = [true]} : tensor<4xf16>, vector<4xf16>
//   CHECK-DAG:   %[[ACC:.+]] = vector.transfer_read %arg2[%c0], %[[CSTF32]] {in_bounds = [true]} : tensor<4xf32>, vector<4xf32>
//       CHECK:   %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>
//       CHECK:   vector.transfer_write %[[MMA]], %arg2[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>

// -----

func.func @barrier_region(%init: tensor<6x6xf32>) -> tensor<3x2xf32> {
  %0 = iree_gpu.barrier_region ins(%init : tensor<6x6xf32>) {
  ^bb0(%intermediate: tensor<6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, 0] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
    iree_gpu.yield %slice : tensor<3x2xf32>
  } : tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.vectorize_iree_gpu
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @barrier_region
//       CHECK:   %[[SHUFFLE:.+]] = iree_gpu.barrier_region
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<6x6xf32>):
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INTERMEDIATE]][0, 0] [3, 2] [1, 1]
//       CHECK:       %[[READ:.+]] = vector.transfer_read {{.*}} : tensor<3x2xf32>, vector<3x2xf32>
//       CHECK:       iree_gpu.yield %[[READ]] : vector<3x2xf32>
//       CHECK:   } : vector<3x2xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<3x2xf32>
//       CHECK:   vector.transfer_write %[[SHUFFLE]], %[[EMPTY]]

// -----

func.func @multi_result_barrier_region(%init: tensor<6x6xf32>) -> (index, tensor<3x2xf32>) {
  %0:2 = iree_gpu.barrier_region ins(%init : tensor<6x6xf32>) {
  ^bb0(%intermediate: tensor<6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, 0] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
    %c0 = arith.constant 0 : index
    iree_gpu.yield %c0, %slice : index, tensor<3x2xf32>
  } : index, tensor<3x2xf32>
  return %0#0, %0#1 : index, tensor<3x2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.vectorize_iree_gpu
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @multi_result_barrier_region
//       CHECK:   %[[SHUFFLE:.+]]:2 = iree_gpu.barrier_region
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<6x6xf32>):
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INTERMEDIATE]][0, 0] [3, 2] [1, 1]
//       CHECK:       %[[READ:.+]] = vector.transfer_read {{.*}} : tensor<3x2xf32>, vector<3x2xf32>
//       CHECK:       iree_gpu.yield %c0, %[[READ]] : index, vector<3x2xf32>
//       CHECK:   } : index, vector<3x2xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<3x2xf32>
//       CHECK:   vector.transfer_write %[[SHUFFLE]]#1, %[[EMPTY]]
