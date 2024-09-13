// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-gpu-combine-barrier-regions))" --split-input-file | FileCheck %s

func.func @combine_barrier_region(%arg0: tensor<6xf32>, %arg1: tensor<7xf32>) -> (tensor<1xf32>, tensor<2xf32>) {
  %0 = iree_gpu.barrier_region ins(%arg0 : tensor<6xf32>) {
  ^bb0(%intermediate: tensor<6xf32>):
    %slice = tensor.extract_slice %intermediate[1] [1] [1] : tensor<6xf32> to tensor<1xf32>
    iree_gpu.yield %slice : tensor<1xf32>
  } : tensor<1xf32>
  %1 = iree_gpu.barrier_region ins(%arg1 : tensor<7xf32>) {
  ^bb0(%intermediate: tensor<7xf32>):
    %slice = tensor.extract_slice %intermediate[2] [2] [2] : tensor<7xf32> to tensor<2xf32>
    iree_gpu.yield %slice : tensor<2xf32>
  } : tensor<2xf32>
  return %0, %1 : tensor<1xf32>, tensor<2xf32>
}

// CHECK-LABEL: func @combine_barrier_region
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<6xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<7xf32>
//       CHECK:   %[[B:.+]]:2 = iree_gpu.barrier_region ins(%[[ARG0]], %[[ARG1]] : tensor<6xf32>, tensor<7xf32>) {
//       CHECK:     ^bb0(%[[I0:.+]]: tensor<6xf32>, %[[I1:.+]]: tensor<7xf32>):
//       CHECK:       %[[S0:.+]] = tensor.extract_slice %[[I0]][1] [1] [1]
//       CHECK:       %[[S1:.+]] = tensor.extract_slice %[[I1]][2] [2] [2]
//       CHECK:       iree_gpu.yield %[[S0]], %[[S1]] : tensor<1xf32>, tensor<2xf32>
//       CHECK:   } : tensor<1xf32>, tensor<2xf32>
//       CHECK:   return %[[B]]#0, %[[B]]#1
