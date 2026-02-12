// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-gpu-generate-schedule-ir))" --split-input-file | FileCheck %s

// Test that the GenerateScheduleIR pass replaces a matmul contraction with
// the structured schedule: nested pcf.generic (subgroup + lane) + scf.for K loop
// with 8 phase barriers.

func.func @matmul_f16_f32(%lhs: tensor<256x128xf16>,
                           %rhs: tensor<128x256xf16>,
                           %out: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<256x128xf16>, tensor<128x256xf16>)
                          outs(%out : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func.func @matmul_f16_f32
//  CHECK-SAME:   %[[LHS:.+]]: tensor<256x128xf16>
//  CHECK-SAME:   %[[RHS:.+]]: tensor<128x256xf16>
//  CHECK-SAME:   %[[OUT:.+]]: tensor<256x256xf32>
//       CHECK:   %[[RESULT:.+]] = pcf.generic
//  CHECK-SAME:     scope(#iree_gpu.subgroup_scope)
//       CHECK:     execute(%[[DEST_REF:.+]] = %[[OUT]])
//  CHECK-SAME:       [%[[SG_ID:.+]]: index, %[[SG_COUNT:.+]]: index]
//       CHECK:     pcf.alloc() : !pcf.sref<256x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:     pcf.alloc() : !pcf.sref<64x256xf16, #iree_gpu.subgroup_scope>
//       CHECK:     pcf.generic
//  CHECK-SAME:       scope(#iree_gpu.lane_scope)
//       CHECK:       execute[%[[LANE_ID:.+]]: index, %[[LANE_COUNT:.+]]: index]
//       CHECK:       %[[C0:.+]] = arith.constant 0 : index
//       CHECK:       %[[KSTEP:.+]] = arith.constant 64 : index
//       CHECK:       %[[KBOUND:.+]] = arith.constant 128 : index
//       CHECK:       scf.for %[[K:.+]] = %[[C0]] to %[[KBOUND]] step %[[KSTEP]]
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//       CHECK:       }
//       CHECK:       pcf.return
//       CHECK:     pcf.return
//       CHECK:   return %[[RESULT]]

// -----

// Test with different K dimension size (K=64, one iteration with kTile=64).

func.func @matmul_single_k_tile(%lhs: tensor<128x64xf16>,
                                 %rhs: tensor<64x128xf16>,
                                 %out: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x128xf16>)
                          outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @matmul_single_k_tile
//       CHECK:   pcf.generic
//  CHECK-SAME:     scope(#iree_gpu.subgroup_scope)
//       CHECK:     pcf.alloc() : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:     pcf.alloc() : !pcf.sref<64x128xf16, #iree_gpu.subgroup_scope>
//       CHECK:     pcf.generic
//  CHECK-SAME:       scope(#iree_gpu.lane_scope)
//       CHECK:       %[[C0:.+]] = arith.constant 0 : index
//       CHECK:       %[[KSTEP:.+]] = arith.constant 64 : index
//       CHECK:       %[[KBOUND:.+]] = arith.constant 64 : index
//       CHECK:       scf.for %{{.+}} = %[[C0]] to %[[KBOUND]] step %[[KSTEP]]
//  CHECK-COUNT-8:      pcf.barrier(#iree_gpu.subgroup_scope)
//       CHECK:       }

// -----

// Test with linalg.generic that has matmul semantics (contraction interface).

#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>

func.func @generic_contraction(%lhs: tensor<64x128xf16>,
                                %rhs: tensor<128x64xf16>,
                                %out: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : tensor<64x128xf16>, tensor<128x64xf16>)
    outs(%out : tensor<64x64xf32>) {
  ^bb0(%a: f16, %b: f16, %c: f32):
    %ext_a = arith.extf %a : f16 to f32
    %ext_b = arith.extf %b : f16 to f32
    %mul = arith.mulf %ext_a, %ext_b : f32
    %add = arith.addf %c, %mul : f32
    linalg.yield %add : f32
  } -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @generic_contraction
//       CHECK:   pcf.generic
//  CHECK-SAME:     scope(#iree_gpu.subgroup_scope)
//       CHECK:     pcf.alloc() : !pcf.sref<64x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:     pcf.alloc() : !pcf.sref<64x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:     pcf.generic
//  CHECK-SAME:       scope(#iree_gpu.lane_scope)
//       CHECK:       arith.constant 128 : index
//       CHECK:       scf.for
//  CHECK-COUNT-8:      pcf.barrier(#iree_gpu.subgroup_scope)
