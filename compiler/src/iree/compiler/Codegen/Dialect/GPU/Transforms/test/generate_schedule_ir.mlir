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
//       CHECK:     %[[LDS_LHS:.+]] = pcf.alloc() : !pcf.sref<256x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:     %[[LDS_RHS:.+]] = pcf.alloc() : !pcf.sref<64x256xf16, #iree_gpu.subgroup_scope>
//       CHECK:     pcf.generic
//  CHECK-SAME:       scope(#iree_gpu.lane_scope)
//       CHECK:       execute[%[[LANE_ID:.+]]: index, %[[LANE_COUNT:.+]]: index]
//       CHECK:       %[[ACC_INIT:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf32>
//
//  Prologue: load first K tile into LDS.
//       CHECK:       tensor.extract_slice %[[LHS]]
//  CHECK-SAME:         tensor<256x128xf16> to tensor<256x64xf16>
//       CHECK:       tensor.extract_slice %[[RHS]]
//  CHECK-SAME:         tensor<128x256xf16> to tensor<64x256xf16>
//       CHECK:       pcf.write_slice {{.*}} into %[[LDS_LHS]]
//       CHECK:       pcf.write_slice {{.*}} into %[[LDS_RHS]]
//       CHECK:       pcf.barrier(#iree_gpu.subgroup_scope)
//
//       CHECK:       scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ACC:.+]] = %[[ACC_INIT]])
//
//  P1: global load LHS(k+1) + LDS read q0.
//       CHECK:         tensor.extract_slice %[[LHS]]
//       CHECK:         pcf.read_slice %[[LDS_LHS]][0, 0]
//       CHECK:         pcf.read_slice %[[LDS_RHS]][0, 0]
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//  P2: compute WMMA q0.
//       CHECK:         vector.contract
//  CHECK-SAME:           vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//  P3: global load RHS(k+1) + LDS read q1.
//       CHECK:         tensor.extract_slice %[[RHS]]
//       CHECK:         pcf.read_slice %[[LDS_LHS]][0, 16]
//       CHECK:         pcf.read_slice %[[LDS_RHS]][16, 0]
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//  P4: compute WMMA q1.
//       CHECK:         vector.contract
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//  P5: LDS write staged LHS + LDS read q2.
//       CHECK:         pcf.write_slice {{.*}} into %[[LDS_LHS]]
//       CHECK:         pcf.read_slice %[[LDS_LHS]][0, 32]
//       CHECK:         pcf.read_slice %[[LDS_RHS]][32, 0]
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//  P6: compute WMMA q2 + LDS write staged RHS + LDS read q3.
//       CHECK:         vector.contract
//       CHECK:         pcf.write_slice {{.*}} into %[[LDS_RHS]]
//       CHECK:         pcf.read_slice %[[LDS_LHS]][0, 48]
//       CHECK:         pcf.read_slice %[[LDS_RHS]][48, 0]
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//  P7: compute WMMA q3.
//       CHECK:         vector.contract
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//  P8: sync barrier.
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//       CHECK:         scf.yield
//       CHECK:       }
//  Result writeback: write final accumulator to output sref.
//       CHECK:       pcf.write_slice %{{.+}} into %[[DEST_REF]]
//  CHECK-SAME:         : vector<16x16xf32> into !pcf.sref<256x256xf32, #iree_gpu.subgroup_scope>
//       CHECK:       pcf.return
//       CHECK:     pcf.return
//   CHECK-NOT:   linalg.matmul
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
//  CHECK-SAME:   %[[LHS2:.+]]: tensor<128x64xf16>
//  CHECK-SAME:   %[[RHS2:.+]]: tensor<64x128xf16>
//       CHECK:   pcf.generic
//  CHECK-SAME:     scope(#iree_gpu.subgroup_scope)
//       CHECK:     %[[LDS_LHS2:.+]] = pcf.alloc() : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:     %[[LDS_RHS2:.+]] = pcf.alloc() : !pcf.sref<64x128xf16, #iree_gpu.subgroup_scope>
//       CHECK:     pcf.generic
//  CHECK-SAME:       scope(#iree_gpu.lane_scope)
//  Prologue: initial LDS fill.
//       CHECK:       tensor.extract_slice %[[LHS2]]
//       CHECK:       tensor.extract_slice %[[RHS2]]
//       CHECK:       pcf.write_slice {{.*}} into %[[LDS_LHS2]]
//       CHECK:       pcf.write_slice {{.*}} into %[[LDS_RHS2]]
//       CHECK:       pcf.barrier
//       CHECK:       scf.for {{.*}} iter_args
//  Verify global loads, LDS reads/writes, and compute all present.
//       CHECK:         tensor.extract_slice %[[LHS2]]
//       CHECK:         pcf.read_slice %[[LDS_LHS2]]
//       CHECK:         vector.contract
//       CHECK:         tensor.extract_slice %[[RHS2]]
//       CHECK:         pcf.read_slice %[[LDS_LHS2]]
//       CHECK:         vector.contract
//       CHECK:         pcf.write_slice {{.*}} into %[[LDS_LHS2]]
//       CHECK:         vector.contract
//       CHECK:         pcf.write_slice {{.*}} into %[[LDS_RHS2]]
//       CHECK:         vector.contract
//       CHECK:         scf.yield
//       CHECK:       }
//       CHECK:       pcf.write_slice
//  CHECK-SAME:         : vector<16x16xf32> into !pcf.sref<128x128xf32, #iree_gpu.subgroup_scope>
//       CHECK:       pcf.return

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
//  CHECK-SAME:   %[[LHS3:.+]]: tensor<64x128xf16>
//  CHECK-SAME:   %[[RHS3:.+]]: tensor<128x64xf16>
//       CHECK:   pcf.generic
//  CHECK-SAME:     scope(#iree_gpu.subgroup_scope)
//       CHECK:     %[[LDS_LHS3:.+]] = pcf.alloc() : !pcf.sref<64x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:     %[[LDS_RHS3:.+]] = pcf.alloc() : !pcf.sref<64x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:     pcf.generic
//  CHECK-SAME:       scope(#iree_gpu.lane_scope)
//  Prologue + K-loop with full pipeline.
//       CHECK:       tensor.extract_slice %[[LHS3]]
//       CHECK:       pcf.write_slice {{.*}} into %[[LDS_LHS3]]
//       CHECK:       pcf.barrier
//       CHECK:       scf.for {{.*}} iter_args
//  CHECK-COUNT-4:      vector.contract {{.*}} vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
//       CHECK:       pcf.write_slice
//  CHECK-SAME:         : vector<16x16xf32> into !pcf.sref<64x64xf32, #iree_gpu.subgroup_scope>
