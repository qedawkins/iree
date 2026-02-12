// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-gpu-generate-schedule-ir))" --split-input-file | FileCheck %s
// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-gpu-generate-schedule-ir),one-shot-bufferize{bufferize-function-boundaries},iree-pcf-convert-sref-to-memref,iree-pcf-lower-structural-pcf,inline)" \
// RUN:   --split-input-file | FileCheck %s --check-prefix=PIPELINE

// Test that the GenerateScheduleIR pass replaces a matmul contraction with
// the structured schedule: nested pcf.generic (subgroup + lane) with scf.for
// K loop using load-barrier-compute-barrier pattern.

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
//  Initialize region: allocate LDS shared memory.
//       CHECK:     initialize {
//       CHECK:       pcf.alloc() : !pcf.sref<256x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:       pcf.alloc() : !pcf.sref<64x256xf16, #iree_gpu.subgroup_scope>
//       CHECK:       pcf.yield
//       CHECK:     } -> (%[[LDS_LHS:.+]]: !pcf.sref<256x64xf16, #iree_gpu.subgroup_scope>,
//  CHECK-SAME:          %[[LDS_RHS:.+]]: !pcf.sref<64x256xf16, #iree_gpu.subgroup_scope>)
//       CHECK:     execute(%[[DEST_REF:.+]] = %[[OUT]])
//  CHECK-SAME:       [%[[SG_ID:.+]]: index, %[[SG_COUNT:.+]]: index]
//       CHECK:     pcf.generic
//  CHECK-SAME:       scope(#iree_gpu.lane_scope)
//       CHECK:       execute[%[[LANE_ID:.+]]: index, %[[LANE_COUNT:.+]]: index]
//       CHECK:       %[[ACC_INIT:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf32>
//
//  K loop: load global->LDS, barrier, 4 quarter-K compute phases, barrier.
//       CHECK:       scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ACC:.+]] = %[[ACC_INIT]])
//  Global-to-LDS loads.
//       CHECK:         tensor.extract_slice %[[LHS]]
//  CHECK-SAME:           tensor<256x128xf16> to tensor<256x64xf16>
//       CHECK:         tensor.extract_slice %[[RHS]]
//  CHECK-SAME:           tensor<128x256xf16> to tensor<64x256xf16>
//       CHECK:         pcf.write_slice {{.*}} into %[[LDS_LHS]]
//       CHECK:         pcf.write_slice {{.*}} into %[[LDS_RHS]]
//       CHECK:         pcf.barrier(#iree_gpu.subgroup_scope)
//  4 quarter-K compute phases: read pairs from LDS and contract.
//  CHECK-COUNT-4:      vector.contract {{.*}} vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
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

// PIPELINE test: verify full lowering from PCF to standard MLIR.
// After bufferize + convert-sref-to-memref + lower-structural-pcf + inline:
//   - No PCF ops remain.
//   - Function signature is bufferized (memref args).
//   - LDS is workgroup-addressed memref.alloc.
//   - Reads from LDS are vector.transfer_read.
//   - Writes to LDS are memref.copy.
//   - Compute phases are vector.contract.
//   - Barriers are gpu.barrier.
//   - Result writeback is vector.transfer_write.
//
// PIPELINE-LABEL: func.func @matmul_f16_f32
//  PIPELINE-SAME:   %[[A:.+]]: memref<256x128xf16
//  PIPELINE-SAME:   %[[B:.+]]: memref<128x256xf16
//  PIPELINE-SAME:   %[[C:.+]]: memref<256x256xf32
//       PIPELINE:   memref.alloc(){{.*}}: memref<256x64xf16, #gpu.address_space<workgroup>>
//       PIPELINE:   memref.alloc(){{.*}}: memref<64x256xf16, #gpu.address_space<workgroup>>
// K-loop with copy, barrier, 4 contracts, barrier.
//       PIPELINE:   scf.for {{.*}} iter_args
//       PIPELINE:     memref.copy
//       PIPELINE:     memref.copy
//       PIPELINE:     gpu.barrier
//  PIPELINE-COUNT-4: vector.contract {{.*}} vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
//       PIPELINE:     gpu.barrier
//       PIPELINE:     scf.yield
//       PIPELINE:   }
// Result writeback.
//       PIPELINE:   vector.transfer_write
//  PIPELINE-SAME:     vector<16x16xf32>, memref<16x16xf32
// No PCF ops remain.
//   PIPELINE-NOT:   pcf.
//       PIPELINE:   return %[[C]]

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
//       CHECK:     initialize {
//       CHECK:       pcf.alloc() : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:       pcf.alloc() : !pcf.sref<64x128xf16, #iree_gpu.subgroup_scope>
//       CHECK:       pcf.yield
//       CHECK:     } -> (%[[LDS_LHS2:.+]]: !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>,
//  CHECK-SAME:          %[[LDS_RHS2:.+]]: !pcf.sref<64x128xf16, #iree_gpu.subgroup_scope>)
//       CHECK:     pcf.generic
//  CHECK-SAME:       scope(#iree_gpu.lane_scope)
//  K-loop (one iteration when K=64, step=64).
//       CHECK:       scf.for {{.*}} iter_args
//       CHECK:         tensor.extract_slice %[[LHS2]]
//       CHECK:         tensor.extract_slice %[[RHS2]]
//       CHECK:         pcf.write_slice {{.*}} into %[[LDS_LHS2]]
//       CHECK:         pcf.write_slice {{.*}} into %[[LDS_RHS2]]
//       CHECK:         pcf.barrier
//  CHECK-COUNT-4:      vector.contract {{.*}} vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
//       CHECK:         pcf.barrier
//       CHECK:         scf.yield
//       CHECK:       }
//       CHECK:       pcf.write_slice
//  CHECK-SAME:         : vector<16x16xf32> into !pcf.sref<128x128xf32, #iree_gpu.subgroup_scope>
//       CHECK:       pcf.return

// PIPELINE-LABEL: func.func @matmul_single_k_tile
//  PIPELINE-SAME:   memref<128x64xf16
//  PIPELINE-SAME:   memref<64x128xf16
//  PIPELINE-SAME:   memref<128x128xf32
//       PIPELINE:   memref.alloc(){{.*}}: memref<128x64xf16, #gpu.address_space<workgroup>>
//       PIPELINE:   memref.alloc(){{.*}}: memref<64x128xf16, #gpu.address_space<workgroup>>
// K-loop unrolled (K=64, step=64): just copy, barrier, 4 contracts, write.
//       PIPELINE:   memref.copy
//       PIPELINE:   memref.copy
//       PIPELINE:   gpu.barrier
//  PIPELINE-COUNT-4: vector.contract
//       PIPELINE:   vector.transfer_write
//   PIPELINE-NOT:   pcf.

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
//       CHECK:     initialize {
//       CHECK:       pcf.alloc() : !pcf.sref<64x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:       pcf.alloc() : !pcf.sref<64x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:       pcf.yield
//       CHECK:     } -> (%[[LDS_LHS3:.+]]: !pcf.sref<64x64xf16, #iree_gpu.subgroup_scope>,
//  CHECK-SAME:          %[[LDS_RHS3:.+]]: !pcf.sref<64x64xf16, #iree_gpu.subgroup_scope>)
//       CHECK:     pcf.generic
//  CHECK-SAME:       scope(#iree_gpu.lane_scope)
//  K-loop with load-barrier-compute-barrier.
//       CHECK:       scf.for {{.*}} iter_args
//       CHECK:         tensor.extract_slice %[[LHS3]]
//       CHECK:         pcf.write_slice {{.*}} into %[[LDS_LHS3]]
//       CHECK:         pcf.barrier
//  CHECK-COUNT-4:      vector.contract {{.*}} vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
//       CHECK:         pcf.barrier
//       CHECK:         scf.yield
//       CHECK:       pcf.write_slice
//  CHECK-SAME:         : vector<16x16xf32> into !pcf.sref<64x64xf32, #iree_gpu.subgroup_scope>

// PIPELINE-LABEL: func.func @generic_contraction
//  PIPELINE-SAME:   memref<64x128xf16
//  PIPELINE-SAME:   memref<128x64xf16
//  PIPELINE-SAME:   memref<64x64xf32
//       PIPELINE:   memref.alloc(){{.*}}: memref<64x64xf16, #gpu.address_space<workgroup>>
//       PIPELINE:   scf.for {{.*}} iter_args
//       PIPELINE:     gpu.barrier
//  PIPELINE-COUNT-4: vector.contract
//       PIPELINE:     gpu.barrier
//       PIPELINE:     scf.yield
//       PIPELINE:   vector.transfer_write
//   PIPELINE-NOT:   pcf.
