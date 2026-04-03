// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-multi-level-tiling{subgroup-tile-sizes=32,32,0 lane-tile-sizes=8,8,0 reduction-tile-sizes=0,0,16})" --split-input-file | FileCheck %s
// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-multi-level-tiling{subgroup-tile-sizes=0,32,32,0 lane-tile-sizes=0,8,8,0 reduction-tile-sizes=0,0,0,16})" --split-input-file | FileCheck %s --check-prefix=BATCH
// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-multi-level-tiling{subgroup-tile-sizes=64,16,0 lane-tile-sizes=8,4,0 reduction-tile-sizes=0,0,16})" --split-input-file | FileCheck %s --check-prefix=ASYM

// Basic matmul with multi-level tiling.
// Subgroup tiles [32,32,0], Lane tiles [8,8,0], Reduction tile [0,0,16].
// Expected: pcf.generic(sg) { pcf.generic(lane) { init; for { matmul } wb } }

func.func @matmul(%lhs: tensor<64x128xf16>, %rhs: tensor<128x64xf16>,
                  %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf16>, tensor<128x64xf16>)
                          outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// CHECK-LABEL: func @matmul
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9_]+]]: tensor<64x128xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9_]+]]: tensor<128x64xf16>
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<64x64xf32>
//       CHECK:   %[[RESULT:.+]] = pcf.generic scope(#pcf.sequential)
//       CHECK:     execute(%[[LHS_REF:.+]] <- %[[LHS]], %[[RHS_REF:.+]] <- %[[RHS]], %[[OUT_REF:.+]] = %[[INIT]])
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           %[[INIT_TILE:.+]] = pcf.read_slice %[[OUT_REF]]
//       CHECK:           scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[INIT_TILE]])
//       CHECK:             pcf.read_slice %[[LHS_REF]]
//       CHECK:             pcf.read_slice %[[RHS_REF]]
//       CHECK:             linalg.matmul
//       CHECK:             scf.yield
//       CHECK:           pcf.write_slice {{.*}} into %[[OUT_REF]]
//       CHECK:           pcf.return
//       CHECK:       pcf.return
//       CHECK:   return %[[RESULT]]

// -----

// Fill + matmul: fill stays outside the pcf.generic nest.

func.func @fill_matmul(%lhs: tensor<64x128xf16>, %rhs: tensor<128x64xf16>,
                       %dest: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<64x64xf32>) -> tensor<64x64xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf16>, tensor<128x64xf16>)
                          outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// CHECK-LABEL: func @fill_matmul
//       CHECK:   linalg.fill
//       CHECK:   pcf.generic scope(#pcf.sequential)
//       CHECK:     pcf.generic scope(#pcf.sequential)
//   CHECK-NOT:       linalg.fill
//       CHECK:       scf.for
//   CHECK-NOT:         linalg.fill
//       CHECK:         linalg.matmul

// -----

// Dynamic shapes: matmul with unknown dimensions.

func.func @matmul_dynamic(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>,
                          %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>)
                          outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// Verify dynamic shapes produce the full nesting structure with correct
// scf.for bounds and write_slice.
// CHECK-LABEL: func @matmul_dynamic
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9_]+]]: tensor<?x?xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9_]+]]: tensor<?x?xf16>
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<?x?xf32>
//       CHECK:   %[[R:.+]] = pcf.generic scope(#pcf.sequential)
//       CHECK:     execute(%{{.+}} <- %[[LHS]], %{{.+}} <- %[[RHS]], %{{.+}} = %[[INIT]])
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           %[[INIT_TILE:.+]] = pcf.read_slice
//       CHECK:           scf.for {{.*}} iter_args({{.*}} = %[[INIT_TILE]])
//       CHECK:             pcf.read_slice
//       CHECK:             pcf.read_slice
//       CHECK:             linalg.matmul
//       CHECK:             scf.yield
//       CHECK:           pcf.write_slice
//       CHECK:           pcf.return
//       CHECK:       pcf.return
//       CHECK:   return %[[R]]

// -----

// Non-divisible dimensions: 63x63 output with 32x32 subgroup tiles.
// Boundary handling should use affine.min for tile size clamping.

func.func @matmul_nondivisible(%lhs: tensor<63x127xf16>,
                                %rhs: tensor<127x63xf16>,
                                %init: tensor<63x63xf32>) -> tensor<63x63xf32> {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<63x127xf16>, tensor<127x63xf16>)
                          outs(%init : tensor<63x63xf32>) -> tensor<63x63xf32>
  return %result : tensor<63x63xf32>
}

// CHECK-LABEL: func @matmul_nondivisible
//       CHECK:   pcf.generic scope(#pcf.sequential)
//       CHECK:     execute
//       CHECK:       affine.min
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           affine.min
//       CHECK:           pcf.read_slice
//       CHECK:           scf.for
//       CHECK:             pcf.read_slice
//       CHECK:             pcf.read_slice
//       CHECK:             linalg.matmul
//       CHECK:           pcf.write_slice
//       CHECK:           pcf.return
//       CHECK:       pcf.return

// -----

// Batch matmul: batch dimension is untiled (0) at all levels.
// Uses BATCH prefix (second RUN line with 4D tile sizes).

func.func @batch_matmul(%lhs: tensor<2x64x128xf16>, %rhs: tensor<2x128x64xf16>,
                         %init: tensor<2x64x64xf32>) -> tensor<2x64x64xf32> {
  %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x64x128xf16>, tensor<2x128x64xf16>)
                                outs(%init : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>
  return %result : tensor<2x64x64xf32>
}

// BATCH-LABEL: func @batch_matmul
//  BATCH-SAME:   %[[LHS:[A-Za-z0-9_]+]]: tensor<2x64x128xf16>
//  BATCH-SAME:   %[[RHS:[A-Za-z0-9_]+]]: tensor<2x128x64xf16>
//  BATCH-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<2x64x64xf32>
//       BATCH:   pcf.generic scope(#pcf.sequential)
//       BATCH:     execute(%{{.+}} <- %[[LHS]], %{{.+}} <- %[[RHS]], %{{.+}} = %[[INIT]])
//       BATCH:       pcf.generic scope(#pcf.sequential)
//       BATCH:         execute
//       BATCH:           pcf.read_slice
//       BATCH:           scf.for
//       BATCH:             pcf.read_slice
//       BATCH:             pcf.read_slice
//       BATCH:             linalg.batch_matmul
//       BATCH:             scf.yield
//       BATCH:           pcf.write_slice
//       BATCH:           pcf.return
//       BATCH:       pcf.return

// -----

// Asymmetric tile sizes: M=64, N=16 at subgroup level; M=8, N=4 at lane level.
// Verifies the pass handles non-square tiling correctly.
// Uses ASYM prefix (third RUN line with subgroup=64,16,0 lane=8,4,0).

func.func @matmul_asymmetric(%lhs: tensor<64x128xf16>, %rhs: tensor<128x64xf16>,
                              %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf16>, tensor<128x64xf16>)
                          outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// ASYM-LABEL: func @matmul_asymmetric
//  ASYM-SAME:   %[[LHS:[A-Za-z0-9_]+]]: tensor<64x128xf16>
//  ASYM-SAME:   %[[RHS:[A-Za-z0-9_]+]]: tensor<128x64xf16>
//  ASYM-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<64x64xf32>
//       ASYM:   pcf.generic scope(#pcf.sequential)
//       ASYM:     execute(%{{.+}} <- %[[LHS]], %{{.+}} <- %[[RHS]], %{{.+}} = %[[INIT]])
//       ASYM:       pcf.generic scope(#pcf.sequential)
//       ASYM:         execute
//       ASYM:           pcf.read_slice
//       ASYM:           scf.for
//       ASYM:             pcf.read_slice
//       ASYM:             pcf.read_slice
//       ASYM:             linalg.matmul
//       ASYM:             scf.yield
//       ASYM:           pcf.write_slice
//       ASYM:           pcf.return
//       ASYM:       pcf.return
