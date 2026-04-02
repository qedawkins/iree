// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-multi-level-tiling{subgroup-tile-sizes=32,32,0 lane-tile-sizes=8,8,0 reduction-tile-sizes=0,0,16})" --split-input-file | FileCheck %s

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
//       CHECK:       scf.for
//       CHECK:         linalg.matmul

// -----

// Dynamic shapes: matmul with unknown dimensions.

func.func @matmul_dynamic(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>,
                          %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>)
                          outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul_dynamic
//       CHECK:   pcf.generic scope(#pcf.sequential)
//       CHECK:     pcf.generic scope(#pcf.sequential)
//       CHECK:       pcf.read_slice
//       CHECK:       scf.for
//       CHECK:         pcf.read_slice
//       CHECK:         pcf.read_slice
//       CHECK:         linalg.matmul
//       CHECK:       pcf.write_slice
