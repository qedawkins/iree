// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-multi-level-tiling)" --split-input-file | FileCheck %s

// Basic matmul with multi-level tiling.
// Subgroup tiles [32,32,0], Lane tiles [8,8,0], Reduction tile [0,0,16].
// Expected: pcf.generic(sg) { pcf.generic(lane) { init; for { matmul } wb } }

func.func @matmul(%lhs: tensor<64x128xf16>, %rhs: tensor<128x64xf16>,
                  %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.matmul {
    lowering_config = {
      subgroup = [32, 32, 0], lane = [8, 8, 0], reduction = [0, 0, 16]
    }
  } ins(%lhs, %rhs : tensor<64x128xf16>, tensor<128x64xf16>)
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
//       CHECK:           %[[M_OFF:.+]] = affine.apply
//       CHECK:           %[[M_SZ:.+]] = affine.min
//       CHECK:           %[[N_OFF:.+]] = affine.apply
//       CHECK:           %[[N_SZ:.+]] = affine.min
//       CHECK:           %[[INIT_TILE:.+]] = pcf.read_slice %[[OUT_REF]][%[[M_OFF]], %[[N_OFF]]] [%[[M_SZ]], %[[N_SZ]]] [1, 1]
//       CHECK:           %[[REDUCE:.+]] = scf.for %[[K_IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ACC:.+]] = %[[INIT_TILE]])
//       CHECK:             %[[LHS_TILE:.+]] = pcf.read_slice %[[LHS_REF]][%[[M_OFF]], %[[K_IV]]] [%[[M_SZ]], 16] [1, 1]
//       CHECK:             %[[RHS_TILE:.+]] = pcf.read_slice %[[RHS_REF]][%[[K_IV]], %[[N_OFF]]] [16, %[[N_SZ]]] [1, 1]
//       CHECK:             %[[MATMUL:.+]] = linalg.matmul {{.*}} ins(%[[LHS_TILE]], %[[RHS_TILE]]
//       CHECK:               outs(%[[ACC]]
//       CHECK:             scf.yield %[[MATMUL]]
//       CHECK:           pcf.write_slice %[[REDUCE]] into %[[OUT_REF]][%[[M_OFF]], %[[N_OFF]]] [%[[M_SZ]], %[[N_SZ]]] [1, 1]
//       CHECK:           pcf.return
//       CHECK:       pcf.return
//       CHECK:   return %[[RESULT]]

// -----

// Fill + matmul: fill stays outside the pcf.generic nest.

func.func @fill_matmul(%lhs: tensor<64x128xf16>, %rhs: tensor<128x64xf16>,
                       %dest: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<64x64xf32>) -> tensor<64x64xf32>
  %result = linalg.matmul {
    lowering_config = {
      subgroup = [32, 32, 0], lane = [8, 8, 0], reduction = [0, 0, 16]
    }
  } ins(%lhs, %rhs : tensor<64x128xf16>, tensor<128x64xf16>)
                          outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// CHECK-LABEL: func @fill_matmul
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9_]+]]: tensor<64x128xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9_]+]]: tensor<128x64xf16>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<64x64xf32>
//       CHECK:   %[[CST:.+]] = arith.constant
//       CHECK:   %[[FILLED:.+]] = linalg.fill ins(%[[CST]]
//       CHECK:   %[[RESULT:.+]] = pcf.generic scope(#pcf.sequential)
//       CHECK:     execute(%[[LHS_REF:.+]] <- %[[LHS]], %[[RHS_REF:.+]] <- %[[RHS]], %[[OUT_REF:.+]] = %[[FILLED]])
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           %[[INIT_TILE:.+]] = pcf.read_slice %[[OUT_REF]]
//       CHECK:           %[[REDUCE:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[INIT_TILE]])
//       CHECK:             %[[LHS_TILE:.+]] = pcf.read_slice %[[LHS_REF]]
//       CHECK:             %[[RHS_TILE:.+]] = pcf.read_slice %[[RHS_REF]]
//       CHECK:             %[[MATMUL:.+]] = linalg.matmul {{.*}} ins(%[[LHS_TILE]], %[[RHS_TILE]]
//       CHECK:               outs(%[[ACC]]
//       CHECK:             scf.yield %[[MATMUL]]
//       CHECK:           pcf.write_slice %[[REDUCE]] into %[[OUT_REF]]
//       CHECK:   return %[[RESULT]]

// -----

// Purely parallel op: no reduction dimension means no reduction loop.

func.func @pointwise_add_no_reduction(
    %lhs: tensor<64x64xf32>, %rhs: tensor<64x64xf32>,
    %dest: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"],
    lowering_config = {subgroup = [32, 32], lane = [8, 8], reduction = [0, 0]}
  } ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%dest : tensor<64x64xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %sum = arith.addf %in0, %in1 : f32
    linalg.yield %sum : f32
  } -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// CHECK-LABEL: func @pointwise_add_no_reduction
//  CHECK-SAME:   %[[PW_LHS_ARG:[A-Za-z0-9_]+]]: tensor<64x64xf32>
//  CHECK-SAME:   %[[PW_RHS_ARG:[A-Za-z0-9_]+]]: tensor<64x64xf32>
//  CHECK-SAME:   %[[PW_DEST_ARG:[A-Za-z0-9_]+]]: tensor<64x64xf32>
//       CHECK:   %[[PW_RESULT:.+]] = pcf.generic scope(#pcf.sequential)
//       CHECK:     execute(%{{.+}} <- %[[PW_LHS_ARG]], %{{.+}} <- %[[PW_RHS_ARG]], %[[PW_OUT_REF:[A-Za-z0-9_]+]] = %[[PW_DEST_ARG]])
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           %[[PW_LHS_TILE:.+]] = pcf.read_slice %{{.+}}[%{{.+}}, %{{.+}}] [%{{.+}}, %{{.+}}] [1, 1] : !pcf.sref<64x64xf32, #pcf.sequential> to tensor<?x?xf32>
//       CHECK:           %[[PW_RHS_TILE:.+]] = pcf.read_slice %{{.+}}[%{{.+}}, %{{.+}}] [%{{.+}}, %{{.+}}] [1, 1] : !pcf.sref<64x64xf32, #pcf.sequential> to tensor<?x?xf32>
//       CHECK:           %[[PW_OUT_TILE:.+]] = pcf.read_slice %[[PW_OUT_REF]][%{{.+}}, %{{.+}}] [%{{.+}}, %{{.+}}] [1, 1] : !pcf.sref<64x64xf32, sync(#pcf.sequential)> to tensor<?x?xf32>
//       CHECK:           %[[PW_GENERIC:.+]] = linalg.generic {{.*}} ins(%[[PW_LHS_TILE]], %[[PW_RHS_TILE]]
//       CHECK:             outs(%[[PW_OUT_TILE]]
//   CHECK-NOT:           scf.for
//       CHECK:           pcf.write_slice %[[PW_GENERIC]] into %[[PW_OUT_REF]]
//       CHECK:   return %[[PW_RESULT]]

// -----

// Control-flow: tiling should happen in the then-region only.

func.func @matmul_in_if(%cond: i1, %lhs: tensor<64x128xf16>,
                        %rhs: tensor<128x64xf16>,
                        %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = scf.if %cond -> tensor<64x64xf32> {
    %then = linalg.matmul {
      lowering_config = {
        subgroup = [32, 32, 0], lane = [8, 8, 0], reduction = [0, 0, 16]
      }
    } ins(%lhs, %rhs : tensor<64x128xf16>, tensor<128x64xf16>)
                          outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %then : tensor<64x64xf32>
  } else {
    scf.yield %init : tensor<64x64xf32>
  }
  return %result : tensor<64x64xf32>
}

// CHECK-LABEL: func @matmul_in_if
//  CHECK-SAME:   %[[COND:[A-Za-z0-9_]+]]: i1
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9_]+]]: tensor<64x128xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9_]+]]: tensor<128x64xf16>
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<64x64xf32>
//       CHECK:   %[[RESULT:.+]] = scf.if %[[COND]] -> (tensor<64x64xf32>) {
//       CHECK:     %[[THEN:.+]] = pcf.generic scope(#pcf.sequential)
//       CHECK:       execute(%[[LHS_REF:.+]] <- %[[LHS]], %[[RHS_REF:.+]] <- %[[RHS]], %[[OUT_REF:.+]] = %[[INIT]])
//       CHECK:         pcf.generic scope(#pcf.sequential)
//       CHECK:           scf.for
//       CHECK:             linalg.matmul
//       CHECK:     scf.yield %[[THEN]]
//       CHECK:   } else {
//   CHECK-NOT:     pcf.generic scope(#pcf.sequential)
//       CHECK:     scf.yield %[[INIT]]
//       CHECK:   }
//       CHECK:   return %[[RESULT]]

// -----

// Dynamic shapes: matmul with unknown dimensions.

func.func @matmul_dynamic(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>,
                          %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul {
    lowering_config = {
      subgroup = [32, 32, 0], lane = [8, 8, 0], reduction = [0, 0, 16]
    }
  } ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>)
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
  %result = linalg.matmul {
    lowering_config = {
      subgroup = [32, 32, 0], lane = [8, 8, 0], reduction = [0, 0, 16]
    }
  } ins(%lhs, %rhs : tensor<63x127xf16>, tensor<127x63xf16>)
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

func.func @batch_matmul(%lhs: tensor<2x64x128xf16>, %rhs: tensor<2x128x64xf16>,
                         %init: tensor<2x64x64xf32>) -> tensor<2x64x64xf32> {
  %result = linalg.batch_matmul {
    lowering_config = {
      subgroup = [0, 32, 32, 0], lane = [0, 8, 8, 0], reduction = [0, 0, 0, 16]
    }
  } ins(%lhs, %rhs : tensor<2x64x128xf16>, tensor<2x128x64xf16>)
                                outs(%init : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>
  return %result : tensor<2x64x64xf32>
}

// CHECK-LABEL: func @batch_matmul
//  CHECK-SAME:   %[[B_LHS:[A-Za-z0-9_]+]]: tensor<2x64x128xf16>
//  CHECK-SAME:   %[[B_RHS:[A-Za-z0-9_]+]]: tensor<2x128x64xf16>
//  CHECK-SAME:   %[[B_INIT:[A-Za-z0-9_]+]]: tensor<2x64x64xf32>
//       CHECK:   %[[B_RES:.+]] = pcf.generic scope(#pcf.sequential)
//       CHECK:     execute(%[[B_LHS_REF:.+]] <- %[[B_LHS]], %[[B_RHS_REF:.+]] <- %[[B_RHS]], %[[B_OUT_REF:.+]] = %[[B_INIT]])
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           %[[B_M_OFF:.+]] = affine.apply
//       CHECK:           %[[B_M_SZ:.+]] = affine.min
//       CHECK:           %[[B_N_OFF:.+]] = affine.apply
//       CHECK:           %[[B_N_SZ:.+]] = affine.min
//       CHECK:           %[[B_INIT_TILE:.+]] = pcf.read_slice %[[B_OUT_REF]][0, %[[B_M_OFF]], %[[B_N_OFF]]] [2, %[[B_M_SZ]], %[[B_N_SZ]]] [1, 1, 1]
//       CHECK:           %[[B_RED:.+]] = scf.for %[[B_K_IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[B_ACC:.+]] = %[[B_INIT_TILE]])
//       CHECK:             %[[B_LHS_TILE:.+]] = pcf.read_slice %[[B_LHS_REF]][0, %[[B_M_OFF]], %[[B_K_IV]]] [2, %[[B_M_SZ]], 16] [1, 1, 1]
//       CHECK:             %[[B_RHS_TILE:.+]] = pcf.read_slice %[[B_RHS_REF]][0, %[[B_K_IV]], %[[B_N_OFF]]] [2, 16, %[[B_N_SZ]]] [1, 1, 1]
//       CHECK:             %[[B_MATMUL:.+]] = linalg.batch_matmul {{.*}} ins(%[[B_LHS_TILE]], %[[B_RHS_TILE]]
//       CHECK:               outs(%[[B_ACC]]
//       CHECK:             scf.yield %[[B_MATMUL]]
//       CHECK:           pcf.write_slice %[[B_RED]] into %[[B_OUT_REF]][0, %[[B_M_OFF]], %[[B_N_OFF]]] [2, %[[B_M_SZ]], %[[B_N_SZ]]] [1, 1, 1]
//       CHECK:   return %[[B_RES]]

// -----

// Asymmetric tile sizes: M=64, N=16 at subgroup level; M=8, N=4 at lane level.
// Verifies the pass handles non-square tiling correctly.

func.func @matmul_asymmetric(%lhs: tensor<64x128xf16>, %rhs: tensor<128x64xf16>,
                              %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.matmul {
    lowering_config = {
      subgroup = [64, 16, 0], lane = [8, 4, 0], reduction = [0, 0, 16]
    }
  } ins(%lhs, %rhs : tensor<64x128xf16>, tensor<128x64xf16>)
                          outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// CHECK-LABEL: func @matmul_asymmetric
//  CHECK-SAME:   %[[A_LHS:[A-Za-z0-9_]+]]: tensor<64x128xf16>
//  CHECK-SAME:   %[[A_RHS:[A-Za-z0-9_]+]]: tensor<128x64xf16>
//  CHECK-SAME:   %[[A_INIT:[A-Za-z0-9_]+]]: tensor<64x64xf32>
//       CHECK:   %[[A_RES:.+]] = pcf.generic scope(#pcf.sequential)
//       CHECK:     execute(%[[A_LHS_REF:.+]] <- %[[A_LHS]], %[[A_RHS_REF:.+]] <- %[[A_RHS]], %[[A_OUT_REF:.+]] = %[[A_INIT]])
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           %[[A_M_OFF:.+]] = affine.apply
//       CHECK:           %[[A_M_SZ:.+]] = affine.min
//       CHECK:           %[[A_N_OFF:.+]] = affine.apply
//       CHECK:           %[[A_N_SZ:.+]] = affine.min
//       CHECK:           %[[A_INIT_TILE:.+]] = pcf.read_slice %[[A_OUT_REF]][%[[A_M_OFF]], %[[A_N_OFF]]] [%[[A_M_SZ]], %[[A_N_SZ]]] [1, 1]
//       CHECK:           %[[A_RED:.+]] = scf.for %[[A_K_IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[A_ACC:.+]] = %[[A_INIT_TILE]])
//       CHECK:             %[[A_LHS_TILE:.+]] = pcf.read_slice %[[A_LHS_REF]][%[[A_M_OFF]], %[[A_K_IV]]] [%[[A_M_SZ]], 16] [1, 1]
//       CHECK:             %[[A_RHS_TILE:.+]] = pcf.read_slice %[[A_RHS_REF]][%[[A_K_IV]], %[[A_N_OFF]]] [16, %[[A_N_SZ]]] [1, 1]
//       CHECK:             %[[A_MATMUL:.+]] = linalg.matmul {{.*}} ins(%[[A_LHS_TILE]], %[[A_RHS_TILE]]
//       CHECK:               outs(%[[A_ACC]]
//       CHECK:             scf.yield %[[A_MATMUL]]
//       CHECK:           pcf.write_slice %[[A_RED]] into %[[A_OUT_REF]][%[[A_M_OFF]], %[[A_N_OFF]]] [%[[A_M_SZ]], %[[A_N_SZ]]] [1, 1]
//       CHECK:   return %[[A_RES]]

// -----

// Generic reduction: ensure a non-contraction reduction op also receives
// subgroup/lane/reduction tiling with explicit producer-consumer checks.

func.func @row_reduce_sum(%input: tensor<64x128xf32>,
                          %init: tensor<64xf32>) -> tensor<64xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"],
    lowering_config = {subgroup = [32, 0], lane = [8, 0], reduction = [0, 16]}
  } ins(%input : tensor<64x128xf32>) outs(%init : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  } -> tensor<64xf32>
  return %result : tensor<64xf32>
}

// CHECK-LABEL: func @row_reduce_sum
//  CHECK-SAME:   %[[R_INPUT:[A-Za-z0-9_]+]]: tensor<64x128xf32>
//  CHECK-SAME:   %[[R_INIT:[A-Za-z0-9_]+]]: tensor<64xf32>
//       CHECK:   %[[R_RES:.+]] = pcf.generic scope(#pcf.sequential)
//       CHECK:     execute(%[[R_IN_REF:.+]] <- %[[R_INPUT]], %[[R_OUT_REF:.+]] = %[[R_INIT]])
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           %[[R_M_OFF:.+]] = affine.apply
//       CHECK:           %[[R_M_SZ:.+]] = affine.min
//       CHECK:           %[[R_INIT_TILE:.+]] = pcf.read_slice %[[R_OUT_REF]][%[[R_M_OFF]]] [%[[R_M_SZ]]] [1]
//       CHECK:           %[[R_RED:.+]] = scf.for %[[R_K_IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[R_ACC:.+]] = %[[R_INIT_TILE]])
//       CHECK:             %[[R_IN_TILE:.+]] = pcf.read_slice %[[R_IN_REF]][%[[R_M_OFF]], %[[R_K_IV]]] [%[[R_M_SZ]], 16] [1, 1]
//       CHECK:             %[[R_GEN:.+]] = linalg.generic {{.*}} ins(%[[R_IN_TILE]]
//       CHECK:               outs(%[[R_ACC]]
//       CHECK:             scf.yield %[[R_GEN]]
//       CHECK:           pcf.write_slice %[[R_RED]] into %[[R_OUT_REF]][%[[R_M_OFF]]] [%[[R_M_SZ]]] [1]
//       CHECK:   return %[[R_RES]]

// -----

// Conv2D with multiple reduction dimensions (KH, KW, IC).
// 7 iteration dims: d0(N), d1(OH), d2(OW), d3(OC), d4(KH), d5(KW), d6(IC).
// Subgroup tiles [0,4,4,16,0,0,0], Lane tiles [0,2,2,4,0,0,0],
// Reduction tiles [0,0,0,0,1,1,8] to tile filter spatial dims to unit.

func.func @conv_2d_nhwc_hwcf(
    %input: tensor<1x14x14x32xf16>,
    %filter: tensor<3x3x32x64xf16>,
    %init: tensor<1x12x12x64xf32>) -> tensor<1x12x12x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>,
       strides = dense<1> : tensor<2xi64>,
       lowering_config = {
         subgroup = [0, 4, 4, 16, 0, 0, 0],
         lane = [0, 2, 2, 4, 0, 0, 0],
         reduction = [0, 0, 0, 0, 1, 1, 8]
       }}
      ins(%input, %filter : tensor<1x14x14x32xf16>, tensor<3x3x32x64xf16>)
      outs(%init : tensor<1x12x12x64xf32>) -> tensor<1x12x12x64xf32>
  return %result : tensor<1x12x12x64xf32>
}

// CHECK-LABEL: func @conv_2d_nhwc_hwcf
//  CHECK-SAME:   %[[C_INPUT:[A-Za-z0-9_]+]]: tensor<1x14x14x32xf16>
//  CHECK-SAME:   %[[C_FILTER:[A-Za-z0-9_]+]]: tensor<3x3x32x64xf16>
//  CHECK-SAME:   %[[C_INIT:[A-Za-z0-9_]+]]: tensor<1x12x12x64xf32>
//       CHECK:   %[[C_RES:.+]] = pcf.generic scope(#pcf.sequential)
//       CHECK:     execute(%[[C_IN_REF:.+]] <- %[[C_INPUT]], %[[C_FILTER_REF:.+]] <- %[[C_FILTER]], %[[C_OUT_REF:.+]] = %[[C_INIT]])
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           %[[C_OH_OFF:.+]] = affine.apply
//       CHECK:           %[[C_OH_SZ:.+]] = affine.min
//       CHECK:           %[[C_OW_OFF:.+]] = affine.apply
//       CHECK:           %[[C_OW_SZ:.+]] = affine.min
//       CHECK:           %[[C_OC_OFF:.+]] = affine.apply
//       CHECK:           %[[C_OC_SZ:.+]] = affine.min
//       CHECK:           %[[C_INIT_TILE:.+]] = pcf.read_slice %[[C_OUT_REF]][0, %[[C_OH_OFF]], %[[C_OW_OFF]], %[[C_OC_OFF]]] [1, %[[C_OH_SZ]], %[[C_OW_SZ]], %[[C_OC_SZ]]] [1, 1, 1, 1]
//       CHECK:           %[[C_KH_LOOP:.+]] = scf.for %[[C_KH:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[C_ACC0:.+]] = %[[C_INIT_TILE]])
//       CHECK:             %[[C_KW_LOOP:.+]] = scf.for %[[C_KW:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[C_ACC1:.+]] = %[[C_ACC0]])
//       CHECK:               %[[C_IC_LOOP:.+]] = scf.for %[[C_IC:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[C_ACC2:.+]] = %[[C_ACC1]])
//       CHECK:                 %[[C_IN_TILE:.+]] = pcf.read_slice %[[C_IN_REF]][0, %{{.+}}, %{{.+}}, %[[C_IC]]] [1, %[[C_OH_SZ]], %[[C_OW_SZ]], 8] [1, 1, 1, 1]
//       CHECK:                 %[[C_FILTER_TILE:.+]] = pcf.read_slice %[[C_FILTER_REF]][%[[C_KH]], %[[C_KW]], %[[C_IC]], %[[C_OC_OFF]]] [1, 1, 8, %[[C_OC_SZ]]] [1, 1, 1, 1]
//       CHECK:                 %[[C_CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//  CHECK-SAME:                   strides = dense<1> : tensor<2xi64>
//  CHECK-SAME:                   ins(%[[C_IN_TILE]], %[[C_FILTER_TILE]]
//  CHECK-SAME:                   outs(%[[C_ACC2]]
//       CHECK:                 scf.yield %[[C_CONV]]
//       CHECK:           pcf.write_slice %[[C_KH_LOOP]] into %[[C_OUT_REF]][0, %[[C_OH_OFF]], %[[C_OW_OFF]], %[[C_OC_OFF]]] [1, %[[C_OH_SZ]], %[[C_OW_SZ]], %[[C_OC_SZ]]] [1, 1, 1, 1]
//       CHECK:   return %[[C_RES]]

// -----

// Non-unit strides with non-divisible output dimensions must preserve
// stride-2 semantics while clamping tiles on boundaries.

func.func @conv_2d_nhwc_hwcf_stride2_nondivisible(
    %input: tensor<1x15x15x32xf16>,
    %filter: tensor<3x3x32x64xf16>,
    %init: tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>,
       strides = dense<2> : tensor<2xi64>,
       lowering_config = {
         subgroup = [0, 4, 4, 16, 0, 0, 0],
         lane = [0, 2, 2, 4, 0, 0, 0],
         reduction = [0, 0, 0, 0, 1, 1, 8]
       }}
      ins(%input, %filter : tensor<1x15x15x32xf16>, tensor<3x3x32x64xf16>)
      outs(%init : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  return %result : tensor<1x7x7x64xf32>
}

// CHECK-LABEL: func @conv_2d_nhwc_hwcf_stride2_nondivisible
//  CHECK-SAME:   %[[S2_INPUT:[A-Za-z0-9_]+]]: tensor<1x15x15x32xf16>
//  CHECK-SAME:   %[[S2_FILTER:[A-Za-z0-9_]+]]: tensor<3x3x32x64xf16>
//  CHECK-SAME:   %[[S2_INIT:[A-Za-z0-9_]+]]: tensor<1x7x7x64xf32>
//       CHECK:   %[[S2_RES:.+]] = pcf.generic scope(#pcf.sequential)
//       CHECK:     execute(%[[S2_IN_REF:.+]] <- %[[S2_INPUT]], %[[S2_FILTER_REF:.+]] <- %[[S2_FILTER]], %[[S2_OUT_REF:.+]] = %[[S2_INIT]])
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           %[[S2_OH_OFF:.+]] = affine.apply
//       CHECK:           %[[S2_OH_SZ:.+]] = affine.min
//       CHECK:           %[[S2_OW_OFF:.+]] = affine.apply
//       CHECK:           %[[S2_OW_SZ:.+]] = affine.min
//       CHECK:           %[[S2_OC_OFF:.+]] = affine.apply
//       CHECK:           %[[S2_OC_SZ:.+]] = affine.min
//       CHECK:           %[[S2_INIT_TILE:.+]] = pcf.read_slice %[[S2_OUT_REF]][0, %[[S2_OH_OFF]], %[[S2_OW_OFF]], %[[S2_OC_OFF]]] [1, %[[S2_OH_SZ]], %[[S2_OW_SZ]], %[[S2_OC_SZ]]] [1, 1, 1, 1]
//       CHECK:           %[[S2_KH_LOOP:.+]] = scf.for %[[S2_KH:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[S2_ACC0:.+]] = %[[S2_INIT_TILE]])
//       CHECK:             %[[S2_KW_LOOP:.+]] = scf.for %[[S2_KW:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[S2_ACC1:.+]] = %[[S2_ACC0]])
//       CHECK:               %[[S2_IC_LOOP:.+]] = scf.for %[[S2_IC:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[S2_ACC2:.+]] = %[[S2_ACC1]])
//       CHECK:                 %[[S2_IN_TILE:.+]] = pcf.read_slice %[[S2_IN_REF]][0, %{{.+}}, %{{.+}}, %[[S2_IC]]] [1, %{{.+}}, %{{.+}}, 8] [1, 1, 1, 1]
//       CHECK:                 %[[S2_FILTER_TILE:.+]] = pcf.read_slice %[[S2_FILTER_REF]][%[[S2_KH]], %[[S2_KW]], %[[S2_IC]], %[[S2_OC_OFF]]] [1, 1, 8, %[[S2_OC_SZ]]] [1, 1, 1, 1]
//       CHECK:                 %[[S2_CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//  CHECK-SAME:                   strides = dense<2> : tensor<2xi64>
//  CHECK-SAME:                   ins(%[[S2_IN_TILE]], %[[S2_FILTER_TILE]]
//  CHECK-SAME:                   outs(%[[S2_ACC2]]
//       CHECK:                 scf.yield %[[S2_CONV]]
//       CHECK:           pcf.write_slice %[[S2_KH_LOOP]] into %[[S2_OUT_REF]][0, %[[S2_OH_OFF]], %[[S2_OW_OFF]], %[[S2_OC_OFF]]] [1, %[[S2_OH_SZ]], %[[S2_OW_SZ]], %[[S2_OC_SZ]]] [1, 1, 1, 1]
//       CHECK:   return %[[S2_RES]]
