// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-pcf, iree-codegen-gpu-apply-multi-level-tiling, cse))" --mlir-print-local-scope --split-input-file | FileCheck %s

// Basic matmul: workgroup=[64,64,0], subgroup=[32,32,0], thread=[8,8,0],
// reduction=[0,0,16].

func.func @matmul_pipeline(%lhs: tensor<256x512xf16>, %rhs: tensor<512x256xf16>,
                           %init: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
  %result = linalg.matmul {
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [64, 64, 0],
      reduction = [0, 0, 16],
      subgroup = [32, 32, 0],
      thread = [8, 8, 0]
    }>
  } ins(%lhs, %rhs : tensor<256x512xf16>, tensor<512x256xf16>)
    outs(%fill : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func @matmul_pipeline
//  CHECK-SAME:   %[[M_LHS:[A-Za-z0-9_]+]]: tensor<256x512xf16>
//  CHECK-SAME:   %[[M_RHS:[A-Za-z0-9_]+]]: tensor<512x256xf16>
//  CHECK-SAME:   %[[M_INIT:[A-Za-z0-9_]+]]: tensor<256x256xf32>
//       CHECK:   %[[M_WG:.+]] = pcf.loop scope(#iree_codegen.workgroup_scope<linearize>) count(%{{.+}}, %{{.+}})
//       CHECK:     execute(%[[M_LHS_REF:.+]] <- %[[M_LHS]], %[[M_RHS_REF:.+]] <- %[[M_RHS]], %[[M_OUT_REF:.+]] = %[[M_INIT]])
//       CHECK:       %[[M_WG_M_OFF:.+]] = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%{{.+}}]
//       CHECK:       %[[M_WG_M_SZ:.+]] = affine.min affine_map<()[s0] -> (64, s0 * -64 + 256)>()[%{{.+}}]
//       CHECK:       %[[M_WG_N_OFF:.+]] = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%{{.+}}]
//       CHECK:       %[[M_WG_N_SZ:.+]] = affine.min affine_map<()[s0] -> (64, s0 * -64 + 256)>()[%{{.+}}]
//       CHECK:       %[[M_WG_LHS_TILE:.+]] = pcf.read_slice %[[M_LHS_REF]][%[[M_WG_M_OFF]], 0] [%[[M_WG_M_SZ]], 512] [1, 1]
//       CHECK:       %[[M_WG_RHS_TILE:.+]] = pcf.read_slice %[[M_RHS_REF]][0, %[[M_WG_N_OFF]]] [512, %[[M_WG_N_SZ]]] [1, 1]
//       CHECK:       %[[M_WG_OUT_TILE:.+]] = pcf.read_slice %[[M_OUT_REF]][%[[M_WG_M_OFF]], %[[M_WG_N_OFF]]] [%[[M_WG_M_SZ]], %[[M_WG_N_SZ]]] [1, 1]
//       CHECK:       %[[M_WG_FILL:.+]] = linalg.fill
//       CHECK:       %[[M_SG:.+]] = pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:         execute(%[[M_SG_LHS_REF:.+]] <- %[[M_WG_LHS_TILE]], %[[M_SG_RHS_REF:.+]] <- %[[M_WG_RHS_TILE]], %[[M_SG_OUT_REF:.+]] = %[[M_WG_FILL]])
//       CHECK:           %[[M_SG_M_SZ:.+]] = affine.min affine_map<()[s0, s1] -> (32, s0 - s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK:           %[[M_SG_N_SZ:.+]] = affine.min affine_map<()[s0, s1] -> (32, s0 - s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK:           pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK:             %[[M_LN_M_OFF:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK:             %[[M_LN_M_SZ:.+]] = affine.min affine_map<()[s0, s1, s2] -> (8, s0 - s1 * 8 - s2 * 32)>()[%[[M_SG_M_SZ]], %{{.+}}, %{{.+}}]
//       CHECK:             %[[M_LN_N_OFF:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK:             %[[M_LN_N_SZ:.+]] = affine.min affine_map<()[s0, s1, s2] -> (8, s0 - s1 * 8 - s2 * 32)>()[%[[M_SG_N_SZ]], %{{.+}}, %{{.+}}]
//       CHECK:             %[[M_LN_INIT:.+]] = pcf.read_slice %[[M_SG_OUT_REF]][%[[M_LN_M_OFF]], %[[M_LN_N_OFF]]] [%[[M_LN_M_SZ]], %[[M_LN_N_SZ]]] [1, 1]
//       CHECK:             %[[M_RED:.+]] = scf.for %[[M_K_IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[M_ACC:.+]] = %[[M_LN_INIT]])
//       CHECK:               %[[M_LN_LHS:.+]] = pcf.read_slice %[[M_SG_LHS_REF]][%[[M_LN_M_OFF]], %[[M_K_IV]]] [%[[M_LN_M_SZ]], 16] [1, 1]
//       CHECK:               %[[M_LN_RHS:.+]] = pcf.read_slice %[[M_SG_RHS_REF]][%[[M_K_IV]], %[[M_LN_N_OFF]]] [16, %[[M_LN_N_SZ]]] [1, 1]
//       CHECK:               %[[M_MATMUL:.+]] = linalg.matmul {lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 16], subgroup = [32, 32, 0], thread = [8, 8, 0], workgroup = [64, 64, 0]}>}
//       CHECK:                 ins(%[[M_LN_LHS]], %[[M_LN_RHS]]
//       CHECK:                 outs(%[[M_ACC]]
//       CHECK:               scf.yield %[[M_MATMUL]]
//       CHECK:             pcf.write_slice %[[M_RED]] into %[[M_SG_OUT_REF]][%[[M_LN_M_OFF]], %[[M_LN_N_OFF]]] [%[[M_LN_M_SZ]], %[[M_LN_N_SZ]]] [1, 1]
//       CHECK:       pcf.write_slice %[[M_SG]] into %[[M_OUT_REF]][%[[M_WG_M_OFF]], %[[M_WG_N_OFF]]] [%[[M_WG_M_SZ]], %[[M_WG_N_SZ]]] [1, 1]
//       CHECK:   return %[[M_WG]]

// -----

// Matmul with promotion.

func.func @matmul_promoted(%lhs: tensor<256x512xf16>, %rhs: tensor<512x256xf16>,
                           %init: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
  %result = linalg.matmul {
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [64, 64, 0],
      reduction = [0, 0, 16],
      subgroup = [32, 32, 0],
      thread = [8, 8, 0],
      promote_operands = [0, 1]
    }>
  } ins(%lhs, %rhs : tensor<256x512xf16>, tensor<512x256xf16>)
    outs(%fill : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func @matmul_promoted
//       CHECK: pcf.generic scope(#iree_gpu.subgroup_scope) initialize {
//       CHECK:   pcf.index_symbol "n0.d0"
//       CHECK:   pcf.index_symbol "n0.d1"
//       CHECK:   pcf.index_symbol "n1.d0"
//       CHECK:   pcf.index_symbol "n1.d1"
//       CHECK:   pcf.yield
//       CHECK: }
//       CHECK: pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK: %[[P_RED:.+]] = scf.for %[[P_K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[P_ACC:.+]] = %{{.+}})
//       CHECK:   %[[P_PROM_LHS:.+]] = iree_gpu.promote_operand #iree_gpu.derived_thread_config %{{.+}} ["n0.d0", "n0.d1"]
//       CHECK:   %[[P_PROM_RHS:.+]] = iree_gpu.promote_operand #iree_gpu.derived_thread_config %{{.+}} ["n1.d0", "n1.d1"]
//       CHECK:   %[[P_LHS_TILE:.+]] = pcf.read_slice %[[P_PROM_LHS]]
//       CHECK:   %[[P_RHS_TILE:.+]] = pcf.read_slice %[[P_PROM_RHS]]
//       CHECK:   %[[P_MATMUL:.+]] = linalg.matmul {lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1], reduction = [0, 0, 16], subgroup = [32, 32, 0], thread = [8, 8, 0], workgroup = [64, 64, 0]}>}
//       CHECK:     ins(%[[P_LHS_TILE]], %[[P_RHS_TILE]]
//       CHECK:     outs(%[[P_ACC]]
//       CHECK:   scf.yield %[[P_MATMUL]]

// -----

// Matmul with MMA kind: subgroup tiling + MMA conversion.

func.func @matmul_mma(%lhs: tensor<256x512xf16>, %rhs: tensor<512x256xf16>,
                      %init: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
  %result = linalg.matmul {
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [64, 64, 0],
      reduction = [0, 0, 16],
      subgroup = [32, 32, 0],
      mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
      promote_operands = [0, 1]
    }>
  } ins(%lhs, %rhs : tensor<256x512xf16>, tensor<512x256xf16>)
    outs(%fill : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func @matmul_mma
//       CHECK: pcf.generic scope(#iree_gpu.subgroup_scope) initialize {
//       CHECK:   pcf.index_symbol "n0.d0"
//       CHECK:   pcf.index_symbol "n0.d1"
//       CHECK:   pcf.index_symbol "n1.d0"
//       CHECK:   pcf.index_symbol "n1.d1"
//       CHECK: }
//       CHECK: %[[MMA_SG_M_OFF:.+]] = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%{{.+}}]
//       CHECK: %[[MMA_SG_M_SZ:.+]] = affine.min affine_map<()[s0, s1] -> (32, s0 - s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: %[[MMA_SG_N_OFF:.+]] = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%{{.+}}]
//       CHECK: %[[MMA_SG_N_SZ:.+]] = affine.min affine_map<()[s0, s1] -> (32, s0 - s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK: %[[MMA_ACC_SUBVIEW:.+]] = pcf.subview %{{.+}}[%[[MMA_SG_M_OFF]], %[[MMA_SG_N_OFF]]] [%[[MMA_SG_M_SZ]], %[[MMA_SG_N_SZ]]] [1, 1]
//       CHECK: %[[MMA_ACC_M_OUTER:.+]] = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%[[MMA_SG_M_SZ]]]
//       CHECK: %[[MMA_ACC_N_OUTER:.+]] = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%[[MMA_SG_N_SZ]]]
//       CHECK: %[[MMA_ACC_PACKED:.+]] = pcf.expand_shape %[[MMA_ACC_SUBVIEW]]
//       CHECK: %[[MMA_DLN:.+]]:3 = affine.delinearize_index %{{.+}} into (4, 16)
//       CHECK: %[[MMA_LANE_M:.+]] = iree_codegen.index_hint %[[MMA_DLN]]#1(#iree_gpu.lane_constant<16>) : index
//       CHECK: %[[MMA_LANE_K:.+]] = iree_codegen.index_hint %[[MMA_DLN]]#2(#iree_gpu.lane_increment<16, aligned>) : index
//       CHECK: %[[MMA_ACC_INNER_M:.+]] = affine.linearize_index disjoint [%[[MMA_LANE_M]], %{{.+}}] by (4, 4) : index
//       CHECK: %[[MMA_ACC_READ:.+]] = pcf.read_slice %[[MMA_ACC_PACKED]][0, %[[MMA_ACC_INNER_M]], 0, %[[MMA_LANE_K]]] [%[[MMA_ACC_M_OUTER]], 4, %[[MMA_ACC_N_OUTER]], 1] [1, 1, 1, 1]
//       CHECK: %[[MMA_ACC_TILE:.+]] = linalg.transpose ins(%[[MMA_ACC_READ]]
//       CHECK-SAME: permutation = [0, 2, 1, 3]
//       CHECK: %[[MMA_RED:.+]] = scf.for %[[MMA_K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[MMA_ACC_ITER:.+]] = %[[MMA_ACC_TILE]]) -> (tensor<?x?x4x1xf32>)
//       CHECK:   %[[MMA_PROM_LHS:.+]] = iree_gpu.promote_operand #iree_gpu.derived_thread_config %{{.+}} ["n0.d0", "n0.d1"]
//       CHECK:   %[[MMA_LHS_SUBVIEW:.+]] = pcf.subview %[[MMA_PROM_LHS]][%[[MMA_SG_M_OFF]], %[[MMA_K]]] [%[[MMA_SG_M_SZ]], 16] [1, 1]
//       CHECK:   %[[MMA_LHS_PACKED:.+]] = pcf.expand_shape %[[MMA_LHS_SUBVIEW]]
//       CHECK:   %[[MMA_LHS_READ:.+]] = pcf.read_slice %[[MMA_LHS_PACKED]][0, %[[MMA_LANE_K]], 0, %[[MMA_ACC_INNER_M]]] [%[[MMA_ACC_M_OUTER]], 1, 1, 4] [1, 1, 1, 1]
//       CHECK:   %[[MMA_LHS_TILE:.+]] = linalg.transpose ins(%[[MMA_LHS_READ]]
//       CHECK-SAME: permutation = [0, 2, 1, 3]
//       CHECK:   %[[MMA_PROM_RHS:.+]] = iree_gpu.promote_operand #iree_gpu.derived_thread_config %{{.+}} ["n1.d0", "n1.d1"]
//       CHECK:   %[[MMA_RHS_SUBVIEW:.+]] = pcf.subview %[[MMA_PROM_RHS]][%[[MMA_K]], %[[MMA_SG_N_OFF]]] [16, %[[MMA_SG_N_SZ]]] [1, 1]
//       CHECK:   %[[MMA_RHS_PACKED:.+]] = pcf.expand_shape %[[MMA_RHS_SUBVIEW]]
//       CHECK:   %[[MMA_RHS_READ:.+]] = pcf.read_slice %[[MMA_RHS_PACKED]][0, %[[MMA_ACC_INNER_M]], 0, %[[MMA_LANE_K]]] [1, 4, %[[MMA_ACC_N_OUTER]], 1] [1, 1, 1, 1]
//       CHECK:   %[[MMA_RHS_TILE:.+]] = linalg.transpose ins(%[[MMA_RHS_READ]]
//       CHECK-SAME: permutation = [0, 2, 3, 1]
//       CHECK:   %[[MMA_INNER:.+]] = iree_codegen.inner_tiled ins(%[[MMA_LHS_TILE]], %[[MMA_RHS_TILE]]) outs(%[[MMA_ACC_ITER]])
//       CHECK-SAME: semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
//       CHECK:   scf.yield %[[MMA_INNER]]
//       CHECK: %[[MMA_RED_INTERLEAVED:.+]] = linalg.transpose ins(%[[MMA_RED]]
//       CHECK-SAME: permutation = [0, 2, 1, 3]
//       CHECK: pcf.write_slice %[[MMA_RED_INTERLEAVED]] into %[[MMA_ACC_PACKED]][0, %[[MMA_ACC_INNER_M]], 0, %[[MMA_LANE_K]]] [%[[MMA_ACC_M_OUTER]], 4, %[[MMA_ACC_N_OUTER]], 1] [1, 1, 1, 1]
//   CHECK-NOT: linalg.pack
//   CHECK-NOT: linalg.unpack
//   CHECK-NOT: tensor.extract_slice
//   CHECK-NOT: tensor.insert_slice
//   CHECK-NOT: scf.forall

// -----

// Batch matmul flavor.

func.func @batch_matmul_pipeline(%lhs: tensor<4x256x512xf16>,
                                 %rhs: tensor<4x512x256xf16>,
                                 %init: tensor<4x256x256xf32>)
    -> tensor<4x256x256xf32> {
  %result = linalg.batch_matmul {
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [0, 64, 64, 0],
      reduction = [0, 0, 0, 16],
      subgroup = [0, 32, 32, 0],
      thread = [0, 8, 8, 0]
    }>
  } ins(%lhs, %rhs : tensor<4x256x512xf16>, tensor<4x512x256xf16>)
    outs(%init : tensor<4x256x256xf32>) -> tensor<4x256x256xf32>
  return %result : tensor<4x256x256xf32>
}

// CHECK-LABEL: func @batch_matmul_pipeline
//       CHECK: pcf.loop scope(#iree_codegen.workgroup_scope<linearize>)
//       CHECK: pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK: %[[B_SG_M_SZ:.+]] = affine.min affine_map<()[s0, s1] -> (32, s0 - s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: %[[B_SG_N_SZ:.+]] = affine.min affine_map<()[s0, s1] -> (32, s0 - s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK: %[[B_LN_M_OFF:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: %[[B_LN_M_SZ:.+]] = affine.min affine_map<()[s0, s1, s2] -> (8, s0 - s1 * 8 - s2 * 32)>()[%[[B_SG_M_SZ]], %{{.+}}, %{{.+}}]
//       CHECK: %[[B_LN_N_OFF:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: %[[B_LN_N_SZ:.+]] = affine.min affine_map<()[s0, s1, s2] -> (8, s0 - s1 * 8 - s2 * 32)>()[%[[B_SG_N_SZ]], %{{.+}}, %{{.+}}]
//       CHECK: %[[B_INIT_TILE:.+]] = pcf.read_slice %{{.+}}[0, %[[B_LN_M_OFF]], %[[B_LN_N_OFF]]] [4, %[[B_LN_M_SZ]], %[[B_LN_N_SZ]]] [1, 1, 1]
//       CHECK: %[[B_RED:.+]] = scf.for %[[B_K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[B_ACC:.+]] = %[[B_INIT_TILE]])
//       CHECK:   %[[B_LHS_TILE:.+]] = pcf.read_slice %{{.+}}[0, %[[B_LN_M_OFF]], %[[B_K]]] [4, %[[B_LN_M_SZ]], 16] [1, 1, 1]
//       CHECK:   %[[B_RHS_TILE:.+]] = pcf.read_slice %{{.+}}[0, %[[B_K]], %[[B_LN_N_OFF]]] [4, 16, %[[B_LN_N_SZ]]] [1, 1, 1]
//       CHECK:   %[[B_MATMUL:.+]] = linalg.batch_matmul
//       CHECK:     ins(%[[B_LHS_TILE]], %[[B_RHS_TILE]]
//       CHECK:     outs(%[[B_ACC]]

// -----

// Purely parallel pointwise flavor (no reduction loop).

func.func @pointwise_pipeline(%lhs: tensor<256x256xf32>,
                              %rhs: tensor<256x256xf32>,
                              %init: tensor<256x256xf32>)
    -> tensor<256x256xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"],
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [64, 64],
      subgroup = [32, 32],
      thread = [8, 8]
    }>
  } ins(%lhs, %rhs : tensor<256x256xf32>, tensor<256x256xf32>)
    outs(%init : tensor<256x256xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %sum = arith.addf %in0, %in1 : f32
    linalg.yield %sum : f32
  } -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func @pointwise_pipeline
//       CHECK: pcf.loop scope(#iree_codegen.workgroup_scope<linearize>)
//       CHECK: pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK: %[[PW_SG_M_SZ:.+]] = affine.min affine_map<()[s0, s1] -> (32, s0 - s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: %[[PW_SG_N_SZ:.+]] = affine.min affine_map<()[s0, s1] -> (32, s0 - s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK: %[[PW_LN_M_OFF:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: %[[PW_LN_M_SZ:.+]] = affine.min affine_map<()[s0, s1, s2] -> (8, s0 - s1 * 8 - s2 * 32)>()[%[[PW_SG_M_SZ]], %{{.+}}, %{{.+}}]
//       CHECK: %[[PW_LN_N_OFF:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: %[[PW_LN_N_SZ:.+]] = affine.min affine_map<()[s0, s1, s2] -> (8, s0 - s1 * 8 - s2 * 32)>()[%[[PW_SG_N_SZ]], %{{.+}}, %{{.+}}]
//       CHECK: %[[PW_LHS_TILE:.+]] = pcf.read_slice %{{.+}}[%[[PW_LN_M_OFF]], %[[PW_LN_N_OFF]]] [%[[PW_LN_M_SZ]], %[[PW_LN_N_SZ]]] [1, 1]
//       CHECK: %[[PW_RHS_TILE:.+]] = pcf.read_slice %{{.+}}[%[[PW_LN_M_OFF]], %[[PW_LN_N_OFF]]] [%[[PW_LN_M_SZ]], %[[PW_LN_N_SZ]]] [1, 1]
//       CHECK: %[[PW_OUT_TILE:.+]] = pcf.read_slice %{{.+}}[%[[PW_LN_M_OFF]], %[[PW_LN_N_OFF]]] [%[[PW_LN_M_SZ]], %[[PW_LN_N_SZ]]] [1, 1]
//       CHECK: %[[PW_GENERIC:.+]] = linalg.generic
//       CHECK:   ins(%[[PW_LHS_TILE]], %[[PW_RHS_TILE]]
//       CHECK:   outs(%[[PW_OUT_TILE]]
//       CHECK: pcf.write_slice %[[PW_GENERIC]] into %{{.+}}[%[[PW_LN_M_OFF]], %[[PW_LN_N_OFF]]] [%[[PW_LN_M_SZ]], %[[PW_LN_N_SZ]]] [1, 1]
//   CHECK-NOT: scf.for

// -----

// Generic reduction flavor.

func.func @row_reduce_pipeline(%input: tensor<256x512xf32>,
                               %init: tensor<256xf32>) -> tensor<256xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"],
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [64, 0],
      subgroup = [32, 0],
      thread = [8, 0],
      reduction = [0, 16]
    }>
  } ins(%input : tensor<256x512xf32>) outs(%init : tensor<256xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  } -> tensor<256xf32>
  return %result : tensor<256xf32>
}

// CHECK-LABEL: func @row_reduce_pipeline
//       CHECK: pcf.loop scope(#iree_codegen.workgroup_scope<linearize>)
//       CHECK: pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK: %[[RR_SG_M_SZ:.+]] = affine.min affine_map<()[s0, s1] -> (32, s0 - s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK: %[[RR_LN_M_OFF:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32)>()[%{{.+}}, %{{.+}}]
//       CHECK: %[[RR_LN_M_SZ:.+]] = affine.min affine_map<()[s0, s1, s2] -> (8, s0 - s1 * 8 - s2 * 32)>()[%[[RR_SG_M_SZ]], %{{.+}}, %{{.+}}]
//       CHECK: %[[RR_INIT_TILE:.+]] = pcf.read_slice %{{.+}}[%[[RR_LN_M_OFF]]] [%[[RR_LN_M_SZ]]] [1]
//       CHECK: %[[RR_RED:.+]] = scf.for %[[RR_K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[RR_ACC:.+]] = %[[RR_INIT_TILE]])
//       CHECK:   %[[RR_IN_TILE:.+]] = pcf.read_slice %{{.+}}[%[[RR_LN_M_OFF]], %[[RR_K]]] [%[[RR_LN_M_SZ]], 16] [1, 1]
//       CHECK:   %[[RR_GENERIC:.+]] = linalg.generic
//       CHECK:     ins(%[[RR_IN_TILE]]
//       CHECK:     outs(%[[RR_ACC]]
//       CHECK:   scf.yield %[[RR_GENERIC]]
//       CHECK: pcf.write_slice %[[RR_RED]] into %{{.+}}[%[[RR_LN_M_OFF]]] [%[[RR_LN_M_SZ]]] [1]
