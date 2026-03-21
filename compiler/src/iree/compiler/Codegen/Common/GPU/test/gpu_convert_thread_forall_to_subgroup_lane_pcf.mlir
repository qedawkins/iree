// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-convert-thread-forall-to-subgroup-lane-pcf))" --split-input-file | FileCheck %s

// Test converting 1D scf.forall with gpu.thread mapping to nested pcf.generic
// with subgroup scope (outer) and lane scope (inner).

func.func @test_1d_thread_mapping(%init: tensor<64xf32>) -> tensor<64xf32> {
  %result = scf.forall (%i) in (64) shared_outs(%out = %init) -> tensor<64xf32> {
    %slice = tensor.extract_slice %out[%i] [1] [1] : tensor<64xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i] [1] [1] : tensor<1xf32> into tensor<64xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return %result : tensor<64xf32>
}

// CHECK-LABEL: func.func @test_1d_thread_mapping
//  CHECK-SAME:   %[[INIT:.+]]: tensor<64xf32>
//       CHECK: %[[RESULT:.+]] = pcf.generic
//  CHECK-SAME:   scope(#iree_gpu.subgroup_scope)
//       CHECK:   execute(%[[REF:.+]] = %[[INIT]])[%[[SUBGROUP_ID:.+]]: index, %[[NUM_SUBGROUPS:.+]]: index]
//       CHECK:   pcf.generic
//  CHECK-SAME:     scope(#iree_gpu.lane_scope)
//       CHECK:     execute[%[[LANE_ID:.+]]: index, %[[SUBGROUP_SIZE:.+]]: index]
//       CHECK:     %[[LIN_ID:.+]] = affine.linearize_index [%[[SUBGROUP_ID]], %[[LANE_ID]]] by (%[[NUM_SUBGROUPS]], %[[SUBGROUP_SIZE]])
//       CHECK:     %[[TOTAL_COUNT:.+]] = arith.muli %[[NUM_SUBGROUPS]], %[[SUBGROUP_SIZE]]
//       CHECK:     %[[TILE_SIZE:.+]] = arith.ceildivui %{{.+}}, %[[TOTAL_COUNT]]
//       CHECK:     %[[START:.+]] = arith.muli %[[LIN_ID]], %[[TILE_SIZE]]
//       CHECK:     %[[END_UNCLAMPED:.+]] = arith.addi %[[START]], %[[TILE_SIZE]]
//       CHECK:     %[[END:.+]] = arith.minui %[[END_UNCLAMPED]]
//       CHECK:     scf.forall (%[[IV:.+]]) = (%[[START]]) to (%[[END]])
//       CHECK:       pcf.write_slice %{{.+}} into %[[REF]][%[[IV]]]
//       CHECK:     pcf.return
//       CHECK:   pcf.return
//       CHECK: return %[[RESULT]]

// -----

// Test that foralls without thread mapping are not converted.

func.func @test_no_thread_mapping(%init: tensor<64xf32>) -> tensor<64xf32> {
  %result = scf.forall (%i) in (64) shared_outs(%out = %init) -> tensor<64xf32> {
    %slice = tensor.extract_slice %out[%i] [1] [1] : tensor<64xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i] [1] [1] : tensor<1xf32> into tensor<64xf32>
    }
  }
  return %result : tensor<64xf32>
}

// CHECK-LABEL: func.func @test_no_thread_mapping
//       CHECK:   scf.forall
//   CHECK-NOT:   pcf.generic

// -----

// Test 2D thread mapping with delinearization.

func.func @test_2d_thread_mapping(%init: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %result = scf.forall (%i, %j) in (64, 128) shared_outs(%out = %init) -> tensor<64x128xf32> {
    %slice = tensor.extract_slice %out[%i, %j] [1, 1] [1, 1] : tensor<64x128xf32> to tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i, %j] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<64x128xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  return %result : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_2d_thread_mapping
//  CHECK-SAME:   %[[INIT:.+]]: tensor<64x128xf32>
//       CHECK: pcf.generic
//  CHECK-SAME:   scope(#iree_gpu.subgroup_scope)
//       CHECK:   execute(%[[REF:.+]] = %[[INIT]])[%[[SUBGROUP_ID:.+]]: index, %[[NUM_SUBGROUPS:.+]]: index]
//       CHECK:   pcf.generic
//  CHECK-SAME:     scope(#iree_gpu.lane_scope)
//       CHECK:     execute[%[[LANE_ID:.+]]: index, %[[SUBGROUP_SIZE:.+]]: index]
//       CHECK:     %[[LIN_ID:.+]] = affine.linearize_index [%[[SUBGROUP_ID]], %[[LANE_ID]]] by (%[[NUM_SUBGROUPS]], %[[SUBGROUP_SIZE]])
//       CHECK:     %[[TOTAL_COUNT:.+]] = arith.muli %[[NUM_SUBGROUPS]], %[[SUBGROUP_SIZE]]
//       CHECK:     scf.forall (%[[IV:.+]]) =
//       CHECK:       %[[INDICES:.+]]:2 = affine.delinearize_index %[[IV]] into (64, 128)
//       CHECK:       pcf.write_slice %{{.+}} into %[[REF]][%[[INDICES]]#0, %[[INDICES]]#1]
//       CHECK:     pcf.return
//       CHECK:   pcf.return

// -----

// Test that warp mapping also uses nested scopes (warp maps to subgroup+lane).

func.func @test_warp_mapping(%init: tensor<32xf32>) -> tensor<32xf32> {
  %result = scf.forall (%i) in (32) shared_outs(%out = %init) -> tensor<32xf32> {
    %slice = tensor.extract_slice %out[%i] [1] [1] : tensor<32xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i] [1] [1] : tensor<1xf32> into tensor<32xf32>
    }
  } {mapping = [#gpu.thread<x>]}
  return %result : tensor<32xf32>
}

// CHECK-LABEL: func.func @test_warp_mapping
//       CHECK: pcf.generic
//  CHECK-SAME:   scope(#iree_gpu.subgroup_scope)
//       CHECK:   pcf.generic
//  CHECK-SAME:     scope(#iree_gpu.lane_scope)

// -----

// Test converting barrier_region + alloc_tensor chain to PCF ops.

func.func @test_barrier_chain_to_pcf(%arg0: tensor<128x128xf16>,
    %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = scf.forall (%i) in (64) shared_outs(%out = %init)
      -> tensor<128x128xf32> {
    %alloc = bufferization.alloc_tensor()
        {memory_space = #gpu.address_space<workgroup>}
        : tensor<128x4xf16>
    %barrier = iree_gpu.barrier_region ins(%alloc : tensor<128x4xf16>) {
    ^bb0(%shared: tensor<128x4xf16>):
      iree_gpu.yield %shared : tensor<128x4xf16>
    } : tensor<128x4xf16>
    %expanded = tensor.expand_shape %barrier [[0, 1], [2]]
        output_shape [8, 16, 4]
        : tensor<128x4xf16> into tensor<8x16x4xf16>
    %slice = tensor.extract_slice %expanded[%i, 0, 0] [1, 16, 4] [1, 1, 1]
        : tensor<8x16x4xf16> to tensor<16x4xf16>
    %read = vector.transfer_read %slice[%c0, %c0], %cst
        {in_bounds = [true, true]}
        : tensor<16x4xf16>, vector<16x4xf16>
    %out_slice = tensor.extract_slice %out[%i, 0] [16, 4] [1, 1]
        : tensor<128x128xf32> to tensor<16x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %out_slice into %out[%i, 0] [16, 4] [1, 1]
          : tensor<16x4xf32> into tensor<128x128xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return %result : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @test_barrier_chain_to_pcf
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:     pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK:       %[[ALLOC:.+]] = pcf.alloc() : !pcf.sref<128x4xf16, #iree_gpu.subgroup_scope>
//       CHECK:       %[[BARRIER:.+]] = iree_gpu.barrier_region ins(%[[ALLOC]] : !pcf.sref<128x4xf16, #iree_gpu.subgroup_scope>)
//       CHECK:       ^bb0(%[[SHARED:.+]]: !pcf.sref<128x4xf16, #iree_gpu.subgroup_scope>):
//       CHECK:         iree_gpu.yield %[[SHARED]]
//       CHECK:       %[[EXPAND:.+]] = pcf.expand_shape %[[BARRIER]] {{\[}}[0, 1], [2]]
//  CHECK-SAME:         : !pcf.sref<128x4xf16, #iree_gpu.subgroup_scope> into !pcf.sref<8x16x4xf16, #iree_gpu.subgroup_scope>
//       CHECK:       %[[SUB:.+]] = pcf.subview %[[EXPAND]]
//  CHECK-SAME:         : !pcf.sref<8x16x4xf16, #iree_gpu.subgroup_scope> to !pcf.sref<16x4xf16, #iree_gpu.subgroup_scope>
//       CHECK:       pcf.read_slice %[[SUB]]
//  CHECK-SAME:         : !pcf.sref<16x4xf16, #iree_gpu.subgroup_scope> to vector<16x4xf16>
