// RUN: iree-opt %s --iree-template-concretize-calls --split-input-file | FileCheck %s

// Test: process_inner_tile concretization with all 5 implementation blocks.
//
// Per the design doc, the template.func has 5 unimplemented blocks:
//   Block 0: Allocate shared memory - () -> !template.type<3>
//   Block 1: Initialize accumulators - (subgroup_id, lane_id, dest) -> !template.type<2>
//   Block 2: Copy inputs to shared memory - (subgroup_id, lane_id, k_idx, allocs) -> ()
//   Block 3: Perform inner_tiled op - (subgroup_id, lane_id, acc, allocs) -> !template.type<2>
//   Block 4: Write results to destinations - (subgroup_id, lane_id, result, dest) -> ()
//
// Type bindings from process_inner_tile (per design doc section 2.2):
//   type<0>: Result types (same as output tensor types)
//   type<1>: pcf.sref with result shape/element, return_only_sync_scope
//   type<2>: Per-thread accumulator tensors (inner tile from MMA)
//   type<3>: pcf.sref for input shared memory

#map_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_acc = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  // Template function following the reference structure.
  // The main region orchestrates: allocate -> init -> loop(copy, compute) -> writeback.
  // Inputs (lhs, rhs) are captured from the enclosing scope.
  // Only the output tensor is passed as an explicit argument.
  template.func @matmul_inner_tile(%out: !template.type<0>) -> !template.type<0> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Convert tensor to pcf.sref for blocks that need shared memory semantics.
    // This cast will be resolved when the template is concretized within pcf.generic.
    %out_ref = builtin.unrealized_conversion_cast %out : !template.type<0> to !template.type<1>

    // Block 0: Allocate shared memory for inputs (returns pcf.sref).
    %allocs = template.branch 0() : () -> !template.type<3>

    // Block 1: Initialize per-thread accumulators from destination (pcf.sref).
    %acc_init = template.branch 1(%c0, %c0, %out_ref) :
        (index, index, !template.type<1>) -> !template.type<2>

    // Loop over k dimension (simplified - just one iteration for test).
    // Block 2: Copy inputs to shared memory (pcf.sref).
    template.branch 2(%c0, %c0, %c0, %allocs) :
        (index, index, index, !template.type<3>) -> ()

    // Block 3: Perform inner-tiled computation.
    %result_acc = template.branch 3(%c0, %c0, %acc_init, %allocs) :
        (index, index, !template.type<2>, !template.type<3>) -> !template.type<2>

    // Block 4: Write results back to destination (pcf.sref).
    template.branch 4(%c0, %c0, %result_acc, %out_ref) :
        (index, index, !template.type<2>, !template.type<1>) -> ()

    template.return %out : !template.type<0>
  } {
  // Block 0: Allocate shared memory (returns pcf.sref).
  ^bb0:
    %0 = template.unimplemented -> !template.type<3>
  // Block 1: Initialize accumulators from destination (pcf.sref).
  ^bb1(%subgroup_id_1: index, %lane_id_1: index, %dest_1: !template.type<1>):
    %1 = template.unimplemented -> !template.type<2>
  // Block 2: Copy inputs to shared memory (pcf.sref).
  ^bb2(%subgroup_id_2: index, %lane_id_2: index, %k_idx_2: index, %allocs_2: !template.type<3>):
    template.unimplemented
  // Block 3: Perform inner_tiled operation.
  ^bb3(%subgroup_id_3: index, %lane_id_3: index, %acc_3: !template.type<2>, %allocs_3: !template.type<3>):
    %2 = template.unimplemented -> !template.type<2>
  // Block 4: Write results to destinations (pcf.sref).
  ^bb4(%subgroup_id_4: index, %lane_id_4: index, %result_4: !template.type<2>, %dest_4: !template.type<1>):
    template.unimplemented
  }

  func.func @test_process_inner_tile_compute(
      %m: index, %n: index, %k: index,
      %lhs: tensor<4x4x16x16xf16>, %rhs: tensor<4x4x16x16xf16>,
      %init: tensor<4x4x16x16xf32>) -> tensor<4x4x16x16xf32> {
    %result = iree_gpu.process_inner_tile
        bounds(%m, %n, %k : index, index, index)
        kind(#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>)
        indexing_maps = [#map_lhs, #map_rhs, #map_acc]
        iterator_types = ["parallel", "parallel", "reduction"]
        outer_dim_distribution = [1, 1]
        ins(%lhs, %rhs : tensor<4x4x16x16xf16>, tensor<4x4x16x16xf16>)
        outs(%init : tensor<4x4x16x16xf32>)
        @matmul_inner_tile -> tensor<4x4x16x16xf32>
    return %result : tensor<4x4x16x16xf32>
  }
}

// The process_inner_tile op should be converted to template.instance.
// The instance receives only output as operand (inputs are captured).
// CHECK-LABEL: func.func @test_process_inner_tile_compute
// CHECK-SAME:    %[[M:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[K:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9_]+]]: tensor<4x4x16x16xf16>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9_]+]]: tensor<4x4x16x16xf16>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9_]+]]: tensor<4x4x16x16xf32>

// CHECK:         %[[RESULT:.*]] = template.instance ins(%[[INIT]] : tensor<4x4x16x16xf32>) -> tensor<4x4x16x16xf32>

// Main region should orchestrate all 5 blocks with correct type bindings.
// Type<1> = !pcf.sref<4x4x16x16xf32> (dest sref), Type<2> = tensor<4x4x4xf32> (distributed accumulator),
// Type<3> = !pcf.sref<4x4x16x16xf16> (shared mem sref).
// CHECK:           %[[INIT_REF:.*]] = builtin.unrealized_conversion_cast %[[INIT]] : tensor<4x4x16x16xf32> to !pcf.sref<4x4x16x16xf32, #pcf.test_scope>
// CHECK:           template.branch 0
// CHECK:           template.branch 1(%{{.*}}, %{{.*}}, %[[INIT_REF]]) : (index, index, !pcf.sref<4x4x16x16xf32, #pcf.test_scope>) -> tensor<4x4x4xf32>
// CHECK:           template.branch 2(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (index, index, index, !pcf.sref<4x4x16x16xf16, #pcf.test_scope>) -> ()
// CHECK:           template.branch 3(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (index, index, tensor<4x4x4xf32>, !pcf.sref<4x4x16x16xf16, #pcf.test_scope>) -> tensor<4x4x4xf32>
// CHECK:           template.branch 4(%{{.*}}, %{{.*}}, %{{.*}}, %[[INIT_REF]]) : (index, index, tensor<4x4x4xf32>, !pcf.sref<4x4x16x16xf32, #pcf.test_scope>) -> ()
// CHECK:           template.return %[[INIT]] : tensor<4x4x16x16xf32>

// Block 0: Allocate shared memory using pcf.alloc (returns pcf.sref).
// CHECK:         %[[ALLOC:.*]] = pcf.alloc() : !pcf.sref<4x4x16x16xf16, #pcf.test_scope>
// CHECK:         template.return %[[ALLOC]] : !pcf.sref<4x4x16x16xf16, #pcf.test_scope>

// Block 1: Initialize accumulators using pcf.read_slice from destination (pcf.sref).
// Type<2> is tensor<4x4x4xf32> (outer 4x4 + distributed inner 4 from MMA).
// Uses affine.delinearize_index on lane_id to compute per-lane offsets.
// For MFMA_F32_16x16x16_F16, delinearizes into (4, 16) -> 64 lanes.
// The read_slice returns tensor<4x4x4x1xf32> which is collapsed to tensor<4x4x4xf32>.
// CHECK:         ^bb1(%[[SG1:.*]]: index, %[[LANE1:.*]]: index, %[[DEST1:.*]]: !pcf.sref<4x4x16x16xf32, #pcf.test_scope>):
// CHECK:           tensor.empty() : tensor<4x4x16x16xf32>
// CHECK:           %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:           %[[DELIN1:.*]]:3 = affine.delinearize_index %[[LANE1]] into (4, 16) : index, index, index
// CHECK:           %[[HINT1_0:.*]] = iree_codegen.index_hint %[[DELIN1]]#0(#iree_gpu.lane_constant<16>) : index
// CHECK:           %[[HINT1_1:.*]] = iree_codegen.index_hint %[[DELIN1]]#1(#iree_gpu.lane_constant<16>) : index
// CHECK:           %[[HINT1_2:.*]] = iree_codegen.index_hint %[[DELIN1]]#2(#iree_gpu.lane_increment<16, aligned>) : index
// CHECK:           %[[LIN1:.*]] = affine.linearize_index disjoint [%[[HINT1_1]], %[[C0_1]]] by (4, 4) : index
// CHECK:           %[[FULL_SLICE:.*]] = pcf.read_slice %[[DEST1]][0, 0, %[[LIN1]], %[[HINT1_2]]] [4, 4, 4, 1] [1, 1, 1, 1] : !pcf.sref<4x4x16x16xf32, #pcf.test_scope> to tensor<4x4x4x1xf32>
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[FULL_SLICE]] {{\[\[}}0], [1], [2, 3]] : tensor<4x4x4x1xf32> into tensor<4x4x4xf32>
// CHECK:           template.return %[[COLLAPSED]] : tensor<4x4x4xf32>

// Block 2: Copy inputs to shared memory using pcf.write_slice.
// Extracts per-lane slices from captured inputs and writes to allocs (pcf.sref).
// CHECK:         ^bb2(%[[SG2:.*]]: index, %[[LANE2:.*]]: index, %[[KIDX2:.*]]: index, %[[ALLOCS2:.*]]: !pcf.sref<4x4x16x16xf16, #pcf.test_scope>):
// CHECK:           %[[C0_2:.*]] = arith.constant 0 : index
// CHECK:           %[[DELIN2_LHS:.*]]:3 = affine.delinearize_index %[[LANE2]] into (4, 16) : index, index, index
// CHECK:           iree_codegen.index_hint %[[DELIN2_LHS]]#0
// CHECK:           iree_codegen.index_hint %[[DELIN2_LHS]]#1
// CHECK:           iree_codegen.index_hint %[[DELIN2_LHS]]#2
// CHECK:           %[[LIN2_LHS:.*]] = affine.linearize_index disjoint
// CHECK:           %[[EXT2_LHS:.*]] = tensor.extract_slice %[[LHS]][0, 0, %{{.*}}, %[[LIN2_LHS]]] [4, 4, 1, 4] [1, 1, 1, 1] : tensor<4x4x16x16xf16> to tensor<4x4x1x4xf16>
// CHECK:           pcf.write_slice %[[EXT2_LHS]] into %[[ALLOCS2]][0, 0, %{{.*}}, %[[LIN2_LHS]]] [4, 4, 1, 4] [1, 1, 1, 1] : tensor<4x4x1x4xf16> into !pcf.sref<4x4x16x16xf16, #pcf.test_scope>
// CHECK:           %[[DELIN2_RHS:.*]]:3 = affine.delinearize_index %[[LANE2]] into (4, 16) : index, index, index
// CHECK:           %[[LIN2_RHS:.*]] = affine.linearize_index disjoint
// CHECK:           %[[EXT2_RHS:.*]] = tensor.extract_slice %[[RHS]][0, 0, %[[LIN2_RHS]], %{{.*}}] [4, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4x16x16xf16> to tensor<4x4x4x1xf16>
// CHECK:           pcf.write_slice %[[EXT2_RHS]] into %[[ALLOCS2]][0, 0, %[[LIN2_RHS]], %{{.*}}] [4, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4x4x1xf16> into !pcf.sref<4x4x16x16xf16, #pcf.test_scope>
// CHECK:           template.return

// Block 3: Perform distributed inner_tiled computation.
// Slices captured inputs using lane_id, creates inner_tiled with distributed=true.
// CHECK:         ^bb3(%[[SG3:.*]]: index, %[[LANE3:.*]]: index, %[[ACC3:.*]]: tensor<4x4x4xf32>, %[[ALLOCS3:.*]]: !pcf.sref<4x4x16x16xf16, #pcf.test_scope>):
// CHECK:           %[[DELIN3_LHS:.*]]:3 = affine.delinearize_index %[[LANE3]] into (4, 16) : index, index, index
// CHECK:           %[[LIN3_LHS:.*]] = affine.linearize_index disjoint
// CHECK:           %[[EXT3_LHS:.*]] = tensor.extract_slice %[[LHS]][0, 0, %{{.*}}, %[[LIN3_LHS]]] [4, 4, 1, 4] [1, 1, 1, 1] : tensor<4x4x16x16xf16> to tensor<4x4x1x4xf16>
// CHECK:           %[[DELIN3_RHS:.*]]:3 = affine.delinearize_index %[[LANE3]] into (4, 16) : index, index, index
// CHECK:           %[[LIN3_RHS:.*]] = affine.linearize_index disjoint
// CHECK:           %[[EXT3_RHS:.*]] = tensor.extract_slice %[[RHS]][0, 0, %[[LIN3_RHS]], %{{.*}}] [4, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4x16x16xf16> to tensor<4x4x4x1xf16>
// CHECK:           %[[INNER_RESULT:.*]] = iree_codegen.inner_tiled ins(%[[EXT3_LHS]], %[[EXT3_RHS]]) outs(%[[ACC3]])
// CHECK-SAME:        indexing_maps = [#map, #map1, #map2]
// CHECK-SAME:        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// CHECK-SAME:        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
// CHECK-SAME:        : tensor<4x4x1x4xf16>, tensor<4x4x4x1xf16> into tensor<4x4x4xf32>
// CHECK:           template.return %[[INNER_RESULT]] : tensor<4x4x4xf32>

// Block 4: Write results to destinations using pcf.write_slice.
// Uses tensor.expand_shape to match ranks, then pcf.write_slice to write.
// CHECK:         ^bb4(%[[SG4:.*]]: index, %[[LANE4:.*]]: index, %[[RES4:.*]]: tensor<4x4x4xf32>, %[[DEST4:.*]]: !pcf.sref<4x4x16x16xf32, #pcf.test_scope>):
// CHECK:           tensor.empty() : tensor<4x4x16x16xf32>
// CHECK:           %[[C0_4:.*]] = arith.constant 0 : index
// CHECK:           %[[DELIN4:.*]]:3 = affine.delinearize_index %[[LANE4]] into (4, 16) : index, index, index
// CHECK:           iree_codegen.index_hint %[[DELIN4]]#0
// CHECK:           iree_codegen.index_hint %[[DELIN4]]#1
// CHECK:           iree_codegen.index_hint %[[DELIN4]]#2
// CHECK:           %[[LIN4:.*]] = affine.linearize_index disjoint
// CHECK:           %[[EXPANDED:.*]] = tensor.expand_shape %[[RES4]] {{\[\[}}0], [1], [2, 3]] output_shape [4, 4, 4, 1] : tensor<4x4x4xf32> into tensor<4x4x4x1xf32>
// CHECK:           pcf.write_slice %[[EXPANDED]] into %[[DEST4]][0, 0, %[[LIN4]], %{{.*}}] [4, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4x4x1xf32> into !pcf.sref<4x4x16x16xf32, #pcf.test_scope>
// CHECK:           template.return

// CHECK:         return %[[RESULT]]
