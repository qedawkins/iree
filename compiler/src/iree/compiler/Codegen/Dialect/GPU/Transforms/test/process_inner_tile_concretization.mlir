// RUN: iree-opt %s --iree-template-concretize-calls --split-input-file | FileCheck %s

// Test: process_inner_tile concretization with 7-block pingpong layout.
//
// The template.func has 7 blocks, 5 of which are unimplemented:
//   Block 0: Init accumulators from dest sref - (sg_id, lane_id, dest) -> type<0>
//   Block 1: Copy LHS to shared (concrete, not populated by process_inner_tile)
//   Block 2: Copy RHS to shared (concrete, not populated by process_inner_tile)
//   Block 3: Read LHS from shared - (buf_idx, sg_id, lane_id, alloc) -> type<1>
//   Block 4: Read RHS from shared - (buf_idx, sg_id, lane_id, alloc) -> type<2>
//   Block 5: Compute MMA - (acc: type<0>, lhs: type<1>, rhs: type<2>) -> type<0>
//   Block 6: Write results to dest - (sg_id, lane_id, result: type<0>, dest) -> ()
//
// Type bindings (3 distributed types with outer-then-inner layout):
//   type<0>: Per-subgroup distributed accumulator: tensor<2x1x4x1xf32>
//   type<1>: Per-subgroup distributed LHS: tensor<2x1x1x4xf16>
//   type<2>: Per-subgroup distributed RHS: tensor<1x1x4x1xf16>
//
// For MFMA_F32_16x16x16_F16 with bounds [32, 16, 16] and dist [1, 1]:
//   ACC: outer_M=32/16/1=2, outer_N=16/16/1=1, inner=[4,1] -> tensor<2x1x4x1xf32>
//   LHS: outer_M=32/16/1=2, outer_K=16/16=1, inner=[1,4] -> tensor<2x1x1x4xf16>
//   RHS: outer_K=16/16=1, outer_N=16/16/1=1, inner=[4,1] -> tensor<1x1x4x1xf16>

#map_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_acc = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  template.func @pingpong_template(
      %out: tensor<32x16xf32>, %k: index,
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>
  ) -> tensor<32x16xf32> {
    // Minimal main region.
    template.return %out : tensor<32x16xf32>
  } {
  // Block 0: Init accumulators from dest sref.
  ^bb0(%sg0: index, %lane0: index, %dest0: !pcf.sref<32x16xf32, #pcf.test_scope>):
    %0 = template.unimplemented -> !template.type<0>
  // Block 1: Copy LHS (concrete - not populated by process_inner_tile).
  ^bb1(%buf1: index, %k1: index, %sg1: index, %lane1: index, %src1: tensor<32x16xf16>, %alloc1: !pcf.sref<2x32x16xf16, #pcf.test_scope>):
    template.return
  // Block 2: Copy RHS (concrete - not populated by process_inner_tile).
  ^bb2(%buf2: index, %k2: index, %sg2: index, %lane2: index, %src2: tensor<16x16xf16>, %alloc2: !pcf.sref<2x16x16xf16, #pcf.test_scope>):
    template.return
  // Block 3: Read LHS from shared memory.
  ^bb3(%buf3: index, %sg3: index, %lane3: index, %alloc3: !pcf.sref<2x32x16xf16, #pcf.test_scope>):
    %1 = template.unimplemented -> !template.type<1>
  // Block 4: Read RHS from shared memory.
  ^bb4(%buf4: index, %sg4: index, %lane4: index, %alloc4: !pcf.sref<2x16x16xf16, #pcf.test_scope>):
    %2 = template.unimplemented -> !template.type<2>
  // Block 5: Compute MMA.
  ^bb5(%acc5: !template.type<0>, %lhs5: !template.type<1>, %rhs5: !template.type<2>):
    %3 = template.unimplemented -> !template.type<0>
  // Block 6: Write results to dest sref.
  ^bb6(%sg6: index, %lane6: index, %res6: !template.type<0>, %dest6: !pcf.sref<32x16xf32, #pcf.test_scope>):
    template.unimplemented
  }

  func.func @test_pingpong_concretization(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) -> tensor<32x16xf32> {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %result = iree_gpu.process_inner_tile
        bounds(%c32, %c16, %c16 : index, index, index)
        kind(#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>)
        indexing_maps = [#map_lhs, #map_rhs, #map_acc]
        iterator_types = ["parallel", "parallel", "reduction"]
        outer_dim_distribution = [1, 1]
        ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
        outs(%init : tensor<32x16xf32>)
        @pingpong_template -> tensor<32x16xf32>
    return %result : tensor<32x16xf32>
  }
}

// The process_inner_tile op should be converted to template.instance.
// getCallOperands() returns [outputs, reduction_bounds, inputs].
// CHECK-LABEL: func.func @test_pingpong_concretization
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9_]+]]: tensor<32x16xf16>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9_]+]]: tensor<16x16xf16>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9_]+]]: tensor<32x16xf32>

// CHECK:         %[[C16:.*]] = arith.constant 16 : index
// CHECK:         %[[RESULT:.*]] = template.instance ins(%[[INIT]], %[[C16]], %[[LHS]], %[[RHS]] : tensor<32x16xf32>, index, tensor<32x16xf16>, tensor<16x16xf16>) -> tensor<32x16xf32>

// Main region should pass through.
// CHECK:           template.return %[[INIT]] : tensor<32x16xf32>

// Block 0: Init accumulators. Reads from dest sref using per-lane MMA offsets.
// Distributed acc type: tensor<2x1x4x1xf32>.
// CHECK:         ^bb0(%[[SG0:.*]]: index, %[[LANE0:.*]]: index, %[[DEST0:.*]]: !pcf.sref<32x16xf32, #pcf.test_scope>):
// CHECK:           affine.delinearize_index %[[LANE0]] into (4, 16)
// CHECK:           tensor.empty() : tensor<2x1x4x1xf32>
// CHECK:           scf.for
// CHECK:             pcf.read_slice %[[DEST0]]{{.*}} : !pcf.sref<32x16xf32, #pcf.test_scope> to tensor<4x1xf32>
// CHECK:             tensor.insert_slice {{.*}} into {{.*}}[{{.*}}, {{.*}}, 0, 0] [1, 1, 4, 1]
// CHECK:           template.return %{{.*}} : tensor<2x1x4x1xf32>

// Blocks 1-2: Copy blocks are left as-is (concrete).
// CHECK:         ^bb1({{.*}}):
// CHECK-NEXT:      template.return
// CHECK:         ^bb2({{.*}}):
// CHECK-NEXT:      template.return

// Block 3: Read LHS from shared memory. buf_idx is extra leading offset.
// Distributed LHS type: tensor<2x1x1x4xf16>.
// CHECK:         ^bb3(%[[BUF3:.*]]: index, %[[SG3:.*]]: index, %[[LANE3:.*]]: index, %[[ALLOC3:.*]]: !pcf.sref<2x32x16xf16, #pcf.test_scope>):
// CHECK:           affine.delinearize_index %[[LANE3]] into (4, 16)
// CHECK:           tensor.empty() : tensor<2x1x1x4xf16>
// CHECK:           scf.for
// CHECK:             pcf.read_slice %[[ALLOC3]][%[[BUF3]], {{.*}}] {{.*}} : !pcf.sref<2x32x16xf16, #pcf.test_scope> to tensor<1x1x4xf16>
// CHECK:             tensor.collapse_shape {{.*}} : tensor<1x1x4xf16> into tensor<1x4xf16>
// CHECK:             tensor.insert_slice {{.*}} into {{.*}}[{{.*}}, {{.*}}, 0, 0] [1, 1, 1, 4]
// CHECK:           template.return %{{.*}} : tensor<2x1x1x4xf16>

// Block 4: Read RHS from shared memory.
// Distributed RHS type: tensor<1x1x4x1xf16>.
// CHECK:         ^bb4(%[[BUF4:.*]]: index, %[[SG4:.*]]: index, %[[LANE4:.*]]: index, %[[ALLOC4:.*]]: !pcf.sref<2x16x16xf16, #pcf.test_scope>):
// CHECK:           affine.delinearize_index %[[LANE4]] into (4, 16)
// CHECK:           tensor.empty() : tensor<1x1x4x1xf16>
// CHECK:           scf.for
// CHECK:             pcf.read_slice %[[ALLOC4]][%[[BUF4]], {{.*}}] {{.*}} : !pcf.sref<2x16x16xf16, #pcf.test_scope> to tensor<1x4x1xf16>
// CHECK:             tensor.collapse_shape {{.*}} : tensor<1x4x1xf16> into tensor<4x1xf16>
// CHECK:             tensor.insert_slice {{.*}} into {{.*}}[{{.*}}, {{.*}}, 0, 0] [1, 1, 4, 1]
// CHECK:           template.return %{{.*}} : tensor<1x1x4x1xf16>

// Block 5: Compute MMA using inner_tiled.
// CHECK:         ^bb5(%[[ACC5:.*]]: tensor<2x1x4x1xf32>, %[[LHS5:.*]]: tensor<2x1x1x4xf16>, %[[RHS5:.*]]: tensor<1x1x4x1xf16>):
// CHECK:           %[[INNER:.*]] = iree_codegen.inner_tiled ins(%[[LHS5]], %[[RHS5]]) outs(%[[ACC5]])
// CHECK-SAME:        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// CHECK-SAME:        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
// CHECK-SAME:        : tensor<2x1x1x4xf16>, tensor<1x1x4x1xf16> into tensor<2x1x4x1xf32>
// CHECK:           template.return %[[INNER]] : tensor<2x1x4x1xf32>

// Block 6: Write results to dest sref using per-lane MMA offsets.
// CHECK:         ^bb6(%[[SG6:.*]]: index, %[[LANE6:.*]]: index, %[[RES6:.*]]: tensor<2x1x4x1xf32>, %[[DEST6:.*]]: !pcf.sref<32x16xf32, #pcf.test_scope>):
// CHECK:           affine.delinearize_index %[[LANE6]] into (4, 16)
// CHECK:           scf.for
// CHECK:             tensor.extract_slice %[[RES6]]{{.*}} : tensor<2x1x4x1xf32> to tensor<4x1xf32>
// CHECK:             pcf.write_slice {{.*}} into %[[DEST6]]{{.*}} : tensor<4x1xf32> into !pcf.sref<32x16xf32, #pcf.test_scope>
// CHECK:           template.return

// CHECK:         return %[[RESULT]]
