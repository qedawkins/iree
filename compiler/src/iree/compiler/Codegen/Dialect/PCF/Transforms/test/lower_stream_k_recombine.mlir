// RUN: iree-opt --split-input-file \
// RUN:   --iree-pcf-lower-stream-k-recombine %s \
// RUN:   | FileCheck %s

// ============================================================================
// Test 1: Basic lowering of stream_k_recombine for f32 matmul tile.
//
// Expected lowered structure:
//   1. Atomic RMW increment on counter (via pcf.get_memref + memref.atomic_rmw).
//   2. Check: is_only = (num_in_group == 1).
//   3. Check: is_last = (old_count == num_in_group - 1).
//   4. If not sole contributor: write partial to scratch slot, release fence.
//      If last: acquire fence + accumulate all scratch slots + writeback.
//   5. If sole contributor: direct writeback.
// ============================================================================

func.func @basic_lowering(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %off_m: index, %off_n: index) {
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref
      group(%num_in_group)
      combiner {
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          pcf.yield %sum : f32
      }
      writeback {
        ^bb0(%final: tensor<64x64xf32>):
          pcf.write_slice %final into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
              : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  return
}

// Atomic increment on counter via get_memref.
// CHECK-LABEL: func @basic_lowering
//       CHECK:   pcf.get_memref %{{.+}}[] [] []
//       CHECK:   %[[OLD:.+]] = memref.atomic_rmw addi
//  CHECK-SAME:     : (i32, memref<i32
//
// Sole contributor check.
//       CHECK:   %[[IS_ONLY:.+]] = arith.cmpi eq, %{{.+}}, %c1
//
// Last contributor check.
//       CHECK:   arith.cmpi eq
//
// Not-sole branch: scratch write + release fence.
//       CHECK:   scf.if
//       CHECK:     pcf.write_slice %{{.+}} into %{{.+}}
//       CHECK:     pcf.fence release
//
// Last-contributor branch: acquire fence + accumulation + writeback.
//       CHECK:     scf.if
//       CHECK:       pcf.fence acquire
//       CHECK:       pcf.read_slice
//       CHECK:       scf.for
//       CHECK:         pcf.read_slice
//       CHECK:         linalg.generic
//       CHECK:           arith.addf
//       CHECK:       linalg.generic
//       CHECK:         arith.addf
//       CHECK:       pcf.write_slice
//
// Sole contributor writeback.
//       CHECK:   scf.if %[[IS_ONLY]]
//       CHECK:     pcf.write_slice

// -----

// ============================================================================
// Test 2: Sole contributor fast path (num_in_group == 1 statically).
//
// When num_in_group is statically 1, the entire scratch path should
// be optimized away. Only the writeback remains.
// ============================================================================

func.func @sole_contributor_static(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %off_m: index, %off_n: index) {
  %c1 = arith.constant 1 : index
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref
      group(%c1)
      combiner {
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          pcf.yield %sum : f32
      }
      writeback {
        ^bb0(%final: tensor<64x64xf32>):
          pcf.write_slice %final into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
              : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  return
}

// With num_in_group == 1, scratch and counter should not be touched.
// CHECK-LABEL: func @sole_contributor_static
// CHECK-NOT:     memref.atomic_rmw
// CHECK-NOT:     pcf.fence
// CHECK:         pcf.write_slice
// CHECK-NOT:     memref.atomic_rmw

// -----

// ============================================================================
// Test 3: Writeback with epilogue ops (bias add + relu).
//
// The writeback region should be inlined into both the last-contributor
// and sole-contributor branches. Epilogue ops must appear in both.
// ============================================================================

func.func @writeback_with_epilogue(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %bias: tensor<64xf32>,
    %off_m: index, %off_n: index) {
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref
      group(%num_in_group)
      combiner {
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          pcf.yield %sum : f32
      }
      writeback {
        ^bb0(%final: tensor<64x64xf32>):
          %c0 = arith.constant 0.0 : f32
          %biased = linalg.generic {
              indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                               affine_map<(d0, d1) -> (d1)>,
                               affine_map<(d0, d1) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel"]}
              ins(%final, %bias : tensor<64x64xf32>, tensor<64xf32>)
              outs(%final : tensor<64x64xf32>) {
            ^bb0(%in0: f32, %in1: f32, %out: f32):
              %add = arith.addf %in0, %in1 : f32
              %relu = arith.maximumf %add, %c0 : f32
              linalg.yield %relu : f32
          } -> tensor<64x64xf32>
          pcf.write_slice %biased into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
              : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  return
}

// Epilogue ops (linalg.generic for bias+relu) should appear in the
// last-contributor branch.
// CHECK-LABEL: func @writeback_with_epilogue
//       CHECK:   memref.atomic_rmw
//       CHECK:   scf.if
//       CHECK:     pcf.write_slice {{.*}} into %{{.+}}
//       CHECK:     pcf.fence release
//       CHECK:     scf.if
//       CHECK:       pcf.fence acquire
//       CHECK:       linalg.generic
//       CHECK:         arith.addf
//       CHECK:       linalg.generic
//       CHECK:         arith.maximumf
//       CHECK:       pcf.write_slice
//
// Sole contributor also gets the epilogue.
//       CHECK:   scf.if
//       CHECK:     linalg.generic
//       CHECK:       arith.maximumf
//       CHECK:     pcf.write_slice
