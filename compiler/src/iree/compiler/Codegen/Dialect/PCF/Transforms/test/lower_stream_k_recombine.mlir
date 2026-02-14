// RUN: iree-opt --split-input-file \
// RUN:   --iree-pcf-lower-stream-k-recombine %s \
// RUN:   | FileCheck %s

// ============================================================================
// Test 1: Basic lowering of stream_k_recombine for f32 matmul tile.
//
// Expected lowered 3-zone structure:
//   Zone A: scf.if isSplit { pcf.write_slice to scratch }
//   Barrier (intra-workgroup sync)
//   scf.if isSplit {
//     Zone B: thread_id==0 { release fence + atomic + isLast check }
//     Zone C: if isLast { acquire fence + combine loop + writeback }
//   } else {
//     Direct writeback (distributed, for fusion via 696)
//   }
//   Barrier (final sync)
// ============================================================================

func.func @basic_lowering(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %counter_idx: index, %ordinal: index,
    %off_m: index, %off_n: index) {
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref[%counter_idx]
      group(%num_in_group)
      ordinal(%ordinal)
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

// isSplit condition.
// CHECK-LABEL: func @basic_lowering
//       CHECK:   %[[IS_SPLIT:.+]] = arith.cmpi ne, %{{.+}}, %c1
//
// Zone A: Distributed scratch write (before barrier, before atomic).
//       CHECK:   scf.if %[[IS_SPLIT]]
//       CHECK:     pcf.write_slice %{{.+}} into %{{.+}}
//
// Intra-workgroup barrier.
//       CHECK:   gpu.barrier
//
// Split/non-split branch.
//       CHECK:   scf.if %[[IS_SPLIT]]
//
// Thread-0 guard for Zone B+C.
//       CHECK:     scf.if
//
// Zone B: Release fence + atomic.
//       CHECK:       pcf.fence release
//       CHECK:       pcf.get_memref %{{.+}}[] [] []
//       CHECK:       memref.atomic_rmw addi
//
// Zone C: Recombine + writeback (last contributor).
//       CHECK:       scf.if
//       CHECK:         pcf.fence acquire
//       CHECK:         pcf.read_slice
//       CHECK:         scf.for
//       CHECK:           pcf.read_slice
//       CHECK:           linalg.generic
//       CHECK:             arith.addf
//       CHECK:         pcf.write_slice
//
// Non-split else branch: direct writeback (distributed).
//       CHECK:   } else {
//       CHECK:     pcf.write_slice
//
// Final barrier.
//       CHECK:   gpu.barrier

// -----

// ============================================================================
// Test 2: Sole contributor fast path (num_in_group == 1 statically).
//
// When num_in_group is statically 1, the entire scratch path should
// be optimized away. Only the writeback remains. No barriers needed.
// ============================================================================

func.func @sole_contributor_static(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %counter_idx: index, %ordinal: index,
    %off_m: index, %off_n: index) {
  %c1 = arith.constant 1 : index
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref[%counter_idx]
      group(%c1)
      ordinal(%ordinal)
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
// CHECK-NOT:     gpu.barrier
// CHECK:         pcf.write_slice
// CHECK-NOT:     memref.atomic_rmw

// -----

// ============================================================================
// Test 3: Writeback with epilogue ops (bias add + relu).
//
// The writeback region should be inlined into both the last-contributor
// (Zone C) and non-split (else branch) paths. Epilogue ops must appear
// in both.
// ============================================================================

func.func @writeback_with_epilogue(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %bias: tensor<64xf32>,
    %counter_idx: index, %ordinal: index,
    %off_m: index, %off_n: index) {
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref[%counter_idx]
      group(%num_in_group)
      ordinal(%ordinal)
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

// Zone A: scratch write.
// CHECK-LABEL: func @writeback_with_epilogue
//       CHECK:   scf.if
//       CHECK:     pcf.write_slice {{.*}} into %{{.+}}
//
// Barrier.
//       CHECK:   gpu.barrier
//
// Zone B: atomic (inside thread-0 guard inside split branch).
//       CHECK:   scf.if
//       CHECK:     scf.if
//       CHECK:       pcf.fence release
//       CHECK:       memref.atomic_rmw
//
// Zone C: Recombine + writeback with epilogue (last contributor).
//       CHECK:       scf.if
//       CHECK:         pcf.fence acquire
//       CHECK:         linalg.generic
//       CHECK:           arith.addf
//       CHECK:         linalg.generic
//       CHECK:           arith.maximumf
//       CHECK:         pcf.write_slice
//
// Non-split else branch also gets the epilogue.
//       CHECK:   } else {
//       CHECK:     linalg.generic
//       CHECK:       arith.maximumf
//       CHECK:     pcf.write_slice
