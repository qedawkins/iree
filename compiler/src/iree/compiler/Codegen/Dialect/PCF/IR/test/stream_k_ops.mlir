// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// Basic f32 recombine with addf combiner and simple writeback.
// CHECK-LABEL: @basic_recombine_f32
util.func private @basic_recombine_f32(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %counter_idx: index, %ordinal: index,
    %off_m: index, %off_n: index) {
  // CHECK: pcf.stream_k_recombine %[[PARTIAL:.+]]
  // CHECK-NEXT: into %[[OUT:.+]] [%[[OFF_M:.+]], %[[OFF_N:.+]]] [64, 64] [1, 1]
  // CHECK-NEXT: scratch %[[SCRATCH:.+]] counter %[[CTR:.+]][%[[IDX:.+]]]
  // CHECK-NEXT: group(%[[NUM:.+]])
  // CHECK-NEXT: ordinal(%[[ORD:.+]])
  // CHECK-NEXT: combiner
  // CHECK-NEXT: ^bb0(%[[LHS:.+]]: f32, %[[RHS:.+]]: f32):
  // CHECK-NEXT:   arith.addf %[[LHS]], %[[RHS]]
  // CHECK-NEXT:   pcf.yield %{{.+}} : f32
  // CHECK-NEXT: } writeback {
  // CHECK-NEXT: ^bb0(%[[FINAL:.+]]: tensor<64x64xf32>):
  // CHECK-NEXT:   pcf.write_slice %[[FINAL]] into %[[OUT]][%[[OFF_M]], %[[OFF_N]]] [64, 64] [1, 1] : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
  // CHECK-NEXT:   pcf.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
  // CHECK-NEXT: scratch_type !pcf.sref<64x64xf32, #pcf.test_scope> counter_type !pcf.sref<i32, #pcf.test_scope>
  pcf.stream_k_recombine %partial
      into %out_ref [%off_m, %off_n] [64, 64] [1, 1]
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
  util.return
}

// -----

// Integer element type with addi combiner.
// CHECK-LABEL: @basic_recombine_i32
util.func private @basic_recombine_i32(
    %partial: tensor<32x32xi32>,
    %out_ref: !pcf.sref<128x128xi32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<32x32xi32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %counter_idx: index, %ordinal: index,
    %off_m: index, %off_n: index) {
  // CHECK: pcf.stream_k_recombine %{{.+}}
  // CHECK: : tensor<32x32xi32> into !pcf.sref<128x128xi32, #pcf.test_scope>
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [32, 32] [1, 1]
      scratch %scratch_ref counter %counter_ref[%counter_idx]
      group(%num_in_group)
      ordinal(%ordinal)
      combiner {
        ^bb0(%lhs: i32, %rhs: i32):
          %sum = arith.addi %lhs, %rhs : i32
          pcf.yield %sum : i32
      }
      writeback {
        ^bb0(%final: tensor<32x32xi32>):
          pcf.write_slice %final into %out_ref[%off_m, %off_n] [32, 32] [1, 1]
              : tensor<32x32xi32> into !pcf.sref<128x128xi32, #pcf.test_scope>
          pcf.yield
      }
      : tensor<32x32xi32> into !pcf.sref<128x128xi32, #pcf.test_scope>
      scratch_type !pcf.sref<32x32xi32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// All offsets and sizes are dynamic.
// CHECK-LABEL: @dynamic_offsets_sizes
util.func private @dynamic_offsets_sizes(
    %partial: tensor<?x?xf32>,
    %out_ref: !pcf.sref<?x?xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<?x?xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %counter_idx: index, %ordinal: index,
    %off_m: index, %off_n: index,
    %sz_m: index, %sz_n: index) {
  // CHECK: pcf.stream_k_recombine %{{.+}}
  // CHECK-NEXT: into %{{.+}} [%{{.+}}, %{{.+}}] [%{{.+}}, %{{.+}}] [1, 1]
  // CHECK: : tensor<?x?xf32> into !pcf.sref<?x?xf32, #pcf.test_scope>
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [%sz_m, %sz_n] [1, 1]
      scratch %scratch_ref counter %counter_ref[%counter_idx]
      group(%num_in_group)
      ordinal(%ordinal)
      combiner {
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          pcf.yield %sum : f32
      }
      writeback {
        ^bb0(%final: tensor<?x?xf32>):
          pcf.write_slice %final into %out_ref[%off_m, %off_n] [%sz_m, %sz_n] [1, 1]
              : tensor<?x?xf32> into !pcf.sref<?x?xf32, #pcf.test_scope>
          pcf.yield
      }
      : tensor<?x?xf32> into !pcf.sref<?x?xf32, #pcf.test_scope>
      scratch_type !pcf.sref<?x?xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// Writeback region with epilogue operations (bias add + relu).
// CHECK-LABEL: @writeback_with_epilogue
util.func private @writeback_with_epilogue(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %counter_idx: index, %ordinal: index,
    %bias: tensor<64xf32>,
    %off_m: index, %off_n: index) {
  // CHECK: pcf.stream_k_recombine
  // CHECK: writeback
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  // CHECK: pcf.write_slice
  // CHECK: pcf.yield
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
          // Bias add + ReLU epilogue.
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
  util.return
}

// -----

// f16 element type.
// CHECK-LABEL: @f16_element_type
util.func private @f16_element_type(
    %partial: tensor<64x64xf16>,
    %out_ref: !pcf.sref<256x256xf16, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf16, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %counter_idx: index, %ordinal: index,
    %off_m: index, %off_n: index) {
  // CHECK: pcf.stream_k_recombine
  // CHECK: : tensor<64x64xf16> into !pcf.sref<256x256xf16, #pcf.test_scope>
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref[%counter_idx]
      group(%num_in_group)
      ordinal(%ordinal)
      combiner {
        ^bb0(%lhs: f16, %rhs: f16):
          %sum = arith.addf %lhs, %rhs : f16
          pcf.yield %sum : f16
      }
      writeback {
        ^bb0(%final: tensor<64x64xf16>):
          pcf.write_slice %final into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
              : tensor<64x64xf16> into !pcf.sref<256x256xf16, #pcf.test_scope>
          pcf.yield
      }
      : tensor<64x64xf16> into !pcf.sref<256x256xf16, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf16, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// 1D partial tile (pure reduction to vector).
// CHECK-LABEL: @recombine_1d
util.func private @recombine_1d(
    %partial: tensor<128xf32>,
    %out_ref: !pcf.sref<1024xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<128xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %counter_idx: index, %ordinal: index,
    %off: index) {
  // CHECK: pcf.stream_k_recombine
  // CHECK-NEXT: into %{{.+}} [%{{.+}}] [128] [1]
  // CHECK: : tensor<128xf32> into !pcf.sref<1024xf32, #pcf.test_scope>
  pcf.stream_k_recombine %partial
      into %out_ref[%off] [128] [1]
      scratch %scratch_ref counter %counter_ref[%counter_idx]
      group(%num_in_group)
      ordinal(%ordinal)
      combiner {
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          pcf.yield %sum : f32
      }
      writeback {
        ^bb0(%final: tensor<128xf32>):
          pcf.write_slice %final into %out_ref[%off] [128] [1]
              : tensor<128xf32> into !pcf.sref<1024xf32, #pcf.test_scope>
          pcf.yield
      }
      : tensor<128xf32> into !pcf.sref<1024xf32, #pcf.test_scope>
      scratch_type !pcf.sref<128xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// 3D partial tile shape.
// CHECK-LABEL: @recombine_3d
util.func private @recombine_3d(
    %partial: tensor<4x8x16xf32>,
    %out_ref: !pcf.sref<16x32x64xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<4x8x16xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %counter_idx: index, %ordinal: index,
    %o0: index, %o1: index, %o2: index) {
  // CHECK: pcf.stream_k_recombine
  // CHECK-NEXT: into %{{.+}} [%{{.+}}, %{{.+}}, %{{.+}}] [4, 8, 16] [1, 1, 1]
  // CHECK: : tensor<4x8x16xf32> into !pcf.sref<16x32x64xf32, #pcf.test_scope>
  pcf.stream_k_recombine %partial
      into %out_ref[%o0, %o1, %o2] [4, 8, 16] [1, 1, 1]
      scratch %scratch_ref counter %counter_ref[%counter_idx]
      group(%num_in_group)
      ordinal(%ordinal)
      combiner {
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          pcf.yield %sum : f32
      }
      writeback {
        ^bb0(%final: tensor<4x8x16xf32>):
          pcf.write_slice %final into %out_ref[%o0, %o1, %o2] [4, 8, 16] [1, 1, 1]
              : tensor<4x8x16xf32> into !pcf.sref<16x32x64xf32, #pcf.test_scope>
          pcf.yield
      }
      : tensor<4x8x16xf32> into !pcf.sref<16x32x64xf32, #pcf.test_scope>
      scratch_type !pcf.sref<4x8x16xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}
