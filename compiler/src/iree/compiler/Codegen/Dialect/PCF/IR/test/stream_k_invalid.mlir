// RUN: iree-opt --split-input-file %s --verify-diagnostics

// Combiner region takes wrong number of arguments (1 instead of 2).
util.func private @combiner_wrong_num_args(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %off_m: index, %off_n: index) {
  // expected-error@+1 {{combiner region must take exactly 2 arguments}}
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref
      group(%num_in_group)
      combiner {
        ^bb0(%lhs: f32):
          pcf.yield %lhs : f32
      }
      writeback {
        ^bb0(%final: tensor<64x64xf32>):
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// Combiner arguments have wrong type (i32 instead of f32 matching tile).
util.func private @combiner_wrong_arg_type(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %off_m: index, %off_n: index) {
  // expected-error@+1 {{combiner argument type must match partial tile element type}}
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref
      group(%num_in_group)
      combiner {
        ^bb0(%lhs: i32, %rhs: i32):
          %sum = arith.addi %lhs, %rhs : i32
          pcf.yield %sum : i32
      }
      writeback {
        ^bb0(%final: tensor<64x64xf32>):
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// Combiner yields wrong type (i32 instead of f32).
util.func private @combiner_wrong_result_type(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %off_m: index, %off_n: index) {
  // expected-error@+1 {{combiner must yield a single value matching the element type}}
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref
      group(%num_in_group)
      combiner {
        ^bb0(%lhs: f32, %rhs: f32):
          %c1 = arith.constant 1 : i32
          pcf.yield %c1 : i32
      }
      writeback {
        ^bb0(%final: tensor<64x64xf32>):
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// Writeback region takes wrong argument type.
util.func private @writeback_wrong_arg_type(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %off_m: index, %off_n: index) {
  // expected-error@+1 {{writeback argument type must match partial tile type}}
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
        ^bb0(%final: tensor<64x64xi32>):
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// Output sref rank does not match partial tile rank.
util.func private @output_sref_rank_mismatch(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %off: index) {
  // expected-error@+1 {{output sref rank must be >= partial tile rank}}
  pcf.stream_k_recombine %partial
      into %out_ref[%off] [64] [1]
      scratch %scratch_ref counter %counter_ref
      group(%num_in_group)
      combiner {
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          pcf.yield %sum : f32
      }
      writeback {
        ^bb0(%final: tensor<64x64xf32>):
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// Output sref element type does not match partial tile element type.
util.func private @output_sref_eltype_mismatch(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xi32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %off_m: index, %off_n: index) {
  // expected-error@+1 {{output sref element type must match partial tile element type}}
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
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xi32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// Scratch sref element type mismatch.
util.func private @scratch_sref_wrong_eltype(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xi32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %off_m: index, %off_n: index) {
  // expected-error@+1 {{scratch sref element type must match partial tile element type}}
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
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xi32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// Number of offsets does not match dest sref rank.
util.func private @offsets_count_mismatch(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<i32, #pcf.test_scope>,
    %num_in_group: index,
    %off_m: index, %off_n: index, %off_extra: index) {
  // expected-error@+1 {{expected number of offsets to match output sref rank}}
  pcf.stream_k_recombine %partial
      into %out_ref[%off_m, %off_n, %off_extra] [64, 64] [1, 1]
      scratch %scratch_ref counter %counter_ref
      group(%num_in_group)
      combiner {
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          pcf.yield %sum : f32
      }
      writeback {
        ^bb0(%final: tensor<64x64xf32>):
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<i32, #pcf.test_scope>
  util.return
}

// -----

// Counter sref has wrong rank (non-scalar). Must be rank-0 for atomic RMW.
util.func private @counter_wrong_rank(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<4xi32, #pcf.test_scope>,
    %num_in_group: index,
    %off_m: index, %off_n: index) {
  // expected-error@+1 {{counter sref must be scalar (rank-0) for atomic increment}}
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
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<4xi32, #pcf.test_scope>
  util.return
}

// -----

// Counter sref has wrong element type (f32 instead of integer).
util.func private @counter_wrong_eltype(
    %partial: tensor<64x64xf32>,
    %out_ref: !pcf.sref<256x256xf32, #pcf.test_scope>,
    %scratch_ref: !pcf.sref<64x64xf32, #pcf.test_scope>,
    %counter_ref: !pcf.sref<f32, #pcf.test_scope>,
    %num_in_group: index,
    %off_m: index, %off_n: index) {
  // expected-error@+1 {{counter sref element type must be integer (i32 or i64)}}
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
          pcf.yield
      }
      : tensor<64x64xf32> into !pcf.sref<256x256xf32, #pcf.test_scope>
      scratch_type !pcf.sref<64x64xf32, #pcf.test_scope>
      counter_type !pcf.sref<f32, #pcf.test_scope>
  util.return
}
