// RUN: iree-opt --split-input-file --iree-pcf-fuse-consumers %s | FileCheck %s

// Tests that pcf.stream_k_recombine is decomposed into:
//   1. Conditional scratch write (if split).
//   2. Workgroup barrier.
//   3. Conditional atomic + recombine (thread-0 predicated).
//   4. Else: writeback for non-split path.
//   5. Final workgroup barrier.

#wg = #iree_codegen.workgroup_scope<linearize>

// Use function args for num_in_group/ordinal to prevent constant folding.
func.func @decompose_stream_k_recombine(
    %out: tensor<128x128xf32>,
    %num_in_group_arg: index,
    %ordinal_arg: index) -> tensor<128x128xf32> {
  %result = pcf.generic scope(#wg) initialize {
    %scratch = pcf.alloc() : !pcf.sref<1024x64xf32, #wg>
    %counter = pcf.alloc() : !pcf.sref<4xi32, #wg>
    pcf.yield %scratch, %counter
        : !pcf.sref<1024x64xf32, #wg>, !pcf.sref<4xi32, #wg>
  } -> (%arg0: !pcf.sref<1024x64xf32, #wg>,
        %arg1: !pcf.sref<4xi32, #wg>)
    execute(%ref = %out)[%id: index, %count: index]
         : (!pcf.sref<128x128xf32, #wg>)
        -> (tensor<128x128xf32>) {
    %cst = arith.constant 0.000000e+00 : f32

    // Compute a partial tile (fill for simplicity).
    %empty = tensor.empty() : tensor<64x64xf32>
    %partial = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>)
        -> tensor<64x64xf32>

    %c0 = arith.constant 0 : index

    pcf.stream_k_recombine %partial
        into %ref [0, 0] [64, 64] [1, 1]
        scratch %arg0 counter %arg1[%c0]
        group(%num_in_group_arg)
        ordinal(%ordinal_arg)
        combiner {
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      pcf.yield %sum : f32
    } writeback {
    ^bb0(%final: tensor<64x64xf32>):
      pcf.write_slice %final into %ref[0, 0] [64, 64] [1, 1]
          : tensor<64x64xf32>
          into !pcf.sref<128x128xf32, #wg>
      pcf.yield
    }
        : tensor<64x64xf32>
        into !pcf.sref<128x128xf32, #wg>
        scratch_type !pcf.sref<1024x64xf32, #wg>
        counter_type !pcf.sref<4xi32, #wg>

    pcf.return
  }
  return %result : tensor<128x128xf32>
}

// CHECK-LABEL: @decompose_stream_k_recombine

// Step 1: Split condition (num_in_group != 1).
//  CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//      CHECK: %[[IS_SPLIT:.*]] = arith.cmpi ne, %{{.*}}, %[[C1]] : index

// Step 2: Conditional scratch write / else writeback.
//      CHECK: scf.if %[[IS_SPLIT]] {
//      CHECK:   pcf.write_slice %{{.*}} into %{{.*}}
//      CHECK: } else {
//      CHECK:   pcf.write_slice %{{.*}} into %{{.*}}
//      CHECK: }

// Step 3: Workgroup barrier after scratch writes.
//      CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]

// Step 4: Conditional atomic + recombine for split tiles.
//      CHECK: scf.if %[[IS_SPLIT]] {
//      CHECK:   %[[TID:.*]] = gpu.thread_id x
//      CHECK:   %[[IS_T0:.*]] = arith.cmpi eq, %[[TID]]
//      CHECK:   scf.if %[[IS_T0]] {
// Release fence.
//      CHECK:     pcf.fence release %{{.*}}
// Atomic increment.
//      CHECK:     pcf.get_memref
//      CHECK:     %[[OLD:.*]] = memref.atomic_rmw addi
// Last contributor check.
//      CHECK:     %[[OLD_IDX:.*]] = arith.index_cast %[[OLD]]
//      CHECK:     arith.cmpi eq, %[[OLD_IDX]]
//      CHECK:     scf.if
// Acquire fence.
//      CHECK:       pcf.fence acquire %{{.*}}
// Read first partial tile.
//      CHECK:       pcf.read_slice
// Accumulation loop.
//      CHECK:       scf.for
//      CHECK:         pcf.read_slice
//      CHECK:         linalg.generic
//      CHECK:           arith.addf
//      CHECK:           linalg.yield
//      CHECK:         scf.yield
// Writeback of accumulated result.
//      CHECK:       pcf.write_slice
//      CHECK:     }
//      CHECK:   }
//      CHECK: }

// Step 5: Final workgroup barrier.
//      CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]

// The stream_k_recombine op should be gone.
// CHECK-NOT: pcf.stream_k_recombine
