// RUN: iree-opt --split-input-file --iree-pcf-fuse-consumers %s | FileCheck %s

// Tests that pcf.stream_k_recombine is fused into its producer pcf.generic:
//   1. Producer gets sync_on_return=true.
//   2. Conditional scratch write inside the producer body.
//   3. Conditional atomic + recombine + writeback (then) / writeback (else).
//   4. Final workgroup barrier.

#wg = #iree_codegen.workgroup_scope<linearize>

// Use function args for srefs so the recombine can sit outside the producer.
func.func @fuse_stream_k_recombine(
    %init: tensor<64x64xf32>,
    %out_ref: !pcf.sref<128x128xf32, #wg>,
    %scratch: !pcf.sref<1024x64xf32, #wg>,
    %counter: !pcf.sref<4xi32, #wg>,
    %num_in_group: index,
    %ordinal: index) {
  %c0 = arith.constant 0 : index

  // Producer generic computing a partial tile.
  %partial = pcf.generic scope(#wg)
    execute(%tile_ref = %init)[%id: index, %count: index]
         : (!pcf.sref<64x64xf32, sync(#wg)>)
        -> (tensor<64x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<64x64xf32>
    %fill = linalg.fill ins(%cst : f32)
        outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    pcf.write_slice %fill into %tile_ref[0, 0] [64, 64] [1, 1]
        : tensor<64x64xf32>
        into !pcf.sref<64x64xf32, sync(#wg)>
    pcf.return
  }

  // Recombine consuming the producer's result.
  pcf.stream_k_recombine %partial
      into %out_ref [0, 0] [64, 64] [1, 1]
      scratch %scratch counter %counter[%c0]
      group(%num_in_group)
      ordinal(%ordinal)
      combiner {
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    pcf.yield %sum : f32
  } writeback {
  ^bb0(%final: tensor<64x64xf32>):
    pcf.write_slice %final into %out_ref[0, 0] [64, 64] [1, 1]
        : tensor<64x64xf32>
        into !pcf.sref<128x128xf32, #wg>
    pcf.yield
  }
      : tensor<64x64xf32>
      into !pcf.sref<128x128xf32, #wg>
      scratch_type !pcf.sref<1024x64xf32, #wg>
      counter_type !pcf.sref<4xi32, #wg>

  return
}

// CHECK-LABEL: @fuse_stream_k_recombine

// Step 1: Split condition (num_in_group != 1).
//  CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//      CHECK: %[[IS_SPLIT:.*]] = arith.cmpi ne, %{{.*}}, %[[C1]] : index

// Step 2: Producer generic has sync_on_return=true.
//      CHECK: pcf.generic sync scope

// Step 2b: Inside producer, conditional scratch write.
//      CHECK: scf.if %[[IS_SPLIT]] {
//      CHECK:   pcf.write_slice %{{.*}} into %{{.*}}
//      CHECK: }

// Original write_slice to output ref unchanged.
//      CHECK: pcf.write_slice %{{.*}} into %{{.*}}
//      CHECK: pcf.return

// Step 3: Post-producer conditional.
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
// First writeback.
//      CHECK:       pcf.write_slice
//      CHECK:     }
//      CHECK:   }
//      CHECK: } else {
// Second writeback (non-split path).
//      CHECK:   pcf.write_slice
//      CHECK: }

// Step 4: Final workgroup barrier.
//      CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]

// The stream_k_recombine op should be gone.
// CHECK-NOT: pcf.stream_k_recombine
