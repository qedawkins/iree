// RUN: iree-opt --split-input-file --iree-pcf-fuse-consumers %s | FileCheck %s

// Tests that FuseStreamKRecombineIntoGeneric handles the case where
// numInGroup and contributorOrdinal are computed AFTER the producer
// pcf.generic.  The pattern must hoist these computations (and their
// transitive dependencies) before the producer in order to fire.
//
// The key arithmetic uses function arguments (%tile_idx, %tiles_per_wg, %id)
// to prevent constant folding from collapsing the hoisted ops.

#wg = #iree_codegen.workgroup_scope<linearize>

func.func @fuse_recombine_with_hoisting(
    %init: tensor<64x64xf32>,
    %out_ref: !pcf.sref<128x128xf32, #wg>,
    %scratch: !pcf.sref<1024x64xf32, #wg>,
    %counter: !pcf.sref<4xi32, #wg>,
    %tile_idx: index, %tiles_per_wg: index, %id: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f32

  // Producer generic -- computes a partial tile.
  %partial = pcf.generic scope(#wg)
    execute(%tile_ref = %init)[%wid: index, %wcount: index]
         : (!pcf.sref<64x64xf32, sync(#wg)>)
        -> (tensor<64x64xf32>) {
    %empty = tensor.empty() : tensor<64x64xf32>
    %fill = linalg.fill ins(%cst : f32)
        outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    pcf.write_slice %fill into %tile_ref[0, 0] [64, 64] [1, 1]
        : tensor<64x64xf32>
        into !pcf.sref<64x64xf32, sync(#wg)>
    pcf.return
  }

  // ---- Operands that are defined AFTER the producer ----
  // These mimic the real pipeline where numInGroup and
  // contributorOrdinal are computed from tile_idx after the
  // inner pcf.generic returns.
  %hi = arith.addi %tile_idx, %c3 : index
  %first_wg = arith.divui %tile_idx, %tiles_per_wg : index
  %last_wg = arith.divui %hi, %tiles_per_wg : index
  %span = arith.subi %last_wg, %first_wg : index
  %num_in_group = arith.addi %span, %c1 : index
  %ordinal = arith.subi %id, %first_wg : index

  // Scratch offset for this tile (must also be hoisted).
  %c64 = arith.constant 64 : index
  %scratch_offset = arith.muli %ordinal, %c64 : index

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

// Verify: hoisted arithmetic BEFORE the producer generic.
// CHECK-LABEL: @fuse_recombine_with_hoisting

// Hoisted numInGroup and ordinal computations.
//       CHECK:   arith.addi %arg4
//       CHECK:   arith.divui
//       CHECK:   arith.divui
//       CHECK:   arith.subi
//  CHECK-DAG:   %[[NIG:.*]] = arith.addi
//  CHECK-DAG:   %[[ORD:.*]] = arith.subi %arg6

// Split condition.
//       CHECK:   %[[IS_SPLIT:.*]] = arith.cmpi ne, %[[NIG]]
//       CHECK-SAME: : index

// Producer generic has sync_on_return=true.
//       CHECK:   pcf.generic sync true scope

// Inside producer: conditional scratch write.
//       CHECK:     scf.if %[[IS_SPLIT]] {
//       CHECK:       pcf.write_slice {{.*}} into %arg2
//       CHECK:     }

// Original write_slice to output ref unchanged.
//       CHECK:     pcf.write_slice {{.*}} into %ref
//       CHECK:     pcf.return

// Broadcast dword allocation for atomic result broadcast.
//       CHECK:   pcf.alloc() : !pcf.sref<1xi32

// Post-producer conditional: split path.
//       CHECK:   scf.if %[[IS_SPLIT]] {

// Thread-0 predicated atomic with broadcast.
//       CHECK:     gpu.thread_id x
//       CHECK:     arith.cmpi eq
//       CHECK:     scf.if
//       CHECK:       pcf.fence release
//       CHECK:       memref.atomic_rmw addi
//       CHECK:       memref.store

// Barrier: all threads sync after broadcast dword write.
//       CHECK:     gpu.barrier memfence [#gpu.address_space<workgroup>]

// ALL threads load broadcast result and check last contributor.
//       CHECK:     memref.load
//       CHECK:     arith.index_cast
//       CHECK:     arith.cmpi eq

// ALL threads: if last, acquire fence + recombine + writeback.
//       CHECK:     scf.if
//       CHECK:       pcf.fence acquire
//       CHECK:       pcf.read_slice
//       CHECK:       scf.for
//       CHECK:         pcf.read_slice
//       CHECK:         linalg.generic
//       CHECK:           arith.addf
//       CHECK:         scf.yield

// Writeback.
//       CHECK:       pcf.write_slice {{.*}} into %arg1

// Else branch: non-split path (direct writeback).
//       CHECK:   } else {
//       CHECK:     pcf.write_slice {{.*}} into %arg1
//       CHECK:   }

// Final workgroup barrier.
//       CHECK:   gpu.barrier memfence [#gpu.address_space<workgroup>]

// The stream_k_recombine op should be gone.
//   CHECK-NOT:   pcf.stream_k_recombine
