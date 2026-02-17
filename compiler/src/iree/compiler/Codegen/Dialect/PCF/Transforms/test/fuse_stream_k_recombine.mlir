// RUN: iree-opt --split-input-file --iree-pcf-fuse-consumers %s | FileCheck %s

// Tests that FuseStreamKRecombineIntoGeneric decomposes stream_k_recombine
// into a fully distributed 4-phase pattern:
//   Phase 1: Distributed scratch write via pcf.generic scope nests.
//   Phase 2: Barrier + thread-0 atomic + broadcast dword.
//   Phase 3: Distributed recombine from scratch (if last contributor).
//   Phase 4: Distributed writeback (non-split path).
//
// Also verifies operand hoisting: numInGroup and contributorOrdinal are
// computed AFTER the producer but must be hoisted before it.

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

  // Operands defined AFTER the producer (must be hoisted).
  %hi = arith.addi %tile_idx, %c3 : index
  %first_wg = arith.divui %tile_idx, %tiles_per_wg : index
  %last_wg = arith.divui %hi, %tiles_per_wg : index
  %span = arith.subi %last_wg, %first_wg : index
  %num_in_group = arith.addi %span, %c1 : index
  %ordinal = arith.subi %id, %first_wg : index

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

// CHECK-LABEL: @fuse_recombine_with_hoisting

// Hoisted operands and split condition.
//       CHECK:   arith.subi
//       CHECK:   %[[NIG:.*]] = arith.addi %{{.*}}, %c1
//       CHECK:   %[[IS_SPLIT:.*]] = arith.cmpi ne, %[[NIG]]

// Producer generic has sync_on_return=true.
//       CHECK:   pcf.generic sync true scope

// Inside producer: conditional scratch write.
//       CHECK:     scf.if %[[IS_SPLIT]] {
//       CHECK:       pcf.write_slice {{.*}} into %arg2
//       CHECK:     }

// Original write_slice to output ref unchanged.
//       CHECK:     pcf.write_slice {{.*}} into %ref
//       CHECK:     pcf.return

// Broadcast dword allocation.
//       CHECK:   pcf.alloc() : !pcf.sref<1xi32

// Post-producer: split vs non-split branch.
//       CHECK:   scf.if %[[IS_SPLIT]] {

// ── Phase 1: Distributed scratch write ──
//       CHECK:     pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:       pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK:         scf.for
//       CHECK:           tensor.extract_slice
//       CHECK:           pcf.write_slice {{.*}} into %arg2
//       CHECK:         pcf.return
//       CHECK:       pcf.return

// ── Phase 2: Barrier + atomic + broadcast ──
//       CHECK:     gpu.barrier memfence [#gpu.address_space<global>]
//       CHECK:     gpu.thread_id x
//       CHECK:     arith.cmpi eq
//       CHECK:     scf.if
//       CHECK:       memref.generic_atomic_rmw
//       CHECK:       arith.remui
//       CHECK:       memref.store
//       CHECK:     gpu.barrier memfence [#gpu.address_space<workgroup>, #gpu.address_space<global>]
//       CHECK:     memref.load

// ── Phase 3: Distributed recombine (if last contributor) ──
//       CHECK:     arith.cmpi eq
//       CHECK:     scf.if
//       CHECK:       pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:         pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK:           scf.for
//       CHECK:             pcf.read_slice {{.*}} : !pcf.sref<1024x64xf32
//       CHECK:             scf.for
//       CHECK:               pcf.read_slice
//       CHECK:               arith.addf
//       CHECK:               scf.yield
//       CHECK:             pcf.write_slice {{.*}} into %arg1
//       CHECK:           pcf.return
//       CHECK:         pcf.return

// ── Phase 4: Distributed non-split writeback (else branch) ──
//       CHECK:   } else {
//       CHECK:     pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:       pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK:         scf.for
//       CHECK:           tensor.extract_slice
//       CHECK:           pcf.write_slice {{.*}} into %arg1
//       CHECK:         pcf.return
//       CHECK:       pcf.return
//       CHECK:   }

// Final workgroup barrier.
//       CHECK:   gpu.barrier memfence [#gpu.address_space<workgroup>]

// The stream_k_recombine op should be gone.
//   CHECK-NOT:   pcf.stream_k_recombine
