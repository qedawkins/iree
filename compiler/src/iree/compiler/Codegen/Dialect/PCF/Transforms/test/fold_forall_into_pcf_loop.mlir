// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-fold-forall-into-pcf-loop)" --split-input-file | FileCheck %s

// Test folding scf.forall containing pcf.loop into a single pcf.generic.
// The pcf.loop produces a tile that gets inserted back into the output.

func.func @fold_forall_into_pcf_loop(%init: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %c4 = arith.constant 4 : index
  %0 = scf.forall (%id0, %id1) in (4, 8) shared_outs(%iter = %init) -> (tensor<16x32xf32>) {
    // The pcf.loop produces a 4x4 tile that gets inserted at [id0, id1].
    %tile_init = tensor.extract_slice %iter[%id0, %id1] [4, 4] [1, 1]
        : tensor<16x32xf32> to tensor<4x4xf32>
    %loop_result = pcf.loop scope(#pcf.sequential) count(%c4)
        execute(%ref = %tile_init)[%loop_id: index]
            : (!pcf.sref<4x4xf32, sync(#pcf.sequential)>)
           -> (tensor<4x4xf32>) {
      %slice = tensor.extract_slice %init[%id0, %loop_id] [1, 4] [1, 1]
          : tensor<16x32xf32> to tensor<1x4xf32>
      pcf.write_slice %slice into %ref[%loop_id, 0] [1, 4] [1, 1]
          : tensor<1x4xf32> into !pcf.sref<4x4xf32, sync(#pcf.sequential)>
      pcf.return
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %loop_result into %iter[%id0, %id1] [4, 4] [1, 1]
          : tensor<4x4xf32> into tensor<16x32xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<0>, #iree_codegen.local_mapping<1>]}
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: @fold_forall_into_pcf_loop
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<16x32xf32>

//       CHECK:   %[[GENERIC:.+]] = pcf.generic
//       CHECK:     scope(#pcf.sequential)
//       CHECK:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[INIT]])[%[[ID:[A-Za-z0-9_]+]]: index, %{{.*}}: index]
//       CHECK:          : (!pcf.sref<16x32xf32, sync(#pcf.sequential)>)
//       CHECK:         -> (tensor<16x32xf32>) {
//       CHECK:     scf.for
//       CHECK:       affine.delinearize_index
//       CHECK:       pcf.write_slice
//       CHECK:     pcf.return
//       CHECK:   return %[[GENERIC]]

// -----

// Test with multiple results from pcf.loop.

func.func @fold_forall_multiple_results(%init0: tensor<16xf32>, %init1: tensor<16xf32>)
    -> (tensor<16xf32>, tensor<16xf32>) {
  %c2 = arith.constant 2 : index
  %0:2 = scf.forall (%id) in (4) shared_outs(%iter0 = %init0, %iter1 = %init1)
      -> (tensor<16xf32>, tensor<16xf32>) {
    %tile_init0 = tensor.extract_slice %iter0[%id] [4] [1]
        : tensor<16xf32> to tensor<4xf32>
    %tile_init1 = tensor.extract_slice %iter1[%id] [4] [1]
        : tensor<16xf32> to tensor<4xf32>
    %loop_result:2 = pcf.loop scope(#pcf.sequential) count(%c2)
        execute(%ref0 = %tile_init0, %ref1 = %tile_init1)[%loop_id: index]
            : (!pcf.sref<4xf32, sync(#pcf.sequential)>,
               !pcf.sref<4xf32, sync(#pcf.sequential)>)
           -> (tensor<4xf32>, tensor<4xf32>) {
      %slice0 = tensor.extract_slice %init0[%loop_id] [2] [1]
          : tensor<16xf32> to tensor<2xf32>
      %slice1 = tensor.extract_slice %init1[%loop_id] [2] [1]
          : tensor<16xf32> to tensor<2xf32>
      pcf.write_slice %slice0 into %ref0[%loop_id] [2] [1]
          : tensor<2xf32> into !pcf.sref<4xf32, sync(#pcf.sequential)>
      pcf.write_slice %slice1 into %ref1[%loop_id] [2] [1]
          : tensor<2xf32> into !pcf.sref<4xf32, sync(#pcf.sequential)>
      pcf.return
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %loop_result#0 into %iter0[%id] [4] [1]
          : tensor<4xf32> into tensor<16xf32>
      tensor.parallel_insert_slice %loop_result#1 into %iter1[%id] [4] [1]
          : tensor<4xf32> into tensor<16xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return %0#0, %0#1 : tensor<16xf32>, tensor<16xf32>
}

// CHECK-LABEL: @fold_forall_multiple_results
//  CHECK-SAME:   %[[INIT0:[A-Za-z0-9_]+]]: tensor<16xf32>
//  CHECK-SAME:   %[[INIT1:[A-Za-z0-9_]+]]: tensor<16xf32>

//       CHECK:   %[[GENERIC:.+]]:2 = pcf.generic
//       CHECK:     scope(#pcf.sequential)
//       CHECK:     execute(%[[REF0:[A-Za-z0-9_]+]] = %[[INIT0]], %[[REF1:[A-Za-z0-9_]+]] = %[[INIT1]])[%[[ID:[A-Za-z0-9_]+]]: index, %{{.*}}: index]
//       CHECK:          : (!pcf.sref<16xf32, sync(#pcf.sequential)>, !pcf.sref<16xf32, sync(#pcf.sequential)>)
//       CHECK:         -> (tensor<16xf32>, tensor<16xf32>) {
//       CHECK:     scf.for
//       CHECK:       pcf.write_slice {{.*}} into %[[REF0]]
//       CHECK:       pcf.write_slice {{.*}} into %[[REF1]]
//       CHECK:     pcf.return
//       CHECK:   return %[[GENERIC]]#0, %[[GENERIC]]#1
