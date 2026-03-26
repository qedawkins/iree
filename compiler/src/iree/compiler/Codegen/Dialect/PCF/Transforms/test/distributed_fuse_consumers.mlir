// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-distributed-fuse-consumers)" --split-input-file | FileCheck %s

// NOTE: Distributed consumer fusion currently has known bugs with linalg ops
// that call getResultTilePosition on cloned ops with incorrect operand indices.
// This test file documents the pass registration and will be expanded as the
// underlying implementation is fixed.

// Verify the pass runs without error when there is nothing to fuse.
func.func @no_consumers_to_fuse(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %cst = arith.constant dense<5.0> : tensor<4x8xf32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @no_consumers_to_fuse
//       CHECK:  pcf.generic scope(#pcf.test_scope)
//       CHECK:    pcf.write_slice
//       CHECK:    pcf.return
