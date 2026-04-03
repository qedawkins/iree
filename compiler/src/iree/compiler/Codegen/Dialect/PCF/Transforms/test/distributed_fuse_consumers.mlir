// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-distributed-fuse-consumers)" --split-input-file | FileCheck %s

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

// -----

// Fuse an elementwise consumer (arith.mulf linalg.generic) into a pcf.generic.
// The consumer's constant input becomes a new readonly sref arg and the
// consumer's output becomes a new readwrite sref arg tied to the init.
func.func @fuse_elementwise_consumer(%input: tensor<16x32xf32>,
                                     %dest: tensor<16x32xf32>)
    -> tensor<16x32xf32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%in_ref <- %input, %out_ref = %dest)
         [%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<16x32xf32, #pcf.test_scope>,
            !pcf.sref<16x32xf32, sync(#pcf.test_scope)>)
        -> (tensor<16x32xf32>) {
    %tile = pcf.read_slice %in_ref[%id0, %id1] [4, 8] [1, 1]
        : !pcf.sref<16x32xf32, #pcf.test_scope> to tensor<4x8xf32>
    %dest_tile = pcf.read_slice %out_ref[%id0, %id1] [4, 8] [1, 1]
        : !pcf.sref<16x32xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %add = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%tile : tensor<4x8xf32>) outs(%dest_tile : tensor<4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    } -> tensor<4x8xf32>
    pcf.write_slice %add into %out_ref[%id0, %id1] [4, 8] [1, 1]
        : tensor<4x8xf32>
          into !pcf.sref<16x32xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  %cst = arith.constant dense<2.0> : tensor<16x32xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%cst : tensor<16x32xf32>) outs(%0 : tensor<16x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.mulf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<16x32xf32>
  return %result : tensor<16x32xf32>
}

// After fusion the pcf.generic should have:
//   - 2 readonly inputs (original %input + consumer's %cst).
//   - 2 readwrite outputs (original %dest + consumer's output tied to %dest).
//   - The consumer's linalg.generic (mulf) fused inside the body.
//
// CHECK-LABEL: @fuse_elementwise_consumer
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<16x32xf32>, %[[DEST:.+]]: tensor<16x32xf32>)
//       CHECK: %[[CST:.+]] = arith.constant dense<2.{{0*}}e+00>
//       CHECK: pcf.generic scope(#pcf.test_scope)
//       CHECK:   execute(%{{.+}} <- %[[INPUT]], %{{.+}} <- %[[CST]], %{{.+}} = %[[DEST]], %{{.+}} = %[[DEST]])
//       CHECK:     pcf.read_slice %{{.+}}[%{{.+}}, %{{.+}}] [4, 8]
//       CHECK:     pcf.read_slice %{{.+}}[%{{.+}}, %{{.+}}] [4, 8]
//       CHECK:     linalg.generic
//       CHECK:       arith.addf
//       CHECK:     pcf.read_slice %{{.+}}[%{{.+}}, %{{.+}}] [4, 8]
//       CHECK:     linalg.generic
//       CHECK:       arith.mulf
//       CHECK:     pcf.write_slice
//       CHECK:     pcf.write_slice
//       CHECK:     pcf.return
