// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-distributed-fuse-consumers)" --split-input-file | FileCheck %s

// NOTE: Distributed consumer fusion has a bug: after fusing, the
// dropUnusedResults canonicalization crashes with a heap-buffer-overflow
// in Transforms.cpp:dropUnusedResults<GenericOp> (operand index out of
// bounds after the generic is modified). The crash occurs when the greedy
// rewriter runs canonicalization patterns on the modified pcf.generic.
//
// TODO(shared-exec): Fix the dropUnusedResults crash, then enable the
// commented-out @fuse_elementwise_consumer test below.

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

// Commented-out test case: fuse an elementwise consumer into a pcf.generic.
// Enable once the dropUnusedResults crash is fixed.
//
// func.func @fuse_elementwise_consumer(%input: tensor<16x32xf32>,
//                                      %dest: tensor<16x32xf32>)
//     -> tensor<16x32xf32> {
//   %0 = pcf.generic scope(#pcf.test_scope)
//     execute(%in_ref <- %input, %out_ref = %dest)
//          [%id0: index, %id1: index, %n0: index, %n1: index]
//          : (!pcf.sref<16x32xf32, #pcf.test_scope>,
//             !pcf.sref<16x32xf32, sync(#pcf.test_scope)>)
//         -> (tensor<16x32xf32>) {
//     %tile = pcf.read_slice %in_ref[%id0, %id1] [4, 8] [1, 1]
//         : !pcf.sref<16x32xf32, #pcf.test_scope> to tensor<4x8xf32>
//     %dest_tile = pcf.read_slice %out_ref[%id0, %id1] [4, 8] [1, 1]
//         : !pcf.sref<16x32xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
//     %add = linalg.generic {
//       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
//                        affine_map<(d0, d1) -> (d0, d1)>],
//       iterator_types = ["parallel", "parallel"]
//     } ins(%tile : tensor<4x8xf32>) outs(%dest_tile : tensor<4x8xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       %1 = arith.addf %in, %out : f32
//       linalg.yield %1 : f32
//     } -> tensor<4x8xf32>
//     pcf.write_slice %add into %out_ref[%id0, %id1] [4, 8] [1, 1]
//         : tensor<4x8xf32>
//           into !pcf.sref<16x32xf32, sync(#pcf.test_scope)>
//     pcf.return
//   }
//   %cst = arith.constant dense<2.0> : tensor<16x32xf32>
//   %result = linalg.generic {
//     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
//                      affine_map<(d0, d1) -> (d0, d1)>],
//     iterator_types = ["parallel", "parallel"]
//   } ins(%cst : tensor<16x32xf32>) outs(%0 : tensor<16x32xf32>) {
//   ^bb0(%in: f32, %out: f32):
//     %1 = arith.mulf %in, %out : f32
//     linalg.yield %1 : f32
//   } -> tensor<16x32xf32>
//   return %result : tensor<16x32xf32>
// }
//
// Expected: the elementwise consumer (arith.mulf) should be fused into the
// pcf.generic, with a new readwrite sref for the consumer's output.
