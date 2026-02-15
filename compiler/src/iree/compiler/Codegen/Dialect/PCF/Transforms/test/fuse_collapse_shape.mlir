// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-fuse-consumers)" --split-input-file | FileCheck %s

// Test: Fuse tensor.collapse_shape into pcf.generic with tied init.
// The 2D result is collapsed to 1D, and the write_slice offsets/sizes are
// linearized accordingly. The constant source also gets collapsed.

func.func @fuse_collapse_shape_into_generic(%arg0: tensor<8x10xi32>) -> tensor<80xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
        -> (tensor<8x10xi32>) {
    %cst = arith.constant dense<5> : tensor<4x10xi32>
    pcf.write_slice %cst into %ref[%id0, 0] [4, 10] [1, 1] : tensor<4x10xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1 : tensor<80xi32>
}

// CHECK-LABEL: @fuse_collapse_shape_into_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x10xi32>

// The constant gets collapsed by canonicalization.
//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<40xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} : tensor<8x10xi32> into tensor<80xi32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<80xi32>)
//       CHECK:    %[[FLAT_OFF:.+]] = affine.apply
//       CHECK:    pcf.write_slice %[[CST]] into %[[REF]][%[[FLAT_OFF]]] [40] [1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Test: Fuse tensor.collapse_shape into pcf.loop.

func.func @fuse_collapse_shape_into_loop(%arg0: tensor<8x10xi32>, %n0: index, %n1: index) -> tensor<80xi32> {
  %0 = pcf.loop scope(#pcf.test_scope) count(%n0, %n1)
    execute(%ref = %arg0)[%id0: index, %id1: index]
            : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
           -> (tensor<8x10xi32>) {
    %cst = arith.constant dense<5> : tensor<4x10xi32>
    pcf.write_slice %cst into %ref[%id0, 0] [4, 10] [1, 1] : tensor<4x10xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1 : tensor<80xi32>
}

// CHECK-LABEL: @fuse_collapse_shape_into_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x10xi32>

//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<40xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} : tensor<8x10xi32> into tensor<80xi32>
//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<80xi32>)
//       CHECK:    %[[FLAT_OFF:.+]] = affine.apply
//       CHECK:    pcf.write_slice %[[CST]] into %[[REF]][%[[FLAT_OFF]]] [40] [1]
//       CHECK:    pcf.return
//       CHECK:  return %[[LOOP]]

// -----

// Test: Fuse tensor.collapse_shape with multiple write_slices.

func.func @fuse_collapse_shape_multiple_write_slices(%arg0: tensor<8x10xi32>) -> tensor<80xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
        -> (tensor<8x10xi32>) {
    %cst1 = arith.constant dense<5> : tensor<3x10xi32>
    %cst2 = arith.constant dense<7> : tensor<5x10xi32>
    pcf.write_slice %cst1 into %ref[%id0, 0] [3, 10] [1, 1] : tensor<3x10xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.write_slice %cst2 into %ref[%id1, 0] [5, 10] [1, 1] : tensor<5x10xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1 : tensor<80xi32>
}

// CHECK-LABEL: @fuse_collapse_shape_multiple_write_slices
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x10xi32>

// Constants are collapsed by canonicalization.
//   CHECK-DAG:  %[[CST1:.+]] = arith.constant dense<5> : tensor<30xi32>
//   CHECK-DAG:  %[[CST2:.+]] = arith.constant dense<7> : tensor<50xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]{{\]}}
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])
//       CHECK:    -> (tensor<80xi32>)
//       CHECK:    pcf.write_slice %[[CST1]] into %[[REF]]{{.*}} [30] [1]
//       CHECK:    pcf.write_slice %[[CST2]] into %[[REF]]{{.*}} [50] [1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Negative test: producer result is directly returned (multiple uses prevent
// fusion). Using a direct return of %0 ensures the second use cannot be fused
// by other patterns.

func.func @no_fuse_collapse_shape_multiple_uses(%arg0: tensor<8x10xi32>) -> (tensor<80xi32>, tensor<8x10xi32>) {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
        -> (tensor<8x10xi32>) {
    %cst = arith.constant dense<5> : tensor<4x10xi32>
    pcf.write_slice %cst into %ref[%id0, 0] [4, 10] [1, 1] : tensor<4x10xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1, %0 : tensor<80xi32>, tensor<8x10xi32>
}

// CHECK-LABEL: @no_fuse_collapse_shape_multiple_uses

//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//       CHECK:  %[[COLLAPSE:.+]] = tensor.collapse_shape %[[GENERIC]] {{\[}}[0, 1]{{\]}}
//       CHECK:  return %[[COLLAPSE]], %[[GENERIC]]

// -----

// Negative test: inner dimension not fully covered by write_slice.

func.func @no_fuse_collapse_shape_partial_inner_dim(%arg0: tensor<8x10xi32>) -> tensor<80xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
        -> (tensor<8x10xi32>) {
    %cst = arith.constant dense<5> : tensor<4x5xi32>
    pcf.write_slice %cst into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1 : tensor<80xi32>
}

// CHECK-LABEL: @no_fuse_collapse_shape_partial_inner_dim

//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//       CHECK:  %[[COLLAPSE:.+]] = tensor.collapse_shape %[[GENERIC]] {{\[}}[0, 1]{{\]}}
//       CHECK:  return %[[COLLAPSE]]
