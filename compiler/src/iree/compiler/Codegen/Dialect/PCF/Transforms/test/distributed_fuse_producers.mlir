// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-distributed-fuse-producers)" --split-input-file | FileCheck %s

// Basic: fuse a linalg.fill producer into a pcf.generic's init. The scalar
// fill value is passed through directly (not an sref), while the DPS init
// feeds the existing readwrite sref.
func.func @fuse_fill_into_generic(%dest: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %fill)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %slice = pcf.read_slice %ref[%id0, %id1] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result = linalg.exp ins(%slice : tensor<4x8xf32>) outs(%slice : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @fuse_fill_into_generic
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-NOT:  linalg.fill
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[READ:.+]] = pcf.read_slice %[[REF]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    %[[TILED_FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[READ]]
//       CHECK:    %[[EXP:.+]] = linalg.exp ins(%[[TILED_FILL]]{{.*}} outs(%[[TILED_FILL]]
//       CHECK:    pcf.write_slice %[[EXP]] into %[[REF]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Fuse a linalg.fill producer into a pcf.loop's init.
func.func @fuse_fill_into_loop(%dest: tensor<8x16xf32>, %n0: index, %n1: index) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.loop scope(#pcf.test_scope) count(%n0, %n1)
    execute(%ref = %fill)[%id0: index, %id1: index]
            : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
           -> (tensor<8x16xf32>) {
    %slice = pcf.read_slice %ref[%id0, %id1] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result = linalg.exp ins(%slice : tensor<4x8xf32>) outs(%slice : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @fuse_fill_into_loop
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-NOT:  linalg.fill
//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[READ:.+]] = pcf.read_slice %[[REF]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    %[[TILED_FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[READ]]
//       CHECK:    %[[EXP:.+]] = linalg.exp ins(%[[TILED_FILL]]{{.*}} outs(%[[TILED_FILL]]
//       CHECK:    pcf.write_slice %[[EXP]] into %[[REF]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    pcf.return
//       CHECK:  return %[[LOOP]]

// -----

// Producer kept when it has other uses.
func.func @keep_producer_with_other_uses(%dest: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %fill)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %slice = pcf.read_slice %ref[%id0, %id1] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result = linalg.exp ins(%slice : tensor<4x8xf32>) outs(%slice : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0, %fill : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: @keep_producer_with_other_uses
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:  %[[FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[DEST]]
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    pcf.read_slice
//       CHECK:    linalg.fill ins(%[[CST]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]], %[[FILL]]
