// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-fuse-producers)" --split-input-file | FileCheck %s

// Positive Tests:
//* - linalg.fill producer into pcf.generic
//* - linalg.fill producer into pcf.loop
//* - Producer with multiple read sites
//* - Producer erased when no other uses
//* - Producer kept when other uses exist

// Basic: fuse a linalg.fill producer into a pcf.generic's init.
func.func @fuse_fill_into_generic(%arg0: tensor<8x16xf32>, %dest: tensor<8x16xf32>) -> tensor<8x16xf32> {
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
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x16xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-NOT:  linalg.fill
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[TILED_FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%{{.+}} : tensor<4x8xf32>)
//       CHECK:    %[[EXP:.+]] = linalg.exp ins(%[[TILED_FILL]]{{.*}} outs(%[[TILED_FILL]]
//       CHECK:    pcf.write_slice %[[EXP]] into %[[REF]][%[[ID0]], %[[ID1]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Basic: fuse a linalg.fill producer into a pcf.loop's init.
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
//       CHECK:    %[[TILED_FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%{{.+}} : tensor<4x8xf32>)
//       CHECK:    %[[EXP:.+]] = linalg.exp ins(%[[TILED_FILL]]{{.*}} outs(%[[TILED_FILL]]
//       CHECK:    pcf.write_slice %[[EXP]] into %[[REF]][%[[ID0]], %[[ID1]]]
//       CHECK:    pcf.return
//       CHECK:  return %[[LOOP]]

// -----

// Multiple read_slice sites for the same init value.
func.func @fuse_fill_multiple_reads(%dest: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %fill)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %slice0 = pcf.read_slice %ref[%id0, 0] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result0 = linalg.exp ins(%slice0 : tensor<4x8xf32>) outs(%slice0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result0 into %ref[%id0, 0] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    %slice1 = pcf.read_slice %ref[%id0, 8] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result1 = linalg.exp ins(%slice1 : tensor<4x8xf32>) outs(%slice1 : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result1 into %ref[%id0, 8] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @fuse_fill_multiple_reads
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-NOT:  linalg.fill
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])
//       CHECK:    %[[FILL0:.+]] = linalg.fill ins(%[[CST]]
//       CHECK:    %[[EXP0:.+]] = linalg.exp ins(%[[FILL0]]
//       CHECK:    pcf.write_slice %[[EXP0]]
//       CHECK:    %[[FILL1:.+]] = linalg.fill ins(%[[CST]]
//       CHECK:    %[[EXP1:.+]] = linalg.exp ins(%[[FILL1]]
//       CHECK:    pcf.write_slice %[[EXP1]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

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
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])
//       CHECK:    %[[TILED_FILL:.+]] = linalg.fill ins(%[[CST]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]], %[[FILL]]

// -----

// Negative Tests:
//  - No producer (init is a block argument)
//  - Non-DPS producer
//  - Producer operand does not dominate the scoped op
//  - No read_slice on the sref

// Negative: init is a block argument (no defining op).
func.func @no_fuse_block_arg_init(%init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %init)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %slice = pcf.read_slice %ref[%id0, %id1] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    pcf.write_slice %slice into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @no_fuse_block_arg_init
//       CHECK:  pcf.generic
//       CHECK:    pcf.read_slice
//       CHECK:    pcf.write_slice

// -----

// Negative: no read_slice on the sref (only writes).
func.func @no_fuse_no_reads(%dest: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %fill)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %cst2 = arith.constant dense<1.0> : tensor<4x8xf32>
    pcf.write_slice %cst2 into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @no_fuse_no_reads
//       CHECK:  %[[FILL:.+]] = linalg.fill
//       CHECK:  %[[GENERIC:.+]] = pcf.generic
//  CHECK-NEXT:    execute(%{{.+}} = %[[FILL]])
//       CHECK:  return %[[GENERIC]]
