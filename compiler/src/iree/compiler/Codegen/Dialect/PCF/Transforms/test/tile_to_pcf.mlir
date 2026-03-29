// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-tile-to-pcf{tile-sizes=4,8})" --split-input-file | FileCheck %s

// Basic: tile a linalg.fill into a pcf.loop. The scalar input is passed through
// directly (not an sref arg), while the DPS init becomes a readwrite sref.
func.func @tile_fill(%dest: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %fill : tensor<16x32xf32>
}

// CHECK-LABEL: @tile_fill
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<16x32xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.sequential)
//  CHECK-SAME:    count(
//       CHECK:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index]
//       CHECK:         : (!pcf.sref<16x32xf32, sync(#pcf.sequential)>)
//       CHECK:        -> (tensor<16x32xf32>) {
//       CHECK:      %[[TILE:.+]] = pcf.read_slice %[[REF]]
//       CHECK:      %[[FILLED:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[TILE]]
//       CHECK:      pcf.write_slice %[[FILLED]] into %[[REF]]
//       CHECK:      pcf.return
//       CHECK:  return %[[LOOP]]

// -----

// Tile a linalg.copy. Both input and output are shaped, so the input becomes a
// readonly sref arg and the output becomes a readwrite sref arg.
func.func @tile_copy(%src: tensor<16x32xf32>, %dest: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %copy = linalg.copy ins(%src : tensor<16x32xf32>) outs(%dest : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %copy : tensor<16x32xf32>
}

// CHECK-LABEL: @tile_copy
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9_]+]]: tensor<16x32xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<16x32xf32>

//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.sequential)
//  CHECK-SAME:    count(
//       CHECK:    execute(%[[IN_REF:.+]] <- %[[SRC]], %[[OUT_REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index]
//       CHECK:         : (!pcf.sref<16x32xf32, #pcf.sequential>, !pcf.sref<16x32xf32, sync(#pcf.sequential)>)
//       CHECK:        -> (tensor<16x32xf32>) {
//       CHECK:      %[[IN_TILE:.+]] = pcf.read_slice %[[IN_REF]]
//       CHECK:      %[[OUT_TILE:.+]] = pcf.read_slice %[[OUT_REF]]
//       CHECK:      %[[COPIED:.+]] = linalg.copy ins(%[[IN_TILE]]{{.*}} outs(%[[OUT_TILE]]
//       CHECK:      pcf.write_slice %[[COPIED]] into %[[OUT_REF]]
//       CHECK:      pcf.return
//       CHECK:  return %[[LOOP]]

// -----

// Tile a linalg.generic with static shapes. Verifies readonly sref for input
// and readwrite sref for output.
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @tile_generic(%lhs: tensor<16x32xf32>, %dest: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs : tensor<16x32xf32>) outs(%dest : tensor<16x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: @tile_generic
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9_]+]]: tensor<16x32xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<16x32xf32>

//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.sequential)
//  CHECK-SAME:    count(
//       CHECK:    execute(%[[IN_REF:.+]] <- %[[LHS]], %[[OUT_REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index]
//       CHECK:         : (!pcf.sref<16x32xf32, #pcf.sequential>, !pcf.sref<16x32xf32, sync(#pcf.sequential)>)
//       CHECK:        -> (tensor<16x32xf32>) {
//       CHECK:      %[[LHS_TILE:.+]] = pcf.read_slice %[[IN_REF]]
//       CHECK:      %[[DEST_TILE:.+]] = pcf.read_slice %[[OUT_REF]]
//       CHECK:      linalg.generic
//  CHECK-SAME:        ins(%[[LHS_TILE]]
//  CHECK-SAME:        outs(%[[DEST_TILE]]
//       CHECK:      pcf.write_slice
//       CHECK:      pcf.return
//       CHECK:  return %[[LOOP]]
