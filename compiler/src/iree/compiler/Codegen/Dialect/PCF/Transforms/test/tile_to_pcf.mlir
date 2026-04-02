// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-tile-to-pcf{tile-sizes=4,8})" --split-input-file | FileCheck %s
// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-tile-to-pcf{tile-sizes=4,8,0})" --split-input-file | FileCheck %s --check-prefix=CHECK3D

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

// -----

// Dynamic shapes: tile a linalg.generic with dynamic tensor dimensions.
// The sref types should have dynamic dims and the loop should handle them.
func.func @tile_dynamic(%lhs: tensor<?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs : tensor<?x?xf32>) outs(%dest : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @tile_dynamic
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<?x?xf32>
//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.sequential)
//       CHECK:    execute(%{{.+}} <- %[[LHS]], %{{.+}} = %[[DEST]])
//       CHECK:         : (!pcf.sref<?x?xf32, #pcf.sequential>, !pcf.sref<?x?xf32, sync(#pcf.sequential)>)
//       CHECK:        -> (tensor<?x?xf32>) {
//       CHECK:      pcf.read_slice
//       CHECK:      pcf.read_slice
//       CHECK:      linalg.generic
//       CHECK:      pcf.write_slice
//       CHECK:      pcf.return
//       CHECK:  return %[[LOOP]]

// -----

// Reduction dimension: tile a matmul with 3 tile sizes (M=4, N=8, K=0).
// K is untiled (zero tile size) so the tiled matmul has full K extent.
// Uses CHECK3D prefix (second RUN line with tile-sizes=4,8,0).
func.func @tile_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x16xf32>,
                        %dest: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<16x32xf32>, tensor<32x16xf32>)
                      outs(%dest : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK3D-LABEL: @tile_matmul
// CHECK3D-SAME:   %[[LHS:[A-Za-z0-9_]+]]: tensor<16x32xf32>
// CHECK3D-SAME:   %[[RHS:[A-Za-z0-9_]+]]: tensor<32x16xf32>
// CHECK3D-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<16x16xf32>
//      CHECK3D:  pcf.loop scope(#pcf.sequential)
//      CHECK3D:    execute(%{{.+}} <- %[[LHS]], %{{.+}} <- %[[RHS]], %{{.+}} = %[[DEST]])
//      CHECK3D:      pcf.read_slice
//      CHECK3D:      pcf.read_slice
//      CHECK3D:      pcf.read_slice
//      CHECK3D:      linalg.matmul
//      CHECK3D:      pcf.write_slice
//      CHECK3D:      pcf.return
