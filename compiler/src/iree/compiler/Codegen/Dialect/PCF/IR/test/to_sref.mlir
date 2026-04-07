// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// Basic to_sref from tensor.
util.func private @to_sref_tensor(%input: tensor<128x256xf16>) {
  %sref = pcf.to_sref %input : tensor<128x256xf16> -> !pcf.sref<128x256xf16, #pcf.test_scope>
  util.return
}

// CHECK-LABEL: @to_sref_tensor
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<128x256xf16>
//       CHECK:   %[[SREF:[A-Za-z0-9_]+]] = pcf.to_sref %[[INPUT]] : tensor<128x256xf16> -> !pcf.sref<128x256xf16, #pcf.test_scope>
//       CHECK:   util.return

// -----

// to_sref with dynamic dimensions.
util.func private @to_sref_dynamic(%input: tensor<?x?xf32>) {
  %sref = pcf.to_sref %input : tensor<?x?xf32> -> !pcf.sref<?x?xf32, #pcf.sequential>
  util.return
}

// CHECK-LABEL: @to_sref_dynamic
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[SREF:[A-Za-z0-9_]+]] = pcf.to_sref %[[INPUT]] : tensor<?x?xf32> -> !pcf.sref<?x?xf32, #pcf.sequential>
//       CHECK:   util.return

// -----

// to_sref from memref.
util.func private @to_sref_memref(%input: memref<4x8xi32>) {
  %sref = pcf.to_sref %input : memref<4x8xi32> -> !pcf.sref<4x8xi32, #pcf.test_scope>
  util.return
}

// CHECK-LABEL: @to_sref_memref
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: memref<4x8xi32>
//       CHECK:   %[[SREF:[A-Za-z0-9_]+]] = pcf.to_sref %[[INPUT]] : memref<4x8xi32> -> !pcf.sref<4x8xi32, #pcf.test_scope>
//       CHECK:   util.return

// -----

// to_sref from partially-dynamic memref.
util.func private @to_sref_dynamic_memref(%input: memref<?x8xf32>) {
  %sref = pcf.to_sref %input : memref<?x8xf32> -> !pcf.sref<?x8xf32, #pcf.test_scope>
  util.return
}

// CHECK-LABEL: @to_sref_dynamic_memref
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: memref<?x8xf32>
//       CHECK:   %[[SREF:[A-Za-z0-9_]+]] = pcf.to_sref %[[INPUT]] : memref<?x8xf32> -> !pcf.sref<?x8xf32, #pcf.test_scope>
//       CHECK:   util.return

// -----

// rank-0 tensor capture.
util.func private @to_sref_rank0(%input: tensor<f32>) {
  %sref = pcf.to_sref %input : tensor<f32> -> !pcf.sref<f32, #pcf.sequential>
  util.return
}

// CHECK-LABEL: @to_sref_rank0
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<f32>
//       CHECK:   %[[SREF:[A-Za-z0-9_]+]] = pcf.to_sref %[[INPUT]] : tensor<f32> -> !pcf.sref<f32, #pcf.sequential>
//       CHECK:   util.return
