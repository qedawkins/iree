// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// Basic to_sref from tensor.
util.func private @to_sref_tensor(%input: tensor<128x256xf16>) {
  %sref = pcf.to_sref %input : tensor<128x256xf16> -> !pcf.sref<128x256xf16, #pcf.test_scope>
  util.return
}

// CHECK-LABEL: @to_sref_tensor
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<128x256xf16>
//       CHECK:   pcf.to_sref %[[INPUT]] : tensor<128x256xf16> -> !pcf.sref<128x256xf16, #pcf.test_scope>

// -----

// to_sref with dynamic dimensions.
util.func private @to_sref_dynamic(%input: tensor<?x?xf32>) {
  %sref = pcf.to_sref %input : tensor<?x?xf32> -> !pcf.sref<?x?xf32, #pcf.sequential>
  util.return
}

// CHECK-LABEL: @to_sref_dynamic
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<?x?xf32>
//       CHECK:   pcf.to_sref %[[INPUT]] : tensor<?x?xf32> -> !pcf.sref<?x?xf32, #pcf.sequential>

// -----

// to_sref from memref.
util.func private @to_sref_memref(%input: memref<4x8xi32>) {
  %sref = pcf.to_sref %input : memref<4x8xi32> -> !pcf.sref<4x8xi32, #pcf.test_scope>
  util.return
}

// CHECK-LABEL: @to_sref_memref
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: memref<4x8xi32>
//       CHECK:   pcf.to_sref %[[INPUT]] : memref<4x8xi32> -> !pcf.sref<4x8xi32, #pcf.test_scope>
