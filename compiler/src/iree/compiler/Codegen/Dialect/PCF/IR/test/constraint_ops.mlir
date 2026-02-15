// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @constrain_shared_layout_basic
// CHECK-SAME: %[[INPUT:.+]]: !pcf.sref<128x32xf16, #pcf.test_scope>
util.func private @constrain_shared_layout_basic(%input: !pcf.sref<128x32xf16, #pcf.test_scope>) -> !pcf.sref<128x32xf16, #pcf.test_scope> {
  // CHECK: pcf.constrain_shared_layout %[[INPUT]]
  // CHECK-SAME: layout(#pcf.shared_layout<{stride = [36, 1], swizzle = none}>)
  // CHECK-SAME: : !pcf.sref<128x32xf16, #pcf.test_scope>
  %0 = pcf.constrain_shared_layout %input
      layout(#pcf.shared_layout<{stride = [36, 1], swizzle = none}>)
      : !pcf.sref<128x32xf16, #pcf.test_scope>
  util.return %0 : !pcf.sref<128x32xf16, #pcf.test_scope>
}

// -----

// CHECK-LABEL: @constrain_shared_layout_no_swizzle
// CHECK-SAME: %[[INPUT:.+]]: !pcf.sref<64x64xf32, #pcf.test_scope>
util.func private @constrain_shared_layout_no_swizzle(%input: !pcf.sref<64x64xf32, #pcf.test_scope>) -> !pcf.sref<64x64xf32, #pcf.test_scope> {
  // CHECK: pcf.constrain_shared_layout %[[INPUT]]
  // CHECK-SAME: layout(#pcf.shared_layout<{stride = [64, 1]}>)
  // CHECK-SAME: : !pcf.sref<64x64xf32, #pcf.test_scope>
  %0 = pcf.constrain_shared_layout %input
      layout(#pcf.shared_layout<{stride = [64, 1]}>)
      : !pcf.sref<64x64xf32, #pcf.test_scope>
  util.return %0 : !pcf.sref<64x64xf32, #pcf.test_scope>
}

// -----

// CHECK-LABEL: @constrain_shared_layout_xor_swizzle
util.func private @constrain_shared_layout_xor_swizzle(%input: !pcf.sref<128x32xf16, #pcf.test_scope>) -> !pcf.sref<128x32xf16, #pcf.test_scope> {
  // CHECK: pcf.constrain_shared_layout %{{.+}}
  // CHECK-SAME: layout(#pcf.shared_layout<{stride = [32, 1], swizzle = xor_128}>)
  %0 = pcf.constrain_shared_layout %input
      layout(#pcf.shared_layout<{stride = [32, 1], swizzle = xor_128}>)
      : !pcf.sref<128x32xf16, #pcf.test_scope>
  util.return %0 : !pcf.sref<128x32xf16, #pcf.test_scope>
}

// -----

// Test 3D sref with shared layout.
// CHECK-LABEL: @constrain_shared_layout_3d
util.func private @constrain_shared_layout_3d(%input: !pcf.sref<4x32x16xf16, #pcf.test_scope>) -> !pcf.sref<4x32x16xf16, #pcf.test_scope> {
  // CHECK: pcf.constrain_shared_layout %{{.+}}
  // CHECK-SAME: layout(#pcf.shared_layout<{stride = [512, 16, 1]}>)
  %0 = pcf.constrain_shared_layout %input
      layout(#pcf.shared_layout<{stride = [512, 16, 1]}>)
      : !pcf.sref<4x32x16xf16, #pcf.test_scope>
  util.return %0 : !pcf.sref<4x32x16xf16, #pcf.test_scope>
}
