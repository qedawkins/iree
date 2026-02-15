// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

util.func private @shaped_ref_with_no_sync(!pcf.sref<1x?x3x?xi32, #pcf.test_scope>)
// CHECK: @shaped_ref_with_no_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, #pcf.test_scope>

util.func private @shaped_ref_with_type_sync(!pcf.sref<1x?x3x?xi32, #pcf.test_scope, i32>)
// CHECK: @shaped_ref_with_type_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, #pcf.test_scope, i32>

util.func private @shaped_ref_with_attr_sync(!pcf.sref<1x?x3x?xi32, #pcf.test_scope, 42>)
// CHECK: @shaped_ref_with_attr_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, #pcf.test_scope, 42 : i64>

util.func private @shaped_ref_with_parent_sync(!pcf.sref<1x?x3x?xi32, sync(#pcf.test_scope)>)
// CHECK: @shaped_ref_with_parent_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, sync(#pcf.test_scope)>

// -----

// Round-trip: sref with readwrite accessor mode.
util.func private @sref_readwrite(!pcf.sref<128x128xf32, #pcf.test_scope, readwrite>)
// CHECK: @sref_readwrite
// CHECK-SAME: !pcf.sref<128x128xf32, #pcf.test_scope, readwrite>

// -----

// Round-trip: sref with readonly accessor mode.
util.func private @sref_readonly(!pcf.sref<128x64xf16, #pcf.test_scope, readonly>)
// CHECK: @sref_readonly
// CHECK-SAME: !pcf.sref<128x64xf16, #pcf.test_scope, readonly>

// -----

// Backward compat: sref with NO accessor mode (existing syntax).
util.func private @sref_no_accessor(!pcf.sref<128x128xf32, #pcf.test_scope>)
// CHECK: @sref_no_accessor
// CHECK-SAME: !pcf.sref<128x128xf32, #pcf.test_scope>

// -----

// Accessor mode with sync scope.
util.func private @sref_accessor_and_sync(!pcf.sref<128x128xf32, sync(#pcf.test_scope), readwrite>)
// CHECK: @sref_accessor_and_sync
// CHECK-SAME: !pcf.sref<128x128xf32, sync(#pcf.test_scope), readwrite>

// -----

// Round-trip: bundle type with scope and ID.
util.func private @bundle_type(!pcf.bundle<#pcf.test_scope, 0>)
// CHECK: @bundle_type
// CHECK-SAME: !pcf.bundle<#pcf.test_scope, 0>

// -----

// Bundle with larger ID.
util.func private @bundle_type_id3(!pcf.bundle<#pcf.test_scope, 3>)
// CHECK: @bundle_type_id3
// CHECK-SAME: !pcf.bundle<#pcf.test_scope, 3>
