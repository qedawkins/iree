// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-convert-sref-to-memref)" --split-input-file --verify-diagnostics | FileCheck %s

// Test that pcf.fence on subgroup scope srefs lowers to iree_codegen.fence
// with the workgroup address space.
util.func private @subgroup_scope_fence_release(%d0: index) {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    initialize {
      %alloc = pcf.alloc(%d0) : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>
      pcf.yield %alloc : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>
    } -> (%aref: !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>)
    execute[%id: index, %n: index] {
    pcf.fence release %aref : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @subgroup_scope_fence_release
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//       CHECK:     iree_codegen.fence release #gpu.address_space<workgroup>
//       CHECK:     pcf.return

// -----

// Test acquire fence.
util.func private @subgroup_scope_fence_acquire(%d0: index) {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    initialize {
      %alloc = pcf.alloc(%d0) : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>
      pcf.yield %alloc : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>
    } -> (%aref: !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>)
    execute[%id: index, %n: index] {
    pcf.fence acquire %aref : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @subgroup_scope_fence_acquire
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//       CHECK:     iree_codegen.fence acquire #gpu.address_space<workgroup>
//       CHECK:     pcf.return

// -----

// Test deduplication: multiple srefs with the same scope produce one fence.
util.func private @subgroup_scope_fence_dedup(%d0: index, %d1: index) {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    initialize {
      %alloc0 = pcf.alloc(%d0) : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>
      %alloc1 = pcf.alloc(%d1) : !pcf.sref<?x4xf16, #iree_gpu.subgroup_scope>
      pcf.yield %alloc0, %alloc1 : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>, !pcf.sref<?x4xf16, #iree_gpu.subgroup_scope>
    } -> (%aref0: !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>, %aref1: !pcf.sref<?x4xf16, #iree_gpu.subgroup_scope>)
    execute[%id: index, %n: index] {
    pcf.fence release %aref0, %aref1 : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>, !pcf.sref<?x4xf16, #iree_gpu.subgroup_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @subgroup_scope_fence_dedup
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//       CHECK:     iree_codegen.fence release #gpu.address_space<workgroup>
//   CHECK-NOT:     iree_codegen.fence
//       CHECK:     pcf.return
