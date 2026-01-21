// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-pcf-resolve-tokens, iree-pcf-convert-sref-to-memref, iree-pcf-lower-structural-pcf))" --split-input-file | FileCheck %s

// Test that PCF lowering passes correctly lower workgroup-scope PCF ops.
// These passes are integrated into the LLVMCPU backend pipeline.
// Note: We test the passes directly rather than the full pipeline because
// the full pipeline requires more complete IR context (executables, etc.).

func.func @pcf_workgroup_loop(%arg0: memref<64xf32>) {
  %c64 = arith.constant 64 : index
  pcf.loop scope(#iree_codegen.workgroup_scope) count(%c64)
    execute[%iv: index] {
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %arg0[%iv] : memref<64xf32>
    pcf.return
  }
  return
}

// CHECK-LABEL: @pcf_workgroup_loop
// Verify workgroup IDs are materialized.
//       CHECK:   hal.interface.workgroup.id
//       CHECK:   hal.interface.workgroup.count

// -----

func.func @pcf_workgroup_generic(%arg0: memref<64xf32>) {
  pcf.generic scope(#iree_codegen.workgroup_scope)
    execute[%id: index, %n: index] {
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %arg0[%id] : memref<64xf32>
    pcf.return
  }
  return
}

// CHECK-LABEL: @pcf_workgroup_generic
// Verify workgroup IDs are materialized.
//       CHECK:   hal.interface.workgroup.id
//       CHECK:   hal.interface.workgroup.count
