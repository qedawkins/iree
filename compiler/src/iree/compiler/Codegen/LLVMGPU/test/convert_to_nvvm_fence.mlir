// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_60 --iree-convert-to-nvvm %s | FileCheck %s

// Test that iree_codegen.fence lowers to llvm.fence for NVVM targets.
builtin.module {
  func.func @fence_workgroup_release() {
    iree_codegen.fence release #gpu.address_space<workgroup>
    return
  }
}
// CHECK-LABEL: llvm.func @fence_workgroup_release
//       CHECK:   llvm.fence syncscope("workgroup") release
//       CHECK:   llvm.return

// -----

builtin.module {
  func.func @fence_workgroup_acquire() {
    iree_codegen.fence acquire #gpu.address_space<workgroup>
    return
  }
}
// CHECK-LABEL: llvm.func @fence_workgroup_acquire
//       CHECK:   llvm.fence syncscope("workgroup") acquire
//       CHECK:   llvm.return

// -----

// Test that global address space produces a fence without syncscope.
builtin.module {
  func.func @fence_global() {
    iree_codegen.fence release #gpu.address_space<global>
    return
  }
}
// CHECK-LABEL: llvm.func @fence_global
//       CHECK:   llvm.fence release
//   CHECK-NOT:   syncscope
//       CHECK:   llvm.return
