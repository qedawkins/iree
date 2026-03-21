// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-gpu-wrap-in-shared-executor))" --split-input-file | FileCheck %s

// Test wrapping a workgroup-scoped pcf.generic in shared_executor.

func.func @wrap_generic(%tensor: tensor<128xf32>) -> tensor<128xf32> {
  %0 = pcf.generic scope(#iree_codegen.workgroup_scope)
    execute(%ref = %tensor)[%wg_id: index, %wg_count: index]
         : (!pcf.sref<128xf32, #iree_codegen.workgroup_scope>)
        -> (tensor<128xf32>) {
    %c0 = arith.constant 0 : index
    pcf.return
  }
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: func.func @wrap_generic
//       CHECK:   pcf.generic scope(#iree_codegen.workgroup_scope)
//       CHECK:     execute(%[[REF:.+]] = %{{.*}})[%{{.*}}: index, %{{.*}}: index]
//       CHECK:       pcf.shared_executor scope(#iree_gpu.thread_scope)
//       CHECK:         execute[%{{.*}}: !pcf.threadgroup<#iree_gpu.thread_scope>] {
//       CHECK:           arith.constant 0
//       CHECK:           pcf.return
//       CHECK:         }
//       CHECK:       pcf.return

// -----

// Test wrapping a workgroup-scoped pcf.loop in shared_executor.

func.func @wrap_loop(%tensor: tensor<64xf32>, %count: index) -> tensor<64xf32> {
  %0 = pcf.loop scope(#iree_codegen.workgroup_scope) count(%count)
    execute(%ref = %tensor)[%wg_id: index]
         : (!pcf.sref<64xf32, #iree_codegen.workgroup_scope>)
        -> (tensor<64xf32>) {
    %c0 = arith.constant 0 : index
    pcf.return
  }
  return %0 : tensor<64xf32>
}

// CHECK-LABEL: func.func @wrap_loop
//       CHECK:   pcf.loop scope(#iree_codegen.workgroup_scope) count(%{{.*}})
//       CHECK:     execute(%[[REF:.+]] = %{{.*}})[%{{.*}}: index]
//       CHECK:       pcf.shared_executor scope(#iree_gpu.thread_scope)
//       CHECK:         execute[%{{.*}}: !pcf.threadgroup<#iree_gpu.thread_scope>] {
//       CHECK:           arith.constant 0
//       CHECK:           pcf.return
//       CHECK:         }
//       CHECK:       pcf.return

// -----

// Test that non-workgroup-scoped pcf.generic is not wrapped.

func.func @no_wrap_subgroup(%tensor: tensor<128xf32>) -> tensor<128xf32> {
  %0 = pcf.generic scope(#iree_gpu.subgroup_scope)
    execute(%ref = %tensor)[%sg_id: index, %num_sg: index]
         : (!pcf.sref<128xf32, #iree_gpu.subgroup_scope>)
        -> (tensor<128xf32>) {
    %c0 = arith.constant 0 : index
    pcf.return
  }
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: func.func @no_wrap_subgroup
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//   CHECK-NOT:     pcf.shared_executor
//       CHECK:     arith.constant 0
//       CHECK:     pcf.return

// -----

// Test that an already-wrapped generic is not double-wrapped.

func.func @no_double_wrap(%tensor: tensor<128xf32>) -> tensor<128xf32> {
  %0 = pcf.generic scope(#iree_codegen.workgroup_scope)
    execute(%ref = %tensor)[%wg_id: index, %wg_count: index]
         : (!pcf.sref<128xf32, #iree_codegen.workgroup_scope>)
        -> (tensor<128xf32>) {
    pcf.shared_executor scope(#iree_gpu.thread_scope)
      execute[%tg: !pcf.threadgroup<#iree_gpu.thread_scope>] {
      %c0 = arith.constant 0 : index
      pcf.return
    }
    pcf.return
  }
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: func.func @no_double_wrap
//       CHECK:   pcf.generic scope(#iree_codegen.workgroup_scope)
//       CHECK:     pcf.shared_executor scope(#iree_gpu.thread_scope)
//       CHECK:       arith.constant 0
//       CHECK:       pcf.return
//       CHECK:     pcf.return

// -----

// Test wrapping an empty body (only terminator) is a no-op.

func.func @empty_body(%tensor: tensor<128xf32>) -> tensor<128xf32> {
  %0 = pcf.generic scope(#iree_codegen.workgroup_scope)
    execute(%ref = %tensor)[%wg_id: index, %wg_count: index]
         : (!pcf.sref<128xf32, #iree_codegen.workgroup_scope>)
        -> (tensor<128xf32>) {
    pcf.return
  }
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: func.func @empty_body
//       CHECK:   pcf.generic scope(#iree_codegen.workgroup_scope)
//   CHECK-NOT:     pcf.shared_executor
//       CHECK:     pcf.return
