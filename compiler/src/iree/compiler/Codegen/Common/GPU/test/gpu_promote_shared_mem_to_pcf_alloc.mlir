// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-gpu-promote-shared-mem-to-pcf-alloc))" --split-input-file | FileCheck %s

// Test converting bufferization.alloc_tensor + transfer_write + value_barrier
// + transfer_read chain to pcf.alloc + sref ops.

func.func @promote_shared_mem(%output: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f16
  %0 = pcf.generic scope(#iree_codegen.workgroup_scope)
    execute(%out_ref = %output)[%wg_id: index, %wg_count: index]
         : (!pcf.sref<128x128xf32, #iree_codegen.workgroup_scope>)
        -> (tensor<128x128xf32>) {
    pcf.shared_executor scope(#iree_gpu.thread_scope)
      execute[%tg: !pcf.threadgroup<#iree_gpu.thread_scope>] {
      %c0 = arith.constant 0 : index
      %vec = arith.constant dense<1.0> : vector<64x128xf16>
      %smem = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>}
          : tensor<64x128xf16, #gpu.address_space<workgroup>>
      %written = vector.transfer_write %vec, %smem[%c0, %c0]
          {in_bounds = [true, true]}
          : vector<64x128xf16>, tensor<64x128xf16, #gpu.address_space<workgroup>>
      %barrier = iree_gpu.value_barrier %written
          : tensor<64x128xf16, #gpu.address_space<workgroup>>
      %reread = vector.transfer_read %barrier[%c0, %c0], %cst
          {in_bounds = [true, true]}
          : tensor<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
      pcf.return
    }
    pcf.return
  }
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @promote_shared_mem
//       CHECK:   pcf.shared_executor scope(#iree_gpu.thread_scope)
//       CHECK:     initialize {
//       CHECK:       %[[ALLOC:.+]] = pcf.alloc() : !pcf.sref<64x128xf16, #iree_gpu.thread_scope>
//       CHECK:       pcf.yield %[[ALLOC]] : !pcf.sref<64x128xf16, #iree_gpu.thread_scope>
//       CHECK:     } -> (%[[SMEM:.+]]: !pcf.sref<64x128xf16, #iree_gpu.thread_scope>)
//       CHECK:     execute[%{{.*}}: !pcf.threadgroup<#iree_gpu.thread_scope>] {
//       CHECK:       %[[VEC:.+]] = arith.constant dense<1.{{.*}}> : vector<64x128xf16>
//       CHECK:       iree_vector_ext.transfer_write %[[VEC]], %[[SMEM]][%{{.*}}, %{{.*}}] {in_bounds = [true, true]{{.*}}} : vector<64x128xf16>, !pcf.sref<64x128xf16, #iree_gpu.thread_scope>
//       CHECK:       gpu.barrier
//       CHECK:       iree_vector_ext.transfer_read %[[SMEM]][%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true]{{.*}}} : !pcf.sref<64x128xf16, #iree_gpu.thread_scope>, vector<64x128xf16>
//   CHECK-NOT:   bufferization.alloc_tensor
//   CHECK-NOT:   iree_gpu.value_barrier
