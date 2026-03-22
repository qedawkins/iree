// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-gpu-absorb-tensor-reads-into-pcf))" --split-input-file | FileCheck %s

// Test converting vector.transfer_read on a captured tensor to
// iree_vector_ext.transfer_read on a pcf.sref.

func.func @absorb_input_read(%input: tensor<256x256xf16>,
                              %output: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f16
  %0 = pcf.generic scope(#iree_codegen.workgroup_scope)
    execute(%out_ref = %output)[%wg_id: index, %wg_count: index]
         : (!pcf.sref<128x128xf32, #iree_codegen.workgroup_scope>)
        -> (tensor<128x128xf32>) {
    pcf.shared_executor scope(#iree_gpu.thread_scope)
      execute[%tg: !pcf.threadgroup<#iree_gpu.thread_scope>] {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %v = vector.transfer_read %input[%c0, %c64], %cst {in_bounds = [true, true]}
          : tensor<256x256xf16>, vector<64x128xf16>
      pcf.return
    }
    pcf.return
  }
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @absorb_input_read
//  CHECK-SAME:   %[[INPUT:.+]]: tensor<256x256xf16>
//       CHECK:   pcf.shared_executor scope(#iree_gpu.thread_scope)
//       CHECK:     execute[%{{.*}}: !pcf.threadgroup<#iree_gpu.thread_scope>] {
//       CHECK:       %[[SREF:.+]] = pcf.to_sref %[[INPUT]] : tensor<256x256xf16> -> !pcf.sref<256x256xf16, #iree_gpu.thread_scope>
//       CHECK:       iree_vector_ext.transfer_read %[[SREF]][%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true]
//  CHECK-SAME:           : !pcf.sref<256x256xf16, #iree_gpu.thread_scope>, vector<64x128xf16>
//   CHECK-NOT:   vector.transfer_read {{.*}} : tensor<
//       CHECK:       pcf.return

// -----

// Test converting vector.transfer_write + pcf.write_slice to
// iree_vector_ext.transfer_write on sref.

func.func @absorb_output_write(%output: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = pcf.generic scope(#iree_codegen.workgroup_scope)
    execute(%out_ref = %output)[%wg_id: index, %wg_count: index]
         : (!pcf.sref<128x128xf32, #iree_codegen.workgroup_scope>)
        -> (tensor<128x128xf32>) {
    pcf.shared_executor scope(#iree_gpu.thread_scope)
      execute(%se_ref = %output)[%tg: !pcf.threadgroup<#iree_gpu.thread_scope>]
           : (!pcf.sref<128x128xf32, #iree_gpu.thread_scope>)
          -> (tensor<128x128xf32>) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %vec = arith.constant dense<1.0> : vector<64x64xf32>
      %empty = tensor.empty() : tensor<64x64xf32>
      %written = vector.transfer_write %vec, %empty[%c0, %c0]
          {in_bounds = [true, true]}
          : vector<64x64xf32>, tensor<64x64xf32>
      pcf.write_slice %written into %se_ref [%c16, %c32] [64, 64] [1, 1]
          : tensor<64x64xf32> into !pcf.sref<128x128xf32, #iree_gpu.thread_scope>
      pcf.return
    }
    pcf.return
  }
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @absorb_output_write
//       CHECK:   pcf.shared_executor scope(#iree_gpu.thread_scope)
//       CHECK:     execute(%[[SE_REF:.+]] = %{{.*}})[%{{.*}}: !pcf.threadgroup<#iree_gpu.thread_scope>]
//       CHECK:       %[[C16:.+]] = arith.constant 16 : index
//       CHECK:       %[[C32:.+]] = arith.constant 32 : index
//       CHECK:       %[[VEC:.+]] = arith.constant dense<1.{{.*}}> : vector<64x64xf32>
//       CHECK:       iree_vector_ext.transfer_write %[[VEC]], %[[SE_REF]][%[[C16]], %[[C32]]]
//  CHECK-SAME:           {in_bounds = [true, true]
//  CHECK-SAME:           : vector<64x64xf32>, !pcf.sref<128x128xf32, #iree_gpu.thread_scope>
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   pcf.write_slice

// -----

// Test that vector.transfer_read on a tensor defined inside the
// shared_executor is NOT converted.

func.func @no_convert_local_tensor(%output: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = pcf.generic scope(#iree_codegen.workgroup_scope)
    execute(%out_ref = %output)[%wg_id: index, %wg_count: index]
         : (!pcf.sref<128x128xf32, #iree_codegen.workgroup_scope>)
        -> (tensor<128x128xf32>) {
    pcf.shared_executor scope(#iree_gpu.thread_scope)
      execute[%tg: !pcf.threadgroup<#iree_gpu.thread_scope>] {
      %local = tensor.empty() : tensor<64x64xf32>
      %c0 = arith.constant 0 : index
      %v = vector.transfer_read %local[%c0, %c0], %cst {in_bounds = [true, true]}
          : tensor<64x64xf32>, vector<64x64xf32>
      pcf.return
    }
    pcf.return
  }
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @no_convert_local_tensor
//       CHECK:   pcf.shared_executor
//   CHECK-NOT:     pcf.to_sref
//       CHECK:     vector.transfer_read
//  CHECK-SAME:       : tensor<64x64xf32>
