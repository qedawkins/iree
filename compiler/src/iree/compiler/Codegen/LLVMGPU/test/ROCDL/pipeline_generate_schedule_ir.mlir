// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1201 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target{for-rocdl=true}), iree-pcf-convert-sref-to-memref, iree-pcf-lower-structural-pcf))))" %s | FileCheck %s

// Test that the LLVMGPUGenerateScheduleIR pipeline produces correct lowered IR
// for a matmul dispatch targeting gfx1201 (RDNA4).
//
// Pipeline: GenerateScheduleIR → GPU bufferize → ConvertSRefToMemRef → LowerStructuralPCF
//
// After lowering:
//   - LDS allocations are memref.alloc with workgroup address space.
//   - Global loads become memref.subview.
//   - LDS reads are vector.transfer_read from workgroup memory.
//   - LDS writes are memref.copy.
//   - Compute is vector.contract.
//   - Barriers are gpu.barrier.
//   - Result writeback is vector.transfer_write.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_f16_f32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f16_f32()
        attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUGenerateScheduleIR workgroup_size = [32, 1, 1] subgroup_size = 32>} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xf16>> -> tensor<256x128xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf16>> -> tensor<128x256xf16>
        %5 = tensor.empty() : tensor<256x256xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
        %7 = linalg.matmul ins(%3, %4 : tensor<256x128xf16>, tensor<128x256xf16>)
                           outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func @matmul_f16_f32
//   CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
// LDS allocations with workgroup address space.
//   CHECK-DAG:   memref.alloc(){{.*}}: memref<256x64xf16, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc(){{.*}}: memref<64x256xf16, #gpu.address_space<workgroup>>
// Prologue: copy first K tile to LDS.
//       CHECK:   memref.copy
//       CHECK:   memref.copy
//       CHECK:   gpu.barrier
// K-loop with 4 quarter-K compute phases.
//       CHECK:   scf.for {{.*}} iter_args
//  CHECK-COUNT-4: vector.contract {{.*}} vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
//       CHECK:   scf.yield
// Result writeback.
//       CHECK:   vector.transfer_write
// No PCF ops remain after lowering.
//   CHECK-NOT:   pcf.
//       CHECK:   return
