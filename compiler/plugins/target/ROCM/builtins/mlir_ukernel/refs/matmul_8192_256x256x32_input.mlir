// 8192x8192x8192 f16 matmul with forced 256x256x32 tile for gfx950.
// LLVMGPUTileAndFuse pipeline to get FuseAndHoistParallelLoops IR.
//
// Subgroup layout: 2M x 4N subgroups, each handles 8x4 = 32 MFMA-16x16.
// 8 subgroups x 64 lanes = 512 threads.
// K_block = 32 = MFMA_K (single dot per reduction step).

#config = #iree_gpu.lowering_config<{
  workgroup = [256, 256, 0],
  reduction = [0, 0, 1],
  subgroup = [8, 4, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>,
  promote_operands = [0, 1]
}>
#translation = #iree_codegen.translation_info<
  pipeline = LLVMGPUTileAndFuse
  workgroup_size = [512, 1, 1]
  subgroup_size = 64, {
    gpu_pipeline_options = #iree_gpu.pipeline_options<
      prefetch_num_stages = 0,
      no_reduce_shared_memory_bank_conflicts = false
    >
  }
>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_8192x8192x8192_f16_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_8192x8192x8192_f16_f32 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root(%arg1, %arg2)
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_8192x8192x8192_f16_f32() attributes {translation_info = #translation} {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192x8192xf32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>> -> tensor<8192x8192xf16>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>> -> tensor<8192x8192xf16>
      %5 = tensor.empty() : tensor<8192x8192xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
      %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<8192x8192xf16>, tensor<8192x8192xf16>) outs(%6 : tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
      iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : tensor<8192x8192xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192x8192xf32>>
      return
    }
  }
}
}
