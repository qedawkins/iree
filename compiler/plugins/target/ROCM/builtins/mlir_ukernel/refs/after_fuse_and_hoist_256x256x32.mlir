// -----// IR Dump After GPUFuseAndHoistParallelLoopsPass (iree-codegen-gpu-fuse-and-hoist-parallel-loops) //----- //
func.func @matmul_8192x8192x8192_f16_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 0, no_reduce_shared_memory_bank_conflicts = false>}>} {
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x16x512x16xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>> -> tensor<8192x8192xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>> -> tensor<8192x8192xf16>
  %5 = tensor.empty() : tensor<512x16x512x16xf32>
  %6 = scf.forall (%arg0, %arg1) in (32, 32) shared_outs(%arg2 = %5) -> (tensor<512x16x512x16xf32>) {
    %7 = affine.apply affine_map<(d0) -> (d0 * 16)>(%arg1)
    %8 = affine.apply affine_map<(d0) -> (d0 * 16)>(%arg0)
    %extracted_slice = tensor.extract_slice %arg2[%8, 0, %7, 0] [16, 16, 16, 16] [1, 1, 1, 1] : tensor<512x16x512x16xf32> to tensor<16x16x16x16xf32>
    %9 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<256x32xf16>
    %10 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32x256xf16>
    %11 = scf.forall (%arg3, %arg4, %arg5) in (2, 4, 64) shared_outs(%arg6 = %extracted_slice) -> (tensor<16x16x16x16xf32>) {
      %12:3 = affine.delinearize_index %arg5 into (4, 16) : index, index, index
      %13 = iree_codegen.index_hint %12#1(#iree_gpu.lane_constant<16>) : index
      %14 = iree_codegen.index_hint %12#2(#iree_gpu.lane_increment<16, aligned>) : index
      %15 = affine.linearize_index disjoint [%13, %c0] by (4, 4) : index
      %16 = tensor.empty() : tensor<8x4x4x1xf32>
      %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<8x4x4x1xf32>) -> tensor<8x4x4x1xf32>
      %18 = scf.for %arg7 = %c0 to %c256 step %c1 iter_args(%arg8 = %17) -> (tensor<8x4x4x1xf32>) {
        %23 = iree_gpu.barrier_region ins(%9 : tensor<256x32xf16>) {
        ^bb0(%arg9: tensor<256x32xf16>):
          %31 = scf.forall (%arg10) in (2) shared_outs(%arg11 = %arg9) -> (tensor<256x32xf16>) {
            %32 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 + d2 * 256 + d3 * 64)>(%arg10, %arg5, %arg3, %arg4)
            %33:2 = affine.delinearize_index %32 into (256, 4) : index, index
            %34 = affine.apply affine_map<(d0) -> (d0 * 8)>(%33#1)
            %extracted_slice_6 = tensor.extract_slice %arg11[%33#0, %34] [1, 8] [1, 1] : tensor<256x32xf16> to tensor<1x8xf16>
            %35 = affine.apply affine_map<(d0)[s0] -> (d0 * 256 + s0)>(%arg0)[%33#0]
            %36 = affine.apply affine_map<(d0, d1) -> (d0 * 32 + d1 * 8)>(%arg7, %33#1)
            %extracted_slice_7 = tensor.extract_slice %3[%35, %36] [1, 8] [1, 1] : tensor<8192x8192xf16> to tensor<1x8xf16>
            %37 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_7 : tensor<1x8xf16>) outs(%extracted_slice_6 : tensor<1x8xf16>) -> tensor<1x8xf16>
            scf.forall.in_parallel {
              tensor.parallel_insert_slice %37 into %arg11[%33#0, %34] [1, 8] [1, 1] : tensor<1x8xf16> into tensor<256x32xf16>
            }
          } {unroll_loop}
          iree_gpu.yield %31 : tensor<256x32xf16>
        } : tensor<256x32xf16>
        %24 = iree_gpu.barrier_region ins(%10 : tensor<32x256xf16>) {
        ^bb0(%arg9: tensor<32x256xf16>):
          %31 = scf.forall (%arg10) in (2) shared_outs(%arg11 = %arg9) -> (tensor<32x256xf16>) {
            %32 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 + d2 * 256 + d3 * 64)>(%arg10, %arg5, %arg3, %arg4)
            %33:2 = affine.delinearize_index %32 into (32, 32) : index, index
            %34 = affine.apply affine_map<(d0) -> (d0 * 8)>(%33#1)
            %extracted_slice_6 = tensor.extract_slice %arg11[%33#0, %34] [1, 8] [1, 1] : tensor<32x256xf16> to tensor<1x8xf16>
            %35 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg7)[%33#0]
            %36 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1 * 256)>(%33#1, %arg1)
            %extracted_slice_7 = tensor.extract_slice %4[%35, %36] [1, 8] [1, 1] : tensor<8192x8192xf16> to tensor<1x8xf16>
            %37 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_7 : tensor<1x8xf16>) outs(%extracted_slice_6 : tensor<1x8xf16>) -> tensor<1x8xf16>
            scf.forall.in_parallel {
              tensor.parallel_insert_slice %37 into %arg11[%33#0, %34] [1, 8] [1, 1] : tensor<1x8xf16> into tensor<32x256xf16>
            }
          } {unroll_loop}
          iree_gpu.yield %31 : tensor<32x256xf16>
        } : tensor<32x256xf16>
        %25 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg4)
        %26 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
        %27 = affine.linearize_index disjoint [%13, %c0] by (4, 8) : index
        %expanded = tensor.expand_shape %24 [[0, 1], [2, 3]] output_shape [1, 32, 16, 16] : tensor<32x256xf16> into tensor<1x32x16x16xf16>
        %extracted_slice_1 = tensor.extract_slice %expanded[0, %27, %25, %14] [1, 8, 4, 1] [1, 1, 1, 1] : tensor<1x32x16x16xf16> to tensor<1x8x4x1xf16>
        %expanded_2 = tensor.expand_shape %23 [[0, 1], [2, 3]] output_shape [16, 16, 1, 32] : tensor<256x32xf16> into tensor<16x16x1x32xf16>
        %extracted_slice_3 = tensor.extract_slice %expanded_2[%26, %14, 0, %27] [8, 1, 1, 8] [1, 1, 1, 1] : tensor<16x16x1x32xf16> to tensor<8x1x1x8xf16>
        %28 = tensor.empty() : tensor<8x1x1x8xf16>
        %transposed_4 = linalg.transpose ins(%extracted_slice_3 : tensor<8x1x1x8xf16>) outs(%28 : tensor<8x1x1x8xf16>) permutation = [0, 2, 1, 3] 
        %29 = tensor.empty() : tensor<1x4x1x8xf16>
        %transposed_5 = linalg.transpose ins(%extracted_slice_1 : tensor<1x8x4x1xf16>) outs(%29 : tensor<1x4x1x8xf16>) permutation = [0, 2, 3, 1] 
        %30 = iree_codegen.inner_tiled ins(%transposed_4, %transposed_5) outs(%arg8) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>, promote_operands = [0, 1], reduction = [0, 0, 1], subgroup = [8, 4, 0], workgroup = [256, 256, 0]}>, semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>} : tensor<8x1x1x8xf16>, tensor<1x4x1x8xf16> into tensor<8x4x4x1xf32>
        scf.yield %30 : tensor<8x4x4x1xf32>
      }
      %19 = tensor.empty() : tensor<8x4x4x1xf32>
      %transposed = linalg.transpose ins(%18 : tensor<8x4x4x1xf32>) outs(%19 : tensor<8x4x4x1xf32>) permutation = [0, 2, 1, 3] 
      %20 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      %21 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg4)
      %extracted_slice_0 = tensor.extract_slice %arg6[%20, %15, %21, %14] [8, 4, 4, 1] [1, 1, 1, 1] : tensor<16x16x16x16xf32> to tensor<8x4x4x1xf32>
      %22 = linalg.copy ins(%transposed : tensor<8x4x4x1xf32>) outs(%extracted_slice_0 : tensor<8x4x4x1xf32>) -> tensor<8x4x4x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %22 into %arg6[%20, %15, %21, %14] [8, 4, 4, 1] [1, 1, 1, 1] : tensor<8x4x4x1xf32> into tensor<16x16x16x16xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg2[%8, 0, %7, 0] [16, 16, 16, 16] [1, 1, 1, 1] : tensor<16x16x16x16xf32> into tensor<512x16x512x16xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [512, 16, 512, 16], strides = [1, 1, 1, 1] : tensor<512x16x512x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x16x512x16xf32>>
  return
}

