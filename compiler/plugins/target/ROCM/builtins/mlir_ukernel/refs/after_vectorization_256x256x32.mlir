// -----// IR Dump After GenericVectorizationPass (iree-codegen-generic-vectorization) //----- //
func.func @matmul_8192x8192x8192_f16_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 0, no_reduce_shared_memory_bank_conflicts = false>}>} {
  %0 = ub.poison : f16
  %cst = arith.constant dense<0.000000e+00> : vector<8x4x4x1xf32>
  %1 = ub.poison : f32
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x16x512x16xf32>>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>> -> tensor<8192x8192xf16>
  %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>> -> tensor<8192x8192xf16>
  %7 = tensor.empty() : tensor<512x16x512x16xf32>
  %8 = scf.forall (%arg0, %arg1) in (32, 32) shared_outs(%arg2 = %7) -> (tensor<512x16x512x16xf32>) {
    %9 = affine.apply affine_map<(d0) -> (d0 * 16)>(%arg1)
    %10 = affine.apply affine_map<(d0) -> (d0 * 16)>(%arg0)
    %extracted_slice = tensor.extract_slice %arg2[%10, 0, %9, 0] [16, 16, 16, 16] [1, 1, 1, 1] : tensor<512x16x512x16xf32> to tensor<16x16x16x16xf32>
    %11 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<256x32xf16>
    %12 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32x256xf16>
    %13 = scf.forall (%arg3, %arg4, %arg5) in (2, 4, 64) shared_outs(%arg6 = %extracted_slice) -> (tensor<16x16x16x16xf32>) {
      %14:3 = affine.delinearize_index %arg5 into (4, 16) : index, index, index
      %15 = iree_codegen.index_hint %14#1(#iree_gpu.lane_constant<16>) : index
      %16 = iree_codegen.index_hint %14#2(#iree_gpu.lane_increment<16, aligned>) : index
      %17 = affine.linearize_index disjoint [%15, %c0] by (4, 4) : index
      %18 = tensor.empty() : tensor<8x4x4x1xf32>
      %19 = vector.transfer_write %cst, %18[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<8x4x4x1xf32>, tensor<8x4x4x1xf32>
      %20 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg4)
      %21 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      %22 = affine.linearize_index disjoint [%15, %c0] by (4, 8) : index
      %23 = scf.for %arg7 = %c0 to %c256 step %c1 iter_args(%arg8 = %19) -> (tensor<8x4x4x1xf32>) {
        %28:2 = iree_gpu.barrier_region ins(%11, %12 : tensor<256x32xf16>, tensor<32x256xf16>) {
        ^bb0(%arg9: tensor<256x32xf16>, %arg10: tensor<32x256xf16>):
          %35 = scf.forall (%arg11) in (2) shared_outs(%arg12 = %arg9) -> (tensor<256x32xf16>) {
            %37 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 + d2 * 256 + d3 * 64)>(%arg11, %arg5, %arg3, %arg4)
            %38:2 = affine.delinearize_index %37 into (256, 4) : index, index
            %39 = affine.apply affine_map<(d0) -> (d0 * 8)>(%38#1)
            %extracted_slice_5 = tensor.extract_slice %arg12[%38#0, %39] [1, 8] [1, 1] : tensor<256x32xf16> to tensor<1x8xf16>
            %40 = affine.apply affine_map<(d0)[s0] -> (d0 * 256 + s0)>(%arg0)[%38#0]
            %41 = affine.apply affine_map<(d0, d1) -> (d0 * 32 + d1 * 8)>(%arg7, %38#1)
            %extracted_slice_6 = tensor.extract_slice %5[%40, %41] [1, 8] [1, 1] : tensor<8192x8192xf16> to tensor<1x8xf16>
            %42 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_6 : tensor<1x8xf16>) outs(%extracted_slice_5 : tensor<1x8xf16>) -> tensor<1x8xf16>
            scf.forall.in_parallel {
              tensor.parallel_insert_slice %42 into %arg12[%38#0, %39] [1, 8] [1, 1] : tensor<1x8xf16> into tensor<256x32xf16>
            }
          } {unroll_loop}
          %36 = scf.forall (%arg11) in (2) shared_outs(%arg12 = %arg10) -> (tensor<32x256xf16>) {
            %37 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 + d2 * 256 + d3 * 64)>(%arg11, %arg5, %arg3, %arg4)
            %38:2 = affine.delinearize_index %37 into (32, 32) : index, index
            %39 = affine.apply affine_map<(d0) -> (d0 * 8)>(%38#1)
            %extracted_slice_5 = tensor.extract_slice %arg12[%38#0, %39] [1, 8] [1, 1] : tensor<32x256xf16> to tensor<1x8xf16>
            %40 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg7)[%38#0]
            %41 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1 * 256)>(%38#1, %arg1)
            %extracted_slice_6 = tensor.extract_slice %6[%40, %41] [1, 8] [1, 1] : tensor<8192x8192xf16> to tensor<1x8xf16>
            %42 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_6 : tensor<1x8xf16>) outs(%extracted_slice_5 : tensor<1x8xf16>) -> tensor<1x8xf16>
            scf.forall.in_parallel {
              tensor.parallel_insert_slice %42 into %arg12[%38#0, %39] [1, 8] [1, 1] : tensor<1x8xf16> into tensor<32x256xf16>
            }
          } {unroll_loop}
          iree_gpu.yield %35, %36 : tensor<256x32xf16>, tensor<32x256xf16>
        } : tensor<256x32xf16>, tensor<32x256xf16>
        %expanded = tensor.expand_shape %28#1 [[0, 1], [2, 3]] output_shape [1, 32, 16, 16] : tensor<32x256xf16> into tensor<1x32x16x16xf16>
        %extracted_slice_2 = tensor.extract_slice %expanded[0, %22, %20, %16] [1, 8, 4, 1] [1, 1, 1, 1] : tensor<1x32x16x16xf16> to tensor<1x8x4x1xf16>
        %expanded_3 = tensor.expand_shape %28#0 [[0, 1], [2, 3]] output_shape [16, 16, 1, 32] : tensor<256x32xf16> into tensor<16x16x1x32xf16>
        %extracted_slice_4 = tensor.extract_slice %expanded_3[%21, %16, 0, %22] [8, 1, 1, 8] [1, 1, 1, 1] : tensor<16x16x1x32xf16> to tensor<8x1x1x8xf16>
        %29 = vector.transfer_read %extracted_slice_4[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : tensor<8x1x1x8xf16>, vector<8x1x1x8xf16>
        %30 = vector.transfer_read %extracted_slice_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : tensor<1x8x4x1xf16>, vector<1x8x4x1xf16>
        %31 = vector.transpose %30, [0, 2, 3, 1] : vector<1x8x4x1xf16> to vector<1x4x1x8xf16>
        %32 = vector.transfer_read %arg8[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : tensor<8x4x4x1xf32>, vector<8x4x4x1xf32>
        %33 = iree_codegen.inner_tiled ins(%29, %31) outs(%32) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>, semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>} : vector<8x1x1x8xf16>, vector<1x4x1x8xf16> into vector<8x4x4x1xf32>
        %34 = vector.transfer_write %33, %arg8[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<8x4x4x1xf32>, tensor<8x4x4x1xf32>
        scf.yield %34 : tensor<8x4x4x1xf32>
      }
      %24 = vector.transfer_read %23[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : tensor<8x4x4x1xf32>, vector<8x4x4x1xf32>
      %25 = vector.transpose %24, [0, 2, 1, 3] : vector<8x4x4x1xf32> to vector<8x4x4x1xf32>
      %26 = vector.transfer_write %25, %18[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<8x4x4x1xf32>, tensor<8x4x4x1xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%21, %17, %20, %16] [8, 4, 4, 1] [1, 1, 1, 1] : tensor<16x16x16x16xf32> to tensor<8x4x4x1xf32>
      %27 = linalg.copy ins(%26 : tensor<8x4x4x1xf32>) outs(%extracted_slice_1 : tensor<8x4x4x1xf32>) -> tensor<8x4x4x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %27 into %arg6[%21, %17, %20, %16] [8, 4, 4, 1] [1, 1, 1, 1] : tensor<8x4x4x1xf32> into tensor<16x16x16x16xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %13 into %arg2[%10, 0, %9, 0] [16, 16, 16, 16] [1, 1, 1, 1] : tensor<16x16x16x16xf32> into tensor<512x16x512x16xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  iree_tensor_ext.dispatch.tensor.store %8, %4, offsets = [0, 0, 0, 0], sizes = [512, 16, 512, 16], strides = [1, 1, 1, 1] : tensor<512x16x512x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x16x512x16xf32>>
  return
}

