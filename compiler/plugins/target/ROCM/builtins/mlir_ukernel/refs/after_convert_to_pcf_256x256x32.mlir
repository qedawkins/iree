#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0) -> (d0 * 8)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 + d2 * 256 + d3 * 64)>
#map4 = affine_map<(d0)[s0] -> (d0 * 256 + s0)>
#map5 = affine_map<(d0, d1) -> (d0 * 32 + d1 * 8)>
#map6 = affine_map<(d0)[s0] -> (d0 * 32 + s0)>
#map7 = affine_map<(d0, d1) -> (d0 * 8 + d1 * 256)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map9 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 0, no_reduce_shared_memory_bank_conflicts = false>}>
module {
  func.func @matmul_8192x8192x8192_f16_f32() attributes {translation_info = #translation} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<8x4x4x1xf32>
    %1 = ub.poison : f32
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>>
    %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>>
    %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x16x512x16xf32>>
    %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>> -> tensor<8192x8192xf16>
    %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xf16>> -> tensor<8192x8192xf16>
    %7 = tensor.empty() : tensor<512x16x512x16xf32>
    %8 = scf.forall (%arg0, %arg1) in (32, 32) shared_outs(%arg2 = %7) -> (tensor<512x16x512x16xf32>) {
      %9 = affine.apply #map(%arg1)
      %10 = affine.apply #map(%arg0)
      %extracted_slice = tensor.extract_slice %arg2[%10, 0, %9, 0] [16, 16, 16, 16] [1, 1, 1, 1] : tensor<512x16x512x16xf32> to tensor<16x16x16x16xf32>
      %11 = pcf.alloc() : !pcf.sref<256x32xf16, #iree_gpu.subgroup_scope>
      %12 = pcf.alloc() : !pcf.sref<32x256xf16, #iree_gpu.subgroup_scope>
      %13 = pcf.generic scope(#iree_gpu.subgroup_scope) 
        execute(%ref = %extracted_slice)[%id: index, %count: index]
             : (!pcf.sref<16x16x16x16xf32, sync(#iree_gpu.subgroup_scope)>)
            -> (tensor<16x16x16x16xf32>) {
        pcf.generic scope(#iree_gpu.lane_scope) 
          execute[%id_1: index, %count_2: index] {
          %14 = affine.linearize_index [%id, %id_1] by (%count, %count_2) : index
          %15 = arith.muli %count, %count_2 : index
          %c1_3 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %16 = arith.muli %c1_3, %c2 : index
          %c4 = arith.constant 4 : index
          %17 = arith.muli %16, %c4 : index
          %c64 = arith.constant 64 : index
          %18 = arith.muli %17, %c64 : index
          %19 = arith.ceildivui %18, %15 : index
          %20 = arith.muli %14, %19 : index
          %21 = arith.addi %20, %19 : index
          %22 = arith.minui %21, %18 : index
          scf.forall (%arg3) = (%20) to (%22) step (1) {
            %23:3 = affine.delinearize_index %arg3 into (2, 4, 64) : index, index, index
            %24:3 = affine.delinearize_index %23#2 into (4, 16) : index, index, index
            %25 = iree_codegen.index_hint %24#1(#iree_gpu.lane_constant<16>) : index
            %26 = iree_codegen.index_hint %24#2(#iree_gpu.lane_increment<16, aligned>) : index
            %27 = affine.linearize_index disjoint [%25, %c0] by (4, 4) : index
            %28 = tensor.empty() : tensor<8x4x4x1xf32>
            %29 = vector.transfer_write %cst, %28[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<8x4x4x1xf32>, tensor<8x4x4x1xf32>
            %30 = affine.apply #map1(%23#1)
            %31 = affine.apply #map2(%23#0)
            %32 = affine.linearize_index disjoint [%25, %c0] by (4, 8) : index
            %33 = scf.for %arg4 = %c0 to %c256 step %c1 iter_args(%arg5 = %29) -> (tensor<8x4x4x1xf32>) {
              %38:2 = iree_gpu.barrier_region ins(%11, %12 : !pcf.sref<256x32xf16, #iree_gpu.subgroup_scope>, !pcf.sref<32x256xf16, #iree_gpu.subgroup_scope>) {
              ^bb0(%arg6: !pcf.sref<256x32xf16, #iree_gpu.subgroup_scope>, %arg7: !pcf.sref<32x256xf16, #iree_gpu.subgroup_scope>):
                %49 = tensor.empty() : tensor<256x32xf16>
                %50 = scf.forall (%arg8) in (2) shared_outs(%arg9 = %49) -> (tensor<256x32xf16>) {
                  %53 = affine.apply #map3(%arg8, %23#2, %23#0, %23#1)
                  %54:2 = affine.delinearize_index %53 into (256, 4) : index, index
                  %55 = affine.apply #map2(%54#1)
                  %extracted_slice_5 = tensor.extract_slice %arg9[%54#0, %55] [1, 8] [1, 1] : tensor<256x32xf16> to tensor<1x8xf16>
                  %56 = affine.apply #map4(%arg0)[%54#0]
                  %57 = affine.apply #map5(%arg4, %54#1)
                  %extracted_slice_6 = tensor.extract_slice %5[%56, %57] [1, 8] [1, 1] : tensor<8192x8192xf16> to tensor<1x8xf16>
                  %58 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_6 : tensor<1x8xf16>) outs(%extracted_slice_5 : tensor<1x8xf16>) -> tensor<1x8xf16>
                  pcf.write_slice %58 into %11[%54#0, %55] [1, 8] [1, 1] : tensor<1x8xf16> into !pcf.sref<256x32xf16, #iree_gpu.subgroup_scope>
                  scf.forall.in_parallel {
                  }
                } {unroll_loop}
                %51 = tensor.empty() : tensor<32x256xf16>
                %52 = scf.forall (%arg8) in (2) shared_outs(%arg9 = %51) -> (tensor<32x256xf16>) {
                  %53 = affine.apply #map3(%arg8, %23#2, %23#0, %23#1)
                  %54:2 = affine.delinearize_index %53 into (32, 32) : index, index
                  %55 = affine.apply #map2(%54#1)
                  %extracted_slice_5 = tensor.extract_slice %arg9[%54#0, %55] [1, 8] [1, 1] : tensor<32x256xf16> to tensor<1x8xf16>
                  %56 = affine.apply #map6(%arg4)[%54#0]
                  %57 = affine.apply #map7(%54#1, %arg1)
                  %extracted_slice_6 = tensor.extract_slice %6[%56, %57] [1, 8] [1, 1] : tensor<8192x8192xf16> to tensor<1x8xf16>
                  %58 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_6 : tensor<1x8xf16>) outs(%extracted_slice_5 : tensor<1x8xf16>) -> tensor<1x8xf16>
                  pcf.write_slice %58 into %12[%54#0, %55] [1, 8] [1, 1] : tensor<1x8xf16> into !pcf.sref<32x256xf16, #iree_gpu.subgroup_scope>
                  scf.forall.in_parallel {
                  }
                } {unroll_loop}
                iree_gpu.yield %arg6, %arg7 : !pcf.sref<256x32xf16, #iree_gpu.subgroup_scope>, !pcf.sref<32x256xf16, #iree_gpu.subgroup_scope>
              } : !pcf.sref<256x32xf16, #iree_gpu.subgroup_scope>, !pcf.sref<32x256xf16, #iree_gpu.subgroup_scope>
              %39 = pcf.expand_shape %38#1 [[0, 1], [2, 3]] : !pcf.sref<32x256xf16, #iree_gpu.subgroup_scope> into !pcf.sref<1x32x16x16xf16, #iree_gpu.subgroup_scope>
              %40 = pcf.subview %39[0, %32, %30, %26] [1, 8, 4, 1] [1, 1, 1, 1] : !pcf.sref<1x32x16x16xf16, #iree_gpu.subgroup_scope> to !pcf.sref<1x8x4x1xf16, #iree_gpu.subgroup_scope>
              %41 = pcf.expand_shape %38#0 [[0, 1], [2, 3]] : !pcf.sref<256x32xf16, #iree_gpu.subgroup_scope> into !pcf.sref<16x16x1x32xf16, #iree_gpu.subgroup_scope>
              %42 = pcf.subview %41[%31, %26, 0, %32] [8, 1, 1, 8] [1, 1, 1, 1] : !pcf.sref<16x16x1x32xf16, #iree_gpu.subgroup_scope> to !pcf.sref<8x1x1x8xf16, #iree_gpu.subgroup_scope>
              %43 = pcf.read_slice %42[%c0, %c0, %c0, %c0] [8, 1, 1, 8] [1, 1, 1, 1] : !pcf.sref<8x1x1x8xf16, #iree_gpu.subgroup_scope> to vector<8x1x1x8xf16>
              %44 = pcf.read_slice %40[%c0, %c0, %c0, %c0] [1, 8, 4, 1] [1, 1, 1, 1] : !pcf.sref<1x8x4x1xf16, #iree_gpu.subgroup_scope> to vector<1x8x4x1xf16>
              %45 = vector.transpose %44, [0, 2, 3, 1] : vector<1x8x4x1xf16> to vector<1x4x1x8xf16>
              %46 = vector.transfer_read %arg5[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : tensor<8x4x4x1xf32>, vector<8x4x4x1xf32>
              %47 = iree_codegen.inner_tiled ins(%43, %45) outs(%46) {indexing_maps = [#map8, #map9, #map10], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>, semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>} : vector<8x1x1x8xf16>, vector<1x4x1x8xf16> into vector<8x4x4x1xf32>
              %48 = vector.transfer_write %47, %arg5[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<8x4x4x1xf32>, tensor<8x4x4x1xf32>
              scf.yield %48 : tensor<8x4x4x1xf32>
            }
            %34 = vector.transfer_read %33[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : tensor<8x4x4x1xf32>, vector<8x4x4x1xf32>
            %35 = vector.transpose %34, [0, 2, 1, 3] : vector<8x4x4x1xf32> to vector<8x4x4x1xf32>
            %36 = vector.transfer_write %35, %28[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<8x4x4x1xf32>, tensor<8x4x4x1xf32>
            %extracted_slice_4 = tensor.extract_slice %extracted_slice[%31, %27, %30, %26] [8, 4, 4, 1] [1, 1, 1, 1] : tensor<16x16x16x16xf32> to tensor<8x4x4x1xf32>
            %37 = linalg.copy ins(%36 : tensor<8x4x4x1xf32>) outs(%extracted_slice_4 : tensor<8x4x4x1xf32>) -> tensor<8x4x4x1xf32>
            pcf.write_slice %37 into %ref[%31, %27, %30, %26] [8, 4, 4, 1] [1, 1, 1, 1] : tensor<8x4x4x1xf32> into !pcf.sref<16x16x16x16xf32, sync(#iree_gpu.subgroup_scope)>
          }
          pcf.return
        }  
        pcf.return
      }  
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %13 into %arg2[%10, 0, %9, 0] [16, 16, 16, 16] [1, 1, 1, 1] : tensor<16x16x16x16xf32> into tensor<512x16x512x16xf32>
      }
    } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
    iree_tensor_ext.dispatch.tensor.store %8, %4, offsets = [0, 0, 0, 0], sizes = [512, 16, 512, 16], strides = [1, 1, 1, 1] : tensor<512x16x512x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x16x512x16xf32>>
    return
  }
}

