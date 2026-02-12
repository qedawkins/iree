// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-stream-k-tile)))))" %s | FileCheck %s

// NOTE: This file tests the stream-K tiling transformation triggered by the
// `streamed_reduction` tiling level in the lowering config.

// ============================================================================
// Test 1: Standard matmul 128x128x256 with tile 64x64x64.
//
// output_tiles = ceil(128/64) * ceil(128/64) = 2*2 = 4
// k_tiles = ceil(256/64) = 4
// total_work = 4 * 4 = 16
// num_workgroups = 8 (saturate GPU)
// items_per_wg = ceil(16/8) = 2
// ============================================================================

#pipeline_layout_3 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

// The lowering config should include a streamed_reduction level.
// Exact attribute format TBD — placeholder below.
#stream_k_config = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  streamed_reduction = [0, 0, 64],
  thread = [8, 4]
}>

hal.executable public @matmul_128x128x256_stream_k {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul ordinal(0) layout(#pipeline_layout_3)
        count(%device: !hal.device) -> (index, index, index) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      hal.return %c8, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @matmul()
          attributes {translation_info = #iree_codegen.translation_info<
              pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1]
              subgroup_size = 64>} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>>
        %rhs = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(1)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(2)
            alignment(64) offset(%c0)
            : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0],
            sizes = [128, 256], strides = [1, 1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>>
                -> tensor<128x256xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0, 0],
            sizes = [256, 128], strides = [1, 1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xf32>>
                -> tensor<256x128xf32>
        %init = tensor.empty() : tensor<128x128xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x128xf32>)
            -> tensor<128x128xf32>
        %mm = linalg.matmul {lowering_config = #stream_k_config}
            ins(%lhs_t, %rhs_t : tensor<128x256xf32>, tensor<256x128xf32>)
            outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %mm, %out, offsets = [0, 0],
            sizes = [128, 128], strides = [1, 1]
            : tensor<128x128xf32>
                -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }
}

// Stream-K tiling should produce:
// 1. A pcf.loop with workgroup scope and total_work count.
// 2. Decode arithmetic: divmod decomposition of linear index.
// 3. Input slicing using TilingInterface.
// 4. Compute of partial tile (the matmul on tile-sized operands).
// 5. pcf.stream_k_recombine with combiner (addf) and writeback.

// CHECK-LABEL: func @matmul
//
// Scratch and counter allocations.
//       CHECK:   pcf.alloc() : !pcf.sref<256x64xf32
//       CHECK:   pcf.alloc() : !pcf.sref<i32
//
// Loop with total_work = 16.
//       CHECK:   %[[C16:.+]] = arith.constant 16 : index
//       CHECK:   pcf.loop scope({{.+}}) count(%[[C16]])
//       CHECK:     execute({{.+}})[%[[WORK_IDX:.+]]: index]
//
// Decode: linear_idx -> out_idx, k_idx.
//       CHECK:     %[[OUT_IDX:.+]] = arith.divui %[[WORK_IDX]]
//       CHECK:     %[[K_IDX:.+]] = arith.remui %[[WORK_IDX]]
//
// Decompose out_idx into per-parallel-dim coordinates (remui then divui).
//       CHECK:     arith.remui %[[OUT_IDX]]
//       CHECK:     arith.divui %[[OUT_IDX]]
//
// Group size arithmetic.
//       CHECK:     hal.interface.workgroup.count
//       CHECK:     arith.divui
//       CHECK:     arith.subi
//       CHECK:     arith.addi
//
// Tiled inputs via extract_slice.
//       CHECK:     tensor.extract_slice
//       CHECK:     tensor.extract_slice
//
// Compute partial tile (matmul on sliced operands).
//       CHECK:     linalg.matmul
//
// Recombine.
//       CHECK:     pcf.stream_k_recombine
//       CHECK:       combiner
//       CHECK:         arith.addf
//       CHECK:       writeback
//       CHECK:         pcf.write_slice
//       CHECK:     pcf.return

// ============================================================================
// TODO: Add more test cases as implementation matures:
//
// - matmul_256x64x1024: Tall-skinny output, large K.
// - batch_matmul: 3D problem with batch dim.
// - dynamic_shapes: Dynamic M, N, K.
// - single_workgroup: num_workgroups = 1 degenerate case.
// - more_wgs_than_work: total_work < num_workgroups.
// - tile_not_dividing: Dims not divisible by tile sizes.
// ============================================================================
