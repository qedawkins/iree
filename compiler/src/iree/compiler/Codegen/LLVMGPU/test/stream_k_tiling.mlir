// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1150 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-stream-k-tile)))))" %s | FileCheck %s

// NOTE: This file tests the stream-K tiling transformation triggered by the
// `streamed_reduction` tiling level in the lowering config.

// ============================================================================
// Test 1: Standard matmul 128x128x256 with tile 64x64x64.
//
// output_tiles = ceil(128/64) * ceil(128/64) = 2*2 = 4
// k_tiles = ceil(256/64) = 4
// total_work = 4 * 4 = 16
// cuCount = 512 (default, no chip for gfx1150)
// maxNumInGroup = min(512, 4) = 4
// scratch: [4*64, 64] = [256, 64] f32
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
// 1. A workgroup count hint (cuCount = 512 for gfx1150 default).
// 2. A pcf.generic with initializer (scratch + counter allocs).
// 3. Work range computation (items_per_wg, my_start, my_end).
// 4. Outer scf.for over output tiles.
// 5. Inner scf.for accumulating k-tile partials with iter_args.
// 6. pcf.stream_k_recombine per output tile with combiner + writeback.

// CHECK-LABEL: func @matmul
//
// Workgroup count hint.
//       CHECK:   iree_codegen.workgroup_count_hint
//
// pcf.generic with initializer region.
//       CHECK:   pcf.generic
//  CHECK-SAME:     scope(#iree_codegen.workgroup_scope<linearize>)
//       CHECK:     initialize
//       CHECK:       pcf.alloc() : !pcf.sref<256x64xf32
//       CHECK:       pcf.alloc() : !pcf.sref<i32
//       CHECK:       pcf.yield
//       CHECK:     execute
//
// Work range computation.
//       CHECK:       arith.constant 16 : index
//       CHECK:       arith.constant 4 : index
//       CHECK:       arith.constant 1 : index
//
// Outer scf.for over output tiles.
//       CHECK:       scf.for
//
// Inner scf.for with iter_args (zero tile accumulator).
//       CHECK:         linalg.fill
//       CHECK:         scf.for
//  CHECK-SAME:           iter_args
//
// Tiled matmul (extract_slice + matmul).
//       CHECK:           tensor.extract_slice
//       CHECK:           tensor.extract_slice
//       CHECK:           linalg.matmul
//       CHECK:           scf.yield
//
// Recombine (after inner loop, inside outer loop).
//       CHECK:         pcf.stream_k_recombine
//       CHECK:           combiner
//       CHECK:             arith.addf
//       CHECK:           writeback
//       CHECK:             pcf.write_slice
//       CHECK:       pcf.return

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
