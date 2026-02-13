// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1150 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module( \
// RUN:     func.func(iree-llvmgpu-stream-k-tile), \
// RUN:     iree-pcf-lower-stream-k-recombine))))" %s | FileCheck %s

// ============================================================================
// Test 1: Stream-K tiling + recombine lowering for 128x128x256 matmul.
//
// This chains two passes:
//   1. iree-llvmgpu-stream-k-tile: Produces pcf.generic + pcf.stream_k_recombine.
//   2. iree-pcf-lower-stream-k-recombine: Lowers recombine to atomics + branching.
//
// After both passes, we should see:
//   - pcf.generic with initializer (from tiling).
//   - No pcf.stream_k_recombine ops (all lowered).
//   - Outer scf.for over output tiles with inner scf.for accumulation.
//   - Atomic RMW on counter sref (from lowering).
//   - scf.if branching (sole/last/not-last contributor logic).
//   - pcf.write_slice in writeback branch.
//   - pcf.fence for memory ordering.
// ============================================================================

#pipeline_layout_4 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#stream_k_config = #iree_gpu.lowering_config<{
  workgroup = [64, 64, 0],
  streamed_reduction = [0, 0, 64],
  thread = [8, 4]
}>

hal.executable public @matmul_stream_k_integration {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul ordinal(0) layout(#pipeline_layout_4)
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
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>>
        %rhs = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(1)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x128xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(2)
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

// After tiling + lowering:
// - pcf.generic with initializer present (from tiling).
// - No stream_k_recombine (lowered away).
// - Allocs in initializer for partial results and counter.
// - Nested scf.for loops (outer output tiles, inner k-tiles).
// - Thread guard (gpu.thread_id + scf.if) around recombine ops.
// - Atomic counter increment inside thread guard.
// - Branching for writeback inside thread guard.
// - gpu.barrier after thread guard for synchronization.

// CHECK-LABEL: func @matmul
//       CHECK:   iree_codegen.workgroup_count_hint
//       CHECK:   pcf.generic
//       CHECK:     initialize
//       CHECK:       pcf.alloc() : !pcf.sref<{{.+}}, #iree_codegen.workgroup_scope
//       CHECK:       pcf.alloc() : !pcf.sref<4xi32, #iree_codegen.workgroup_scope
//       CHECK:     execute
// CHECK-NOT:       pcf.stream_k_recombine
//       CHECK:       scf.for
//       CHECK:         scf.for
//       CHECK:           linalg.matmul
//       CHECK:         gpu.thread_id
//       CHECK:         scf.if
//       CHECK:           pcf.get_memref
//       CHECK:           memref.atomic_rmw
//       CHECK:           arith.cmpi
//       CHECK:           scf.if
//       CHECK:             pcf.write_slice
//       CHECK:         gpu.barrier
//       CHECK:       pcf.return
