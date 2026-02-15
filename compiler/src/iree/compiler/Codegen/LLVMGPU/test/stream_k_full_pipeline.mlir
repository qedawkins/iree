// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1150 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant( \
// RUN:     builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   --mlir-print-ir-after=iree-codegen-gpu-fuse-consumers %s 2>&1 | \
// RUN: FileCheck %s

// ============================================================================
// Full pipeline test for Stream-K matmul: tiling -> forall-to-generic-nest
// -> fusion -> bufferization.
//
// Verifies that pcf.stream_k_recombine is fully fused into its producer
// pcf.generic with the correct distributed pattern:
//   1. Distributed scratch writes inside the producer (all threads).
//   2. Thread-0 only: atomic + broadcast via shared memory dword.
//   3. All threads: read broadcast, check isLast.
//   4. All threads (if isLast): accumulate + writeback.
//   5. Else (non-split): direct writeback.
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

hal.executable public @matmul_stream_k_full_pipeline {
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

// No pcf.stream_k_recombine should remain after fusion.
// CHECK-LABEL: func @matmul

// Workgroup scope with scratch and counter allocations.
//       CHECK:   pcf.generic scope(#iree_codegen.workgroup_scope<linearize>)
//       CHECK:     initialize
//       CHECK:       pcf.alloc() : !pcf.sref<{{.+}}, #iree_codegen.workgroup_scope
//       CHECK:       pcf.alloc() : !pcf.sref<4xi32, #iree_codegen.workgroup_scope
//       CHECK:     execute

// Producer generic with sync_on_return=true.
//       CHECK:     pcf.generic sync true scope(#iree_gpu.subgroup_scope)
//       CHECK:       pcf.generic scope(#iree_gpu.lane_scope)

// Distributed compute (matmul via vector.contract).
//       CHECK:           vector.contract

// Distributed scratch write inside producer (conditional on isSplit).
//       CHECK:           scf.if
//       CHECK:             pcf.write_slice {{.*}} into %arg0
//       CHECK:           pcf.write_slice {{.*}} into %ref

// Broadcast dword allocation for atomic result.
//       CHECK:     pcf.alloc() : !pcf.sref<1xi32

// Outer split condition.
//       CHECK:     scf.if
// Thread-0 only: atomic increment + store to broadcast dword.
//       CHECK:       gpu.thread_id x
//       CHECK:       scf.if
//       CHECK:         pcf.fence release
//       CHECK:         memref.atomic_rmw addi
//       CHECK:         memref.store
//       CHECK:       }

// Barrier: broadcast dword visible to all threads.
//       CHECK:       gpu.barrier

// All threads: load broadcast, check isLast.
//       CHECK:       memref.load
//       CHECK:       arith.index_cast
//       CHECK:       arith.cmpi eq

// All threads (if isLast): acquire fence + accumulate + writeback.
//       CHECK:       scf.if
//       CHECK:         pcf.fence acquire
//  Accumulation loop: read partials from scratch and combine.
//       CHECK:         scf.for
//       CHECK:           linalg.generic
//       CHECK:             arith.addf
//  First writeback.
//       CHECK:         pcf.write_slice {{.*}} into %ref
//       CHECK:       }

// Else: non-split direct writeback.
//       CHECK:     } else {
//       CHECK:       pcf.write_slice {{.*}} into %ref
//       CHECK:     }

// Final barrier.
//       CHECK:     gpu.barrier

// No recombine ops remain.
//   CHECK-NOT:     pcf.stream_k_recombine
//       CHECK:     pcf.return
