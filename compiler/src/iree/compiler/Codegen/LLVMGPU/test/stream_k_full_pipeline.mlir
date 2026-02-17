// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant( \
// RUN:     builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s

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
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
  promote_operands = [0, 1],
  reduction = [0, 0, 16],
  streamed_reduction = [0, 0, 16],
  subgroup = [2, 4, 0],
  workgroup = [64, 128, 0]
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
              pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1]
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

// Verify the fully lowered pipeline output after bufferization.
// CHECK-LABEL: func @matmul

// Workgroup scope with scratch and counter allocations.
//       CHECK:   pcf.generic scope(#iree_codegen.workgroup_scope<linearize>)
//       CHECK:     initialize
//       CHECK:       pcf.alloc() : !pcf.sref<2048x128xf32, #iree_codegen.workgroup_scope
//       CHECK:       pcf.alloc() : !pcf.sref<2xi32, #iree_codegen.workgroup_scope
//       CHECK:     execute

// Producer generic with sync_on_return=true containing MFMA compute.
//       CHECK:     pcf.generic sync true scope(#iree_gpu.subgroup_scope)
//       CHECK:       pcf.generic scope(#iree_gpu.lane_scope)

// MFMA intrinsics (not vector.contract).
//       CHECK:           amdgpu.mfma 16x16x4 {{.*}} : f32, f32, vector<4xf32>
//   CHECK-NOT:           vector.contract
//
// Collapse decomposition inside producer: scf.for loops with collapse_shape
// and TWO scf.if guards (isSplit -> scratch, isNotSplit -> output sref).
//       CHECK:           scf.for
//       CHECK:             scf.for
//       CHECK:               memref.collapse_shape
//       CHECK:               scf.if
//       CHECK:                 pcf.write_slice {{.*}} into %arg0
//       CHECK:               scf.if
//       CHECK:                 pcf.write_slice {{.*}} into %ref

// Outer split condition for post-k-loop phases.
//       CHECK:     scf.if

// ── Phase 2: Barrier + atomic + broadcast ──
//       CHECK:       gpu.barrier memfence [#gpu.address_space<global>]
//       CHECK:       gpu.thread_id x
//       CHECK:       arith.cmpi eq
//       CHECK:       scf.if
//       CHECK:         memref.generic_atomic_rmw
//       CHECK:         arith.remui
//       CHECK:         memref.store
//       CHECK:       gpu.barrier memfence [#gpu.address_space<workgroup>, #gpu.address_space<global>]
//       CHECK:       memref.load

// ── Phase 3: Distributed recombine (if last contributor) ──
//       CHECK:       arith.cmpi eq
//       CHECK:       scf.if
//       CHECK:         pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:           pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK:             scf.for
//       CHECK:               pcf.get_memref %arg0
//       CHECK:               scf.for
//       CHECK:                 pcf.get_memref %arg0
//       CHECK:                 arith.addf
//       CHECK:               pcf.write_slice {{.*}} into %ref

// ── Phase 4: Distributed non-split writeback (else branch) ──
//       CHECK:     } else {
//       CHECK:       pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:         pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK:           scf.for
//       CHECK:             pcf.write_slice {{.*}} into %ref
//       CHECK:     }

// Final barrier.
//       CHECK:     gpu.barrier

// No recombine ops remain.
//   CHECK-NOT:     pcf.stream_k_recombine
//       CHECK:     pcf.return
