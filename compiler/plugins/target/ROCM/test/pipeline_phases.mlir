// Test that the rocm.pipeline_options attribute on hal.executable.variant
// controls which phases of the translation pipeline run when invoked
// through the HAL executable translation pass.

// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(hal.executable(iree-hal-translate-all-executables))" \
// RUN:   %s | FileCheck %s

// Input IR is pre-configured (translation_info + lowering_config already set)
// so that the translation pipeline can lower it properly.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "gfx942",
    features = "",
    wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8,
      storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
      dot = dp4xi8toi32, subgroup_size_choices = [64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024,
      max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#map = affine_map<(d0) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

// Default options: run all phases. Should produce LLVM+ROCDL IR with
// rocdl.kernel attribute, rocdl.workitem.id, and buffer resource ops.
//
// CHECK-LABEL: hal.executable public @full_pipeline
//       CHECK:   hal.executable.variant public @rocm
//  CHECK-SAME:     options(#rocm.pipeline_options<>)
//       CHECK:     llvm.func @entry
//  CHECK-SAME:       rocdl.kernel
//       CHECK:       rocdl.workitem.id.x
//       CHECK:       rocdl.make.buffer.rsrc
//       CHECK:       llvm.fadd %{{.*}}, %{{.*}} : vector<1xf32>
//       CHECK:       llvm.store %{{.*}}, %{{.*}} : vector<1xf32>, !llvm.ptr<7>
//       CHECK:       llvm.return
hal.executable @full_pipeline {
  hal.executable.variant public @rocm target(#executable_target_rocm_hsaco_fb)
      options(#rocm.pipeline_options<>) {
    hal.executable.export public @entry ordinal(0) layout(#pipeline_layout)
        count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
        %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
            to memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            alignment(64) offset(%c0)
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
        %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
            to memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
        %4 = iree_codegen.load_from_buffer %1
            : memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
            -> tensor<64xf32>
        %5 = tensor.empty() : tensor<64xf32>
        %6 = linalg.generic {
            indexing_maps = [#map, #map], iterator_types = ["parallel"]}
            ins(%4 : tensor<64xf32>) outs(%5 : tensor<64xf32>)
            attrs = {lowering_config =
              #iree_gpu.lowering_config<{thread = [1], workgroup = [64]}>} {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %in, %in : f32
          linalg.yield %7 : f32
        } -> tensor<64xf32>
        iree_codegen.store_to_buffer %6, %3
            : tensor<64xf32>
            into memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
        return
      }
    }
  }
}

// -----

// compile_to = configuration_controlled_translation: Phase 1 only.
// Should tile and distribute to threads but NOT lower to LLVM. The output
// should have gpu.thread_id, vector.transfer_read/write, and arith ops
// but no llvm dialect ops.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "gfx942",
    features = "",
    wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8,
      storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
      dot = dp4xi8toi32, subgroup_size_choices = [64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024,
      max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#map = affine_map<(d0) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

// CHECK-LABEL: hal.executable public @phase1_only
//       CHECK:   hal.executable.variant public @rocm
//  CHECK-SAME:     options(#rocm.pipeline_options<compile_to = configuration_controlled_translation>)
//       CHECK:     func.func @entry
//  CHECK-SAME:       gpu.known_block_size = array<i32: 64, 1, 1>
//       CHECK:       gpu.thread_id x
//       CHECK:       amdgpu.fat_raw_buffer_cast
//       CHECK:       vector.transfer_read
//       CHECK:       arith.addf
//       CHECK:       vector.transfer_write
//   CHECK-NOT:       llvm.func
//   CHECK-NOT:       rocdl.kernel
hal.executable @phase1_only {
  hal.executable.variant public @rocm target(#executable_target_rocm_hsaco_fb)
      options(#rocm.pipeline_options<compile_to = configuration_controlled_translation>) {
    hal.executable.export public @entry ordinal(0) layout(#pipeline_layout)
        count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
        %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
            to memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            alignment(64) offset(%c0)
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
        %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
            to memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
        %4 = iree_codegen.load_from_buffer %1
            : memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
            -> tensor<64xf32>
        %5 = tensor.empty() : tensor<64xf32>
        %6 = linalg.generic {
            indexing_maps = [#map, #map], iterator_types = ["parallel"]}
            ins(%4 : tensor<64xf32>) outs(%5 : tensor<64xf32>)
            attrs = {lowering_config =
              #iree_gpu.lowering_config<{thread = [1], workgroup = [64]}>} {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %in, %in : f32
          linalg.yield %7 : f32
        } -> tensor<64xf32>
        iree_codegen.store_to_buffer %6, %3
            : tensor<64xf32>
            into memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
        return
      }
    }
  }
}

// -----

// compile_from = llvm_translation: Phase 2 only, starting from Phase 1
// output (already tiled/distributed). Should lower to LLVM+ROCDL.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "gfx942",
    features = "",
    wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8,
      storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic,
      dot = dp4xi8toi32, subgroup_size_choices = [64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024,
      max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK-LABEL: hal.executable public @phase2_only
//       CHECK:   hal.executable.variant public @rocm
//  CHECK-SAME:     options(#rocm.pipeline_options<compile_from = llvm_translation>)
//       CHECK:     llvm.func @entry
//  CHECK-SAME:       rocdl.kernel
//       CHECK:       rocdl.workitem.id.x
//       CHECK:       rocdl.make.buffer.rsrc
//       CHECK:       llvm.fadd %{{.*}}, %{{.*}} : vector<1xf32>
//       CHECK:       llvm.store %{{.*}}, %{{.*}} : vector<1xf32>, !llvm.ptr<7>
//       CHECK:       llvm.return
hal.executable @phase2_only {
  hal.executable.variant public @rocm target(#executable_target_rocm_hsaco_fb)
      options(#rocm.pipeline_options<compile_from = llvm_translation>) {
    hal.executable.export public @entry ordinal(0) layout(#pipeline_layout)
        count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    } attributes {subgroup_size = 64 : index,
                  workgroup_size = [64 : index, 1 : index, 1 : index]}
    builtin.module {
      func.func @entry()
          attributes {gpu.known_block_size = array<i32: 64, 1, 1>} {
        %c0 = arith.constant 0 : index
        %0 = ub.poison : f32
        %thread_id_x = gpu.thread_id x upper_bound 64
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            {iree_gpu.use_rocdl_buffer_instructions}
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
        %assume_align = memref.assume_alignment %1, 64
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
        %2 = amdgpu.fat_raw_buffer_cast %assume_align resetOffset
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
            to memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            alignment(64) offset(%c0)
            {iree_gpu.use_rocdl_buffer_instructions}
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
        %assume_align_0 = memref.assume_alignment %3, 64
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
        %4 = amdgpu.fat_raw_buffer_cast %assume_align_0 resetOffset
            : memref<64xf32, #hal.descriptor_type<storage_buffer>>
            to memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
        gpu.barrier memfence [#gpu.address_space<workgroup>]
        %5 = vector.transfer_read %2[%thread_id_x], %0 {in_bounds = [true]}
            : memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>,
              vector<1xf32>
        %6 = arith.addf %5, %5 : vector<1xf32>
        vector.transfer_write %6, %4[%thread_id_x] {in_bounds = [true]}
            : vector<1xf32>,
              memref<64xf32, #amdgpu.address_space<fat_raw_buffer>>
        gpu.barrier memfence [#gpu.address_space<workgroup>]
        return
      }
    }
  }
}
