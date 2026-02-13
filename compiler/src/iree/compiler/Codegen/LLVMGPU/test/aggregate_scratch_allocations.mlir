// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module( \
// RUN:     iree-llvmgpu-aggregate-scratch-allocations))))" %s | FileCheck %s

// ============================================================================
// Test 1: Two alloc_scratch ops (256x64xf16 tile + scalar i32 counter).
//
// scratch: 256*64*2 = 32768 bytes at offset 0.
// counter: 4 bytes at offset alignTo(32768, 16) = 32768.
// total: alignTo(32768 + 4, 16) = 32784.
// ============================================================================

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable public @two_alloc_scratch {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @main ordinal(0) layout(#pipeline_layout)
    count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    scratch(%device: !hal.device) -> index {
      %c0 = arith.constant 0 : index
      hal.return %c0 : index
    }
    builtin.module {
      func.func @main() {
        %c0 = arith.constant 0 : index
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<128x128xf16>
        %tile_scratch = iree_codegen.alloc_scratch() : memref<256x64xf16>
        %counter = iree_codegen.alloc_scratch() : memref<i32>
        %val = memref.load %tile_scratch[%c0, %c0] : memref<256x64xf16>
        memref.store %val, %out[%c0, %c0] : memref<128x128xf16>
        return
      }
    }
  }
}

// scratch_size region updated with total = 32784.
// CHECK-LABEL: hal.executable public @two_alloc_scratch
//       CHECK:   scratch(%{{.+}}: !hal.device) -> index
//       CHECK:     %[[TOTAL:.+]] = arith.constant 32784 : index
//       CHECK:     hal.return %[[TOTAL]]

// Function body: scratch binding + two memref.view ops.
//       CHECK:   func.func @main
//       CHECK:     %[[BUF:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(3) alignment(64) offset({{.+}}) : memref<32784xi8>
//       CHECK:     memref.view %[[BUF]][{{.+}}][] : memref<32784xi8> to memref<256x64xf16>
//       CHECK:     %[[OFF1:.+]] = arith.constant 32768 : index
//       CHECK:     memref.view %[[BUF]][%[[OFF1]]][] : memref<32784xi8> to memref<i32>
//   CHECK-NOT:     iree_codegen.alloc_scratch

// -----

// ============================================================================
// Test 2: No alloc_scratch ops — pass is a no-op.
// ============================================================================

#pipeline_layout2 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable public @no_alloc_scratch {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @main ordinal(0) layout(#pipeline_layout2)
    count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @main() {
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @no_alloc_scratch
//   CHECK-NOT:   iree_codegen.alloc_scratch
//   CHECK-NOT:   memref.view

// -----

// ============================================================================
// Test 3: Single alloc_scratch (tile only, no counter).
//
// scratch: 64*64*4 = 16384 bytes at offset 0.
// total: alignTo(16384, 16) = 16384.
// ============================================================================

#pipeline_layout3 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable public @single_alloc_scratch {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @main ordinal(0) layout(#pipeline_layout3)
    count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    scratch(%device: !hal.device) -> index {
      %c0 = arith.constant 0 : index
      hal.return %c0 : index
    }
    builtin.module {
      func.func @main() {
        %c0 = arith.constant 0 : index
        %out = hal.interface.binding.subspan layout(#pipeline_layout3) binding(0) alignment(64) offset(%c0) : memref<64x64xf32>
        %scratch = iree_codegen.alloc_scratch() : memref<64x64xf32>
        %val = memref.load %scratch[%c0, %c0] : memref<64x64xf32>
        memref.store %val, %out[%c0, %c0] : memref<64x64xf32>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @single_alloc_scratch
//       CHECK:   scratch(%{{.+}}: !hal.device) -> index
//       CHECK:     %[[T:.+]] = arith.constant 16384 : index
//       CHECK:     hal.return %[[T]]
//       CHECK:   func.func @main
//       CHECK:     hal.interface.binding.subspan layout({{.+}}) binding(3) {{.*}} : memref<16384xi8>
//       CHECK:     memref.view {{.*}} : memref<16384xi8> to memref<64x64xf32>
//   CHECK-NOT:     iree_codegen.alloc_scratch

// -----

// ============================================================================
// Test 4: Alignment padding (first alloc is NOT 16-byte aligned).
//
// alloc 1: memref<3xf32> = 3 * 4 = 12 bytes at offset 0.
// alloc 2: memref<i32> = 4 bytes at offset alignTo(12, 16) = 16.
// total: alignTo(16 + 4, 16) = 32.
// ============================================================================

#pipeline_layout4 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable public @alignment_padding {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @main ordinal(0) layout(#pipeline_layout4)
    count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    scratch(%device: !hal.device) -> index {
      %c0 = arith.constant 0 : index
      hal.return %c0 : index
    }
    builtin.module {
      func.func @main() {
        %c0 = arith.constant 0 : index
        %buf = hal.interface.binding.subspan layout(#pipeline_layout4) binding(0) alignment(64) offset(%c0) : memref<16xf32>
        %s1 = iree_codegen.alloc_scratch() : memref<3xf32>
        %s2 = iree_codegen.alloc_scratch() : memref<i32>
        %val = memref.load %s1[%c0] : memref<3xf32>
        memref.store %val, %buf[%c0] : memref<16xf32>
        return
      }
    }
  }
}

// scratch_size region updated with total = 32 (12 bytes + padding to 16 + 4 bytes, aligned to 16).
// CHECK-LABEL: hal.executable public @alignment_padding
//       CHECK:   scratch(%{{.+}}: !hal.device) -> index
//       CHECK:     %[[T:.+]] = arith.constant 32 : index
//       CHECK:     hal.return %[[T]]
//       CHECK:   func.func @main
//       CHECK:     %[[BUF:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) {{.*}} : memref<32xi8>
//       CHECK:     memref.view %[[BUF]][{{.*}}][] : memref<32xi8> to memref<3xf32>
//       CHECK:     %[[OFF:.+]] = arith.constant 16 : index
//       CHECK:     memref.view %[[BUF]][%[[OFF]]][] : memref<32xi8> to memref<i32>
//   CHECK-NOT:     iree_codegen.alloc_scratch
