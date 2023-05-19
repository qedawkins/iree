// RUN: iree-opt %s --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%s | \
// RUN: FileCheck --check-prefix=CHECK %s

hal.executable @dispatch_0 {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
    hal.executable.export public @dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dispatch_0() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [64], strides = [1] : !flow.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %empty = tensor.empty() : tensor<64xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
          ins(%2 : tensor<64xf32>) outs(%empty : tensor<64xf32>) {
          ^bb0(%in: f32, %out: f32) :
            linalg.yield %in : f32
        } -> tensor<64xf32>
        flow.dispatch.tensor.store %3, %1, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !flow.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
}

transform.sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):
    /// Just bufferize, we simply want to verify that the memref descriptor types are erased by the pipeline.
    %variant_op_1 = transform.iree.bufferize { target_gpu } %variant_op : (!pdl.operation) -> (!pdl.operation)
}

// CHECK-NOT: hal.descriptor_type
