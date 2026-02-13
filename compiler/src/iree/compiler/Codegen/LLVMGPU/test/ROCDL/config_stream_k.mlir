// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-enable-stream-k=true \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// Tests that --iree-codegen-llvmgpu-enable-stream-k adds a streamed_reduction
// tiling level to the matmul lowering config.

func.func @matmul_stream_k(%lhs: tensor<128x256xf32>, %rhs: tensor<256x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x128xf32>)
      outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %mm : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @matmul_stream_k(
// The config should contain both reduction and streamed_reduction levels.
// CHECK: linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
// CHECK-SAME: streamed_reduction
