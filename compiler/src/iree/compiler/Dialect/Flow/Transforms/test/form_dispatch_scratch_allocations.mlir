// RUN: iree-opt --split-input-file --iree-flow-form-dispatch-scratch-allocations %s | FileCheck %s

// Tests that a matmul-like dispatch gets a scratch_size region and scratch
// buffer argument added.

flow.executable private @matmul_dispatch {
  flow.executable.export public @matmul_dispatch workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    flow.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    func.func @matmul_dispatch(
        %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>>,
        %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>>,
        %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x256xf32>>) {
      %cst = arith.constant 0.0 : f32
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0,
          offsets = [0, 0], sizes = [64, 128], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>>
          -> tensor<64x128xf32>
      %1 = iree_tensor_ext.dispatch.tensor.load %arg1,
          offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>>
          -> tensor<128x256xf32>
      %2 = tensor.empty() : tensor<64x256xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64x256xf32>) -> tensor<64x256xf32>
      %4 = linalg.matmul ins(%0, %1 : tensor<64x128xf32>, tensor<128x256xf32>)
          outs(%3 : tensor<64x256xf32>) -> tensor<64x256xf32>
      iree_tensor_ext.dispatch.tensor.store %4, %arg2,
          offsets = [0, 0], sizes = [64, 256], strides = [1, 1]
          : tensor<64x256xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x256xf32>>
      return
    }
  }
}

// CHECK-LABEL: flow.executable private @matmul_dispatch
// CHECK:         flow.executable.export public @matmul_dispatch
// CHECK-SAME:      workgroups
// CHECK:           scratch_size(%{{.+}}: index, %{{.+}}: index) -> index
util.func public @main(
    %arg0: tensor<64x128xf32>,
    %arg1: tensor<128x256xf32>) -> tensor<64x256xf32>
{
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: %[[SIZE:.+]] = flow.executable.scratch_size @matmul_dispatch::@matmul_dispatch
  // CHECK: %[[SCRATCH:.+]] = tensor.empty(%[[SIZE]]) : tensor<?xi8>
  // CHECK: flow.dispatch @matmul_dispatch::@matmul_dispatch
  // CHECK-SAME: %[[SCRATCH]]
  %0 = flow.dispatch @matmul_dispatch::@matmul_dispatch[%c1, %c2](%arg0, %arg1) :
      (tensor<64x128xf32>, tensor<128x256xf32>) -> tensor<64x256xf32>
  util.return %0 : tensor<64x256xf32>
}

// -----

// Tests that a non-matmul dispatch is NOT modified.

flow.executable private @elementwise_dispatch {
  flow.executable.export public @elementwise_dispatch
  builtin.module {
    func.func @elementwise_dispatch(
        %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>,
        %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0,
          offsets = [0], sizes = [64], strides = [1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
          -> tensor<64xf32>
      %1 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}
          ins(%0 : tensor<64xf32>)
          outs(%0 : tensor<64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %2 = arith.addf %in, %in : f32
          linalg.yield %2 : f32
      } -> tensor<64xf32>
      iree_tensor_ext.dispatch.tensor.store %1, %arg1,
          offsets = [0], sizes = [64], strides = [1]
          : tensor<64xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
      return
    }
  }
}

// CHECK-LABEL: flow.executable private @elementwise_dispatch
// CHECK:         flow.executable.export public @elementwise_dispatch
// CHECK-NOT:     scratch_size
util.func public @elementwise_main(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  // CHECK-NOT: flow.executable.scratch_size
  // CHECK: flow.dispatch @elementwise_dispatch
  // CHECK-NOT: tensor<?xi8>
  %0 = flow.dispatch @elementwise_dispatch::@elementwise_dispatch(%arg0) :
      (tensor<64xf32>) -> tensor<64xf32>
  util.return %0 : tensor<64xf32>
}
