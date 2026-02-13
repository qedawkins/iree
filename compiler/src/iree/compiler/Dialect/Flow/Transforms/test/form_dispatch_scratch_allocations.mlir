// RUN: iree-opt --split-input-file --iree-flow-form-dispatch-scratch-allocations %s | FileCheck %s

// Tests that a matmul-like dispatch gets a scratch_size region and scratch
// buffer argument added. Verifies:
// 1. scratch_size region is added to the export with placeholder computation
// 2. flow.executable.scratch_size op is created before the dispatch
// 3. flow.tensor.empty creates the scratch tensor with dynamic size
// 4. Scratch binding is inserted between input and output args in the function

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

//       CHECK:   flow.executable.export public @matmul_dispatch
//  CHECK-SAME:     workgroups
//       CHECK:     scratch_size(%{{.+}}: index, %{{.+}}: index) -> index
//       CHECK:       %[[C4096:.+]] = arith.constant 4096 : index
//       CHECK:       flow.return %[[C4096]]

// Verify scratch binding inserted between inputs (readonly) and output (writeonly).
//       CHECK:   func.func @matmul_dispatch(
//  CHECK-SAME:     %{{.+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>>,
//  CHECK-SAME:     %{{.+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>>,
//  CHECK-SAME:     %{{.+}}: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi8>>,
//  CHECK-SAME:     %{{.+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x256xf32>>)

// CHECK-LABEL: util.func public @main
util.func public @main(
    %arg0: tensor<64x128xf32>,
    %arg1: tensor<128x256xf32>) -> tensor<64x256xf32>
{
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: %[[SIZE:.+]] = flow.executable.scratch_size @matmul_dispatch::@matmul_dispatch
  // CHECK: %[[SCRATCH:.+]] = flow.tensor.empty : tensor<?xi8>{%[[SIZE]]}
  // CHECK: flow.dispatch @matmul_dispatch::@matmul_dispatch
  // CHECK-SAME: (%{{.+}}, %{{.+}}, %[[SCRATCH]])
  %0 = flow.dispatch @matmul_dispatch::@matmul_dispatch[%c1, %c2](%arg0, %arg1) :
      (tensor<64x128xf32>, tensor<128x256xf32>) -> tensor<64x256xf32>
  util.return %0 : tensor<64x256xf32>
}

// -----

// Tests that a non-matmul dispatch is NOT modified (no-op case).

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
// CHECK-LABEL: util.func public @elementwise_main
util.func public @elementwise_main(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  // CHECK-NOT: flow.executable.scratch_size
  // CHECK-NOT: flow.tensor.empty
  // CHECK: flow.dispatch @elementwise_dispatch
  // CHECK-NOT: tensor<?xi8>
  %0 = flow.dispatch @elementwise_dispatch::@elementwise_dispatch(%arg0) :
      (tensor<64xf32>) -> tensor<64xf32>
  util.return %0 : tensor<64xf32>
}

// -----

// Tests that a dispatch with an existing scratch_size region is NOT modified.
// The pass should skip executables that already have scratch_size populated.

flow.executable private @already_has_scratch {
  flow.executable.export public @already_has_scratch workgroups(%arg0: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    flow.return %c1, %c1, %c1 : index, index, index
  } scratch_size(%arg0: index) -> index {
    %c8192 = arith.constant 8192 : index
    flow.return %c8192 : index
  }
  builtin.module {
    func.func @already_has_scratch(
        %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xf32>>,
        %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x32xf32>>,
        %arg2: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi8>>,
        %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x32xf32>>) {
      %cst = arith.constant 0.0 : f32
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0,
          offsets = [0, 0], sizes = [32, 64], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xf32>>
          -> tensor<32x64xf32>
      %1 = iree_tensor_ext.dispatch.tensor.load %arg1,
          offsets = [0, 0], sizes = [64, 32], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x32xf32>>
          -> tensor<64x32xf32>
      %2 = tensor.empty() : tensor<32x32xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %4 = linalg.matmul ins(%0, %1 : tensor<32x64xf32>, tensor<64x32xf32>)
          outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
      iree_tensor_ext.dispatch.tensor.store %4, %arg3,
          offsets = [0, 0], sizes = [32, 32], strides = [1, 1]
          : tensor<32x32xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x32xf32>>
      return
    }
  }
}

// CHECK-LABEL: flow.executable private @already_has_scratch
//       CHECK:   scratch_size(%{{.+}}: index) -> index
//       CHECK:     %[[C8192:.+]] = arith.constant 8192 : index
//       CHECK:     flow.return %[[C8192]]
// Verify the function signature is unchanged (scratch already present).
//       CHECK:   func.func @already_has_scratch(
//  CHECK-SAME:     !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xf32>>,
//  CHECK-SAME:     !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x32xf32>>,
//  CHECK-SAME:     !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xi8>>,
//  CHECK-SAME:     !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x32xf32>>)
// CHECK-LABEL: util.func public @already_has_scratch_main
util.func public @already_has_scratch_main(
    %arg0: tensor<32x64xf32>,
    %arg1: tensor<64x32xf32>,
    %scratch: tensor<?xi8>,
    %scratch_size: index) -> tensor<32x32xf32>
{
  %c1 = arith.constant 1 : index
  // The dispatch already has scratch plumbed; the pass should not add another.
  // CHECK-NOT: flow.executable.scratch_size @already_has_scratch
  // CHECK: flow.dispatch @already_has_scratch::@already_has_scratch
  %0 = flow.dispatch @already_has_scratch::@already_has_scratch[%c1](%arg0, %arg1, %scratch) :
      (tensor<32x64xf32>, tensor<64x32xf32>, tensor<?xi8>{%scratch_size}) -> tensor<32x32xf32>
  util.return %0 : tensor<32x32xf32>
}
