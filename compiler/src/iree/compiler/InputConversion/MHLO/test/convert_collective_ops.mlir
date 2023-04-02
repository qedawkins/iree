// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-on-tensors --canonicalize -cse %s | FileCheck %s

// CHECK-LABEL: @replica_id
func.func @replica_id() -> tensor<ui32> {
  // CHECK-DAG: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK-DAG: [[RANK:%.+]] = flow.channel.rank [[CHANNEL]] : index
  // CHECK-DAG: [[CAST:%.+]] = arith.index_castui [[RANK]] : index to i32
  // CHECK-DAG: [[TENSOR:%.+]] = tensor.from_elements [[CAST]] : tensor<i32>
  // CHECK-DAG: return [[TENSOR]] : tensor<i32>
  %id = mhlo.replica_id : tensor<ui32>
  return %id : tensor<ui32>
}

// -----

// CHECK-LABEL: @all_reduce_sum
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2304xf32>)
func.func @all_reduce_sum(%input : tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<2304xf32>
  // CHECK: [[ALLREDUCE:%.+]] = flow.collective.all_reduce sum, f32, [[EMPTY]], [[ARG0]], %channel_default  : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> [[EMPTY]] as tensor<2304xf32>
  // CHECK: return [[ALLREDUCE]] : tensor<2304xf32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %sum = mhlo.add %arg0, %arg1 : tensor<f32>
      mhlo.return %sum : tensor<f32>
    }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
        replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
        use_global_device_ids} : (tensor<2304xf32>) -> tensor<2304xf32>
  return %out : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_reduce_product
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2304xf32>)
func.func @all_reduce_product(%input : tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<2304xf32>
  // CHECK: [[OP:%.+]] = flow.collective.all_reduce product, f32, [[EMPTY]], [[ARG0]], %channel_default  : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> [[EMPTY]] as tensor<2304xf32>
  // CHECK: return [[OP]] : tensor<2304xf32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %mul = mhlo.multiply %arg0, %arg1 : tensor<f32>
      mhlo.return %mul : tensor<f32>
    }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
        replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
        use_global_device_ids} : (tensor<2304xf32>) -> tensor<2304xf32>
  return %out : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_reduce_minimum
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2304xf32>)
func.func @all_reduce_minimum(%input : tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<2304xf32>
  // CHECK: [[OP:%.+]] = flow.collective.all_reduce minimum, f32, [[EMPTY]], [[ARG0]], %channel_default  : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> [[EMPTY]] as tensor<2304xf32>
  // CHECK: return [[OP]] : tensor<2304xf32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %mul = mhlo.minimum %arg0, %arg1 : tensor<f32>
      mhlo.return %mul : tensor<f32>
    }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
        replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
        use_global_device_ids} : (tensor<2304xf32>) -> tensor<2304xf32>
  return %out : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_reduce_maximum
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2304xf32>)
func.func @all_reduce_maximum(%input : tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<2304xf32>
  // CHECK: [[OP:%.+]] = flow.collective.all_reduce maximum, f32, [[EMPTY]], [[ARG0]], %channel_default  : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> [[EMPTY]] as tensor<2304xf32>
  // CHECK: return [[OP]] : tensor<2304xf32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %mul = mhlo.maximum %arg0, %arg1 : tensor<f32>
      mhlo.return %mul : tensor<f32>
    }) {channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
        replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
        use_global_device_ids} : (tensor<2304xf32>) -> tensor<2304xf32>
  return %out : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_reduce_maximum_optional_attrs
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2304xf32>)
func.func @all_reduce_maximum_optional_attrs(%input : tensor<2304xf32>) -> tensor<2304xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<2304xf32>
  // CHECK: [[OP:%.+]] = flow.collective.all_reduce maximum, f32, [[EMPTY]], [[ARG0]], %channel_default  : (tensor<2304xf32>, tensor<2304xf32>, !flow.channel) -> [[EMPTY]] as tensor<2304xf32>
  // CHECK: return [[OP]] : tensor<2304xf32>
  %out = "mhlo.all_reduce"(%input) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %mul = mhlo.maximum %arg0, %arg1 : tensor<f32>
      mhlo.return %mul : tensor<f32>
    }) {replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>} : (tensor<2304xf32>) -> tensor<2304xf32>
  return %out : tensor<2304xf32>
}

// -----

// CHECK-LABEL: @all_gather_dim_0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<512xf32>) -> tensor<1024xf32>
func.func @all_gather_dim_0(%input : tensor<512xf32>) -> tensor<1024xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1024xf32>
  // CHECK: [[OP:%.+]] = flow.collective.all_gather f32, [[EMPTY]], [[ARG0]], %channel_default  : (tensor<1024xf32>, tensor<512xf32>, !flow.channel) -> [[EMPTY]] as tensor<1024xf32>
  // CHECK: return [[OP]] : tensor<1024xf32>
  %out = "mhlo.all_gather"(%input) {all_gather_dim = 0 : i64,
     channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
     use_global_device_ids} : (tensor<512xf32>) -> tensor<1024xf32>
  return %out : tensor<1024xf32>
}

// -----

// CHECK-LABEL: @all_gather_dim_1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x2xf32>) -> tensor<2x4xf32>
func.func @all_gather_dim_1(%input : tensor<2x2xf32>) -> tensor<2x4xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: tensor.empty() : tensor<2x2xf32>
  // CHECK: [[TRANSPOSE_ARG:%.+]] = linalg.generic
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<4x2xf32>
  // CHECK: [[OP:%.+]] = flow.collective.all_gather f32, [[EMPTY]], [[TRANSPOSE_ARG]], %channel_default  : (tensor<4x2xf32>, tensor<2x2xf32>, !flow.channel) -> [[EMPTY]] as tensor<4x2xf32>
  // CHECK: tensor.empty() : tensor<2x4xf32>
  // CHECK: [[TRANSPOSE_OUT:%.+]] = linalg.generic
  // CHECK: return [[TRANSPOSE_OUT]] : tensor<2x4xf32>
  %out = "mhlo.all_gather"(%input) {all_gather_dim = 1 : i64,
     channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
     use_global_device_ids} : (tensor<2x2xf32>) -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @all_gather_dim_0_optional_attrs
// CHECK-SAME: ([[ARG0:%.+]]: tensor<512xf32>) -> tensor<1024xf32>
func.func @all_gather_dim_0_optional_attrs(%input : tensor<512xf32>) -> tensor<1024xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1024xf32>
  // CHECK: [[OP:%.+]] = flow.collective.all_gather f32, [[EMPTY]], [[ARG0]], %channel_default  : (tensor<1024xf32>, tensor<512xf32>, !flow.channel) -> [[EMPTY]] as tensor<1024xf32>
  // CHECK: return [[OP]] : tensor<1024xf32>
  %out = "mhlo.all_gather"(%input) {all_gather_dim = 0 : i64,
     replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<512xf32>) -> tensor<1024xf32>
  return %out : tensor<1024xf32>
}

// -----

// CHECK-LABEL: @reduce_scatter_dim_0
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4x2xf32>) -> tensor<2x2xf32>
func.func @reduce_scatter_dim_0(%input : tensor<4x2xf32>) -> tensor<2x2xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: [[OP:%.+]] = flow.collective.reduce_scatter sum, f32, [[EMPTY]], [[ARG0]], %channel_default  : (tensor<2x2xf32>, tensor<4x2xf32>, !flow.channel) -> [[EMPTY]] as tensor<2x2xf32>
  // CHECK: return [[OP]] : tensor<2x2xf32>
  %out = "mhlo.reduce_scatter"(%input) ({
  ^bb0(%arg0: tensor<f32> , %arg1: tensor<f32>) :
    %sum = mhlo.add %arg0, %arg1 : tensor<f32>
    mhlo.return %sum : tensor<f32>
  }) {scatter_dimension = 0 : i64,
      channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids} : (tensor<4x2xf32>) -> tensor<2x2xf32>
  return %out : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @reduce_scatter_dim_1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x4xf32>) -> tensor<2x2xf32>
func.func @reduce_scatter_dim_1(%input : tensor<2x4xf32>) -> tensor<2x2xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: tensor.empty() : tensor<4x2xf32>
  // CHECK: [[TRANSPOSE_ARG:%.+]] = linalg.generic
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: [[OP:%.+]] = flow.collective.reduce_scatter sum, f32, [[EMPTY]], [[TRANSPOSE_ARG]], %channel_default  : (tensor<2x2xf32>, tensor<4x2xf32>, !flow.channel) -> [[EMPTY]] as tensor<2x2xf32>
  // CHECK: [[TRANSPOSE_OUT:%.+]] = linalg.generic
  // CHECK: return [[TRANSPOSE_OUT]] : tensor<2x2xf32>
  %out = "mhlo.reduce_scatter"(%input) ({
  ^bb0(%arg0: tensor<f32> , %arg1: tensor<f32>) :
    %sum = mhlo.add %arg0, %arg1 : tensor<f32>
    mhlo.return %sum : tensor<f32>
  }) {scatter_dimension = 1 : i64,
      channel_handle = #mhlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      use_global_device_ids} : (tensor<2x4xf32>) -> tensor<2x2xf32>
  return %out : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @reduce_scatter_dim_0_optional_attrs
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4x2xf32>) -> tensor<2x2xf32>
func.func @reduce_scatter_dim_0_optional_attrs(%input : tensor<4x2xf32>) -> tensor<2x2xf32> {
  // CHECK: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: [[OP:%.+]] = flow.collective.reduce_scatter sum, f32, [[EMPTY]], [[ARG0]], %channel_default  : (tensor<2x2xf32>, tensor<4x2xf32>, !flow.channel) -> [[EMPTY]] as tensor<2x2xf32>
  // CHECK: return [[OP]] : tensor<2x2xf32>
  %out = "mhlo.reduce_scatter"(%input) ({
  ^bb0(%arg0: tensor<f32> , %arg1: tensor<f32>) :
    %sum = mhlo.add %arg0, %arg1 : tensor<f32>
    mhlo.return %sum : tensor<f32>
  }) {scatter_dimension = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<4x2xf32>) -> tensor<2x2xf32>
  return %out : tensor<2x2xf32>
}
