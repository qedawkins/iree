// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-flow-insert-debug-target-at-ordinal{break-debug-target=@target_func:1 trace-debug-target=@target_func:1})" %s | FileCheck %s --check-prefixes=CHECK,ORDINAL
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-flow-insert-debug-target-at-symbol{break-debug-target=dispatch_1 trace-debug-target=dispatch_1})" %s | FileCheck %s --check-prefixes=CHECK,SYMBOL

/// Multiple functions

// CHECK-LABEL: func.func @target_func
func.func @target_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @dispatch_0::@dispatch_0_entry
  %0 = flow.dispatch @dispatch_0::@dispatch_0_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %1 = flow.dispatch @dispatch_1::@dispatch_1_entry
  %1 = flow.dispatch @dispatch_1::@dispatch_1_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = flow.dispatch @dispatch_2::@dispatch_2_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: %[[EXPORT:.+]] = hal.tensor.export %1 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: return %[[EXPORT]] : !hal.buffer_view
  return %3 : !hal.buffer_view
}

// CHECK-LABEL: func.func @other_func
func.func @other_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @dispatch_1::@dispatch_1_entry
  %0 = flow.dispatch @dispatch_1::@dispatch_1_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>

  // CHECK: %1 = flow.dispatch @dispatch_2::@dispatch_2_entry
  %1 = flow.dispatch @dispatch_2::@dispatch_2_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %2 = flow.dispatch @dispatch_3::@dispatch_3_entry
  %2 = flow.dispatch @dispatch_3::@dispatch_3_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>

  // ORDINAL: %[[ORIGINAL_EXPORT:.+]] = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view
  // SYMBOL:  %[[BREAK_EXPORT:.+]] = hal.tensor.export %0 : tensor<4xf32> -> !hal.buffer_view
  %3 = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view

  /// Only break on the symbol as the ordinal specifies a different function
  // SYMBOL:  return %[[BREAK_EXPORT]] : !hal.buffer_view
  // ORDINAL: return %[[ORIGINAL_EXPORT]] : !hal.buffer_view
  return %3 : !hal.buffer_view
}

// -----

// Break on a dispatch with a different number of results
// CHECK-LABEL: func.func @target_func
func.func @target_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @dispatch_0::@dispatch_0_entry
  %0 = flow.dispatch @dispatch_0::@dispatch_0_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %1:2 = flow.dispatch @dispatch_1::@dispatch_1_entry
  %1:2 = flow.dispatch @dispatch_1::@dispatch_1_entry[%c4] (%arg0) : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  %2 = flow.dispatch @dispatch_2::@dispatch_2_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: %[[EXPORT_0:.+]] = hal.tensor.export %1#0 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: %[[EXPORT_1:.+]] = hal.tensor.export %1#1 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: return %[[EXPORT_0]], %[[EXPORT_1]] : !hal.buffer_view
  return %3 : !hal.buffer_view
}

// -----

// Break/trace on a dispatch not found in the target function should do nothing
// CHECK-LABEL: func.func @target_func
func.func @target_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @dispatch_0::@dispatch_0_entry
  %0 = flow.dispatch @dispatch_0::@dispatch_0_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %1 = hal.tensor.export %0 : tensor<4xf32> -> !hal.buffer_view
  %1 = hal.tensor.export %0 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: return %1 : !hal.buffer_view
  return %1 : !hal.buffer_view
}

// -----

/// Combine tracing and breaking on the same dispatch
// CHECK-LABEL: func.func @target_func
func.func @target_func(%arg0: tensor<4xf32>) -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @dispatch_0::@dispatch_0_entry
  %0 = flow.dispatch @dispatch_0::@dispatch_0_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>

  // ORDINAL: flow.tensor.trace {key = "dispatch_1::dispatch_1_entry::1 inputs"} %arg0 : tensor<4xf32>
  // SYMBOL:  flow.tensor.trace {key = "dispatch_1::dispatch_1_entry inputs"} %arg0 : tensor<4xf32>
  // CHECK: %1 = flow.dispatch @dispatch_1::@dispatch_1_entry
  %1 = flow.dispatch @dispatch_1::@dispatch_1_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // ORDINAL: flow.tensor.trace {key = "dispatch_1::dispatch_1_entry::1 outputs"} %1 : tensor<4xf32>
  // SYMBOL:  flow.tensor.trace {key = "dispatch_1::dispatch_1_entry outputs"} %1 : tensor<4xf32>

  %2 = flow.dispatch @dispatch_2::@dispatch_2_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: %[[EXPORT:.+]] = hal.tensor.export %1 : tensor<4xf32> -> !hal.buffer_view
  // CHECK: return %[[EXPORT]] : !hal.buffer_view
  return %3 : !hal.buffer_view
}
