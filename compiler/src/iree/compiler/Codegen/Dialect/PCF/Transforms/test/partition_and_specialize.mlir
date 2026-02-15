// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-partition-and-specialize{partitioning-strategy=none})" --split-input-file | FileCheck %s

// With strategy=none the pass is a no-op. Verify pass registration and that
// the shared_executor is left untouched.

// CHECK-LABEL: func.func @strategy_none_basic
// CHECK-SAME:    %[[INIT:.+]]: tensor<128x128xf32>
// CHECK:       %[[RESULT:.+]] = pcf.shared_executor scope(#pcf.sequential)
// CHECK:         execute(%{{.*}} = %[[INIT]])[%{{.*}}: index]
// CHECK:           : (!pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
// CHECK:           -> (tensor<128x128xf32>)
// CHECK:         pcf.return
// CHECK:       return %[[RESULT]] : tensor<128x128xf32>
func.func @strategy_none_basic(%init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = pcf.shared_executor scope(#pcf.sequential)
      execute(%ref = %init)[%count: index]
          : (!pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}

// -----

// strategy=none with captures: also a no-op.

// CHECK-LABEL: func.func @strategy_none_with_captures
// CHECK:       pcf.shared_executor scope(#pcf.sequential)
// CHECK:         execute(%{{.*}} from %{{.*}}, %{{.*}} = %{{.*}})[%{{.*}}: index]
// CHECK:           : (!pcf.sref<128x64xf16, #pcf.sequential, readonly>,
// CHECK-SAME:        !pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
// CHECK:           -> (tensor<128x128xf32>)
func.func @strategy_none_with_captures(
    %lhs: tensor<128x64xf16>,
    %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = pcf.shared_executor scope(#pcf.sequential)
      execute(%lhs_ref from %lhs, %out_ref = %init)
          [%count: index]
          : (!pcf.sref<128x64xf16, #pcf.sequential, readonly>,
             !pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}

// -----

// strategy=none with empty body: no-op.

// CHECK-LABEL: func.func @strategy_none_empty_body
// CHECK:       pcf.shared_executor scope(#pcf.sequential)
// CHECK:         execute(%{{.*}} = %{{.*}})[%{{.*}}: index]
func.func @strategy_none_empty_body(%init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = pcf.shared_executor scope(#pcf.sequential)
      execute(%ref = %init)[%count: index]
          : (!pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}
