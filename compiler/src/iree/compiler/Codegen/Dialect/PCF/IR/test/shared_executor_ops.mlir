// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// Single scope, 1D counts.
func.func @shared_executor_basic(%init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = pcf.shared_executor scope(#pcf.sequential)
      execute(%ref = %init)[%count: index]
          : (!pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}
// CHECK-LABEL: @shared_executor_basic
// CHECK: pcf.shared_executor scope(#pcf.sequential)
// CHECK-NEXT: execute(%{{.*}} = %{{.*}})[%{{.*}}: index]

// -----

// Two scopes, 2D + 1D counts.
func.func @shared_executor_multi_scope(%init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = pcf.shared_executor
      scopes(#pcf.sequential, #pcf.sequential)
      execute(%ref = %init)
          [%sg_x: index, %sg_y: index][%lane_count: index]
          : (!pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}
// CHECK-LABEL: @shared_executor_multi_scope
// CHECK: pcf.shared_executor
// CHECK-SAME: scopes(#pcf.sequential, #pcf.sequential)
// CHECK: [%{{.*}}: index, %{{.*}}: index][%{{.*}}: index]

// -----

// Read-only captures using 'from' keyword.
func.func @shared_executor_captures(
    %lhs: tensor<128x64xf16>,
    %rhs: tensor<64x128xf16>,
    %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = pcf.shared_executor scope(#pcf.sequential)
      execute(%lhs_ref from %lhs, %rhs_ref from %rhs, %out_ref = %init)
          [%count: index]
          : (!pcf.sref<128x64xf16, #pcf.sequential, readonly>,
             !pcf.sref<64x128xf16, #pcf.sequential, readonly>,
             !pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}
// CHECK-LABEL: @shared_executor_captures
// CHECK: execute(%{{.*}} from %{{.*}}, %{{.*}} from %{{.*}}, %{{.*}} = %{{.*}})
// CHECK: !pcf.sref<128x64xf16, #pcf.sequential, readonly>
// CHECK: !pcf.sref<128x128xf32, #pcf.sequential, readwrite>

// -----

// Shared executor with initializer region.
func.func @shared_executor_with_init(%init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = pcf.shared_executor scope(#pcf.sequential)
      initialize {
        %alloc = pcf.alloc() : !pcf.sref<64xf32, #pcf.sequential>
        pcf.yield %alloc : !pcf.sref<64xf32, #pcf.sequential>
      } -> (%scratch: !pcf.sref<64xf32, #pcf.sequential>)
      execute(%ref = %init)[%count: index]
          : (!pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}
// CHECK-LABEL: @shared_executor_with_init
// CHECK: pcf.shared_executor scope(#pcf.sequential)
// CHECK-NEXT: initialize
// CHECK: pcf.alloc
// CHECK: pcf.yield
// CHECK: -> (%{{.*}}: !pcf.sref<64xf32, #pcf.sequential>)
// CHECK: execute(%{{.*}} = %{{.*}})[%{{.*}}: index]
