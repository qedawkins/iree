// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-distribute-shared-executor)" --split-input-file | FileCheck %s

// Basic: single scope, single readwrite init, empty body.
// shared_executor is replaced with pcf.generic preserving scope and adding
// thread ID args. The sref block arg and count arg are mapped through.

// CHECK-LABEL: func.func @basic_empty_body
// CHECK-SAME:    %[[INIT:.+]]: tensor<128x128xf32>
// CHECK:       %[[RESULT:.+]] = pcf.generic
// CHECK-SAME:    scope(#pcf.sequential)
// CHECK:         execute(%[[REF:.+]] = %[[INIT]])[%[[ID:.+]]: index, %[[COUNT:.+]]: index]
// CHECK:           : (!pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
// CHECK:           -> (tensor<128x128xf32>)
// CHECK:         pcf.return
// CHECK:       return %[[RESULT]] : tensor<128x128xf32>
func.func @basic_empty_body(%init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = pcf.shared_executor scope(#pcf.sequential)
      execute(%ref = %init)[%count: index]
          : (!pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}

// -----

// Two readwrite inits: both become tied generic refs, both results used.

// CHECK-LABEL: func.func @two_readwrite_inits
// CHECK-SAME:    %[[A:.+]]: tensor<64xf32>, %[[B:.+]]: tensor<32xf32>
// CHECK:       %[[RESULTS:.+]]:2 = pcf.generic
// CHECK-SAME:    scope(#pcf.sequential)
// CHECK:         execute(%{{.*}} = %[[A]], %{{.*}} = %[[B]])[%{{.*}}: index, %{{.*}}: index]
// CHECK:           : (!pcf.sref<64xf32, #pcf.sequential, readwrite>,
// CHECK-SAME:        !pcf.sref<32xf32, #pcf.sequential, readwrite>)
// CHECK:           -> (tensor<64xf32>, tensor<32xf32>)
// CHECK:       return %[[RESULTS]]#0, %[[RESULTS]]#1
func.func @two_readwrite_inits(%a: tensor<64xf32>, %b: tensor<32xf32>)
    -> (tensor<64xf32>, tensor<32xf32>) {
  %r:2 = pcf.shared_executor scope(#pcf.sequential)
      execute(%ref_a = %a, %ref_b = %b)[%count: index]
          : (!pcf.sref<64xf32, #pcf.sequential, readwrite>,
             !pcf.sref<32xf32, #pcf.sequential, readwrite>)
          -> (tensor<64xf32>, tensor<32xf32>) {
    pcf.return
  }
  return %r#0, %r#1 : tensor<64xf32>, tensor<32xf32>
}

// -----

// Captures become tied refs in the generic. The shared_executor only produces
// results for non-capture (readwrite) refs. The generic produces results for
// ALL refs (captures + inits), but only the non-capture results are used.

// CHECK-LABEL: func.func @with_captures
// CHECK-SAME:    %[[LHS:.+]]: tensor<128x64xf16>, %[[INIT:.+]]: tensor<128x128xf32>
// CHECK:       %[[RESULTS:.+]]:2 = pcf.generic
// CHECK-SAME:    scope(#pcf.sequential)
// Capture and init both become tied refs.
// CHECK:         execute(%{{.*}} = %[[LHS]], %{{.*}} = %[[INIT]])
// CHECK:       return %[[RESULTS]]#1 : tensor<128x128xf32>
func.func @with_captures(
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

// Multi-scope shared_executor: only the outermost scope is distributed.
// The inner scope count args are not mapped (Phase 6 handles them).
// Verifies pass doesn't crash on multi-scope inputs.

// CHECK-LABEL: func.func @multi_scope_basic
// CHECK-SAME:    %[[INIT:.+]]: tensor<128x128xf32>
// CHECK:       %[[RESULT:.+]] = pcf.generic
// CHECK-SAME:    scope(#pcf.sequential)
// CHECK:         execute(%{{.*}} = %[[INIT]])[%{{.*}}: index, %{{.*}}: index]
// CHECK:       return %[[RESULT]]
func.func @multi_scope_basic(%init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = pcf.shared_executor
      scopes(#pcf.sequential, #pcf.sequential)
      execute(%ref = %init)
          [%sg_count: index][%lane_count: index]
          : (!pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}
