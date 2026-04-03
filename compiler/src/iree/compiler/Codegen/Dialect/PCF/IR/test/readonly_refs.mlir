// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// Generic with readonly and readwrite operands.
util.func private @generic_readonly_and_readwrite(
    %input: tensor<128x256xf16>, %output: tensor<128x128xf32>) {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%in_ref <- %input, %out_ref = %output)[%id: index, %count: index]
         : (!pcf.sref<128x256xf16, #pcf.test_scope>,
            !pcf.sref<128x128xf32, #pcf.test_scope>)
        -> (tensor<128x128xf32>) {
    util.optimization_barrier %in_ref, %out_ref : !pcf.sref<128x256xf16, #pcf.test_scope>, !pcf.sref<128x128xf32, #pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %0 : tensor<128x128xf32>
  util.return
}

// CHECK-LABEL: @generic_readonly_and_readwrite
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<128x256xf16>
//  CHECK-SAME:   %[[OUTPUT:[A-Za-z0-9]+]]: tensor<128x128xf32>
//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute(%[[IN_REF:[A-Za-z0-9_]+]] <- %[[INPUT]],
//  CHECK-SAME:             %[[OUT_REF:[A-Za-z0-9_]+]] = %[[OUTPUT]])
//  CHECK-SAME:             [%[[ID:.+]]: index, %[[COUNT:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<128x256xf16, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<128x128xf32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (tensor<128x128xf32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[IN_REF]], %[[OUT_REF]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }

// -----

// Generic with only readonly operands (no results).
util.func private @generic_readonly_only(%input: tensor<4x8xf32>) {
  pcf.generic scope(#pcf.test_scope)
    execute(%ref <- %input)[%id: index, %count: index]
         : (!pcf.sref<4x8xf32, #pcf.test_scope>) {
    util.optimization_barrier %ref : !pcf.sref<4x8xf32, #pcf.test_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic_readonly_only
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<4x8xf32>
//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] <- %[[INPUT]])
//  CHECK-SAME:             [%[[ID:.+]]: index, %[[COUNT:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<4x8xf32, #pcf.test_scope>) {
//  CHECK-NEXT:       util.optimization_barrier %[[REF]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }

// -----

// Generic with multiple readonly and multiple readwrite operands.
util.func private @generic_multi_readonly(
    %a: tensor<4xf32>, %b: tensor<8xf16>,
    %c: tensor<4xf32>, %d: tensor<8xf32>) {
  %0:2 = pcf.generic scope(#pcf.test_scope)
    execute(%ra <- %a, %rb <- %b, %rc = %c, %rd = %d)
           [%id: index, %count: index]
         : (!pcf.sref<4xf32, #pcf.test_scope>,
            !pcf.sref<8xf16, #pcf.test_scope>,
            !pcf.sref<4xf32, #pcf.test_scope>,
            !pcf.sref<8xf32, #pcf.test_scope>)
        -> (tensor<4xf32>, tensor<8xf32>) {
    pcf.return
  }
  util.optimization_barrier %0#0, %0#1 : tensor<4xf32>, tensor<8xf32>
  util.return
}

// CHECK-LABEL: @generic_multi_readonly
//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute(%{{.*}} <- %{{.*}}, %{{.*}} <- %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}})
//  CHECK-SAME:             [%{{.*}}: index, %{{.*}}: index]
//  CHECK-NEXT:          : (!pcf.sref<4xf32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<8xf16, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<4xf32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<8xf32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (tensor<4xf32>, tensor<8xf32>) {

// -----

// Generic with initializer and readonly refs.
util.func private @generic_readonly_with_init(
    %input: tensor<128x256xf16>, %output: tensor<128x128xf32>) {
  %0 = pcf.generic scope(#pcf.test_scope) initialize {
      %c42 = arith.constant 42 : index
      pcf.yield %c42 : index
    } -> (%scratch: index)
    execute(%in_ref <- %input, %out_ref = %output)
           [%id: index, %count: index]
         : (!pcf.sref<128x256xf16, #pcf.test_scope>,
            !pcf.sref<128x128xf32, #pcf.test_scope>)
        -> (tensor<128x128xf32>) {
    util.optimization_barrier %scratch : index
    pcf.return
  }
  util.optimization_barrier %0 : tensor<128x128xf32>
  util.return
}

// CHECK-LABEL: @generic_readonly_with_init
//       CHECK:   pcf.generic scope(#pcf.test_scope) initialize {
//  CHECK-NEXT:       %{{.*}} = arith.constant 42
//  CHECK-NEXT:       pcf.yield
//  CHECK-NEXT:     } -> (%{{.*}}: index)
//  CHECK-NEXT:     execute(%{{.*}} <- %{{.*}}, %{{.*}} = %{{.*}})
//  CHECK-SAME:             [%{{.*}}: index, %{{.*}}: index]

// -----

// Loop with readonly and readwrite operands.
util.func private @loop_readonly_and_readwrite(
    %input: tensor<128x256xf16>, %output: tensor<128x128xf32>,
    %n: index) {
  %0 = pcf.loop scope(#pcf.test_scope) count(%n)
    execute(%in_ref <- %input, %out_ref = %output)[%id: index]
         : (!pcf.sref<128x256xf16, #pcf.test_scope>,
            !pcf.sref<128x128xf32, #pcf.test_scope>)
        -> (tensor<128x128xf32>) {
    util.optimization_barrier %in_ref, %out_ref : !pcf.sref<128x256xf16, #pcf.test_scope>, !pcf.sref<128x128xf32, #pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %0 : tensor<128x128xf32>
  util.return
}

// CHECK-LABEL: @loop_readonly_and_readwrite
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<128x256xf16>
//  CHECK-SAME:   %[[OUTPUT:[A-Za-z0-9]+]]: tensor<128x128xf32>
//  CHECK-SAME:   %[[N:[A-Za-z0-9]+]]: index
//       CHECK:   pcf.loop scope(#pcf.test_scope) count(%[[N]])
//  CHECK-NEXT:     execute(%[[IN_REF:[A-Za-z0-9_]+]] <- %[[INPUT]],
//  CHECK-SAME:             %[[OUT_REF:[A-Za-z0-9_]+]] = %[[OUTPUT]])
//  CHECK-SAME:             [%[[ID:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<128x256xf16, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<128x128xf32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (tensor<128x128xf32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[IN_REF]], %[[OUT_REF]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }

// -----

// Loop with only readonly operands (no results).
util.func private @loop_readonly_only(%input: tensor<4x8xf32>, %n: index) {
  pcf.loop scope(#pcf.test_scope) count(%n)
    execute(%ref <- %input)[%id: index]
         : (!pcf.sref<4x8xf32, #pcf.test_scope>) {
    util.optimization_barrier %ref : !pcf.sref<4x8xf32, #pcf.test_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @loop_readonly_only
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<4x8xf32>
//  CHECK-SAME:   %[[N:[A-Za-z0-9]+]]: index
//       CHECK:   pcf.loop scope(#pcf.test_scope) count(%[[N]])
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] <- %[[INPUT]])
//  CHECK-SAME:             [%[[ID:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<4x8xf32, #pcf.test_scope>) {
//  CHECK-NEXT:       util.optimization_barrier %[[REF]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }

// -----

// Loop with multiple readonly refs and multi-dimensional iteration.
util.func private @loop_multi_readonly_multi_dim(
    %a: tensor<4xf32>, %b: tensor<8xf16>,
    %output: tensor<4xf32>,
    %m: index, %n: index) {
  %0 = pcf.loop scope(#pcf.test_scope) count(%m, %n)
    execute(%ra <- %a, %rb <- %b, %ref = %output)[%i: index, %j: index]
         : (!pcf.sref<4xf32, #pcf.test_scope>,
            !pcf.sref<8xf16, #pcf.test_scope>,
            !pcf.sref<4xf32, #pcf.test_scope>)
        -> (tensor<4xf32>) {
    pcf.return
  }
  util.optimization_barrier %0 : tensor<4xf32>
  util.return
}

// CHECK-LABEL: @loop_multi_readonly_multi_dim
//       CHECK:   pcf.loop scope(#pcf.test_scope) count(%{{.*}}, %{{.*}})
//  CHECK-NEXT:     execute(%{{.*}} <- %{{.*}}, %{{.*}} <- %{{.*}}, %{{.*}} = %{{.*}})
//  CHECK-SAME:             [%{{.*}}: index, %{{.*}}: index]
//  CHECK-NEXT:          : (!pcf.sref<4xf32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<8xf16, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<4xf32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (tensor<4xf32>) {

// -----

// Dynamic-shaped tensors: readonly and readwrite operands with unknown dims.
// Verifies that sref types correctly reflect dynamic shapes.
util.func private @generic_readonly_dynamic(
    %input: tensor<?x?xf32>, %output: tensor<?x?xf32>) {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%in_ref <- %input, %out_ref = %output)[%id: index, %count: index]
         : (!pcf.sref<?x?xf32, #pcf.test_scope>,
            !pcf.sref<?x?xf32, #pcf.test_scope>)
        -> (tensor<?x?xf32>) {
    util.optimization_barrier %in_ref, %out_ref : !pcf.sref<?x?xf32, #pcf.test_scope>, !pcf.sref<?x?xf32, #pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %0 : tensor<?x?xf32>
  util.return
}

// CHECK-LABEL: @generic_readonly_dynamic
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[OUTPUT:[A-Za-z0-9]+]]: tensor<?x?xf32>
//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute(%[[IN_REF:[A-Za-z0-9_]+]] <- %[[INPUT]],
//  CHECK-SAME:             %[[OUT_REF:[A-Za-z0-9_]+]] = %[[OUTPUT]])
//  CHECK-SAME:             [%[[ID:.+]]: index, %[[COUNT:.+]]: index]
//  CHECK-NEXT:          : (!pcf.sref<?x?xf32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?x?xf32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (tensor<?x?xf32>) {
//  CHECK-NEXT:       util.optimization_barrier %[[IN_REF]], %[[OUT_REF]]
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }
