// RUN: iree-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// Index symbol in a pcf.generic initializer.
// CHECK-LABEL: @generic_index_symbol
util.func private @generic_index_symbol(%output: tensor<4x8xf32>) {
  %0 = pcf.generic scope(#pcf.sequential) initialize {
      %c4 = arith.constant 4 : index
      pcf.index_symbol "tile_count" = %c4
      %scratch = pcf.alloc() : !pcf.sref<16xf32, #pcf.sequential>
      pcf.yield %scratch : !pcf.sref<16xf32, #pcf.sequential>
    } -> (%s: !pcf.sref<16xf32, #pcf.sequential>)
    execute(%ref = %output)[%id: index, %n: index]
         : (!pcf.sref<4x8xf32, #pcf.sequential>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.optimization_barrier %0 : tensor<4x8xf32>
  util.return
}
// CHECK:   pcf.generic scope(#pcf.sequential) initialize {
// CHECK:       %[[C4:.+]] = arith.constant 4 : index
// CHECK-NEXT:  pcf.index_symbol "tile_count" = %[[C4]]

// -----

// Index symbol in a pcf.shared_executor initializer.
// CHECK-LABEL: @shared_executor_index_symbol
util.func private @shared_executor_index_symbol(%output: tensor<128x128xf32>) {
  %0 = pcf.shared_executor scope(#pcf.sequential)
    initialize {
      %c8 = arith.constant 8 : index
      pcf.index_symbol "num_tiles" = %c8
      %smem = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.sequential>
      pcf.yield %smem : !pcf.sref<128x64xf16, #pcf.sequential>
    } -> (%smem: !pcf.sref<128x64xf16, #pcf.sequential>)
    execute(%ref = %output)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<128x128xf32, #pcf.sequential>)
        -> (tensor<128x128xf32>) {
    pcf.return
  }
  util.optimization_barrier %0 : tensor<128x128xf32>
  util.return
}
// CHECK:   pcf.shared_executor scope(#pcf.sequential) initialize {
// CHECK:       %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT:  pcf.index_symbol "num_tiles" = %[[C8]]

// -----

// Index symbol outside a namespace (should fail verification).
util.func private @index_symbol_outside_namespace(%val: index) {
  // expected-error @+1 {{must be inside a NamespaceOpInterface op}}
  pcf.index_symbol "bad" = %val
  util.return
}

// -----

// Index symbol in the execute body (wrong region).
util.func private @index_symbol_wrong_region(%output: tensor<4x8xf32>) {
  %0 = pcf.generic scope(#pcf.sequential)
    execute(%ref = %output)[%id: index, %n: index]
         : (!pcf.sref<4x8xf32, #pcf.sequential>)
        -> (tensor<4x8xf32>) {
    %c1 = arith.constant 1 : index
    // expected-error @+1 {{must be in the symbol-defining region of the enclosing namespace}}
    pcf.index_symbol "bad" = %c1
    pcf.return
  }
  util.optimization_barrier %0 : tensor<4x8xf32>
  util.return
}
