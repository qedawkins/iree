// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// Basic shared_executor with tied readwrite operand.
util.func private @basic_shared_executor(%init: tensor<4x8xf32>) {
  %0 = pcf.shared_executor scope(#pcf.sequential)
    execute(%ref = %init)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<4x8xf32, #pcf.sequential>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.optimization_barrier %0 : tensor<4x8xf32>
  util.return
}

// CHECK-LABEL: @basic_shared_executor
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9]+]]: tensor<4x8xf32>
//       CHECK:   pcf.shared_executor scope(#pcf.sequential)
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[INIT]])
//  CHECK-SAME:         [%[[TG:.+]]: !pcf.threadgroup<#pcf.sequential>]
//  CHECK-NEXT:          : (!pcf.sref<4x8xf32, #pcf.sequential>)
//  CHECK-NEXT:         -> (tensor<4x8xf32>) {
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }

// -----

// Shared executor with readonly and readwrite operands.
util.func private @readonly_and_readwrite(
    %input: tensor<128x256xf16>, %output: tensor<128x128xf32>) {
  %0 = pcf.shared_executor scope(#pcf.sequential)
    execute(%in_ref <- %input, %out_ref = %output)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<128x256xf16, #pcf.sequential>,
            !pcf.sref<128x128xf32, #pcf.sequential>)
        -> (tensor<128x128xf32>) {
    pcf.return
  }
  util.optimization_barrier %0 : tensor<128x128xf32>
  util.return
}

// CHECK-LABEL: @readonly_and_readwrite
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<128x256xf16>
//  CHECK-SAME:   %[[OUTPUT:[A-Za-z0-9]+]]: tensor<128x128xf32>
//       CHECK:   pcf.shared_executor scope(#pcf.sequential)
//  CHECK-NEXT:     execute(%[[IN_REF:[A-Za-z0-9_]+]] <- %[[INPUT]],
//  CHECK-SAME:             %[[OUT_REF:[A-Za-z0-9_]+]] = %[[OUTPUT]])
//  CHECK-SAME:         [%[[TG:.+]]: !pcf.threadgroup<#pcf.sequential>]
//  CHECK-NEXT:          : (!pcf.sref<128x256xf16, #pcf.sequential>,
//  CHECK-SAME:             !pcf.sref<128x128xf32, #pcf.sequential>)
//  CHECK-NEXT:         -> (tensor<128x128xf32>) {
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }

// -----

// Shared executor with initialize region.
util.func private @with_initializer(%output: tensor<128x128xf32>) {
  %0 = pcf.shared_executor scope(#pcf.sequential)
    initialize {
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

// CHECK-LABEL: @with_initializer
//       CHECK:   pcf.shared_executor scope(#pcf.sequential) initialize {
//  CHECK-NEXT:       %[[SMEM:.+]] = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.sequential>
//  CHECK-NEXT:       pcf.yield %[[SMEM]]
//  CHECK-NEXT:     } -> (%[[SMEM_ARG:.+]]: !pcf.sref<128x64xf16, #pcf.sequential>)
//  CHECK-NEXT:     execute(%{{.*}} = %{{.*}})
//  CHECK-SAME:         [%{{.*}}: !pcf.threadgroup<#pcf.sequential>]

// -----

// Shared executor with no results (only readonly refs).
util.func private @no_results(%input: tensor<4x8xf32>) {
  pcf.shared_executor scope(#pcf.sequential)
    execute(%ref <- %input)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<4x8xf32, #pcf.sequential>) {
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @no_results
//       CHECK:   pcf.shared_executor scope(#pcf.sequential)
//  CHECK-NEXT:     execute(%{{.*}} <- %{{.*}})
//  CHECK-SAME:         [%{{.*}}: !pcf.threadgroup<#pcf.sequential>]
//  CHECK-NEXT:          : (!pcf.sref<4x8xf32, #pcf.sequential>) {
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }

// -----

// Multiple readwrite operands producing multiple results.
util.func private @multi_result(%a: tensor<4xf32>, %b: tensor<8xf32>) {
  %0:2 = pcf.shared_executor scope(#pcf.sequential)
    execute(%ref_a = %a, %ref_b = %b)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<4xf32, #pcf.sequential>,
            !pcf.sref<8xf32, #pcf.sequential>)
        -> (tensor<4xf32>, tensor<8xf32>) {
    pcf.return
  }
  util.optimization_barrier %0#0, %0#1 : tensor<4xf32>, tensor<8xf32>
  util.return
}

// CHECK-LABEL: @multi_result
//       CHECK:   pcf.shared_executor scope(#pcf.sequential)
//  CHECK-NEXT:     execute(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}})
//  CHECK-SAME:         [%{{.*}}: !pcf.threadgroup<#pcf.sequential>]

// -----

// Shared executor with no operands at all (just threadgroup).
util.func private @no_operands() {
  pcf.shared_executor scope(#pcf.sequential)
    execute[%tg: !pcf.threadgroup<#pcf.sequential>] {
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @no_operands
//       CHECK:   pcf.shared_executor scope(#pcf.sequential)
//  CHECK-NEXT:     execute[%{{.*}}: !pcf.threadgroup<#pcf.sequential>] {
//  CHECK-NEXT:       pcf.return
//  CHECK-NEXT:     }

// -----

// Shared executor with sync_on_return.
util.func private @sync_shared_executor(%init: tensor<4x8xf32>) {
  %0 = pcf.shared_executor sync scope(#pcf.sequential)
    execute(%ref = %init)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<4x8xf32, #pcf.sequential>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.optimization_barrier %0 : tensor<4x8xf32>
  util.return
}

// CHECK-LABEL: @sync_shared_executor
//       CHECK:   pcf.shared_executor sync scope(#pcf.sequential)
//  CHECK-NEXT:     execute(%{{.*}} = %{{.*}})
//  CHECK-SAME:         [%{{.*}}: !pcf.threadgroup<#pcf.sequential>]
