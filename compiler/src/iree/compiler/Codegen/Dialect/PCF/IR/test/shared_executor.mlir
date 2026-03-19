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

// Tile group: 1D split into 2 clusters (anonymous namespace).
util.func private @tile_group_1d_split(%tg: !pcf.threadgroup<#pcf.sequential>,
                                        %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.sequential, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.sequential, (d0 -> s0), right>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// CHECK-LABEL: @tile_group_1d_split
//       CHECK:   pcf.shared_executor.tile_group
//  CHECK-SAME:       split {{\[}}[%{{.*}}]]
//  CHECK-NEXT:       (%{{.*}}: !pcf.cluster<#pcf.sequential, (0 -> d0), left>,
//  CHECK-SAME:        %{{.*}}: !pcf.cluster<#pcf.sequential, (d0 -> s0), right>)

// -----

// Tile group: no splits (single cluster covering full range).
util.func private @tile_group_no_split(%tg: !pcf.threadgroup<#pcf.sequential>) {
  pcf.shared_executor.tile_group %tg split [[]]
      (%full: !pcf.cluster<#pcf.sequential, (0 -> s0), full>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// CHECK-LABEL: @tile_group_no_split
//       CHECK:   pcf.shared_executor.tile_group
//  CHECK-SAME:       split {{\[}}[]]

// -----

// Tile group from a 2D cluster, split dim0 only.
util.func private @tile_group_from_cluster(
    %cluster: !pcf.cluster<#pcf.test_scope, (0 -> s0) x (0 -> s1), src>,
    %k: index) {
  pcf.shared_executor.tile_group %cluster split [[%k], []]
      (%top: !pcf.cluster<#pcf.test_scope, (0 -> d0) x (0 -> s1), top>,
       %bot: !pcf.cluster<#pcf.test_scope, (d0 -> s0) x (0 -> s1), bot>) {
    pcf.return
  } : !pcf.cluster<#pcf.test_scope, (0 -> s0) x (0 -> s1), src>
  util.return
}

// CHECK-LABEL: @tile_group_from_cluster
//       CHECK:   pcf.shared_executor.tile_group
//  CHECK-SAME:       split {{\[}}[%{{.*}}], []]

// -----

// Tile group: 2D full split -> 4 clusters.
util.func private @tile_group_2d_full(
    %cluster: !pcf.cluster<#pcf.test_scope, (0 -> s0) x (0 -> s1), src>,
    %j: index, %k: index) {
  pcf.shared_executor.tile_group %cluster split [[%j], [%k]]
      (%c00: !pcf.cluster<#pcf.test_scope, (0 -> d0) x (0 -> d1), c00>,
       %c01: !pcf.cluster<#pcf.test_scope, (0 -> d0) x (d1 -> s1), c01>,
       %c10: !pcf.cluster<#pcf.test_scope, (d0 -> s0) x (0 -> d1), c10>,
       %c11: !pcf.cluster<#pcf.test_scope, (d0 -> s0) x (d1 -> s1), c11>) {
    pcf.return
  } : !pcf.cluster<#pcf.test_scope, (0 -> s0) x (0 -> s1), src>
  util.return
}

// CHECK-LABEL: @tile_group_2d_full
//       CHECK:   pcf.shared_executor.tile_group
//  CHECK-SAME:       split {{\[}}[%{{.*}}], [%{{.*}}]]

// -----

// Tile group with named namespace.
util.func private @tile_group_named_namespace(
    %tg: !pcf.threadgroup<#pcf.sequential>, %k: index) {
  pcf.shared_executor.tile_group %tg ns(tg) split [[%k]]
      (%left: !pcf.cluster<#pcf.sequential, (0 -> d0), tg.left>,
       %right: !pcf.cluster<#pcf.sequential, (d0 -> s0), tg.right>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// CHECK-LABEL: @tile_group_named_namespace
//       CHECK:   pcf.shared_executor.tile_group %{{.*}} ns(tg) split
//  CHECK-SAME:       {{\[}}[%{{.*}}]]
//  CHECK-NEXT:       (%{{.*}}: !pcf.cluster<#pcf.sequential, (0 -> d0), tg.left>,
//  CHECK-SAME:        %{{.*}}: !pcf.cluster<#pcf.sequential, (d0 -> s0), tg.right>)

// -----

// Tile group with anonymous namespace (no ns keyword).
util.func private @tile_group_anonymous_namespace(
    %tg: !pcf.threadgroup<#pcf.sequential>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%a: !pcf.cluster<#pcf.sequential, (0 -> d0), a>,
       %b: !pcf.cluster<#pcf.sequential, (d0 -> s0), b>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// CHECK-LABEL: @tile_group_anonymous_namespace
//   CHECK-NOT:   ns(
//       CHECK:   pcf.shared_executor.tile_group %{{.*}} split

// -----

// run_cluster: basic with result.
util.func private @run_cluster_basic(
    %c: !pcf.cluster<#pcf.sequential, (0 -> d0), shared: {f32}, c0>,
    %k: index) {
  %result = pcf.shared_executor.run_cluster(%c)[%k]
      (%s: f32) {
    pcf.cluster_yield %s : f32
  } : (!pcf.cluster<#pcf.sequential, (0 -> d0), shared: {f32}, c0>)
    -> !pcf.cluster<#pcf.sequential, (0 -> d0), shared: {f32}, c0>
  util.return
}

// CHECK-LABEL: @run_cluster_basic
//       CHECK:   pcf.shared_executor.run_cluster(%{{.*}})[%{{.*}}]
//  CHECK-NEXT:       (%{{.*}}: f32)
//       CHECK:     pcf.cluster_yield %{{.*}} : f32
//       CHECK:   } : (!pcf.cluster<#pcf.sequential, (0 -> d0), shared: {f32}, c0>) -> !pcf.cluster<#pcf.sequential, (0 -> d0), shared: {f32}, c0>

// -----

// run_cluster: no result.
util.func private @run_cluster_void(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>) {
  pcf.shared_executor.run_cluster(%c)[]
      (%s: f32) {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>)
  util.return
}

// CHECK-LABEL: @run_cluster_void
//       CHECK:   pcf.shared_executor.run_cluster(%{{.*}})[]
//  CHECK-NEXT:       (%{{.*}}: f32)
//       CHECK:     pcf.cluster_yield
//       CHECK:   } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>)

// -----

// run_cluster: multiple sources.
util.func private @run_cluster_multi_source(
    %c0: !pcf.cluster<#pcf.sequential, (0 -> d0), shared: {f32}, c0>,
    %c1: !pcf.cluster<#pcf.sequential, (0 -> d0), shared: {index}, c0>,
    %k: index) {
  pcf.shared_executor.run_cluster(%c0, %c1)[%k]
      (%s0: f32, %s1: index) {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> d0), shared: {f32}, c0>,
       !pcf.cluster<#pcf.sequential, (0 -> d0), shared: {index}, c0>)
  util.return
}

// CHECK-LABEL: @run_cluster_multi_source
//       CHECK:   pcf.shared_executor.run_cluster(%{{.*}}, %{{.*}})[%{{.*}}]
//  CHECK-NEXT:       (%{{.*}}: f32, %{{.*}}: index)
//       CHECK:     pcf.cluster_yield
//       CHECK:   } : (!pcf.cluster<#pcf.sequential, (0 -> d0), shared: {f32}, c0>,
//  CHECK-SAME:        !pcf.cluster<#pcf.sequential, (0 -> d0), shared: {index}, c0>)

// -----

// run_thread: basic with result and thread IDs.
util.func private @run_thread_basic(
    %c: !pcf.cluster<#pcf.sequential, (0 -> d0), private: {f32}, c0>,
    %k: index) {
  %result = pcf.shared_executor.run_thread(%c)[%k]
      (%p: f32)[%tid: index] {
    pcf.cluster_yield %p : f32
  } : (!pcf.cluster<#pcf.sequential, (0 -> d0), private: {f32}, c0>)
    -> !pcf.cluster<#pcf.sequential, (0 -> d0), private: {f32}, c0>
  util.return
}

// CHECK-LABEL: @run_thread_basic
//       CHECK:   pcf.shared_executor.run_thread(%{{.*}})[%{{.*}}]
//  CHECK-NEXT:       (%{{.*}}: f32)[%{{.*}}: index]
//       CHECK:     pcf.cluster_yield %{{.*}} : f32
//       CHECK:   } : (!pcf.cluster<#pcf.sequential, (0 -> d0), private: {f32}, c0>) -> !pcf.cluster<#pcf.sequential, (0 -> d0), private: {f32}, c0>

// -----

// run_thread: no result, only thread ID.
util.func private @run_thread_void(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), c0>) {
  pcf.shared_executor.run_thread(%c)[]
      ()[%tid: index] {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), c0>)
  util.return
}

// CHECK-LABEL: @run_thread_void
//       CHECK:   pcf.shared_executor.run_thread(%{{.*}})[]
//  CHECK-NEXT:       ()[%{{.*}}: index]
//       CHECK:   } : (!pcf.cluster<#pcf.sequential, (0 -> s0), c0>)

// -----

// run_thread: multiple sources.
util.func private @run_thread_multi_source(
    %c0: !pcf.cluster<#pcf.sequential, (0 -> d0), private: {f32}, c0>,
    %c1: !pcf.cluster<#pcf.sequential, (0 -> d0), private: {index}, c0>,
    %k: index) {
  pcf.shared_executor.run_thread(%c0, %c1)[%k]
      (%p0: f32, %p1: index)[%tid: index] {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> d0), private: {f32}, c0>,
       !pcf.cluster<#pcf.sequential, (0 -> d0), private: {index}, c0>)
  util.return
}

// CHECK-LABEL: @run_thread_multi_source
//       CHECK:   pcf.shared_executor.run_thread(%{{.*}}, %{{.*}})[%{{.*}}]
//  CHECK-NEXT:       (%{{.*}}: f32, %{{.*}}: index)[%{{.*}}: index]
//       CHECK:     pcf.cluster_yield
//       CHECK:   } : (!pcf.cluster<#pcf.sequential, (0 -> d0), private: {f32}, c0>,
//  CHECK-SAME:        !pcf.cluster<#pcf.sequential, (0 -> d0), private: {index}, c0>)

// -----

// run_cluster: yield values only.
util.func private @run_cluster_values_only(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>) {
  %result = pcf.shared_executor.run_cluster(%c)[]
      (%s: f32) {
    pcf.cluster_yield %s : f32
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>)
    -> !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>
  util.return
}

// CHECK-LABEL: @run_cluster_values_only
//       CHECK:     pcf.cluster_yield %{{.*}} : f32
//       CHECK:   } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>) -> !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>

// -----

// init_subscope: single allocation.
util.func private @init_subscope_single(%tg: !pcf.threadgroup<#pcf.sequential>) {
  %tg2 = pcf.init_subscope %tg {
    %alloc = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.sequential>
    pcf.yield %alloc : !pcf.sref<128x64xf16, #pcf.sequential>
  } -> !pcf.threadgroup<#pcf.sequential, {!pcf.sref<128x64xf16, #pcf.sequential>}>
  util.return
}

// CHECK-LABEL: @init_subscope_single
//  CHECK-SAME:   %[[TG:[A-Za-z0-9]+]]: !pcf.threadgroup<#pcf.sequential>
//       CHECK:   pcf.init_subscope %[[TG]] {
//  CHECK-NEXT:     %[[ALLOC:.+]] = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.sequential>
//  CHECK-NEXT:     pcf.yield %[[ALLOC]] : !pcf.sref<128x64xf16, #pcf.sequential>
//  CHECK-NEXT:   } -> !pcf.threadgroup<#pcf.sequential, {!pcf.sref<128x64xf16, #pcf.sequential>}>

// -----

// init_subscope: multiple allocations.
util.func private @init_subscope_multi(%tg: !pcf.threadgroup<#pcf.sequential>) {
  %tg2 = pcf.init_subscope %tg {
    %a0 = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.sequential>
    %a1 = pcf.alloc() : !pcf.sref<64x32xf32, #pcf.sequential>
    pcf.yield %a0, %a1 : !pcf.sref<128x64xf16, #pcf.sequential>, !pcf.sref<64x32xf32, #pcf.sequential>
  } -> !pcf.threadgroup<#pcf.sequential, {!pcf.sref<128x64xf16, #pcf.sequential>, !pcf.sref<64x32xf32, #pcf.sequential>}>
  util.return
}

// CHECK-LABEL: @init_subscope_multi
//       CHECK:   pcf.init_subscope %{{.*}} {
//       CHECK:     pcf.yield %{{.*}}, %{{.*}} : !pcf.sref<128x64xf16, #pcf.sequential>, !pcf.sref<64x32xf32, #pcf.sequential>
//  CHECK-NEXT:   } -> !pcf.threadgroup<#pcf.sequential, {!pcf.sref<128x64xf16, #pcf.sequential>, !pcf.sref<64x32xf32, #pcf.sequential>}>

