// RUN: iree-opt --iree-pcf-distribute-and-lower --allow-unregistered-dialect --split-input-file %s | FileCheck %s
// RUN: iree-opt --iree-pcf-distribute-and-lower="test-distribution=true" --allow-unregistered-dialect --split-input-file %s | FileCheck %s --check-prefix=DIST

// Two-cluster tile_group lowers to scf.index_switch with 1 case + default.
// CHECK-LABEL: util.func private @two_cluster_tile_group
// CHECK-SAME:    (%{{.+}}: !pcf.threadgroup<#pcf.test_scope>, %[[K:.+]]: index)
// CHECK:         %[[TID:.+]] = builtin.unrealized_conversion_cast to index
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[GE:.+]] = arith.cmpi uge, %[[TID]], %[[K]]
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[IDX:.+]] = arith.select %[[GE]], %[[C1]], %[[C0]]
// CHECK:         scf.index_switch %[[IDX]]
// CHECK-NEXT:    case 0 {
// CHECK:           "test.use_left"
// CHECK:           "test.use_right"
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK:           "test.use_left"
// CHECK:           "test.use_right"
// CHECK:         }
// CHECK-NOT:     pcf.shared_executor.tile_group
util.func private @two_cluster_tile_group(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    "test.use_left"() : () -> ()
    "test.use_right"() : () -> ()
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Three-cluster tile_group lowers to scf.index_switch with 2 cases + default.
// Each case gets the full cloned body (distribution is a separate phase).
// CHECK-LABEL: util.func private @three_cluster_tile_group
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    case 1 {
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK:         }
// CHECK-NOT:     pcf.shared_executor.tile_group
util.func private @three_cluster_tile_group(
    %tg: !pcf.threadgroup<#pcf.test_scope>,
    %k0: index, %k1: index) {
  pcf.shared_executor.tile_group %tg split [[%k0, %k1]]
      (%c0: !pcf.cluster<#pcf.test_scope, (0 -> d0), c0>,
       %c1: !pcf.cluster<#pcf.test_scope, (d0 -> d1), c1>,
       %c2: !pcf.cluster<#pcf.test_scope, (d1 -> s0), c2>) {
    "test.cluster_0"() : () -> ()
    "test.cluster_1"() : () -> ()
    "test.cluster_2"() : () -> ()
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Nested tile_groups: inner lowered first, then outer. Both become
// scf.index_switch. Two switches should appear in the output.
// CHECK-LABEL: util.func private @nested_tile_groups
// CHECK:         scf.index_switch
// CHECK:           scf.index_switch
// CHECK-NOT:     pcf.shared_executor.tile_group
util.func private @nested_tile_groups(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index, %j: index) {
  pcf.shared_executor.tile_group %tg ns(outer) split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), outer.right>) {
    pcf.shared_executor.tile_group %left ns(inner) split [[%j]]
        (%top: !pcf.cluster<#pcf.test_scope, (0 -> d0), inner.top>,
         %bot: !pcf.cluster<#pcf.test_scope, (d0 -> s0), inner.bot>) {
      "test.inner_body"() : () -> ()
      pcf.return
    } : !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Single-cluster tile_group (no split points) lowers to scf.index_switch
// with only a default case.
// CHECK-LABEL: util.func private @single_cluster_tile_group
// CHECK:         scf.index_switch
// CHECK-NEXT:    default {
// CHECK:           "test.only_cluster"
// CHECK:         }
// CHECK-NOT:     pcf.shared_executor.tile_group
util.func private @single_cluster_tile_group(
    %tg: !pcf.threadgroup<#pcf.test_scope>) {
  pcf.shared_executor.tile_group %tg split [[]]
      (%c0: !pcf.cluster<#pcf.test_scope, (0 -> s0), c0>) {
    "test.only_cluster"() : () -> ()
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Test distribution dispatch: run_cluster inside tile_group with
// test-distribution (no-op). The run_cluster targets %left, so it only
// appears in case 0 (left cluster). The default case (right cluster) is
// empty since no ops target the right cluster.
//
// Without test-distribution: same result (distribution skipped).
// CHECK-LABEL: util.func private @tile_group_with_run_cluster
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK:           pcf.shared_executor.run_cluster
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK-NOT:       pcf.shared_executor.run_cluster
// CHECK:         }
// CHECK-NOT:     pcf.shared_executor.tile_group
//
// With test-distribution: distribution dispatch runs (no-op), same output.
// DIST-LABEL: util.func private @tile_group_with_run_cluster
// DIST:         scf.index_switch
// DIST-NEXT:    case 0 {
// DIST:           pcf.shared_executor.run_cluster
// DIST:           scf.yield
// DIST:         }
// DIST-NEXT:    default {
// DIST-NOT:       pcf.shared_executor.run_cluster
// DIST:         }
// DIST-NOT:     pcf.shared_executor.tile_group
util.func private @tile_group_with_run_cluster(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    pcf.shared_executor.run_cluster(%left)[%k]
        () {
      "test.body"() : () -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// run_thread inside tile_group is inlined during structural lowering.
// The run_thread targets %left, so it is only inlined into case 0 (left
// cluster). The default case (right cluster) is empty.
// For cluster "left" with bounds (0 -> d0), lower bound is 0, so
// local_tid = global_tid - affine.apply(0).
//
// CHECK-LABEL: util.func private @run_thread_inline
// CHECK-SAME:    (%{{.+}}: !pcf.threadgroup<#pcf.test_scope>, %[[K:.+]]: index)
// CHECK-DAG:     %[[TID:.+]] = builtin.unrealized_conversion_cast to index
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK:           %[[LB:.+]] = affine.apply
// CHECK:           %[[LOCAL:.+]] = arith.subi %[[TID]], %[[LB]]
// CHECK:           "test.use_tid"(%[[LOCAL]])
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK-NOT:       affine.apply
// CHECK-NOT:       "test.use_tid"
// CHECK:         }
// CHECK-NOT:     pcf.shared_executor.run_thread
// CHECK-NOT:     pcf.cluster_yield
util.func private @run_thread_inline(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    pcf.shared_executor.run_thread(%left)[%k]
        ()[%tid: index] {
      "test.use_tid"(%tid) : (index) -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Two run_thread ops targeting DIFFERENT clusters in the same tile_group.
// This is the critical correctness test: each case should only include
// the run_thread that targets its cluster. Previously, all run_thread ops
// were inlined into every case, causing threads to execute wrong code.
//
// CHECK-LABEL: util.func private @two_run_threads_different_clusters
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK:           "test.left_code"
// CHECK-NOT:       "test.right_code"
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK-NOT:       "test.left_code"
// CHECK:           "test.right_code"
// CHECK:         }
// CHECK-NOT:     pcf.shared_executor.run_thread
util.func private @two_run_threads_different_clusters(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    pcf.shared_executor.run_thread(%left)[%k]
        ()[%tid: index] {
      "test.left_code"(%tid) : (index) -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
    pcf.shared_executor.run_thread(%right)[%k]
        ()[%tid: index] {
      "test.right_code"(%tid) : (index) -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (d0 -> s0), right>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// run_cluster targeting one cluster: only appears in its case, not the other.
// The run_cluster with a matching source is preserved for later lowering.
// The non-matching case gets no run_cluster.
//
// CHECK-LABEL: util.func private @run_cluster_selective
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK-NOT:       pcf.shared_executor.run_cluster
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK:           pcf.shared_executor.run_cluster
// CHECK:         }
// CHECK-NOT:     pcf.shared_executor.tile_group
util.func private @run_cluster_selective(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    pcf.shared_executor.run_cluster(%right)[%k]
        () {
      "test.right_body"() : () -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (d0 -> s0), right>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// shared_executor with a readwrite init and a tile_group inside. After
// lowering, the shared_executor becomes pcf.generic, and the tile_group
// becomes scf.index_switch inside the generic's body.
//
// CHECK-LABEL: util.func private @shared_executor_with_tile_group
// CHECK-SAME:    (%[[INIT:.+]]: tensor<4x8xf32>, %[[K:.+]]: index)
// CHECK:         pcf.generic scope(#pcf.test_scope)
// CHECK:           execute(%{{.+}} = %[[INIT]])
// CHECK:             scf.index_switch
// CHECK:             case 0 {
// CHECK:               "test.body"
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             default {
// CHECK:               "test.body"
// CHECK:             }
// CHECK:           pcf.return
// CHECK-NOT:     pcf.shared_executor
util.func private @shared_executor_with_tile_group(
    %init: tensor<4x8xf32>, %k: index) {
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    execute(%ref = %init)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<4x8xf32, #pcf.test_scope>)
        -> (tensor<4x8xf32>) {
    pcf.shared_executor.tile_group %tg split [[%k]]
        (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
         %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
      "test.body"() : () -> ()
      pcf.return
    } : !pcf.threadgroup<#pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %0 : tensor<4x8xf32>
  util.return
}

// -----

// shared_executor with no operands (no readwrite, no readonly).
// Lowers to pcf.generic with no inits and no results.
//
// CHECK-LABEL: util.func private @shared_executor_no_operands
// CHECK:         pcf.generic scope(#pcf.test_scope)
// CHECK:           execute
// CHECK:             "test.work"
// CHECK:           pcf.return
// CHECK-NOT:     pcf.shared_executor
util.func private @shared_executor_no_operands() {
  pcf.shared_executor scope(#pcf.test_scope)
    execute[%tg: !pcf.threadgroup<#pcf.test_scope>] {
    "test.work"() : () -> ()
    pcf.return
  }
  util.return
}

// -----

// 2D tile_group with 2x2 = 4 clusters from a 2D cluster source.
// Linearized index_switch has 3 cases + default.
// CHECK-LABEL: util.func private @two_dim_tile_group
// CHECK-SAME:    (%{{.+}}: !pcf.cluster<#pcf.test_scope, (0 -> s0) x (0 -> s1), src>, %[[K:.+]]: index, %[[J:.+]]: index)
// CHECK:         %[[TIDS:.+]]:2 = builtin.unrealized_conversion_cast to index, index
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK:           "test.body_2d"
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    case 1 {
// CHECK:           "test.body_2d"
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    case 2 {
// CHECK:           "test.body_2d"
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK:           "test.body_2d"
// CHECK:         }
// CHECK-NOT:     pcf.shared_executor.tile_group
util.func private @two_dim_tile_group(
    %src: !pcf.cluster<#pcf.test_scope, (0 -> s0) x (0 -> s1), src>,
    %k: index, %j: index) {
  pcf.shared_executor.tile_group %src split [[%k], [%j]]
      (%c00: !pcf.cluster<#pcf.test_scope, (0 -> d0) x (0 -> d1), c00>,
       %c01: !pcf.cluster<#pcf.test_scope, (0 -> d0) x (d1 -> s1), c01>,
       %c10: !pcf.cluster<#pcf.test_scope, (d0 -> s0) x (0 -> d1), c10>,
       %c11: !pcf.cluster<#pcf.test_scope, (d0 -> s0) x (d1 -> s1), c11>) {
    "test.body_2d"() : () -> ()
    pcf.return
  } : !pcf.cluster<#pcf.test_scope, (0 -> s0) x (0 -> s1), src>
  util.return
}

// -----

// shared_executor with initializer region lowers to pcf.generic with
// initializer. The initializer body is moved and leading args are mapped.
//
// CHECK-LABEL: util.func private @shared_executor_with_initializer
// CHECK-SAME:    (%[[OUTPUT:.+]]: tensor<128x128xf32>)
// CHECK:         pcf.generic scope(#pcf.test_scope)
// CHECK:           initialize {
// CHECK:             %[[SMEM:.+]] = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.test_scope>
// CHECK:             pcf.yield %[[SMEM]]
// CHECK:           }
// CHECK:           execute(%{{.+}} = %[[OUTPUT]])
// CHECK:             "test.use_smem"
// CHECK:           pcf.return
// CHECK-NOT:     pcf.shared_executor
util.func private @shared_executor_with_initializer(%output: tensor<128x128xf32>) {
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    initialize {
      %smem = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.test_scope>
      pcf.yield %smem : !pcf.sref<128x64xf16, #pcf.test_scope>
    } -> (%smem: !pcf.sref<128x64xf16, #pcf.test_scope>)
    execute(%ref = %output)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<128x128xf32, #pcf.test_scope>)
        -> (tensor<128x128xf32>) {
    "test.use_smem"(%smem) : (!pcf.sref<128x64xf16, #pcf.test_scope>) -> ()
    pcf.return
  }
  util.optimization_barrier %0 : tensor<128x128xf32>
  util.return
}

// -----

// shared_executor with readonly ref lowers to pcf.generic with pcf.to_sref.
// The readonly input tensor is captured from outside the generic.
//
// CHECK-LABEL: util.func private @shared_executor_readonly
// CHECK-SAME:    (%[[INPUT:.+]]: tensor<128x256xf16>, %[[OUTPUT:.+]]: tensor<128x128xf32>)
// CHECK:         pcf.generic scope(#pcf.test_scope)
// CHECK:           execute(%{{.+}} = %[[OUTPUT]])
// CHECK:             %[[SREF:.+]] = pcf.to_sref %[[INPUT]] : tensor<128x256xf16> -> !pcf.sref<128x256xf16, #pcf.test_scope>
// CHECK:           pcf.return
// CHECK-NOT:     pcf.shared_executor
util.func private @shared_executor_readonly(
    %input: tensor<128x256xf16>, %output: tensor<128x128xf32>) {
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    execute(%in_ref <- %input, %out_ref = %output)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<128x256xf16, #pcf.test_scope>,
            !pcf.sref<128x128xf32, #pcf.test_scope>)
        -> (tensor<128x128xf32>) {
    pcf.return
  }
  util.optimization_barrier %0 : tensor<128x128xf32>
  util.return
}

// -----

// shared_executor with readonly ref only (no readwrite, no results).
// Lowers to pcf.generic with no inits and pcf.to_sref inside the body.
//
// CHECK-LABEL: util.func private @shared_executor_readonly_only
// CHECK-SAME:    (%[[INPUT:.+]]: tensor<4x8xf32>)
// CHECK:         pcf.generic scope(#pcf.test_scope)
// CHECK:           execute
// CHECK:             %[[SREF:.+]] = pcf.to_sref %[[INPUT]] : tensor<4x8xf32> -> !pcf.sref<4x8xf32, #pcf.test_scope>
// CHECK:             "test.use"(%[[SREF]])
// CHECK:           pcf.return
// CHECK-NOT:     pcf.shared_executor
util.func private @shared_executor_readonly_only(%input: tensor<4x8xf32>) {
  pcf.shared_executor scope(#pcf.test_scope)
    execute(%ref <- %input)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<4x8xf32, #pcf.test_scope>) {
    "test.use"(%ref) : (!pcf.sref<4x8xf32, #pcf.test_scope>) -> ()
    pcf.return
  }
  util.return
}

// -----

// Equivalence tracing: two independent run_cluster ops with the same cluster
// ID targeting the same cluster arg. The equivalence tracer groups them by
// cluster ID so distribution handles them together. Both should appear only
// in case 0 (left cluster), not in the default case (right cluster).
//
// CHECK-LABEL: util.func private @two_run_clusters_same_id
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK:           pcf.shared_executor.run_cluster
// CHECK:           pcf.shared_executor.run_cluster
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK-NOT:       pcf.shared_executor.run_cluster
// CHECK:         }
util.func private @two_run_clusters_same_id(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    pcf.shared_executor.run_cluster(%left)[%k]
        () {
      "test.first_cluster_op"() : () -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
    pcf.shared_executor.run_cluster(%left)[%k]
        () {
      "test.second_cluster_op"() : () -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// TODO: Test equivalence tracing through scf.for iter_args (run_cluster
// result flowing through loop-carried values to another run_cluster). This
// requires constructing valid IR where a cluster-typed value passes through
// scf.for iter_args. While scf.for accepts cluster-typed iter_args, the
// tile_group lowering converts cluster types to 0 types (erasure), which
// would require an scf.for conversion pattern to handle the type change
// on iter_args. The equivalence analysis itself works (tested separately
// in equivalence_analysis.mlir), but end-to-end lowering of cluster values
// through scf.for is not yet supported.

// -----

// Four-cluster tile_group with 3 split points. Produces an index_switch
// with 3 cases + default.
//
// CHECK-LABEL: util.func private @four_cluster_tile_group
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    case 1 {
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    case 2 {
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK:         }
// CHECK-NOT:     pcf.shared_executor.tile_group
util.func private @four_cluster_tile_group(
    %tg: !pcf.threadgroup<#pcf.test_scope>,
    %k0: index, %k1: index, %k2: index) {
  pcf.shared_executor.tile_group %tg split [[%k0, %k1, %k2]]
      (%c0: !pcf.cluster<#pcf.test_scope, (0 -> d0), c0>,
       %c1: !pcf.cluster<#pcf.test_scope, (d0 -> d1), c1>,
       %c2: !pcf.cluster<#pcf.test_scope, (d1 -> d2), c2>,
       %c3: !pcf.cluster<#pcf.test_scope, (d2 -> s0), c3>) {
    "test.body"() : () -> ()
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Two run_cluster ops targeting different clusters. Each should appear only
// in its own case. This verifies multi-cluster-ID grouping.
//
// CHECK-LABEL: util.func private @multi_cluster_id_groups
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK:           pcf.shared_executor.run_cluster
// CHECK:           "test.left_body"
// CHECK-NOT:       "test.right_body"
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK:           pcf.shared_executor.run_cluster
// CHECK:           "test.right_body"
// CHECK-NOT:       "test.left_body"
// CHECK:         }
util.func private @multi_cluster_id_groups(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    pcf.shared_executor.run_cluster(%left)[%k]
        () {
      "test.left_body"() : () -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
    pcf.shared_executor.run_cluster(%right)[%k]
        () {
      "test.right_body"() : () -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (d0 -> s0), right>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// run_thread inside nested tile_group. The inner tile_group further splits
// a cluster, and a run_thread operates within the inner split.
//
// CHECK-LABEL: util.func private @run_thread_in_nested_tile_group
// CHECK:         scf.index_switch
// CHECK:           scf.index_switch
// CHECK:             arith.subi
// CHECK:             "test.inner_thread"
// CHECK-NOT:     pcf.shared_executor.run_thread
util.func private @run_thread_in_nested_tile_group(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index, %j: index) {
  pcf.shared_executor.tile_group %tg ns(outer) split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), outer.right>) {
    pcf.shared_executor.tile_group %left ns(inner) split [[%j]]
        (%top: !pcf.cluster<#pcf.test_scope, (0 -> d0), inner.top>,
         %bot: !pcf.cluster<#pcf.test_scope, (d0 -> s0), inner.bot>) {
      pcf.shared_executor.run_thread(%top)[%j]
          ()[%tid: index] {
        "test.inner_thread"(%tid) : (index) -> ()
        pcf.cluster_yield
      } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), inner.top>)
      pcf.return
    } : !pcf.cluster<#pcf.test_scope, (0 -> d0), outer.left>
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// shared_executor with both initializer and mixed readonly + readwrite.
// Verifies that the lowering handles all three operand kinds together.
//
// CHECK-LABEL: util.func private @shared_executor_mixed
// CHECK-SAME:    (%[[INPUT:.+]]: tensor<64x256xf16>, %[[OUTPUT:.+]]: tensor<64x64xf32>)
// CHECK:         pcf.generic scope(#pcf.test_scope) initialize {
// CHECK:             %[[SMEM:.+]] = pcf.alloc()
// CHECK:             pcf.yield %[[SMEM]]
// CHECK:           }
// CHECK:           execute(%{{.+}} = %[[OUTPUT]])
// CHECK:             %[[SREF:.+]] = pcf.to_sref %[[INPUT]]
// CHECK:             "test.use_all"
// CHECK:           pcf.return
// CHECK-NOT:     pcf.shared_executor
util.func private @shared_executor_mixed(
    %input: tensor<64x256xf16>, %output: tensor<64x64xf32>) {
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    initialize {
      %smem = pcf.alloc() : !pcf.sref<64x32xf16, #pcf.test_scope>
      pcf.yield %smem : !pcf.sref<64x32xf16, #pcf.test_scope>
    } -> (%smem: !pcf.sref<64x32xf16, #pcf.test_scope>)
    execute(%in_ref <- %input, %out_ref = %output)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<64x256xf16, #pcf.test_scope>,
            !pcf.sref<64x64xf32, #pcf.test_scope>)
        -> (tensor<64x64xf32>) {
    "test.use_all"(%smem, %in_ref, %out_ref)
        : (!pcf.sref<64x32xf16, #pcf.test_scope>,
           !pcf.sref<64x256xf16, #pcf.test_scope>,
           !pcf.sref<64x64xf32, #pcf.test_scope>) -> ()
    pcf.return
  }
  util.optimization_barrier %0 : tensor<64x64xf32>
  util.return
}

// -----

// run_thread with struct block arguments (from source cluster with private
// types). The structural lowering creates unrealized_conversion_cast for
// struct args when inlining the run_thread body.
//
// CHECK-LABEL: util.func private @run_thread_with_struct_args
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK:           %[[CAST:.+]] = builtin.unrealized_conversion_cast to index
// CHECK:           arith.subi
// CHECK:           "test.use_struct"
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NEXT:    default {
// CHECK:         }
// CHECK-NOT:     pcf.shared_executor.run_thread
util.func private @run_thread_with_struct_args(
    %tg: !pcf.threadgroup<#pcf.test_scope>, %k: index) {
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    // A run_thread produces a cluster with private types.
    %c1 = pcf.shared_executor.run_thread(%left)[%k]
        ()[%tid0: index] {
      %idx = arith.constant 42 : index
      pcf.cluster_yield %idx : index
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), left>)
      -> !pcf.cluster<#pcf.test_scope, (0 -> d0), private: {index}, left>
    // Another run_thread consumes the private value as a struct arg.
    pcf.shared_executor.run_thread(%c1)[%k]
        (%p: index)[%tid1: index] {
      "test.use_struct"(%p, %tid1) : (index, index) -> ()
      pcf.cluster_yield
    } : (!pcf.cluster<#pcf.test_scope, (0 -> d0), private: {index}, left>)
    pcf.return
  } : !pcf.threadgroup<#pcf.test_scope>
  util.return
}

// -----

// Telescoping: shared_executor with init_subscope + telescope lowers to
// pcf.generic wrapping a child shared_executor with initializer. The
// init_subscope body moves into the child's initializer, and telescope
// results become the child's threadgroup and leading args.
//
// CHECK-LABEL: util.func private @telescoping_basic
// CHECK-SAME:    (%[[OUTPUT:.+]]: tensor<128xf32>)
// CHECK:         pcf.generic scope(#pcf.test_scope)
// CHECK:           execute(%{{.+}} = %[[OUTPUT]])
// CHECK:             pcf.shared_executor scope(#pcf.test_scope) initialize {
// CHECK:               %[[ALLOC:.+]] = pcf.alloc() : !pcf.sref<64xf16, #pcf.test_scope>
// CHECK:               pcf.yield %[[ALLOC]]
// CHECK:             }
// CHECK:             execute[%[[TG:.+]]: !pcf.threadgroup<#pcf.test_scope>]
// CHECK-NOT:           builtin.unrealized_conversion_cast
// CHECK:               "test.use"(%[[TG]], %{{.+}})
// CHECK:             pcf.return
// CHECK:           pcf.return
// CHECK-NOT:     pcf.shared_executor{{[^.]}}
util.func private @telescoping_basic(%output: tensor<128xf32>) {
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    execute(%ref = %output)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<128xf32, #pcf.test_scope>)
        -> (tensor<128xf32>) {
    %tid = arith.constant 0 : index
    %tg2 = pcf.init_subscope %tg {
      %alloc = pcf.alloc() : !pcf.sref<64xf16, #pcf.test_scope>
      pcf.yield %alloc : !pcf.sref<64xf16, #pcf.test_scope>
    } -> !pcf.threadgroup<#pcf.test_scope, {!pcf.sref<64xf16, #pcf.test_scope>}>
    %child_tg, %sref = pcf.telescope %tg2[%tid]
        : !pcf.threadgroup<#pcf.test_scope, {!pcf.sref<64xf16, #pcf.test_scope>}>
       -> (!pcf.threadgroup<#pcf.test_scope>, !pcf.sref<64xf16, #pcf.test_scope>)
    "test.use"(%child_tg, %sref) : (!pcf.threadgroup<#pcf.test_scope>, !pcf.sref<64xf16, #pcf.test_scope>) -> ()
    pcf.return
  }
  util.optimization_barrier %0 : tensor<128xf32>
  util.return
}

// -----

// Telescope without init_subscope: pure scope conversion with no
// initializer. The child shared_executor has no initializer region.
//
// CHECK-LABEL: util.func private @telescope_no_init_subscope
// CHECK:         pcf.generic scope(#pcf.test_scope)
// CHECK:           execute(%{{.+}} = %{{.+}})
// CHECK:             pcf.shared_executor scope(#pcf.test_scope)
// CHECK-NOT:         initialize
// CHECK:               execute[%[[TG:.+]]: !pcf.threadgroup<#pcf.test_scope>]
// CHECK:               "test.use"(%[[TG]])
// CHECK:             pcf.return
// CHECK:           pcf.return
// CHECK-NOT:     pcf.shared_executor{{[^.]}}
util.func private @telescope_no_init_subscope(%output: tensor<128xf32>) {
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    execute(%ref = %output)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<128xf32, #pcf.test_scope>)
        -> (tensor<128xf32>) {
    %tid = arith.constant 0 : index
    %child_tg = pcf.telescope %tg[%tid]
        : !pcf.threadgroup<#pcf.test_scope> -> !pcf.threadgroup<#pcf.test_scope>
    "test.use"(%child_tg) : (!pcf.threadgroup<#pcf.test_scope>) -> ()
    pcf.return
  }
  util.optimization_barrier %0 : tensor<128xf32>
  util.return
}

// -----

// init_subscope without telescope: no child scope is detected, so the
// shared_executor lowers directly to pcf.generic without creating a
// child shared_executor. The init_subscope remains in the generic body.
//
// CHECK-LABEL: util.func private @init_subscope_no_telescope
// CHECK:         pcf.generic scope(#pcf.test_scope)
// CHECK:           execute(%{{.+}} = %{{.+}})
// CHECK:             pcf.init_subscope
// CHECK:             "test.use"
// CHECK:           pcf.return
// CHECK-NOT:     pcf.shared_executor
util.func private @init_subscope_no_telescope(%output: tensor<128xf32>) {
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    execute(%ref = %output)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<128xf32, #pcf.test_scope>)
        -> (tensor<128xf32>) {
    %tg2 = pcf.init_subscope %tg {
      %alloc = pcf.alloc() : !pcf.sref<64xf16, #pcf.test_scope>
      pcf.yield %alloc : !pcf.sref<64xf16, #pcf.test_scope>
    } -> !pcf.threadgroup<#pcf.test_scope, {!pcf.sref<64xf16, #pcf.test_scope>}>
    "test.use"(%tg2) : (!pcf.threadgroup<#pcf.test_scope, {!pcf.sref<64xf16, #pcf.test_scope>}>) -> ()
    pcf.return
  }
  util.optimization_barrier %0 : tensor<128xf32>
  util.return
}
