// RUN: iree-opt --split-input-file %s --verify-diagnostics

util.func private @scope_mismatch(%dim: index) {
// expected-error@+1 {{expected region ref argument to be of type !pcf.sref with scope #pcf.sequential}}
  pcf.generic scope(#pcf.sequential)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

// expected-note@+1 {{prior use here}}
util.func private @init_type_mismatch(%0: tensor<3xi32>) {
  pcf.generic scope(#pcf.test_scope)
// expected-error@+1 {{expects different type than prior uses: 'tensor<?xi32>' vs 'tensor<3xi32>'}}
    execute(%ref = %0)[%num_threads: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>) {
    pcf.return
  }
  util.return
}

// -----

util.func private @sync_scope_mismatch(%dim: index) {
// expected-error@+1 {{expected region ref argument to sync on return or is unspecified}}
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope, i32>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @arg_shape_mismatch(%dim: index) {
// expected-error@+1 {{region arg at index 0 with type '!pcf.sref<3xi32, #pcf.test_scope>' shape mismatch with tied result of type 'tensor<?xi32>'}}
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<3xi32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @arg_eltype_mismatch(%dim: index) {
// expected-error@+1 {{region arg at index 0 element type mismatch of 'f32' vs 'i32'}}
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?xf32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @empty_count(%dim: index) {
// expected-error@+1 {{expected at least one iteration count argument}}
  pcf.loop scope(#pcf.sequential) count()
    execute(%ref)[]
         : (!pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @missing_execute_keyword() {
  pcf.generic scope(#pcf.test_scope)
    // expected-error@+1 {{custom op 'pcf.generic' expected 'execute'}}
    notexecute(%ref)[%id: index, %n: index]
         : (!pcf.sref<4xi32, #pcf.test_scope>)
        -> (tensor<4xi32>) {
    pcf.return
  }
  util.return
}

// -----

util.func private @result_not_shaped_type() {
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<4xi32, #pcf.test_scope>)
        // expected-error@+1 {{custom op 'pcf.generic' result type must be a shaped type}}
        -> (i32) {
    pcf.return
  }
  util.return
}

// -----

util.func private @dynamic_dim_mismatch(%dim0: index) {
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.test_scope>)
        // expected-error@+1 {{custom op 'pcf.generic' expected 2 dynamic dimension operands for type 'tensor<?x?xi32>', but got 1}}
        -> (tensor<?x?xi32>{%dim0}) {
    pcf.return
  }
  util.return
}

// -----

// Threadgroup scope mismatch with op scope.
util.func private @shared_executor_scope_mismatch(%init: tensor<4x8xf32>) {
  // expected-error@+1 {{threadgroup scope must match op scope}}
  %0 = pcf.shared_executor scope(#pcf.sequential)
    execute(%ref = %init)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<4x8xf32, #pcf.sequential>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}

// -----

// Readwrite sref scope mismatch with op scope.
util.func private @shared_executor_sref_scope_mismatch(%init: tensor<4x8xf32>) {
  // expected-error@+1 {{readwrite sref scope must match op scope}}
  %0 = pcf.shared_executor scope(#pcf.sequential)
    execute(%ref = %init)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<4x8xf32, #pcf.test_scope>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}

// -----

// Tile group: wrong number of block args.
util.func private @tile_group_wrong_num_args(%tg: !pcf.threadgroup<#pcf.sequential>, %k: index) {
  // expected-error@+1 {{expected 2 cluster block arguments but got 1}}
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%only: !pcf.cluster<#pcf.sequential, (0 -> d0), only>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// -----

// Tile group: block arg is not a cluster type.
util.func private @tile_group_non_cluster_arg(%tg: !pcf.threadgroup<#pcf.sequential>) {
  // expected-error@+1 {{block argument must be !pcf.cluster}}
  pcf.shared_executor.tile_group %tg split [[]]
      (%bad: index) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// -----

// Tile group: source has struct elements.
util.func private @tile_group_struct_source(
    %tg: !pcf.threadgroup<#pcf.sequential, {index}>) {
  // expected-error@+1 {{source must not have struct elements}}
  pcf.shared_executor.tile_group %tg split [[]]
      (%c: !pcf.cluster<#pcf.sequential, (0 -> s0), c0>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential, {index}>
  util.return
}

// -----

// Tile group: scope mismatch.
util.func private @tile_group_scope_mismatch(
    %tg: !pcf.threadgroup<#pcf.sequential>, %k: index) {
  // expected-error@+1 {{cluster scope mismatch}}
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%left: !pcf.cluster<#pcf.test_scope, (0 -> d0), left>,
       %right: !pcf.cluster<#pcf.test_scope, (d0 -> s0), right>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// -----

// Tile group: result cluster has struct elements.
util.func private @tile_group_cluster_with_struct(
    %tg: !pcf.threadgroup<#pcf.sequential>) {
  // expected-error@+1 {{result clusters must not have struct elements}}
  pcf.shared_executor.tile_group %tg split [[]]
      (%c: !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {index}, c0>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// -----

// Cluster type: unknown keyword in struct group.
// expected-error@+1 {{expected 'private' or 'shared'}}
util.func private @cluster_unknown_keyword(!pcf.cluster<#pcf.test_scope, (0 -> s0), unknown: {index}, c0>)

// -----

// Cluster type: duplicate keyword in struct group.
// expected-error@+1 {{duplicate keyword 'shared'}}
util.func private @cluster_duplicate_keyword(!pcf.cluster<#pcf.test_scope, (0 -> s0), shared: {index}, shared: {f32}, c0>)

// -----

// run_cluster: result has private types.
util.func private @run_cluster_private_result(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>) {
  // expected-error@+1 {{run_cluster result must not have private types}}
  %r = pcf.shared_executor.run_cluster(%c)[]
      (%s: f32) {
    pcf.cluster_yield %s : f32
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>)
    -> !pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, c0>
  util.return
}

// -----

// run_thread: result has shared types.
util.func private @run_thread_shared_result(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, c0>) {
  // expected-error@+1 {{run_thread result must not have shared types}}
  %r = pcf.shared_executor.run_thread(%c)[]
      (%p: f32)[%tid: index] {
    pcf.cluster_yield %p : f32
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, c0>)
    -> !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>
  util.return
}

// -----

// run_cluster: scope mismatch between sources.
util.func private @run_cluster_scope_mismatch(
    %c0: !pcf.cluster<#pcf.sequential, (0 -> s0), a>,
    %c1: !pcf.cluster<#pcf.test_scope, (0 -> s0), b>) {
  // expected-error@+1 {{all source clusters must have the same scope}}
  pcf.shared_executor.run_cluster(%c0, %c1)[]
      () {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), a>,
       !pcf.cluster<#pcf.test_scope, (0 -> s0), b>)
  util.return
}

// -----

// run_cluster: wrong range value count.
util.func private @run_cluster_wrong_range_count(
    %c: !pcf.cluster<#pcf.sequential, (0 -> d0), c0>,
    %k: index, %extra: index) {
  // expected-error@+1 {{expected 1 range values but got 2}}
  pcf.shared_executor.run_cluster(%c)[%k, %extra]
      () {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> d0), c0>)
  util.return
}

// -----

// run_cluster: block arg type mismatch.
util.func private @run_cluster_arg_mismatch(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>) {
  // expected-error@+1 {{block argument 0 type mismatch}}
  pcf.shared_executor.run_cluster(%c)[]
      (%bad: i32) {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>)
  util.return
}

// -----

// run_cluster: boundsMap mismatch between sources.
util.func private @run_cluster_boundsmap_mismatch(
    %c0: !pcf.cluster<#pcf.sequential, (0 -> s0), a>,
    %c1: !pcf.cluster<#pcf.sequential, (0 -> d0), b>,
    %k: index) {
  // expected-error@+1 {{'pcf.shared_executor.run_cluster' op all source clusters must have the same boundsMap}}
  pcf.shared_executor.run_cluster(%c0, %c1)[%k]
      () {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), a>,
       !pcf.cluster<#pcf.sequential, (0 -> d0), b>)
  util.return
}

// -----

// run_cluster: yield value type mismatch.
util.func private @run_cluster_yield_value_mismatch(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>) {
  %idx = arith.constant 0 : index
  // expected-error@+1 {{'pcf.shared_executor.run_cluster' op yield value types do not match result}}
  %r = pcf.shared_executor.run_cluster(%c)[]
      (%s: f32) {
    pcf.cluster_yield %idx : index
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>)
    -> !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>
  util.return
}

// -----

// run_thread: block arg type mismatch.
util.func private @run_thread_arg_mismatch(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, c0>) {
  // expected-error@+1 {{'pcf.shared_executor.run_thread' op block argument 0 type mismatch}}
  pcf.shared_executor.run_thread(%c)[]
      (%bad: i32)[%tid: index] {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, c0>)
  util.return
}

// -----

// run_cluster: yield with operands but no result.
util.func private @run_cluster_yield_with_operands_no_result(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>) {
  // expected-error@+1 {{cluster_yield must have no operands when parent has no result}}
  pcf.shared_executor.run_cluster(%c)[]
      (%s: f32) {
    pcf.cluster_yield %s : f32
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>)
  util.return
}

// -----

// run_cluster: result scope mismatch.
util.func private @run_cluster_result_scope_mismatch(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>) {
  // expected-error@+1 {{result cluster scope mismatch}}
  %r = pcf.shared_executor.run_cluster(%c)[]
      (%s: f32) {
    pcf.cluster_yield %s : f32
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>)
    -> !pcf.cluster<#pcf.test_scope, (0 -> s0), shared: {f32}, c0>
  util.return
}

// -----

// run_cluster: result boundsMap mismatch.
util.func private @run_cluster_result_boundsmap_mismatch(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>) {
  // expected-error@+1 {{result cluster boundsMap mismatch}}
  %r = pcf.shared_executor.run_cluster(%c)[]
      (%s: f32) {
    pcf.cluster_yield %s : f32
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, c0>)
    -> !pcf.cluster<#pcf.sequential, (0 -> d0), shared: {f32}, c0>
  util.return
}

// -----

// run_thread: wrong number of block args (too many thread IDs).
util.func private @run_thread_wrong_thread_id_count(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, c0>) {
  // expected-error@+1 {{expected 2 block arguments but got 3}}
  pcf.shared_executor.run_thread(%c)[]
      (%p: f32)[%tid0: index, %tid1: index] {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, c0>)
  util.return
}

// -----

// run_cluster: ID mismatch between sources.
util.func private @run_cluster_id_mismatch(
    %c0: !pcf.cluster<#pcf.sequential, (0 -> s0), a>,
    %c1: !pcf.cluster<#pcf.sequential, (0 -> s0), b>) {
  // expected-error@+1 {{all source clusters must have the same ID}}
  pcf.shared_executor.run_cluster(%c0, %c1)[]
      () {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), a>,
       !pcf.cluster<#pcf.sequential, (0 -> s0), b>)
  util.return
}

// -----

// run_cluster: result cluster ID mismatch.
util.func private @run_cluster_result_id_mismatch(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, a>) {
  %cst = arith.constant 0.0 : f32
  // expected-error@+1 {{result cluster ID does not match source cluster ID}}
  %r = pcf.shared_executor.run_cluster(%c)[]
      (%s: f32) {
    pcf.cluster_yield %s : f32
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, a>)
    -> !pcf.cluster<#pcf.sequential, (0 -> s0), shared: {f32}, wrong>
  util.return
}

// -----

// Tile group namespace: duplicate leaf names.
util.func private @tile_group_duplicate_leaf(
    %tg: !pcf.threadgroup<#pcf.sequential>, %k: index) {
  // expected-error@+1 {{duplicate leaf symbol name 'dup' in namespace}}
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%a: !pcf.cluster<#pcf.sequential, (0 -> d0), dup>,
       %b: !pcf.cluster<#pcf.sequential, (d0 -> s0), dup>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// -----

// Tile group namespace: leaf-only ID in named namespace.
util.func private @tile_group_leaf_in_named_ns(
    %tg: !pcf.threadgroup<#pcf.sequential>, %k: index) {
  // expected-error@+1 {{cluster ID 'left' must be qualified with namespace name 'tg'}}
  pcf.shared_executor.tile_group %tg ns(tg) split [[%k]]
      (%a: !pcf.cluster<#pcf.sequential, (0 -> d0), left>,
       %b: !pcf.cluster<#pcf.sequential, (d0 -> s0), tg.right>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// -----

// Tile group namespace: wrong namespace prefix.
util.func private @tile_group_wrong_ns_prefix(
    %tg: !pcf.threadgroup<#pcf.sequential>, %k: index) {
  // expected-error@+1 {{cluster ID first segment 'wrong' does not match namespace name 'tg'}}
  pcf.shared_executor.tile_group %tg ns(tg) split [[%k]]
      (%a: !pcf.cluster<#pcf.sequential, (0 -> d0), wrong.left>,
       %b: !pcf.cluster<#pcf.sequential, (d0 -> s0), tg.right>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// -----

// to_sref: shape mismatch.
util.func private @to_sref_shape_mismatch(%input: tensor<128x256xf16>) {
  // expected-error@+1 {{result sref shape}}
  %sref = pcf.to_sref %input : tensor<128x256xf16> -> !pcf.sref<64x256xf16, #pcf.test_scope>
  util.return
}

// -----

// to_sref: element type mismatch.
util.func private @to_sref_eltype_mismatch(%input: tensor<128x256xf16>) {
  // expected-error@+1 {{result sref element type}}
  %sref = pcf.to_sref %input : tensor<128x256xf16> -> !pcf.sref<128x256xf32, #pcf.test_scope>
  util.return
}

// -----

// Note: The "expected at least one source cluster" verifier check on
// run_cluster/run_thread is only reachable via programmatic construction,
// because the parser requires at least one source type in the type list.

// run_thread: source ID mismatch between sources.
util.func private @run_thread_id_mismatch(
    %c0: !pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, a>,
    %c1: !pcf.cluster<#pcf.sequential, (0 -> s0), private: {i32}, b>) {
  // expected-error@+1 {{'pcf.shared_executor.run_thread' op all source clusters must have the same ID}}
  pcf.shared_executor.run_thread(%c0, %c1)[]
      (%p0: f32, %p1: i32)[%tid: index] {
    pcf.cluster_yield
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, a>,
       !pcf.cluster<#pcf.sequential, (0 -> s0), private: {i32}, b>)
  util.return
}

// -----

// run_thread: result cluster ID mismatch.
util.func private @run_thread_result_id_mismatch(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, a>) {
  // expected-error@+1 {{'pcf.shared_executor.run_thread' op result cluster ID does not match source cluster ID}}
  %r = pcf.shared_executor.run_thread(%c)[]
      (%p: f32)[%tid: index] {
    pcf.cluster_yield %p : f32
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, a>)
    -> !pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, wrong>
  util.return
}

// -----

// run_thread: yield value type mismatch.
util.func private @run_thread_yield_value_mismatch(
    %c: !pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, c0>) {
  %idx = arith.constant 0 : index
  // expected-error@+1 {{'pcf.shared_executor.run_thread' op yield value types do not match result}}
  %r = pcf.shared_executor.run_thread(%c)[]
      (%p: f32)[%tid: index] {
    pcf.cluster_yield %idx : index
  } : (!pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, c0>)
    -> !pcf.cluster<#pcf.sequential, (0 -> s0), private: {f32}, c0>
  util.return
}

// -----

// Tile group namespace: qualified ID in anonymous namespace.
util.func private @tile_group_qualified_in_anon_ns(
    %tg: !pcf.threadgroup<#pcf.sequential>, %k: index) {
  // expected-error@+1 {{cluster ID 'tg.left' must be leaf-only in anonymous namespace}}
  pcf.shared_executor.tile_group %tg split [[%k]]
      (%a: !pcf.cluster<#pcf.sequential, (0 -> d0), tg.left>,
       %b: !pcf.cluster<#pcf.sequential, (d0 -> s0), right>) {
    pcf.return
  } : !pcf.threadgroup<#pcf.sequential>
  util.return
}

// -----

// shared_executor: last region arg is not threadgroup type.
util.func private @shared_executor_last_arg_not_tg(%init: tensor<4x8xf32>) {
  // expected-error@+1 {{expected last region argument to be !pcf.threadgroup, got 'index'}}
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    execute(%ref = %init)
        [%tg: index]
         : (!pcf.sref<4x8xf32, #pcf.test_scope>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}

// -----

// shared_executor: readwrite sref shape mismatch with result.
util.func private @shared_executor_rw_shape_mismatch(%init: tensor<4x8xf32>) {
  // expected-error@+1 {{readwrite sref at index 0 shape mismatch with result type}}
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    execute(%ref = %init)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<8x4xf32, #pcf.test_scope>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}

// -----

// shared_executor: readwrite sref element type mismatch with result.
util.func private @shared_executor_rw_eltype_mismatch(%init: tensor<4x8xf32>) {
  // expected-error@+1 {{readwrite sref at index 0 element type mismatch with result type}}
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    execute(%ref = %init)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<4x8xf16, #pcf.test_scope>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}

// -----

// Note: The "readwrite init type mismatch" verifier check is only reachable
// via programmatic construction, because the parser resolves readwrite init
// operands against result types (SharedExecutorOp::parse line 916), so a
// type mismatch causes a parse-time error before the verifier runs.

// init_subscope: input with struct fields causes parser type mismatch.
// The parser derives the source type from the result type (same scope, no
// struct fields). If the source SSA value has struct fields, the parser
// reports a type mismatch before the verifier runs.
util.func private @init_subscope_input_with_struct(
    // expected-note@below {{prior use here}}
    %tg: !pcf.threadgroup<#pcf.sequential, {index}>) {
  // expected-error@+1 {{expects different type than prior uses}}
  %tg2 = pcf.init_subscope %tg {
    %alloc = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.sequential>
    pcf.yield %alloc : !pcf.sref<128x64xf16, #pcf.sequential>
  } -> !pcf.threadgroup<#pcf.sequential, {!pcf.sref<128x64xf16, #pcf.sequential>}>
  util.return
}

// -----

// init_subscope: scope mismatch between input and result causes parser
// type mismatch. The parser derives source type from result type scope.
util.func private @init_subscope_scope_mismatch(
    // expected-note@below {{prior use here}}
    %tg: !pcf.threadgroup<#pcf.test_scope>) {
  // expected-error@+1 {{expects different type than prior uses}}
  %tg2 = pcf.init_subscope %tg {
    %alloc = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.sequential>
    pcf.yield %alloc : !pcf.sref<128x64xf16, #pcf.sequential>
  } -> !pcf.threadgroup<#pcf.sequential, {!pcf.sref<128x64xf16, #pcf.sequential>}>
  util.return
}

// -----

// init_subscope: yield type mismatch with result struct fields.
util.func private @init_subscope_yield_mismatch(
    %tg: !pcf.threadgroup<#pcf.sequential>) {
  // expected-error@+1 {{yielded type '!pcf.sref<128x64xf16, #pcf.sequential>' at index 0 does not match result struct field type 'index'}}
  %tg2 = pcf.init_subscope %tg {
    %alloc = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.sequential>
    pcf.yield %alloc : !pcf.sref<128x64xf16, #pcf.sequential>
  } -> !pcf.threadgroup<#pcf.sequential, {index}>
  util.return
}

// -----

// telescope: first result not a threadgroup.
util.func private @telescope_first_not_tg(
    %tg: !pcf.threadgroup<#pcf.sequential>, %tid: index) {
  // expected-error@+1 {{first result must be !pcf.threadgroup, got 'index'}}
  %bad = pcf.telescope %tg[%tid]
      : !pcf.threadgroup<#pcf.sequential> -> index
  util.return
}

// -----

// telescope: result count mismatch with struct fields.
util.func private @telescope_result_count_mismatch(
    %tg: !pcf.threadgroup<#pcf.sequential, {index, f32}>, %tid: index) {
  // expected-error@+1 {{expected 3 results (1 threadgroup + 2 struct fields) but got 2}}
  %child_tg, %idx = pcf.telescope %tg[%tid]
      : !pcf.threadgroup<#pcf.sequential, {index, f32}>
     -> (!pcf.threadgroup<#pcf.test_scope>, index)
  util.return
}

// -----

// telescope: struct field type mismatch.
util.func private @telescope_struct_type_mismatch(
    %tg: !pcf.threadgroup<#pcf.sequential, {index}>, %tid: index) {
  // expected-error@+1 {{result type 'f32' at index 1 does not match source struct field type 'index'}}
  %child_tg, %bad = pcf.telescope %tg[%tid]
      : !pcf.threadgroup<#pcf.sequential, {index}>
     -> (!pcf.threadgroup<#pcf.test_scope>, f32)
  util.return
}

// -----

// shared_executor: readonly sref scope mismatch with op scope.
util.func private @shared_executor_readonly_scope_mismatch(
    %input: tensor<4x8xf32>, %output: tensor<4x8xf32>) {
  // expected-error@+1 {{readonly sref scope must match op scope}}
  %0 = pcf.shared_executor scope(#pcf.test_scope)
    execute(%in_ref <- %input, %out_ref = %output)
        [%tg: !pcf.threadgroup<#pcf.test_scope>]
         : (!pcf.sref<4x8xf32, #pcf.sequential>,
            !pcf.sref<4x8xf32, #pcf.test_scope>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}
