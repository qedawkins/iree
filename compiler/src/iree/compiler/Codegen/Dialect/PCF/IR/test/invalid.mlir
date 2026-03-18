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

// Readonly sref scope mismatch with op scope.
util.func private @shared_executor_readonly_scope_mismatch(
    %input: tensor<4x8xf32>, %output: tensor<4x8xf32>) {
  // expected-error@+1 {{readonly sref scope must match op scope}}
  %0 = pcf.shared_executor scope(#pcf.sequential)
    execute(%in_ref <- %input, %out_ref = %output)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<4x8xf32, #pcf.test_scope>,
            !pcf.sref<4x8xf32, #pcf.sequential>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}

// -----

// Readwrite sref shape mismatch with result type.
util.func private @shared_executor_readwrite_shape_mismatch(
    %init: tensor<4x8xf32>) {
  // expected-error@+1 {{readwrite sref at index 0 shape mismatch with result type}}
  %0 = pcf.shared_executor scope(#pcf.sequential)
    execute(%ref = %init)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<8x8xf32, #pcf.sequential>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}

// -----

// Readwrite sref element type mismatch with result type.
util.func private @shared_executor_readwrite_eltype_mismatch(
    %init: tensor<4x8xf32>) {
  // expected-error@+1 {{readwrite sref at index 0 element type mismatch with result type}}
  %0 = pcf.shared_executor scope(#pcf.sequential)
    execute(%ref = %init)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<4x8xf16, #pcf.sequential>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}

// -----

// Last region argument not being !pcf.threadgroup type.
util.func private @shared_executor_not_threadgroup(%init: tensor<4x8xf32>) {
  // expected-error@+1 {{expected last region argument to be !pcf.threadgroup, got 'index'}}
  pcf.shared_executor scope(#pcf.sequential)
    execute[%tg: index] {
    pcf.return
  }
  util.return
}

// -----

// Initializer yield count mismatch with leading args.
util.func private @shared_executor_yield_count_mismatch(
    %output: tensor<4x8xf32>) {
  // expected-error@+1 {{initializer yield operand count (2) does not match num_leading_args (1)}}
  %0 = pcf.shared_executor scope(#pcf.sequential)
    initialize {
      %smem = pcf.alloc() : !pcf.sref<128x64xf16, #pcf.sequential>
      %smem2 = pcf.alloc() : !pcf.sref<64x64xf16, #pcf.sequential>
      pcf.yield %smem, %smem2 : !pcf.sref<128x64xf16, #pcf.sequential>, !pcf.sref<64x64xf16, #pcf.sequential>
    } -> (%smem: !pcf.sref<128x64xf16, #pcf.sequential>)
    execute(%ref = %output)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<4x8xf32, #pcf.sequential>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}

// -----

// Initializer yield type mismatch with leading args.
util.func private @shared_executor_yield_type_mismatch(
    %output: tensor<4x8xf32>) {
  // expected-error@+1 {{initializer yield type '!pcf.sref<64x64xf16, #pcf.sequential>' at index 0 does not match leading arg type '!pcf.sref<128x64xf16, #pcf.sequential>'}}
  %0 = pcf.shared_executor scope(#pcf.sequential)
    initialize {
      %smem = pcf.alloc() : !pcf.sref<64x64xf16, #pcf.sequential>
      pcf.yield %smem : !pcf.sref<64x64xf16, #pcf.sequential>
    } -> (%smem: !pcf.sref<128x64xf16, #pcf.sequential>)
    execute(%ref = %output)
        [%tg: !pcf.threadgroup<#pcf.sequential>]
         : (!pcf.sref<4x8xf32, #pcf.sequential>)
        -> (tensor<4x8xf32>) {
    pcf.return
  }
  util.return
}
