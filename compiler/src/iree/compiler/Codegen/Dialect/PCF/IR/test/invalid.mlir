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

// expected-error@+1 {{invalid accessor mode 'writeonly'}}
util.func private @bad_accessor(!pcf.sref<128x128xf32, sync(#pcf.test_scope), writeonly>)

// -----

// expected-error@+1 {{bundle ID must be non-negative}}
util.func private @negative_bundle_id(!pcf.bundle<#pcf.test_scope, -1>)

// -----

// Bundle scope mismatch with form_bundles scope.
util.func private @form_bundles_scope_mismatch() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    // expected-error @+1 {{block argument 0 has bundle scope '#pcf.test_scope' but expected '#pcf.sequential'}}
    pcf.form_bundles #pcf.sequential sizes [1, 3] {
    ^bb0(%b0: !pcf.bundle<#pcf.test_scope, 0>,
         %b1: !pcf.bundle<#pcf.sequential, 1>):
      pcf.yield
    }
    pcf.return
  }
  util.return
}

// -----

// Bundle ID mismatch.
util.func private @form_bundles_id_mismatch() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    // expected-error @+1 {{block argument 0 has bundle ID 1 but expected 0}}
    pcf.form_bundles #pcf.sequential sizes [1, 3] {
    ^bb0(%b0: !pcf.bundle<#pcf.sequential, 1>,
         %b1: !pcf.bundle<#pcf.sequential, 1>):
      pcf.yield
    }
    pcf.return
  }
  util.return
}

// -----

// Wrong number of block arguments.
util.func private @form_bundles_wrong_arg_count() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    // expected-error @+1 {{expected 2 block arguments (one per bundle) but got 3}}
    pcf.form_bundles #pcf.sequential sizes [1, 3] {
    ^bb0(%b0: !pcf.bundle<#pcf.sequential, 0>,
         %b1: !pcf.bundle<#pcf.sequential, 1>,
         %b2: !pcf.bundle<#pcf.sequential, 2>):
      pcf.yield
    }
    pcf.return
  }
  util.return
}

// -----

// execute_as with duplicate bundle IDs.
util.func private @execute_as_duplicate_bundle() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    pcf.form_bundles #pcf.sequential sizes [1, 3] {
    ^bb0(%b0: !pcf.bundle<#pcf.sequential, 0>,
         %b1: !pcf.bundle<#pcf.sequential, 1>):
      // expected-error @+1 {{duplicate bundle operand with ID 0}}
      pcf.execute_as [%b0 : !pcf.bundle<#pcf.sequential, 0>,
                      %b0 : !pcf.bundle<#pcf.sequential, 0>] {
        pcf.return
      }
      pcf.yield
    }
    pcf.return
  }
  util.return
}

// -----

// Write to readonly sref.
util.func private @write_to_readonly_sref(
    %source: tensor<128x64xf16>) {
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
        : (!pcf.sref<128x64xf16, #pcf.test_scope, readonly>)
        -> (tensor<128x64xf16>) {
    // expected-error @+1 {{'pcf.write_slice' op cannot write to readonly sref}}
    pcf.write_slice %source into %ref [0, 0] [128, 64] [1, 1]
        : tensor<128x64xf16> into !pcf.sref<128x64xf16, #pcf.test_scope, readonly>
    pcf.return
  }
  util.return
}

// -----

// FormBundlesOp scope doesn't match parent GenericOp scope.
util.func private @form_bundles_parent_scope_mismatch() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    // expected-error @+1 {{'pcf.form_bundles' op scope does not match parent generic scope}}
    pcf.form_bundles #pcf.test_scope sizes [1] {
    ^bb0(%b0: !pcf.bundle<#pcf.test_scope, 0>):
      pcf.yield
    }
    pcf.return
  }
  util.return
}

// -----

// FormBundlesOp with bundle size 0.
util.func private @form_bundles_size_zero() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    // expected-error @+1 {{'pcf.form_bundles' op bundle size at index 0 must be >= 1, got 0}}
    pcf.form_bundles #pcf.sequential sizes [0, 3] {
    ^bb0(%b0: !pcf.bundle<#pcf.sequential, 0>,
         %b1: !pcf.bundle<#pcf.sequential, 1>):
      pcf.yield
    }
    pcf.return
  }
  util.return
}

// -----

// SharedExecutorOp capture ref arg not readonly.
func.func @shared_executor_capture_not_readonly(
    %src: tensor<128x64xf16>,
    %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // expected-error @+1 {{'pcf.shared_executor' op capture ref arg at index 0 must have readonly accessor mode}}
  %result = pcf.shared_executor scope(#pcf.sequential)
      execute(%src_ref from %src, %out_ref = %init)[%count: index]
          : (!pcf.sref<128x64xf16, #pcf.sequential, readwrite>,
             !pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}

// -----

// SharedExecutorOp tied ref arg not readwrite.
func.func @shared_executor_tied_not_readwrite(
    %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // expected-error @+1 {{'pcf.shared_executor' op tied ref arg at index 0 must have readwrite accessor mode}}
  %result = pcf.shared_executor scope(#pcf.sequential)
      execute(%ref = %init)[%count: index]
          : (!pcf.sref<128x128xf32, #pcf.sequential, readonly>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}

// -----

// SharedExecutorOp count_dims_per_scope entry is zero.
func.func @shared_executor_count_dims_zero(
    %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // expected-error @+1 {{'pcf.shared_executor' op count_dims_per_scope[0] must be >= 1, got 0}}
  %result = pcf.shared_executor scope(#pcf.sequential)
      execute(%ref = %init)[]
          : (!pcf.sref<128x128xf32, #pcf.sequential, readwrite>)
          -> (tensor<128x128xf32>) {
    pcf.return
  }
  return %result : tensor<128x128xf32>
}
