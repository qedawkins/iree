// RUN: iree-opt --split-input-file %s --verify-diagnostics

util.func private @scope_mismatch() {
// expected-error@+1 {{expected region ref argument to be of type !pcf.sref with scope #pcf.sequential}}
  pcf.generic scope(#pcf.sequential)
    initialize(%ref)[%num_threads: index]
            : (!pcf.sref<?xi32, #pcf.dummy_scope>)
           -> (tensor<?xi32>) {
    pcf.return
  }
  util.return
}

// -----

util.func private @token_scope_mismatch(%0: tensor<?xi32>) {
// expected-error@+1 {{expected region token argument to be of type !pcf.token with scope #pcf.sequential}}
  pcf.generic scope(#pcf.sequential)
    initialize(%ref[%token: !pcf.token<#pcf.dummy_scope>])[%num_threads: index]
            : (!pcf.sref<?xi32, #pcf.sequential>)
           -> (tensor<?xi32>) {
    pcf.return
  }
  util.return
}

// -----

// expected-note@+1 {{prior use here}}
util.func private @init_type_mismatch(%0: tensor<3xi32>) {
  pcf.generic scope(#pcf.dummy_scope)
// expected-error@+1 {{expects different type than prior uses: 'tensor<?xi32>' vs 'tensor<3xi32>'}}
    initialize(%ref = %0)[%num_threads: index]
            : (!pcf.sref<?xi32, #pcf.dummy_scope>)
           -> (tensor<?xi32>) {
    pcf.return
  }
  util.return
}

// -----

util.func private @arg_shape_mismatch() {
// expected-error@+1 {{region arg at index 0 with type '!pcf.sref<3xi32, #pcf.dummy_scope>' shape mismatch with tied result of type 'tensor<?xi32>'}}
  pcf.generic scope(#pcf.dummy_scope)
    initialize(%ref)[%num_threads: index]
            : (!pcf.sref<3xi32, #pcf.dummy_scope>)
           -> (tensor<?xi32>) {
    pcf.return
  }
  util.return
}

// -----

util.func private @arg_eltype_mismatch() {
// expected-error@+1 {{region arg at index 0 element type mismatch of 'f32' vs 'i32'}}
  pcf.generic scope(#pcf.dummy_scope)
    initialize(%ref)[%num_threads: index]
            : (!pcf.sref<?xf32, #pcf.dummy_scope>)
           -> (tensor<?xi32>) {
    pcf.return
  }
  util.return
}
