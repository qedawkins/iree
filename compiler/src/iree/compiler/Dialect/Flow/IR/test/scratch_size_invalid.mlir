// RUN: iree-opt --split-input-file %s --verify-diagnostics

// scratch_size region returns wrong number of values (0 instead of 1).
flow.executable @bad_scratch_size_no_return {
  // expected-error@+1 {{scratch_size region must return exactly one index value}}
  flow.executable.export @dispatch
      scratch_size(%arg0: index) -> () {
        flow.return
      }
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}

// -----

// scratch_size region returns wrong type (i32 instead of index).
flow.executable @bad_scratch_size_wrong_type {
  // expected-error@+1 {{scratch_size region must return exactly one index value}}
  flow.executable.export @dispatch
      scratch_size(%arg0: index) -> i32 {
        %c0 = arith.constant 0 : i32
        flow.return %c0 : i32
      }
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}

// -----

// scratch_size region returns multiple values.
flow.executable @bad_scratch_size_multi_return {
  // expected-error@+1 {{scratch_size region must return exactly one index value}}
  flow.executable.export @dispatch
      scratch_size(%arg0: index) -> (index, index) {
        flow.return %arg0, %arg0 : index, index
      }
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}
