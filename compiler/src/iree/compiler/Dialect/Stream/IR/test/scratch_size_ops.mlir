// RUN: iree-opt --split-input-file %s --verify-diagnostics | FileCheck %s

// ============================================================================
// Stream executable export with scratch_size region.
// Same pattern as Flow but at Stream dialect level.
// ============================================================================

// CHECK-LABEL: stream.executable private @executable_with_scratch_size
stream.executable private @executable_with_scratch_size {
  // CHECK-NEXT: stream.executable.export public @dispatch
  stream.executable.export public @dispatch
      // CHECK-SAME: scratch_size(%[[A0:.+]]: index, %[[A1:.+]]: index, %[[A2:.+]]: index) -> index
      scratch_size(%arg0: index, %arg1: index, %arg2: index) -> index {
        // CHECK-NEXT: %[[C:.+]] = arith.constant 8192
        %c8192 = arith.constant 8192 : index
        // CHECK-NEXT: stream.return %[[C]]
        stream.return %c8192 : index
      }
  builtin.module {
    util.func private @dispatch() {
      util.return
    }
  }
}

// -----

// Both workgroups and scratch_size.
// CHECK-LABEL: stream.executable private @executable_with_both
stream.executable private @executable_with_both {
  stream.executable.export public @dispatch
      // CHECK: workgroups
      workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
        stream.return %arg0, %arg1, %arg2 : index, index, index
      }
      // CHECK: scratch_size
      scratch_size(%s0: index, %s1: index, %s2: index) -> index {
        %c4096 = arith.constant 4096 : index
        stream.return %c4096 : index
      }
  builtin.module {
    util.func private @dispatch() {
      util.return
    }
  }
}

// -----

// Error: scratch_size region must return exactly one index.
stream.executable private @bad_scratch_size {
  // expected-error @+1 {{scratch_size region must return exactly one index value}}
  stream.executable.export public @dispatch
      scratch_size(%arg0: index) -> (index, index) {
        stream.return %arg0, %arg0 : index, index
      }
  builtin.module {
    util.func private @dispatch() {
      util.return
    }
  }
}

// -----

// stream.executable.scratch_size calling op.
stream.executable private @exe_for_call {
  stream.executable.export public @dispatch
      scratch_size(%arg0: index, %arg1: index, %arg2: index) -> index {
        %c2048 = arith.constant 2048 : index
        stream.return %c2048 : index
      }
  builtin.module {
    util.func private @dispatch() {
      util.return
    }
  }
}

// CHECK-LABEL: @stream_scratch_size_call
util.func private @stream_scratch_size_call(%wl0: index, %wl1: index, %wl2: index) -> index {
  // CHECK: %[[SIZE:.+]] = stream.executable.scratch_size @exe_for_call::@dispatch[%{{.+}}, %{{.+}}, %{{.+}}]
  %size = stream.executable.scratch_size @exe_for_call::@dispatch[%wl0, %wl1, %wl2] : index
  // CHECK: util.return %[[SIZE]]
  util.return %size : index
}
