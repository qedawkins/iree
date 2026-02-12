// RUN: iree-opt --split-input-file %s | FileCheck %s

// ============================================================================
// Flow executable export with scratch_size region.
// Mirrors the workgroups region pattern but returns a single index (bytes).
// ============================================================================

// CHECK-LABEL: @export_with_scratch_size
flow.executable @export_with_scratch_size {
  // CHECK: flow.executable.export public @dispatch
  flow.executable.export @dispatch
      // CHECK-SAME: scratch_size(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index) -> index
      scratch_size(%arg0: index, %arg1: index) -> index {
        // CHECK: %[[C1024:.+]] = arith.constant 1024
        %c1024 = arith.constant 1024 : index
        // CHECK: flow.return %[[C1024]]
        flow.return %c1024 : index
      }
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}

// -----

// Export with both workgroups and scratch_size regions.
// CHECK-LABEL: @export_with_both_regions
flow.executable @export_with_both_regions {
  // CHECK: flow.executable.export public @dispatch
  flow.executable.export @dispatch
      // CHECK-SAME: workgroups
      workgroups(%wl0: index, %wl1: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        flow.return %wl0, %wl1, %c1 : index, index, index
      }
      // CHECK: scratch_size(%{{.+}}: index, %{{.+}}: index) -> index
      scratch_size(%arg0: index, %arg1: index) -> index {
        // Scratch = num_output_tiles * tile_bytes.
        %c64 = arith.constant 64 : index
        %c4 = arith.constant 4 : index
        %tile_bytes = arith.muli %c64, %c64 : index
        %tile_bytes_f32 = arith.muli %tile_bytes, %c4 : index
        %num_tiles = arith.muli %arg0, %arg1 : index
        %total = arith.muli %num_tiles, %tile_bytes_f32 : index
        flow.return %total : index
      }
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}

// -----

// Export without scratch_size (backwards compatibility - most exports).
// CHECK-LABEL: @export_no_scratch_size
flow.executable @export_no_scratch_size {
  // CHECK: flow.executable.export public @dispatch
  flow.executable.export @dispatch
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}

// -----

// flow.executable.scratch_size calling op.
// CHECK-LABEL: @scratch_size_call
flow.executable @scratch_size_call_exe {
  flow.executable.export @dispatch
      scratch_size(%arg0: index) -> index {
        %c4096 = arith.constant 4096 : index
        flow.return %c4096 : index
      }
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}
util.func public @scratch_size_call(%wl: index) -> index {
  // CHECK: %[[SIZE:.+]] = flow.executable.scratch_size @scratch_size_call_exe::@dispatch[%{{.+}}] : index
  %size = flow.executable.scratch_size @scratch_size_call_exe::@dispatch[%wl] : index
  // CHECK: util.return %[[SIZE]]
  util.return %size : index
}
