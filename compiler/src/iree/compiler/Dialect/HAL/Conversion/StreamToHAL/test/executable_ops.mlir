// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// Tests that stream.executable.scratch_size converts to inlined scratch_size
// region code via hal.executable.calculate_scratch_size.
// By the time ConvertToHAL runs, MaterializeInterfaces has already converted
// stream.executable -> hal.executable and updated the entry point ref to
// 3-level format (executable::variant::export).

#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @ex {
  hal.executable.variant public @rocm target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    scratch(%device: !hal.device) -> index {
      %c16384 = arith.constant 16384 : index
      hal.return %c16384 : index
    }
    builtin.module {
    }
  }
}

util.global private @device : !hal.device

// CHECK-LABEL: @scratch_size_query
util.func public @scratch_size_query(%workload: index) -> index attributes {
  stream.affinity = #hal.device.affinity<@device>
} {
  // The 3-level entry point ref resolves to the hal.executable.export.
  // ConvertToHAL converts stream.executable.scratch_size ->
  // hal.executable.calculate_scratch_size, then inlines the scratch_size
  // region.
  // CHECK: %[[SIZE:.+]] = arith.constant 16384 : index
  %size = stream.executable.scratch_size @ex::@rocm::@dispatch[%workload] : index
  // CHECK: util.return %[[SIZE]]
  util.return %size : index
}

// CHECK-NOT: stream.executable.scratch_size
