// RUN: iree-opt --split-input-file %s | FileCheck %s

#executable_target_format = #hal.executable.target<"backend", "format">

// ============================================================================
// HAL executable export with scratch_size region.
// Note: HAL uses %device as first arg (like count region).
// ============================================================================

// CHECK-LABEL: @export_with_scratch_size
hal.executable @export_with_scratch_size {
  hal.executable.variant @backend target(#executable_target_format) {
    // CHECK: hal.executable.export public @entry0
    hal.executable.export public @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>,
      #hal.pipeline.binding<storage_buffer>
    ]>)
    // CHECK-SAME: scratch(%[[DEV:.+]]: !hal.device, %[[WL0:.+]]: index, %[[WL1:.+]]: index) -> index
    scratch(%device: !hal.device, %workload0: index, %workload1: index) -> index {
      // CHECK: %[[C:.+]] = arith.constant 16384
      %c16384 = arith.constant 16384 : index
      // CHECK: hal.return %[[C]]
      hal.return %c16384 : index
    }
  }
}

// -----

#executable_target_format = #hal.executable.target<"backend", "format">

// Both count and scratch_size on same export.
// CHECK-LABEL: @export_with_count_and_scratch_size
hal.executable @export_with_count_and_scratch_size {
  hal.executable.variant @backend target(#executable_target_format) {
    hal.executable.export public @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>,
      #hal.pipeline.binding<storage_buffer>
    ]>)
    // CHECK: count(
    count(%device: !hal.device, %wl0: index, %wl1: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %wl0, %wl1, %c1 : index, index, index
    }
    // CHECK: scratch(
    scratch(%dev2: !hal.device, %sl0: index, %sl1: index) -> index {
      %c8192 = arith.constant 8192 : index
      hal.return %c8192 : index
    }
  }
}

// -----

#executable_target_format = #hal.executable.target<"backend", "format">

// All three regions: condition + count + scratch_size.
// CHECK-LABEL: @export_with_all_three_regions
hal.executable @export_with_all_three_regions {
  hal.executable.variant @backend target(#executable_target_format) {
    hal.executable.export public @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>)
    // CHECK: condition(
    condition(%device: !hal.device, %workload: index) -> i1 {
      %c1024 = arith.constant 1024 : index
      %use_me = arith.cmpi slt, %workload, %c1024 : index
      hal.return %use_me : i1
    }
    fallback(@fallback)
    // CHECK: count(
    count(%dev2: !hal.device, %wl0: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %wl0, %c1, %c1 : index, index, index
    }
    // CHECK: scratch(
    scratch(%dev3: !hal.device, %sl0: index) -> index {
      %c4096 = arith.constant 4096 : index
      hal.return %c4096 : index
    }
    hal.executable.export public @fallback ordinal(1) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>)
  }
}
