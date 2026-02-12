// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @ex {
  hal.executable.variant public @variant target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      // Opaque at this point (in some target-specific dialects).
    }
  }
}

// CHECK-LABEL: @calculateWorkgroups
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:  %[[WORKLOAD_0:.+]]: index, %[[WORKLOAD_1:.+]]: index, %[[WORKLOAD_2:.+]]: index)
util.func public @calculateWorkgroups(%device: !hal.device, %workload_0: index, %workload_1: index, %workload_2: index) -> (index, index, index) {
  // CHECK-DAG: %[[WORKGROUP_YZ:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[WORKGROUP_X:.+]] = affine.apply
  %workgroups:3 = hal.executable.calculate_workgroups
      device(%device : !hal.device)
      target(@ex::@variant::@dispatch)
      workload([%workload_0, %workload_1, %workload_2]) : index, index, index
  // CHECK: util.return %[[WORKGROUP_X]], %[[WORKGROUP_YZ]], %[[WORKGROUP_YZ]]
  util.return %workgroups#0, %workgroups#1, %workgroups#2 : index, index, index
}

// -----

#pipeline_layout_scratch = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @ex_scratch {
  hal.executable.variant public @variant target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout_scratch)
      count(%device: !hal.device, %arg0: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %arg0, %c1, %c1 : index, index, index
      }
      scratch(%device: !hal.device, %arg0: index) -> index {
        %c4 = arith.constant 4 : index
        %scratch = arith.muli %arg0, %c4 : index
        hal.return %scratch : index
      }
    builtin.module {
      // Opaque at this point (in some target-specific dialects).
    }
  }
}

// CHECK-LABEL: @calculateScratchSize
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[WORKLOAD:.+]]: index)
util.func public @calculateScratchSize(%device: !hal.device, %workload: index) -> index {
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[SCRATCH:.+]] = arith.muli %[[WORKLOAD]], %[[C4]]
  %scratch_size = hal.executable.calculate_scratch_size
      device(%device : !hal.device)
      target(@ex_scratch::@variant::@dispatch)
      workload([%workload]) : index
  // CHECK: util.return %[[SCRATCH]]
  util.return %scratch_size : index
}
