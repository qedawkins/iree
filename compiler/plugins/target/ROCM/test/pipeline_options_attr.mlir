// RUN: iree-opt --split-input-file %s | FileCheck %s

// Default options: all fields at default values, prints as empty <>.
// CHECK-LABEL: @default_options
hal.executable @default_options {
  // CHECK: options(#rocm.pipeline_options<>)
  hal.executable.variant @rocm
      target(#hal.executable.target<"rocm", "rocm-hsaco-fb">)
      options(#rocm.pipeline_options<>) {
    hal.executable.export public @entry ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>)
    builtin.module {
      func.func @entry() { return }
    }
  }
}

// -----

// compile_to only: compile_from stays at default, so only compile_to prints.
// CHECK-LABEL: @compile_to_phase1
hal.executable @compile_to_phase1 {
  // CHECK: options(#rocm.pipeline_options<compile_to = configuration_controlled_translation>)
  hal.executable.variant @rocm
      target(#hal.executable.target<"rocm", "rocm-hsaco-fb">)
      options(#rocm.pipeline_options<compile_to = configuration_controlled_translation>) {
    hal.executable.export public @entry ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>)
    builtin.module {
      func.func @entry() { return }
    }
  }
}

// -----

// compile_from only: compile_to stays at default, so only compile_from prints.
// CHECK-LABEL: @compile_from_phase2
hal.executable @compile_from_phase2 {
  // CHECK: options(#rocm.pipeline_options<compile_from = llvm_translation>)
  hal.executable.variant @rocm
      target(#hal.executable.target<"rocm", "rocm-hsaco-fb">)
      options(#rocm.pipeline_options<compile_from = llvm_translation>) {
    hal.executable.export public @entry ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>)
    builtin.module {
      func.func @entry() { return }
    }
  }
}

// -----

// Both fields set to non-default values.
// CHECK-LABEL: @both_fields
hal.executable @both_fields {
  // CHECK: options(#rocm.pipeline_options<compile_from = llvm_translation, compile_to = configuration_controlled_translation>)
  hal.executable.variant @rocm
      target(#hal.executable.target<"rocm", "rocm-hsaco-fb">)
      options(#rocm.pipeline_options<compile_from = llvm_translation, compile_to = configuration_controlled_translation>) {
    hal.executable.export public @entry ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>
    ]>)
    builtin.module {
      func.func @entry() { return }
    }
  }
}
