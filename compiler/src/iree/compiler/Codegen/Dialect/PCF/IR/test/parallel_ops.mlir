// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @timepointImmediate
util.func private @timepointImmediate() -> !stream.timepoint {
  // CHECK: = stream.timepoint.immediate => !stream.timepoint
  %0 = stream.timepoint.immediate => !stream.timepoint
  util.return %0 : !stream.timepoint
}
