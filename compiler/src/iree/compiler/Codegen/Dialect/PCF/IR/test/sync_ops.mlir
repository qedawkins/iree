// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

util.func private @fence(%alloc: !pcf.sref<128x64xf16, #pcf.test_scope>) {
  pcf.fence release %alloc : !pcf.sref<128x64xf16, #pcf.test_scope>
  pcf.fence acquire %alloc : !pcf.sref<128x64xf16, #pcf.test_scope>
  util.return
}

// CHECK-LABEL: @fence
//  CHECK-SAME:   %[[ALLOC:[A-Za-z0-9]+]]: !pcf.sref<128x64xf16, #pcf.test_scope>
//       CHECK:   pcf.fence release %[[ALLOC]] : !pcf.sref<128x64xf16, #pcf.test_scope>
//       CHECK:   pcf.fence acquire %[[ALLOC]] : !pcf.sref<128x64xf16, #pcf.test_scope>
