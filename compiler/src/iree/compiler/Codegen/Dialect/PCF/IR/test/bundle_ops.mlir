// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// Basic form_bundles with two bundles.
util.func private @form_bundles_basic() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    pcf.form_bundles #pcf.sequential sizes [1, 3] {
    ^bb0(%b0: !pcf.bundle<#pcf.sequential, 0>,
         %b1: !pcf.bundle<#pcf.sequential, 1>):
      pcf.yield
    }
    pcf.return
  }
  util.return
}
// CHECK-LABEL: @form_bundles_basic
// CHECK: pcf.form_bundles #pcf.sequential sizes [1, 3]
// CHECK-NEXT: ^bb0(%[[B0:.*]]: !pcf.bundle<#pcf.sequential, 0>, %[[B1:.*]]: !pcf.bundle<#pcf.sequential, 1>):

// -----

// execute_as with single bundle.
util.func private @execute_as_single() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    pcf.form_bundles #pcf.sequential sizes [1, 3] {
    ^bb0(%b0: !pcf.bundle<#pcf.sequential, 0>,
         %b1: !pcf.bundle<#pcf.sequential, 1>):
      pcf.execute_as [%b0 : !pcf.bundle<#pcf.sequential, 0>] {
        pcf.return
      }
      pcf.execute_as [%b1 : !pcf.bundle<#pcf.sequential, 1>] {
        pcf.return
      }
      pcf.yield
    }
    pcf.return
  }
  util.return
}
// CHECK-LABEL: @execute_as_single
// CHECK: pcf.execute_as [%[[B0:.*]] : !pcf.bundle<#pcf.sequential, 0>]
// CHECK: pcf.execute_as [%[[B1:.*]] : !pcf.bundle<#pcf.sequential, 1>]

// -----

// execute_as with multiple bundles.
util.func private @execute_as_multi_bundle() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    pcf.form_bundles #pcf.sequential sizes [1, 3] {
    ^bb0(%b0: !pcf.bundle<#pcf.sequential, 0>,
         %b1: !pcf.bundle<#pcf.sequential, 1>):
      pcf.execute_as [%b0 : !pcf.bundle<#pcf.sequential, 0>,
                      %b1 : !pcf.bundle<#pcf.sequential, 1>] {
        pcf.return
      }
      pcf.yield
    }
    pcf.return
  }
  util.return
}
// CHECK-LABEL: @execute_as_multi_bundle
// CHECK: pcf.execute_as [%[[B0:.*]] : !pcf.bundle<#pcf.sequential, 0>, %[[B1:.*]] : !pcf.bundle<#pcf.sequential, 1>]
