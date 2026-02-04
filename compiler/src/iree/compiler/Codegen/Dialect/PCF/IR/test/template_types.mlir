// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @generic_with_template_result
func.func @generic_with_template_result(%init: !template.type<0>) -> !template.type<0> {
  // CHECK: pcf.generic
  %0 = pcf.generic scope(#pcf.sequential)
    execute(%ref = %init)[%id: index, %n: index]
         : (!template.type<0>)
        -> (!template.type<0>) {
    pcf.return
  }
  return %0 : !template.type<0>
}

// -----

// CHECK-LABEL: @loop_with_template_result
func.func @loop_with_template_result(%init: !template.type<1>, %n: index) -> !template.type<1> {
  // CHECK: pcf.loop
  %0 = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%id: index]
         : (!template.type<1>)
        -> (!template.type<1>) {
    pcf.return
  }
  return %0 : !template.type<1>
}

// -----

// CHECK-LABEL: @generic_mixed_template_and_concrete
func.func @generic_mixed_template_and_concrete(%t_init: !template.type<0>, %c_init: tensor<4xf32>) -> (!template.type<0>, tensor<4xf32>) {
  // CHECK: pcf.generic
  %0:2 = pcf.generic scope(#pcf.sequential)
    execute(%t_ref = %t_init, %c_ref = %c_init)[%id: index, %n: index]
         : (!template.type<0>, !pcf.sref<4xf32, #pcf.sequential>)
        -> (!template.type<0>, tensor<4xf32>) {
    pcf.return
  }
  return %0#0, %0#1 : !template.type<0>, tensor<4xf32>
}
