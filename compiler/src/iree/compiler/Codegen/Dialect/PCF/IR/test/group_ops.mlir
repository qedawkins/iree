// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func @form_groups_basic
func.func @form_groups_basic() {
  pcf.form_groups #pcf.test_scope sizes [2, 2] {
  ^bb0(%prod: !pcf.group<0, #pcf.test_scope>,
       %cons: !pcf.group<1, #pcf.test_scope>):
    pcf.yield
  }
  return
}
// CHECK:      pcf.form_groups #pcf.test_scope sizes [2, 2] {
// CHECK-NEXT: ^bb0(%[[PROD:.+]]: !pcf.group<0, #pcf.test_scope>,
// CHECK-SAME:      %[[CONS:.+]]: !pcf.group<1, #pcf.test_scope>):
// CHECK-NEXT:   pcf.yield
// CHECK-NEXT: }

// -----

// CHECK-LABEL: func @form_groups_three_bundles
func.func @form_groups_three_bundles() {
  pcf.form_groups #pcf.test_scope sizes [1, 2, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.test_scope>,
       %b1: !pcf.group<1, #pcf.test_scope>,
       %b2: !pcf.group<2, #pcf.test_scope>):
    pcf.yield
  }
  return
}
// CHECK:      pcf.form_groups #pcf.test_scope sizes [1, 2, 1] {
// CHECK-NEXT: ^bb0(%{{.+}}: !pcf.group<0, #pcf.test_scope>,
// CHECK-SAME:      %{{.+}}: !pcf.group<1, #pcf.test_scope>,
// CHECK-SAME:      %{{.+}}: !pcf.group<2, #pcf.test_scope>):
// CHECK-NEXT:   pcf.yield
// CHECK-NEXT: }

// -----

// CHECK-LABEL: func @execute_as_no_result
func.func @execute_as_no_result() {
  pcf.form_groups #pcf.test_scope sizes [2, 2] {
  ^bb0(%prod: !pcf.group<0, #pcf.test_scope>,
       %cons: !pcf.group<1, #pcf.test_scope>):
    pcf.execute_as [%prod] {
      pcf.return
    } : !pcf.group<0, #pcf.test_scope>
    pcf.yield
  }
  return
}
// CHECK:      pcf.execute_as [%[[PROD:.+]]] {
// CHECK-NEXT:   pcf.return
// CHECK-NEXT: } : !pcf.group<0, #pcf.test_scope>

// -----

// CHECK-LABEL: func @execute_as_with_result
func.func @execute_as_with_result() {
  pcf.form_groups #pcf.test_scope sizes [2, 2] {
  ^bb0(%prod: !pcf.group<0, #pcf.test_scope>,
       %cons: !pcf.group<1, #pcf.test_scope>):
    %updated = pcf.execute_as [%cons]
        -> !pcf.group<1, #pcf.test_scope, {tensor<64x128xf32>}> {
      %zero = arith.constant dense<0.0> : tensor<64x128xf32>
      pcf.yield %zero : tensor<64x128xf32>
    } : !pcf.group<1, #pcf.test_scope>
    pcf.yield
  }
  return
}
// CHECK:      %[[UPDATED:.+]] = pcf.execute_as [%[[CONS:.+]]]
// CHECK-SAME:     -> !pcf.group<1, #pcf.test_scope, {tensor<64x128xf32>}> {
// CHECK:        %[[ZERO:.+]] = arith.constant dense<0.000000e+00>
// CHECK-NEXT:   pcf.yield %[[ZERO]] : tensor<64x128xf32>
// CHECK-NEXT: } : !pcf.group<1, #pcf.test_scope>

// -----

// CHECK-LABEL: func @execute_as_with_struct_input
func.func @execute_as_with_struct_input(
    %b: !pcf.group<0, #pcf.test_scope, {index, tensor<4xf32>}>) {
  %updated = pcf.execute_as [%b]
      -> !pcf.group<0, #pcf.test_scope, {index, tensor<4xf32>}> {
    ^bb0(%off: index, %acc: tensor<4xf32>):
      pcf.yield %off, %acc : index, tensor<4xf32>
  } : !pcf.group<0, #pcf.test_scope, {index, tensor<4xf32>}>
  return
}
// CHECK:      pcf.execute_as [%{{.+}}]
// CHECK-SAME:     -> !pcf.group<0, #pcf.test_scope, {index, tensor<4xf32>}> {
// CHECK-NEXT: ^bb0(%[[OFF:.+]]: index, %[[ACC:.+]]: tensor<4xf32>):
// CHECK-NEXT:   pcf.yield %[[OFF]], %[[ACC]] : index, tensor<4xf32>
// CHECK-NEXT: } : !pcf.group<0, #pcf.test_scope, {index, tensor<4xf32>}>

// -----

// CHECK-LABEL: func @execute_as_no_result_with_struct_input
func.func @execute_as_no_result_with_struct_input(
    %b: !pcf.group<0, #pcf.test_scope, {index, index}>) {
  pcf.execute_as [%b] {
    ^bb0(%off_m: index, %off_n: index):
      pcf.return
  } : !pcf.group<0, #pcf.test_scope, {index, index}>
  return
}
// CHECK:      pcf.execute_as [%{{.+}}] {
// CHECK-NEXT: ^bb0(%[[OFF_M:.+]]: index, %[[OFF_N:.+]]: index):
// CHECK-NEXT:   pcf.return
// CHECK-NEXT: } : !pcf.group<0, #pcf.test_scope, {index, index}>

// -----

// CHECK-LABEL: func @bind_basic
func.func @bind_basic(
    %b: !pcf.group<0, #pcf.test_scope>,
    %off: index, %size: index) {
  %tagged = pcf.bind %b, %off, %size
      : !pcf.group<0, #pcf.test_scope, {index, index}>
  return
}
// CHECK: pcf.bind %{{.+}}, %{{.+}}, %{{.+}} : !pcf.group<0, #pcf.test_scope, {index, index}>

// -----

// CHECK-LABEL: func @bind_replace
func.func @bind_replace(
    %b: !pcf.group<0, #pcf.test_scope, {index}>,
    %new_off: index) {
  %updated = pcf.bind %b, %new_off
      : !pcf.group<0, #pcf.test_scope, {index}> from !pcf.group<0, #pcf.test_scope, {index}>
  return
}
// CHECK: pcf.bind %{{.+}}, %{{.+}} : !pcf.group<0, #pcf.test_scope, {index}> from !pcf.group<0, #pcf.test_scope, {index}>

// -----

// CHECK-LABEL: func @bind_tensor
func.func @bind_tensor(
    %b: !pcf.group<1, #pcf.test_scope>,
    %acc: tensor<64x128xf32>) {
  %tagged = pcf.bind %b, %acc
      : !pcf.group<1, #pcf.test_scope, {tensor<64x128xf32>}>
  return
}
// CHECK: pcf.bind %{{.+}}, %{{.+}} : !pcf.group<1, #pcf.test_scope, {tensor<64x128xf32>}>

// -----

// CHECK-LABEL: func @join_basic
func.func @join_basic(
    %b0: !pcf.group<0, #pcf.test_scope>,
    %b1: !pcf.group<1, #pcf.test_scope>) {
  %grouped = pcf.join %b0, %b1
      : !pcf.group<[0, 1], #pcf.test_scope>
  return
}
// CHECK: pcf.join %{{.+}}, %{{.+}} : !pcf.group<[0, 1], #pcf.test_scope>

// -----

// CHECK-LABEL: func @join_three
func.func @join_three(
    %b0: !pcf.group<0, #pcf.test_scope>,
    %b1: !pcf.group<1, #pcf.test_scope>,
    %b2: !pcf.group<2, #pcf.test_scope>) {
  %grouped = pcf.join %b0, %b1, %b2
      : !pcf.group<[0, 1, 2], #pcf.test_scope>
  return
}
// CHECK: pcf.join %{{.+}}, %{{.+}}, %{{.+}} : !pcf.group<[0, 1, 2], #pcf.test_scope>

// -----

// CHECK-LABEL: func @join_with_struct
func.func @join_with_struct(
    %b0: !pcf.group<0, #pcf.test_scope, {index}>,
    %b1: !pcf.group<1, #pcf.test_scope, {index}>) {
  %grouped = pcf.join %b0, %b1
      : !pcf.group<[0, 1], #pcf.test_scope, {index}>
  return
}
// CHECK: pcf.join %{{.+}}, %{{.+}} : !pcf.group<[0, 1], #pcf.test_scope, {index}>

// -----

// CHECK-LABEL: func @split_basic
func.func @split_basic(
    %grouped: !pcf.group<[0, 1, 2, 3], #pcf.test_scope>) {
  %left, %right = pcf.split %grouped sizes [2, 2]
      : !pcf.group<[0, 1], #pcf.test_scope>,
        !pcf.group<[2, 3], #pcf.test_scope>
  return
}
// CHECK: pcf.split %{{.+}} sizes [2, 2] : !pcf.group<[0, 1], #pcf.test_scope>, !pcf.group<[2, 3], #pcf.test_scope>

// -----

// CHECK-LABEL: func @split_three_way
func.func @split_three_way(
    %grouped: !pcf.group<[0, 1, 2, 3, 4, 5], #pcf.test_scope>) {
  %a, %b, %c = pcf.split %grouped sizes [2, 3, 1]
      : !pcf.group<[0, 1], #pcf.test_scope>,
        !pcf.group<[2, 3, 4], #pcf.test_scope>,
        !pcf.group<5, #pcf.test_scope>
  return
}
// CHECK: pcf.split %{{.+}} sizes [2, 3, 1] : !pcf.group<[0, 1], #pcf.test_scope>, !pcf.group<[2, 3, 4], #pcf.test_scope>, !pcf.group<5, #pcf.test_scope>

// -----

// CHECK-LABEL: func @split_with_struct
func.func @split_with_struct(
    %grouped: !pcf.group<[0, 1], #pcf.test_scope, {index}>) {
  %a, %b = pcf.split %grouped sizes [1, 1]
      : !pcf.group<0, #pcf.test_scope, {index}>,
        !pcf.group<1, #pcf.test_scope, {index}>
  return
}
// CHECK: pcf.split %{{.+}} sizes [1, 1] : !pcf.group<0, #pcf.test_scope, {index}>, !pcf.group<1, #pcf.test_scope, {index}>

// -----

// CHECK-LABEL: func @define_workspace_static
func.func @define_workspace_static(
    %b: !pcf.group<0, #pcf.test_scope>) {
  %work = pcf.define_workspace %b
      iteration_space(128, 128)
      indexing_maps = [affine_map<(m, n) -> (m, n)>]
      : !pcf.group<0, #pcf.test_scope, {index, index, index, index}>
  return
}
// CHECK: pcf.define_workspace %{{.+}} iteration_space(128, 128)
// CHECK-SAME: indexing_maps = [#{{.+}}]
// CHECK-SAME: : !pcf.group<0, #pcf.test_scope, {index, index, index, index}>

// -----

// CHECK-LABEL: func @define_workspace_dynamic
func.func @define_workspace_dynamic(
    %b: !pcf.group<0, #pcf.test_scope>,
    %dim_m: index, %dim_n: index) {
  %work = pcf.define_workspace %b
      iteration_space(%dim_m, %dim_n)
      indexing_maps = [affine_map<(m, n) -> (m, n)>]
      : !pcf.group<0, #pcf.test_scope, {index, index, index, index}>
  return
}
// CHECK-SAME: %[[B:.+]]: !pcf.group<0, #pcf.test_scope>,
// CHECK-SAME: %[[DIM_M:.+]]: index, %[[DIM_N:.+]]: index
// CHECK: pcf.define_workspace %[[B]] iteration_space(%[[DIM_M]], %[[DIM_N]])
// CHECK-SAME: indexing_maps = [#{{.+}}]
// CHECK-SAME: : !pcf.group<0, #pcf.test_scope, {index, index, index, index}>

// -----

// CHECK-LABEL: func @define_workspace_mixed_static_dynamic
func.func @define_workspace_mixed_static_dynamic(
    %b: !pcf.group<0, #pcf.test_scope>,
    %dim_n: index) {
  %work = pcf.define_workspace %b
      iteration_space(128, %dim_n)
      indexing_maps = [affine_map<(m, n) -> (m, n)>]
      : !pcf.group<0, #pcf.test_scope, {index, index, index, index}>
  return
}
// CHECK: pcf.define_workspace %{{.+}} iteration_space(128, %{{.+}})
// CHECK-SAME: indexing_maps = [#{{.+}}]

// -----

// CHECK-LABEL: func @define_workspace_with_symbols
func.func @define_workspace_with_symbols(
    %b: !pcf.group<0, #pcf.test_scope>,
    %dim_m: index, %dim_n: index) {
  %work = pcf.define_workspace %b
      iteration_space(%dim_m, %dim_n)
      indexing_maps = [affine_map<(m) [n] -> (m, n)>]
      : !pcf.group<0, #pcf.test_scope, {index, index}>
  return
}
// CHECK: pcf.define_workspace %{{.+}} iteration_space(%{{.+}}, %{{.+}})
// CHECK-SAME: indexing_maps = [#{{.+}}]
// CHECK-SAME: : !pcf.group<0, #pcf.test_scope, {index, index}>

// -----

// CHECK-LABEL: func @define_workspace_preserves_struct
func.func @define_workspace_preserves_struct(
    %b: !pcf.group<0, #pcf.test_scope, {tensor<4xf32>}>) {
  %work = pcf.define_workspace %b
      iteration_space(64)
      indexing_maps = [affine_map<(m) -> (m)>]
      : !pcf.group<0, #pcf.test_scope, {tensor<4xf32>, index, index}>
  return
}
// CHECK: pcf.define_workspace %{{.+}} iteration_space(64)
// CHECK-SAME: indexing_maps = [#{{.+}}]
// CHECK-SAME: : !pcf.group<0, #pcf.test_scope, {tensor<4xf32>, index, index}>

// -----

// CHECK-LABEL: func @barrier_basic
func.func @barrier_basic() {
  pcf.barrier(#pcf.test_scope)
  return
}
// CHECK: pcf.barrier(#pcf.test_scope)
