// RUN: iree-opt --split-input-file %s --verify-diagnostics

util.func private @scope_mismatch(%dim: index) {
// expected-error@+1 {{expected region ref argument to be of type !pcf.sref with scope #pcf.sequential}}
  pcf.generic scope(#pcf.sequential)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

// expected-note@+1 {{prior use here}}
util.func private @init_type_mismatch(%0: tensor<3xi32>) {
  pcf.generic scope(#pcf.test_scope)
// expected-error@+1 {{expects different type than prior uses: 'tensor<?xi32>' vs 'tensor<3xi32>'}}
    execute(%ref = %0)[%num_threads: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>) {
    pcf.return
  }
  util.return
}

// -----

util.func private @sync_scope_mismatch(%dim: index) {
// expected-error@+1 {{expected region ref argument to sync on return or is unspecified}}
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope, i32>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @arg_shape_mismatch(%dim: index) {
// expected-error@+1 {{region arg at index 0 with type '!pcf.sref<3xi32, #pcf.test_scope>' shape mismatch with tied result of type 'tensor<?xi32>'}}
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<3xi32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @arg_eltype_mismatch(%dim: index) {
// expected-error@+1 {{region arg at index 0 element type mismatch of 'f32' vs 'i32'}}
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?xf32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @empty_count(%dim: index) {
// expected-error@+1 {{expected at least one iteration count argument}}
  pcf.loop scope(#pcf.sequential) count()
    execute(%ref)[]
         : (!pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @missing_execute_keyword() {
  pcf.generic scope(#pcf.test_scope)
    // expected-error@+1 {{custom op 'pcf.generic' expected 'execute'}}
    notexecute(%ref)[%id: index, %n: index]
         : (!pcf.sref<4xi32, #pcf.test_scope>)
        -> (tensor<4xi32>) {
    pcf.return
  }
  util.return
}

// -----

util.func private @result_not_shaped_type() {
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<4xi32, #pcf.test_scope>)
        // expected-error@+1 {{custom op 'pcf.generic' result type must be a shaped type}}
        -> (i32) {
    pcf.return
  }
  util.return
}

// -----

util.func private @dynamic_dim_mismatch(%dim0: index) {
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.test_scope>)
        // expected-error@+1 {{custom op 'pcf.generic' expected 2 dynamic dimension operands for type 'tensor<?x?xi32>', but got 1}}
        -> (tensor<?x?xi32>{%dim0}) {
    pcf.return
  }
  util.return
}

// -----

// Group with negative ID.
// expected-error@+1 {{group IDs must be non-negative}}
func.func @bundle_negative_id(%b: !pcf.group<-1, #pcf.test_scope>) {
  return
}

// -----

// Group with negative grouped ID.
// expected-error@+1 {{group IDs must be non-negative}}
func.func @bundle_negative_grouped_id(%b: !pcf.group<[0, -2], #pcf.test_scope>) {
  return
}

// -----

// Group missing scope attribute.
// expected-error@+1 {{expected scope attribute to implement ScopeAttrInterface}}
func.func @bundle_bad_scope(%b: !pcf.group<0, 42>) {
  return
}

// -----

// Group with empty brackets.
// expected-error@+1 {{expected integer value}}
func.func @bundle_empty_ids(%b: !pcf.group<[], #pcf.test_scope>) {
  return
}

// -----

// Group with duplicate IDs.
// expected-error@+1 {{duplicate group ID: 1}}
func.func @bundle_duplicate_ids(%b: !pcf.group<[0, 1, 1], #pcf.test_scope>) {
  return
}

// -----

// Group with empty struct braces.
// expected-error@+1 {{expected non-function type}}
func.func @bundle_empty_struct(%b: !pcf.group<0, #pcf.test_scope, {}>) {
  return
}

// -----

//===----------------------------------------------------------------------===//
// FormGroupsOp negative tests
//===----------------------------------------------------------------------===//

// Zero group size.
func.func @form_groups_zero_size() {
  // expected-error@+1 {{group size at index 0 must be positive, got 0}}
  pcf.form_groups #pcf.test_scope sizes [0] {
  ^bb0(%b0: !pcf.group<0, #pcf.test_scope>):
    pcf.yield
  }
  return
}

// -----

// Non-positive group size.
func.func @form_groups_nonpositive_size() {
  // expected-error@+1 {{group size at index 1 must be positive, got -1}}
  pcf.form_groups #pcf.test_scope sizes [2, -1] {
  ^bb0(%b0: !pcf.group<0, #pcf.test_scope>,
       %b1: !pcf.group<1, #pcf.test_scope>):
    pcf.yield
  }
  return
}

// -----

// Block argument count mismatch.
func.func @form_groups_arg_count_mismatch() {
  // expected-error@+1 {{expected 2 block arguments, got 1}}
  pcf.form_groups #pcf.test_scope sizes [2, 2] {
  ^bb0(%b0: !pcf.group<0, #pcf.test_scope>):
    pcf.yield
  }
  return
}

// -----

// Block argument is not a group type.
func.func @form_groups_arg_not_bundle() {
  // expected-error@+1 {{block argument 0 must be a !pcf.group type}}
  pcf.form_groups #pcf.test_scope sizes [2] {
  ^bb0(%b0: index):
    pcf.yield
  }
  return
}

// -----

// Block argument scope mismatch.
func.func @form_groups_arg_scope_mismatch() {
  // expected-error@+1 {{block argument 0 scope mismatch}}
  pcf.form_groups #pcf.test_scope sizes [2] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    pcf.yield
  }
  return
}

// -----

// Block argument is a grouped group.
func.func @form_groups_arg_grouped() {
  // expected-error@+1 {{block argument 0 must be a single-ID group, got grouped}}
  pcf.form_groups #pcf.test_scope sizes [2] {
  ^bb0(%b0: !pcf.group<[0, 1], #pcf.test_scope>):
    pcf.yield
  }
  return
}

// -----

// Block argument group ID mismatch.
func.func @form_groups_arg_id_mismatch() {
  // expected-error@+1 {{block argument 0 group ID mismatch: expected 0, got 5}}
  pcf.form_groups #pcf.test_scope sizes [2] {
  ^bb0(%b0: !pcf.group<5, #pcf.test_scope>):
    pcf.yield
  }
  return
}

// -----

// Block argument has struct elements.
func.func @form_groups_arg_has_struct() {
  // expected-error@+1 {{block argument 0 must not have struct elements}}
  pcf.form_groups #pcf.test_scope sizes [2] {
  ^bb0(%b0: !pcf.group<0, #pcf.test_scope, {index}>):
    pcf.yield
  }
  return
}

// -----

// Yield with operands in form_groups body.
func.func @form_groups_yield_with_operands() {
  pcf.form_groups #pcf.test_scope sizes [2] {
  ^bb0(%b0: !pcf.group<0, #pcf.test_scope>):
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{in form_groups body must have zero operands}}
    pcf.yield %c0 : index
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// ExecuteAsOp negative tests
//===----------------------------------------------------------------------===//

// Block argument count mismatch with struct elements.
func.func @execute_as_arg_count_mismatch(
    %b: !pcf.group<0, #pcf.test_scope, {index, index}>) {
  // expected-error@+1 {{expected 2 block arguments matching input group struct elements, got 1}}
  pcf.execute_as [%b] {
  ^bb0(%x: index):
    pcf.return
  } : !pcf.group<0, #pcf.test_scope, {index, index}>
  return
}

// -----

// Block argument type mismatch.
func.func @execute_as_arg_type_mismatch(
    %b: !pcf.group<0, #pcf.test_scope, {index}>) {
  // expected-error@+1 {{block argument 0 type mismatch: expected 'index', got 'f32'}}
  pcf.execute_as [%b] {
  ^bb0(%x: f32):
    pcf.return
  } : !pcf.group<0, #pcf.test_scope, {index}>
  return
}

// -----

// Has result but uses pcf.return instead of pcf.yield.
func.func @execute_as_result_with_return(
    %b: !pcf.group<0, #pcf.test_scope>) {
  // expected-error@+1 {{expected pcf.yield terminator when producing a result, got pcf.return}}
  %r = pcf.execute_as [%b]
      -> !pcf.group<0, #pcf.test_scope, {tensor<4xf32>}> {
    pcf.return
  } : !pcf.group<0, #pcf.test_scope>
  return
}

// -----

// Result group ID mismatch.
func.func @execute_as_result_id_mismatch(
    %b: !pcf.group<0, #pcf.test_scope>) {
  // expected-error@+1 {{result group IDs must match input group IDs}}
  %r = pcf.execute_as [%b]
      -> !pcf.group<1, #pcf.test_scope, {tensor<4xf32>}> {
    %zero = arith.constant dense<0.0> : tensor<4xf32>
    pcf.yield %zero : tensor<4xf32>
  } : !pcf.group<0, #pcf.test_scope>
  return
}

// -----

// Result group scope mismatch.
func.func @execute_as_result_scope_mismatch(
    %b: !pcf.group<0, #pcf.test_scope>) {
  // expected-error@+1 {{result group scope must match input group scope}}
  %r = pcf.execute_as [%b]
      -> !pcf.group<0, #pcf.sequential, {tensor<4xf32>}> {
    %zero = arith.constant dense<0.0> : tensor<4xf32>
    pcf.yield %zero : tensor<4xf32>
  } : !pcf.group<0, #pcf.test_scope>
  return
}

// -----

// Yield operand count mismatch.
func.func @execute_as_yield_count_mismatch(
    %b: !pcf.group<0, #pcf.test_scope>) {
  // expected-error@+1 {{yield operand count 2 does not match result group struct element count 1}}
  %r = pcf.execute_as [%b]
      -> !pcf.group<0, #pcf.test_scope, {tensor<4xf32>}> {
    %zero = arith.constant dense<0.0> : tensor<4xf32>
    %one = arith.constant dense<1.0> : tensor<4xf32>
    pcf.yield %zero, %one : tensor<4xf32>, tensor<4xf32>
  } : !pcf.group<0, #pcf.test_scope>
  return
}

// -----

// Yield operand type mismatch.
func.func @execute_as_yield_type_mismatch(
    %b: !pcf.group<0, #pcf.test_scope>) {
  // expected-error@+1 {{yield operand 0 type mismatch: expected 'tensor<4xf32>', got 'tensor<4xi32>'}}
  %r = pcf.execute_as [%b]
      -> !pcf.group<0, #pcf.test_scope, {tensor<4xf32>}> {
    %zero = arith.constant dense<0> : tensor<4xi32>
    pcf.yield %zero : tensor<4xi32>
  } : !pcf.group<0, #pcf.test_scope>
  return
}

// -----

// No result but uses pcf.yield.
func.func @execute_as_no_result_with_yield(
    %b: !pcf.group<0, #pcf.test_scope>) {
  // expected-error@+1 {{expected pcf.return terminator when producing no result}}
  pcf.execute_as [%b] {
    pcf.yield
  } : !pcf.group<0, #pcf.test_scope>
  return
}

// -----

//===----------------------------------------------------------------------===//
// BindOp negative tests
//===----------------------------------------------------------------------===//

// Input group is grouped.
func.func @bind_input_grouped(
    %b: !pcf.group<[0, 1], #pcf.test_scope>,
    %v: index) {
  // expected-error@+1 {{input group must be a single-ID group, got grouped}}
  %r = pcf.bind %b, %v
      : !pcf.group<[0, 1], #pcf.test_scope, {index}> from !pcf.group<[0, 1], #pcf.test_scope>
  return
}

// -----

// Result ID mismatch.
func.func @bind_result_id_mismatch(
    %b: !pcf.group<0, #pcf.test_scope>,
    %v: index) {
  // expected-error@+1 {{result group IDs must match input group IDs}}
  %r = pcf.bind %b, %v
      : !pcf.group<1, #pcf.test_scope, {index}> from !pcf.group<0, #pcf.test_scope>
  return
}

// -----

// Result scope mismatch.
func.func @bind_result_scope_mismatch(
    %b: !pcf.group<0, #pcf.test_scope>,
    %v: index) {
  // expected-error@+1 {{result group scope must match input group scope}}
  %r = pcf.bind %b, %v
      : !pcf.group<0, #pcf.sequential, {index}> from !pcf.group<0, #pcf.test_scope>
  return
}

// -----

// NOTE: Verifier checks for bind value type mismatch, join operand type
// mismatch (grouped inputs, duplicate IDs, scope/struct mismatches), and split
// result struct mismatch are defense-in-depth for programmatic construction.
// They cannot be tested via lit tests because the MLIR parser catches these
// type inconsistencies before the verifier runs (SSA type resolution, custom
// assembly format type inference, and GroupType parser validation).

// -----

// Replacement count mismatch: input has 2 struct elements but only 1 value.
func.func @bind_replacement_count_mismatch(
    %b: !pcf.group<0, #pcf.test_scope, {index, index}>,
    %v: index) {
  // expected-error@+1 {{input group has 2 struct elements but bind provides 1 values; counts must match when replacing}}
  %r = pcf.bind %b, %v
      : !pcf.group<0, #pcf.test_scope, {index}> from !pcf.group<0, #pcf.test_scope, {index, index}>
  return
}

// -----

//===----------------------------------------------------------------------===//
// JoinOp negative tests
//===----------------------------------------------------------------------===//

// Too few inputs.
func.func @join_too_few(
    %b0: !pcf.group<0, #pcf.test_scope>) {
  // expected-error@+1 {{expected at least 2 input groups, got 1}}
  %r = pcf.join %b0
      : !pcf.group<0, #pcf.test_scope>
  return
}

// -----

//===----------------------------------------------------------------------===//
// SplitOp negative tests
//===----------------------------------------------------------------------===//

// Size must be positive.
func.func @split_nonpositive_size(
    %b: !pcf.group<[0, 1], #pcf.test_scope>) {
  // expected-error@+1 {{size at index 1 must be positive, got 0}}
  %a, %b_out = pcf.split %b sizes [2, 0]
      : !pcf.group<0, #pcf.test_scope>,
        !pcf.group<1, #pcf.test_scope>
  return
}

// -----

// Sum of sizes mismatch.
func.func @split_size_sum_mismatch(
    %b: !pcf.group<[0, 1, 2], #pcf.test_scope>) {
  // expected-error@+1 {{sum of sizes (4) must equal input group ID count (3)}}
  %a, %b_out = pcf.split %b sizes [1, 3]
      : !pcf.group<0, #pcf.test_scope>,
        !pcf.group<[1, 2], #pcf.test_scope>
  return
}

// -----

// Result ID slice mismatch: sizes [3, 1] but result 0 only has 2 IDs.
func.func @split_id_slice_mismatch(
    %b: !pcf.group<[0, 1, 2, 3], #pcf.test_scope>) {
  // expected-error@+1 {{result 0 IDs do not match expected slice}}
  %a, %b_out = pcf.split %b sizes [3, 1]
      : !pcf.group<[0, 1], #pcf.test_scope>,
        !pcf.group<[2, 3], #pcf.test_scope>
  return
}

// -----

// Result scope mismatch.
func.func @split_result_scope_mismatch(
    %b: !pcf.group<[0, 1], #pcf.test_scope>) {
  // expected-error@+1 {{result 1 scope mismatch}}
  %a, %b_out = pcf.split %b sizes [1, 1]
      : !pcf.group<0, #pcf.test_scope>,
        !pcf.group<1, #pcf.sequential>
  return
}

// -----

//===----------------------------------------------------------------------===//
// DefineWorkspaceOp negative tests
//===----------------------------------------------------------------------===//

// Input group is grouped.
func.func @define_workspace_input_grouped(
    %b: !pcf.group<[0, 1], #pcf.test_scope>) {
  // expected-error@+1 {{input group must be a single-ID group, got grouped}}
  %r = pcf.define_workspace %b
      iteration_space(128)
      indexing_maps = [affine_map<(m) -> (m)>]
      : !pcf.group<[0, 1], #pcf.test_scope, {index, index}>
  return
}

// -----

// Result ID mismatch (parser infers input type from result, catches type mismatch).
func.func @define_workspace_id_mismatch(
    %b: !pcf.group<0, #pcf.test_scope>) { // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%b' expects different type than prior uses}}
  %r = pcf.define_workspace %b
      iteration_space(128)
      indexing_maps = [affine_map<(m) -> (m)>]
      : !pcf.group<1, #pcf.test_scope, {index, index}>
  return
}

// -----

// Result scope mismatch (parser infers input type from result, catches type mismatch).
func.func @define_workspace_scope_mismatch(
    %b: !pcf.group<0, #pcf.test_scope>) { // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%b' expects different type than prior uses}}
  %r = pcf.define_workspace %b
      iteration_space(128)
      indexing_maps = [affine_map<(m) -> (m)>]
      : !pcf.group<0, #pcf.sequential, {index, index}>
  return
}

// -----

// No indexing maps.
func.func @define_workspace_no_maps(
    %b: !pcf.group<0, #pcf.test_scope>) {
  // expected-error@+1 {{expected at least one indexing map}}
  %r = pcf.define_workspace %b
      iteration_space(128)
      indexing_maps = []
      : !pcf.group<0, #pcf.test_scope>
  return
}

// -----

// Map result count mismatch with iteration space rank.
func.func @define_workspace_map_result_mismatch(
    %b: !pcf.group<0, #pcf.test_scope>) {
  // expected-error@+1 {{indexing map has 1 results but iteration space has 2 dimensions}}
  %r = pcf.define_workspace %b
      iteration_space(128, 128)
      indexing_maps = [affine_map<(m) -> (m)>]
      : !pcf.group<0, #pcf.test_scope, {index, index}>
  return
}

// -----

// Wrong number of struct elements in result (parser infers input type from
// result, catches type mismatch since inferred input has 2 struct elements).
func.func @define_workspace_struct_count_mismatch(
    %b: !pcf.group<0, #pcf.test_scope>) { // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%b' expects different type than prior uses}}
  %r = pcf.define_workspace %b
      iteration_space(128)
      indexing_maps = [affine_map<(m) -> (m)>]
      : !pcf.group<0, #pcf.test_scope, {index, index, index, index}>
  return
}

// -----

// Appended elements are not index type.
func.func @define_workspace_non_index_workspace_element(
    %b: !pcf.group<0, #pcf.test_scope>) {
  // expected-error@+1 {{workspace struct element 0 must be index type, got 'f32'}}
  %r = pcf.define_workspace %b
      iteration_space(128)
      indexing_maps = [affine_map<(m) -> (m)>]
      : !pcf.group<0, #pcf.test_scope, {f32, index}>
  return
}

// -----

// Input struct prefix not preserved (parser infers input type from result,
// catches type mismatch since inferred input has {index} but arg has {tensor}).
func.func @define_workspace_prefix_mismatch(
    %b: !pcf.group<0, #pcf.test_scope, {tensor<4xf32>}>) { // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%b' expects different type than prior uses}}
  %r = pcf.define_workspace %b
      iteration_space(128)
      indexing_maps = [affine_map<(m) -> (m)>]
      : !pcf.group<0, #pcf.test_scope, {index, index, index}>
  return
}
