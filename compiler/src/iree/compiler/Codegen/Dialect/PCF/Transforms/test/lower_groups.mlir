// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-lower-groups)" --split-input-file | FileCheck %s

// Single group: no switch, execute_as body inlined directly.
// CHECK-LABEL: func @single_group_no_switch
func.func @single_group_no_switch() {
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    pcf.execute_as [%b0] {
      pcf.return
    } : !pcf.group<0, #pcf.sequential>
    pcf.yield
  }
  return
}
// No switch for single group.
// CHECK-NOT: scf.index_switch
// CHECK-NOT: pcf.form_groups
// CHECK-NOT: pcf.execute_as
// CHECK:     return

// -----

// Two groups: produces scf.index_switch with case 0, case 1, default.
// CHECK-LABEL: func @two_groups_index_switch
func.func @two_groups_index_switch() {
  pcf.form_groups #pcf.sequential sizes [1, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>):
    pcf.execute_as [%b0] {
      pcf.return
    } : !pcf.group<0, #pcf.sequential>
    pcf.execute_as [%b1] {
      pcf.return
    } : !pcf.group<1, #pcf.sequential>
    pcf.yield
  }
  return
}
// Group index computed via select chain.
// CHECK:     %[[IDX:.*]] = arith.select
// CHECK:     scf.index_switch %[[IDX]]
// Case 0: execute_as [%b0] inlined (empty), execute_as [%b1] dropped.
// CHECK-NEXT: case 0 {
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// Case 1: execute_as [%b0] dropped, execute_as [%b1] inlined (empty).
// CHECK-NEXT: case 1 {
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: default {
// CHECK-NEXT: }
// CHECK-NOT: pcf.form_groups

// -----

// Single group, execute_as with result. Yielded value is directly available.
// CHECK-LABEL: func @execute_as_with_result
func.func @execute_as_with_result() {
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %updated = pcf.execute_as [%b0]
        -> !pcf.group<0, #pcf.sequential, {index}> {
      %c42 = arith.constant 42 : index
      pcf.yield %c42 : index
    } : !pcf.group<0, #pcf.sequential>
    pcf.yield
  }
  return
}
// No switch, no scf.if, no else block with default values.
// CHECK-NOT: scf.index_switch
// CHECK-NOT: scf.if
// CHECK:     arith.constant 42 : index
// CHECK:     return

// -----

// Bind attaches a value, execute_as receives it as a block arg.
// CHECK-LABEL: func @bind_then_execute_as
func.func @bind_then_execute_as(%val: index) {
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %bound = pcf.bind %b0, %val
        : !pcf.group<0, #pcf.sequential, {index}>
    pcf.execute_as [%bound] {
    ^bb0(%v: index):
      util.optimization_barrier %v : index
      pcf.return
    } : !pcf.group<0, #pcf.sequential, {index}>
    pcf.yield
  }
  return
}
// CHECK-SAME: %[[VAL:.*]]: index
// Bind is eliminated. Value flows directly to inlined body.
// CHECK-NOT: pcf.bind
// CHECK:     util.optimization_barrier %[[VAL]] : index
// CHECK:     return

// -----

// Execute_as with struct passthrough (result captures updated value).
// CHECK-LABEL: func @execute_as_with_struct_passthrough
func.func @execute_as_with_struct_passthrough(%val: index) {
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %bound = pcf.bind %b0, %val
        : !pcf.group<0, #pcf.sequential, {index}>
    %updated = pcf.execute_as [%bound]
        -> !pcf.group<0, #pcf.sequential, {index}> {
    ^bb0(%v: index):
      %new_v = arith.addi %v, %v : index
      pcf.yield %new_v : index
    } : !pcf.group<0, #pcf.sequential, {index}>
    pcf.yield
  }
  return
}
// CHECK-SAME: %[[VAL:.*]]: index
// CHECK-NOT: scf.if
// CHECK:     %[[NEW:.*]] = arith.addi %[[VAL]], %[[VAL]] : index
// CHECK:     return

// -----

// Barrier appears in every case branch (cloned into each).
// CHECK-LABEL: func @barrier_in_both_branches
func.func @barrier_in_both_branches() {
  pcf.form_groups #pcf.sequential sizes [1, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>):
    pcf.execute_as [%b0] {
      pcf.return
    } : !pcf.group<0, #pcf.sequential>
    pcf.barrier(#pcf.sequential)
    pcf.execute_as [%b1] {
      pcf.return
    } : !pcf.group<1, #pcf.sequential>
    pcf.yield
  }
  return
}
// CHECK:     scf.index_switch
// Case 0: execute_as [%b0] inlined (empty), barrier, execute_as [%b1] dropped.
// CHECK-NEXT: case 0 {
// CHECK-NEXT:   pcf.barrier(#pcf.sequential)
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// Case 1: execute_as [%b0] dropped, barrier, execute_as [%b1] inlined (empty).
// CHECK-NEXT: case 1 {
// CHECK-NEXT:   pcf.barrier(#pcf.sequential)
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: default {
// CHECK-NEXT: }

// -----

// Join and split are eliminated. Execute_as uses the reconstructed group.
// CHECK-LABEL: func @join_then_split
func.func @join_then_split() {
  pcf.form_groups #pcf.sequential sizes [1, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>):
    %joined = pcf.join %b0, %b1
        : !pcf.group<[0, 1], #pcf.sequential>
    %a, %b = pcf.split %joined sizes [1, 1]
        : !pcf.group<0, #pcf.sequential>,
          !pcf.group<1, #pcf.sequential>
    pcf.execute_as [%a] {
      pcf.return
    } : !pcf.group<0, #pcf.sequential>
    pcf.yield
  }
  return
}
// CHECK-NOT: pcf.join
// CHECK-NOT: pcf.split
// CHECK:     scf.index_switch
// Case 0: execute_as [%a] inlined (group 0 matches).
// CHECK-NEXT: case 0 {
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// Case 1: execute_as [%a] dropped (group 0 doesn't match region 1).
// CHECK-NEXT: case 1 {
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }

// -----

// Tensor-typed struct element flows through execute_as.
// CHECK-LABEL: func @execute_as_with_tensor_result
func.func @execute_as_with_tensor_result() {
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %updated = pcf.execute_as [%b0]
        -> !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}> {
      %zero = arith.constant dense<0.0> : tensor<4x8xf32>
      pcf.yield %zero : tensor<4x8xf32>
    } : !pcf.group<0, #pcf.sequential>
    pcf.yield
  }
  return
}
// No else block with default tensor. Just the inlined body.
// CHECK-NOT: scf.if
// CHECK:     arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// CHECK:     return

// -----

// define_workspace with a single distributed dim.
// Group 0 has 1 worker. local_id = workerID - 0. size = 128/1. offset = local_id * size.
// CHECK-LABEL: func @define_workspace_single_dim
func.func @define_workspace_single_dim() {
  pcf.form_groups #pcf.sequential sizes [1, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>):
    %ws = pcf.define_workspace %b0
        iteration_space(128)
        indexing_maps = [affine_map<(m) -> (m)>]
        : !pcf.group<0, #pcf.sequential, {index, index}>
    pcf.execute_as [%ws] {
    ^bb0(%off: index, %sz: index):
      util.optimization_barrier %off : index
      util.optimization_barrier %sz : index
      pcf.return
    } : !pcf.group<0, #pcf.sequential, {index, index}>
    pcf.yield
  }
  return
}
// CHECK-NOT: pcf.define_workspace
// CHECK-NOT: pcf.form_groups
// CHECK:     scf.index_switch
// Workspace arithmetic appears in case 0.
// CHECK:     case 0 {
// CHECK:       %[[LID:.*]] = arith.subi {{.*}} overflow<nsw>
// CHECK:       %[[SZ:.*]] = arith.divui
// CHECK:       %[[OFF:.*]] = arith.muli %[[LID]], %[[SZ]] overflow<nsw>
// CHECK:       util.optimization_barrier %[[OFF]] : index
// CHECK:       util.optimization_barrier %[[SZ]] : index
// CHECK:       scf.yield
// CHECK:     }

// -----

// define_workspace with two distributed dims.
// Group 0 has 2 workers. Produces 4 struct values: [off_m, off_n, sz_m, sz_n].
// CHECK-LABEL: func @define_workspace_two_dims
func.func @define_workspace_two_dims() {
  pcf.form_groups #pcf.sequential sizes [2] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %ws = pcf.define_workspace %b0
        iteration_space(128, 128)
        indexing_maps = [affine_map<(m, n) -> (m, n)>]
        : !pcf.group<0, #pcf.sequential, {index, index, index, index}>
    pcf.execute_as [%ws] {
    ^bb0(%off_m: index, %off_n: index, %sz_m: index, %sz_n: index):
      util.optimization_barrier %off_m : index
      util.optimization_barrier %off_n : index
      util.optimization_barrier %sz_m : index
      util.optimization_barrier %sz_n : index
      pcf.return
    } : !pcf.group<0, #pcf.sequential, {index, index, index, index}>
    pcf.yield
  }
  return
}
// Single group: no switch.
// CHECK-NOT: scf.index_switch
// CHECK-NOT: pcf.define_workspace
// Two divui+muli pairs for 2 distributed dims.
// CHECK:     arith.subi {{.*}} overflow<nsw>
// CHECK:     %[[SZ_M:.*]] = arith.divui
// CHECK:     %[[OFF_M:.*]] = arith.muli {{.*}} overflow<nsw>
// CHECK:     %[[SZ_N:.*]] = arith.divui
// CHECK:     %[[OFF_N:.*]] = arith.muli {{.*}} overflow<nsw>
// CHECK:     util.optimization_barrier %[[OFF_M]]
// CHECK:     util.optimization_barrier %[[OFF_N]]
// CHECK:     util.optimization_barrier %[[SZ_M]]
// CHECK:     util.optimization_barrier %[[SZ_N]]

// -----

// define_workspace with symbols (non-distributed dims).
// Map (m) [n] -> (m, n): only m is distributed.
// CHECK-LABEL: func @define_workspace_with_symbol
func.func @define_workspace_with_symbol() {
  pcf.form_groups #pcf.sequential sizes [1, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>):
    %ws = pcf.define_workspace %b1
        iteration_space(128, 128)
        indexing_maps = [affine_map<(m) [n] -> (m, n)>]
        : !pcf.group<1, #pcf.sequential, {index, index}>
    pcf.execute_as [%ws] {
    ^bb0(%off_m: index, %sz_m: index):
      util.optimization_barrier %off_m : index
      util.optimization_barrier %sz_m : index
      pcf.return
    } : !pcf.group<1, #pcf.sequential, {index, index}>
    pcf.yield
  }
  return
}
// Workspace arithmetic appears in case 1 (group 1 matches).
// CHECK:     scf.index_switch
// CHECK:     case 1 {
// CHECK:       arith.subi {{.*}} overflow<nsw>
// CHECK:       %[[SZ:.*]] = arith.divui
// CHECK:       %[[OFF:.*]] = arith.muli {{.*}} overflow<nsw>
// CHECK:       util.optimization_barrier %[[OFF]]
// CHECK:       util.optimization_barrier %[[SZ]]
// CHECK:       scf.yield
// CHECK:     }

// -----

// Bind + define_workspace: existing struct elements are preserved.
// CHECK-LABEL: func @define_workspace_preserves_struct
func.func @define_workspace_preserves_struct(%val: index) {
  pcf.form_groups #pcf.sequential sizes [2] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %bound = pcf.bind %b0, %val
        : !pcf.group<0, #pcf.sequential, {index}>
    %ws = pcf.define_workspace %bound
        iteration_space(128)
        indexing_maps = [affine_map<(m) -> (m)>]
        : !pcf.group<0, #pcf.sequential, {index, index, index}>
    pcf.execute_as [%ws] {
    ^bb0(%v: index, %off: index, %sz: index):
      util.optimization_barrier %v : index
      util.optimization_barrier %off : index
      util.optimization_barrier %sz : index
      pcf.return
    } : !pcf.group<0, #pcf.sequential, {index, index, index}>
    pcf.yield
  }
  return
}
// CHECK-SAME: %[[VAL:.*]]: index
// CHECK-NOT: pcf.bind
// CHECK-NOT: pcf.define_workspace
// The execute_as body receives the bind value plus workspace offsets.
// CHECK:     util.optimization_barrier %[[VAL]] : index
// CHECK:     util.optimization_barrier
// CHECK:     util.optimization_barrier

// -----

// Dynamic iteration space in define_workspace.
// CHECK-LABEL: func @define_workspace_dynamic
func.func @define_workspace_dynamic(%dim: index) {
  pcf.form_groups #pcf.sequential sizes [2] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %ws = pcf.define_workspace %b0
        iteration_space(%dim)
        indexing_maps = [affine_map<(m) -> (m)>]
        : !pcf.group<0, #pcf.sequential, {index, index}>
    pcf.execute_as [%ws] {
    ^bb0(%off: index, %sz: index):
      util.optimization_barrier %off : index
      util.optimization_barrier %sz : index
      pcf.return
    } : !pcf.group<0, #pcf.sequential, {index, index}>
    pcf.yield
  }
  return
}
// CHECK-SAME: %[[DIM:.*]]: index
// Dynamic dim is used directly in division.
// CHECK:     arith.divui %[[DIM]],
// CHECK:     util.optimization_barrier
// CHECK:     util.optimization_barrier

// -----

// Multi-worker groups with different sizes.
// Group 0 has 2 workers, group 1 has 3 workers.
// CHECK-LABEL: func @multi_worker_groups
func.func @multi_worker_groups() {
  pcf.form_groups #pcf.sequential sizes [2, 3] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>):
    pcf.execute_as [%b0] {
      pcf.return
    } : !pcf.group<0, #pcf.sequential>
    pcf.execute_as [%b1] {
      pcf.return
    } : !pcf.group<1, #pcf.sequential>
    pcf.yield
  }
  return
}
// Group index computed via select chain from cumulative sizes.
// Boundary at cumsum=2: if worker_id >= 2, idx = 1.
// CHECK:     %[[IDX:.*]] = arith.select
// CHECK:     scf.index_switch %[[IDX]]
// CHECK-NEXT: case 0 {
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: case 1 {
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }

// -----

// execute_as with i32 result.
// CHECK-LABEL: func @execute_as_with_i32_result
func.func @execute_as_with_i32_result() {
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %updated = pcf.execute_as [%b0]
        -> !pcf.group<0, #pcf.sequential, {i32}> {
      %c1 = arith.constant 1 : i32
      pcf.yield %c1 : i32
    } : !pcf.group<0, #pcf.sequential>
    pcf.yield
  }
  return
}
// No else block. Just the inlined value.
// CHECK-NOT: scf.if
// CHECK:     arith.constant 1 : i32
// CHECK:     return

// -----

// execute_as with f32 result.
// CHECK-LABEL: func @execute_as_with_f32_result
func.func @execute_as_with_f32_result() {
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %updated = pcf.execute_as [%b0]
        -> !pcf.group<0, #pcf.sequential, {f32}> {
      %c1 = arith.constant 1.0 : f32
      pcf.yield %c1 : f32
    } : !pcf.group<0, #pcf.sequential>
    pcf.yield
  }
  return
}
// CHECK-NOT: scf.if
// CHECK:     arith.constant 1.000000e+00 : f32
// CHECK:     return

// -----

// scf.for with a group-typed iter_arg (single group, no switch).
// The group type should be expanded to its struct element types.
// CHECK-LABEL: func @for_single_group_iter_arg
// CHECK-SAME:    %[[INIT:.*]]: tensor<4x8xf32>
func.func @for_single_group_iter_arg(%init: tensor<4x8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %bound = pcf.bind %b0, %init
        : !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>
    %final = scf.for %k = %c0 to %c4 step %c1
        iter_args(%iter = %bound)
        -> (!pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>) {
      %next = pcf.execute_as [%iter]
          -> !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}> {
        ^bb0(%acc: tensor<4x8xf32>):
          %new = util.optimization_barrier %acc : tensor<4x8xf32>
          pcf.yield %new : tensor<4x8xf32>
      } : !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>
      scf.yield %next : !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>
    }
    pcf.yield
  }
  return
}
// No switch for single group. Group type converted to tensor type.
// CHECK-NOT: scf.index_switch
// CHECK-NOT: pcf.group
// CHECK:     scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (tensor<4x8xf32>)
// CHECK:       %[[NEW:.*]] = util.optimization_barrier %[[ACC]]
// CHECK:       scf.yield %[[NEW]] : tensor<4x8xf32>

// -----

// scf.for with two group-typed iter_args across a 2-group switch.
// In each case branch, the matching group's iter_arg expands to its
// tensor type and the non-matching group's iter_arg is erased.
// CHECK-LABEL: func @for_two_groups_iter_args
// CHECK-SAME:    %[[INIT0:.*]]: tensor<4x8xf32>, %[[INIT1:.*]]: tensor<8x4xf32>
func.func @for_two_groups_iter_args(%init0: tensor<4x8xf32>,
                                     %init1: tensor<8x4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  pcf.form_groups #pcf.sequential sizes [1, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>):
    %bound0 = pcf.bind %b0, %init0
        : !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>
    %bound1 = pcf.bind %b1, %init1
        : !pcf.group<1, #pcf.sequential, {tensor<8x4xf32>}>
    %final0, %final1 = scf.for %k = %c0 to %c4 step %c1
        iter_args(%iter0 = %bound0, %iter1 = %bound1)
        -> (!pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>,
            !pcf.group<1, #pcf.sequential, {tensor<8x4xf32>}>) {
      %next0 = pcf.execute_as [%iter0]
          -> !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}> {
        ^bb0(%acc: tensor<4x8xf32>):
          %new = util.optimization_barrier %acc : tensor<4x8xf32>
          pcf.yield %new : tensor<4x8xf32>
      } : !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>

      pcf.barrier(#pcf.sequential)

      %next1 = pcf.execute_as [%iter1]
          -> !pcf.group<1, #pcf.sequential, {tensor<8x4xf32>}> {
        ^bb0(%acc: tensor<8x4xf32>):
          %new = util.optimization_barrier %acc : tensor<8x4xf32>
          pcf.yield %new : tensor<8x4xf32>
      } : !pcf.group<1, #pcf.sequential, {tensor<8x4xf32>}>

      pcf.barrier(#pcf.sequential)

      scf.yield %next0, %next1
          : !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>,
            !pcf.group<1, #pcf.sequential, {tensor<8x4xf32>}>
    }
    pcf.yield
  }
  return
}
// Case 0: only group 0's tensor survives as iter_arg.
// CHECK:     scf.index_switch
// CHECK:     case 0 {
// CHECK:       scf.for {{.*}} iter_args({{.*}} = %[[INIT0]]) -> (tensor<4x8xf32>)
// CHECK:         util.optimization_barrier {{.*}} : tensor<4x8xf32>
// CHECK:         pcf.barrier(#pcf.sequential)
// CHECK-NOT:     util.optimization_barrier{{.*}}tensor<8x4xf32>
// CHECK:         pcf.barrier(#pcf.sequential)
// CHECK:         scf.yield {{.*}} : tensor<4x8xf32>
// CHECK:       scf.yield
// CHECK:     }
// Case 1: only group 1's tensor survives as iter_arg.
// CHECK:     case 1 {
// CHECK:       scf.for {{.*}} iter_args({{.*}} = %[[INIT1]]) -> (tensor<8x4xf32>)
// CHECK-NOT:     util.optimization_barrier{{.*}}tensor<4x8xf32>
// CHECK:         pcf.barrier(#pcf.sequential)
// CHECK:         util.optimization_barrier {{.*}} : tensor<8x4xf32>
// CHECK:         pcf.barrier(#pcf.sequential)
// CHECK:         scf.yield {{.*}} : tensor<8x4xf32>
// CHECK:       scf.yield
// CHECK:     }

// -----

// scf.for with both group-typed and plain index iter_args.
// The group iter_arg is converted, the index iter_arg is preserved.
// CHECK-LABEL: func @for_mixed_iter_args
// CHECK-SAME:    %[[INIT:.*]]: tensor<4x8xf32>, %[[INIT_IDX:.*]]: index
func.func @for_mixed_iter_args(%init: tensor<4x8xf32>,
                                %init_idx: index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %bound = pcf.bind %b0, %init
        : !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>
    %final_g, %final_idx = scf.for %k = %c0 to %c4 step %c1
        iter_args(%iter = %bound, %iter_idx = %init_idx)
        -> (!pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>, index) {
      %next = pcf.execute_as [%iter]
          -> !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}> {
        ^bb0(%acc: tensor<4x8xf32>):
          %new = util.optimization_barrier %acc : tensor<4x8xf32>
          pcf.yield %new : tensor<4x8xf32>
      } : !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>
      %next_idx = arith.addi %iter_idx, %c1 : index
      scf.yield %next, %next_idx
          : !pcf.group<0, #pcf.sequential, {tensor<4x8xf32>}>, index
    }
    pcf.yield
  }
  return
}
// Group iter_arg expanded to tensor, index preserved.
// CHECK-NOT: pcf.group
// CHECK:     scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[IDX:.*]] = %[[INIT_IDX]]) -> (tensor<4x8xf32>, index)
// CHECK:       util.optimization_barrier %[[ACC]]
// CHECK:       arith.addi %[[IDX]]
// CHECK:       scf.yield {{.*}} : tensor<4x8xf32>, index

// -----

// Three groups: produces scf.index_switch with case 0, case 1, case 2, default.
// CHECK-LABEL: func @three_groups_index_switch
func.func @three_groups_index_switch() {
  pcf.form_groups #pcf.sequential sizes [1, 1, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>,
       %b2: !pcf.group<2, #pcf.sequential>):
    pcf.execute_as [%b0] {
      pcf.return
    } : !pcf.group<0, #pcf.sequential>
    pcf.execute_as [%b1] {
      pcf.return
    } : !pcf.group<1, #pcf.sequential>
    pcf.execute_as [%b2] {
      pcf.return
    } : !pcf.group<2, #pcf.sequential>
    pcf.yield
  }
  return
}
// Two select ops form the group index (boundaries at 1 and 2).
// CHECK:     %[[S1:.*]] = arith.select
// CHECK:     %[[S2:.*]] = arith.select
// CHECK:     scf.index_switch %[[S2]]
// CHECK-NEXT: case 0 {
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: case 1 {
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: case 2 {
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: default {
// CHECK-NEXT: }

// -----

// Multi-group: execute_as with result on group 0. In case 1, it is erased.
// CHECK-LABEL: func @execute_as_nonmatch_with_result
func.func @execute_as_nonmatch_with_result() {
  pcf.form_groups #pcf.sequential sizes [1, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>):
    %updated = pcf.execute_as [%b0]
        -> !pcf.group<0, #pcf.sequential, {index}> {
      %c42 = arith.constant 42 : index
      pcf.yield %c42 : index
    } : !pcf.group<0, #pcf.sequential>
    pcf.execute_as [%b1] {
      pcf.return
    } : !pcf.group<1, #pcf.sequential>
    pcf.yield
  }
  return
}
// Case 0: execute_as [%b0] result is inlined, execute_as [%b1] erased.
// CHECK:     scf.index_switch
// CHECK:     case 0 {
// CHECK:       arith.constant 42 : index
// CHECK:       scf.yield
// CHECK:     }
// Case 1: execute_as [%b0] (with result) is erased. No constant 42.
// CHECK:     case 1 {
// CHECK-NOT:   arith.constant 42
// CHECK:       scf.yield
// CHECK:     }

// -----

// Multi-group: bind on group 0. In case 1, bind and execute_as are erased.
// CHECK-LABEL: func @bind_nonmatch_multigroup
// CHECK-SAME:    %[[VAL:.*]]: index
func.func @bind_nonmatch_multigroup(%val: index) {
  pcf.form_groups #pcf.sequential sizes [1, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>):
    %bound = pcf.bind %b0, %val
        : !pcf.group<0, #pcf.sequential, {index}>
    pcf.execute_as [%bound] {
    ^bb0(%v: index):
      util.optimization_barrier %v : index
      pcf.return
    } : !pcf.group<0, #pcf.sequential, {index}>
    pcf.execute_as [%b1] {
      pcf.return
    } : !pcf.group<1, #pcf.sequential>
    pcf.yield
  }
  return
}
// Case 0: bind resolved, value flows to inlined execute_as body.
// CHECK:     scf.index_switch
// CHECK:     case 0 {
// CHECK:       util.optimization_barrier %[[VAL]] : index
// CHECK:       scf.yield
// CHECK:     }
// Case 1: bind and execute_as [%bound] erased. No optimization_barrier.
// CHECK:     case 1 {
// CHECK-NOT:   util.optimization_barrier
// CHECK:       scf.yield
// CHECK:     }

// -----

// Bind with multiple values. Both flow through to execute_as body.
// CHECK-LABEL: func @multi_element_bind
// CHECK-SAME:    %[[V1:.*]]: index, %[[V2:.*]]: f32
func.func @multi_element_bind(%v1: index, %v2: f32) {
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %bound = pcf.bind %b0, %v1, %v2
        : !pcf.group<0, #pcf.sequential, {index, f32}>
    pcf.execute_as [%bound] {
    ^bb0(%a: index, %b: f32):
      util.optimization_barrier %a : index
      util.optimization_barrier %b : f32
      pcf.return
    } : !pcf.group<0, #pcf.sequential, {index, f32}>
    pcf.yield
  }
  return
}
// CHECK-NOT: pcf.bind
// CHECK:     util.optimization_barrier %[[V1]] : index
// CHECK:     util.optimization_barrier %[[V2]] : f32
// CHECK:     return

// -----

// Values from the enclosing func scope are captured in the form_groups body.
// CHECK-LABEL: func @captured_from_enclosing
// CHECK-SAME:    %[[EXT:.*]]: index
func.func @captured_from_enclosing(%ext: index) {
  pcf.form_groups #pcf.sequential sizes [1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>):
    %v = util.optimization_barrier %ext : index
    pcf.execute_as [%b0] {
      pcf.return
    } : !pcf.group<0, #pcf.sequential>
    pcf.yield
  }
  return
}
// CHECK-NOT: pcf.form_groups
// CHECK:     util.optimization_barrier %[[EXT]] : index
// CHECK:     return

// -----

// Nested form_groups: inner is lowered before outer.
// CHECK-LABEL: func @nested_form_groups
func.func @nested_form_groups() {
  pcf.form_groups #pcf.sequential sizes [1, 1] {
  ^bb0(%b0: !pcf.group<0, #pcf.sequential>,
       %b1: !pcf.group<1, #pcf.sequential>):
    pcf.execute_as [%b0] {
      pcf.form_groups #pcf.sequential sizes [1] {
      ^bb0(%inner: !pcf.group<0, #pcf.sequential>):
        pcf.execute_as [%inner] {
          pcf.return
        } : !pcf.group<0, #pcf.sequential>
        pcf.yield
      }
      pcf.return
    } : !pcf.group<0, #pcf.sequential>
    pcf.execute_as [%b1] {
      pcf.return
    } : !pcf.group<1, #pcf.sequential>
    pcf.yield
  }
  return
}
// All group ops are lowered. Outer produces switch, inner fully resolved.
// CHECK:     scf.index_switch
// CHECK-NOT: pcf.form_groups
// CHECK-NOT: pcf.execute_as
// CHECK:     return
