// RUN: iree-opt --iree-pcf-test-value-equivalences --allow-unregistered-dialect --verify-diagnostics --split-input-file %s

// Value passing through scf.for iter_args. The seed value flows through the
// loop-carried variable, so it should be in the same equivalence class.
func.func @for_iter_args(%lb: index, %ub: index, %step: index) {
  // expected-remark @below {{equivalence class: {self}}}
  // expected-remark @below {{seed is leader of its class}}
  %seed = "test.produce"() {test.seed} : () -> (f32)
  %result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %seed) -> f32 {
    scf.yield %arg : f32
  }
  "test.consume"(%result) : (f32) -> ()
  return
}

// -----

// Value passing through scf.if. The seed flows through both branches.
func.func @if_both_branches(%cond: i1) {
  // expected-remark @below {{equivalence class: {self}}}
  // expected-remark @below {{seed is leader of its class}}
  %seed = "test.produce"() {test.seed} : () -> (f32)
  %result = scf.if %cond -> f32 {
    scf.yield %seed : f32
  } else {
    scf.yield %seed : f32
  }
  "test.consume"(%result) : (f32) -> ()
  return
}

// -----

// Two independent seeds with separate equivalence classes.
func.func @two_independent_seeds() {
  // expected-remark @below {{equivalence class: {self}}}
  // expected-remark @below {{seed is leader of its class}}
  %seed0 = "test.produce"() {test.seed} : () -> (f32)
  // expected-remark @below {{equivalence class: {self}}}
  // expected-remark @below {{seed is leader of its class}}
  %seed1 = "test.produce"() {test.seed} : () -> (i32)
  "test.consume"(%seed0) : (f32) -> ()
  "test.consume"(%seed1) : (i32) -> ()
  return
}

// -----

// Value through scf.for inside scf.if -- nested control flow.
func.func @for_inside_if(%cond: i1, %lb: index, %ub: index, %step: index) {
  // expected-remark @below {{equivalence class: {self}}}
  // expected-remark @below {{seed is leader of its class}}
  %seed = "test.produce"() {test.seed} : () -> (f32)
  %result = scf.if %cond -> f32 {
    %inner = scf.for %iv = %lb to %ub step %step iter_args(%arg = %seed) -> f32 {
      scf.yield %arg : f32
    }
    scf.yield %inner : f32
  } else {
    scf.yield %seed : f32
  }
  "test.consume"(%result) : (f32) -> ()
  return
}

// -----

// Value through scf.if inside scf.for -- the other nesting order.
func.func @if_inside_for(%cond: i1, %lb: index, %ub: index, %step: index) {
  // expected-remark @below {{equivalence class: {self}}}
  // expected-remark @below {{seed is leader of its class}}
  %seed = "test.produce"() {test.seed} : () -> (f32)
  %result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %seed) -> f32 {
    %inner = scf.if %cond -> f32 {
      scf.yield %arg : f32
    } else {
      scf.yield %arg : f32
    }
    scf.yield %inner : f32
  }
  "test.consume"(%result) : (f32) -> ()
  return
}

// -----

// Two seeds that converge -- both fed into the same scf.for iter_arg.
// They should end up in the same equivalence class because they are
// both traced to the same loop-carried variable.
func.func @converging_seeds(%cond: i1, %lb: index, %ub: index, %step: index) {
  // expected-remark @below {{seed is equivalent to leader}}
  %seed0 = "test.produce"() {test.seed} : () -> (f32)
  // expected-remark @below {{equivalence class with 2 seeds}}
  // expected-remark @below {{seed is leader of its class}}
  %seed1 = "test.produce"() {test.seed} : () -> (f32)
  %init = scf.if %cond -> f32 {
    scf.yield %seed0 : f32
  } else {
    scf.yield %seed1 : f32
  }
  %result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> f32 {
    scf.yield %arg : f32
  }
  "test.consume"(%result) : (f32) -> ()
  return
}

// -----

// Value through scf.index_switch with multiple cases.
func.func @index_switch(%idx: index) {
  // expected-remark @below {{equivalence class: {self}}}
  // expected-remark @below {{seed is leader of its class}}
  %seed = "test.produce"() {test.seed} : () -> (f32)
  %result = scf.index_switch %idx -> f32
  case 0 {
    scf.yield %seed : f32
  }
  case 1 {
    scf.yield %seed : f32
  }
  default {
    scf.yield %seed : f32
  }
  "test.consume"(%result) : (f32) -> ()
  return
}

// -----

// Value passing through scf.while. The seed flows through the before region
// arg, the condition operand, and the after region arg back to the result.
func.func @while_loop() {
  // expected-remark @below {{equivalence class: {self}}}
  // expected-remark @below {{seed is leader of its class}}
  %seed = "test.produce"() {test.seed} : () -> (f32)
  %result = scf.while (%arg = %seed) : (f32) -> f32 {
    %cond = "test.cond"() : () -> i1
    scf.condition(%cond) %arg : f32
  } do {
  ^bb0(%body_arg: f32):
    scf.yield %body_arg : f32
  }
  "test.consume"(%result) : (f32) -> ()
  return
}
