// RUN: iree-opt --iree-template-concretize-calls --split-input-file %s | FileCheck %s

// Test 1: Simple conversion with no template types.
// The template.call should be converted to template.instance with cloned body.
module {
  template.func @identity(%arg0: index) -> index {
    template.return %arg0 : index
  }

  func.func @test_simple(%idx: index) -> index {
    %result = template.call @identity<>(%idx : index) -> index
    return %result : index
  }
}

// CHECK-LABEL: func.func @test_simple
// CHECK-SAME:    (%[[ARG0:.*]]: index)
// CHECK:         %[[RESULT:.*]] = template.instance ins(%[[ARG0]] : index) -> index {
// CHECK-NEXT:      template.return %[[ARG0]] : index
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[RESULT]] : index

// -----

// Test 2: Conversion with implementation blocks.
// Unimplemented blocks in the func should be replaced with call-provided implementations.
module {
  template.func @with_impl(%arg0: index) -> tensor<4xf32> {
    %result = template.branch 0(%arg0) : (index) -> (tensor<4xf32>)
    template.return %result : tensor<4xf32>
  } {
  ^bb0(%idx: index):
    template.unimplemented -> tensor<4xf32>
  }

  func.func @test_with_impl(%idx: index) -> tensor<4xf32> {
    %result = template.call @with_impl<>(%idx : index) -> tensor<4xf32> {
    ^bb0(%inner_idx: index):
      %empty = tensor.empty() : tensor<4xf32>
      template.return %empty : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @test_with_impl
// CHECK:         template.instance ins(%{{.*}} : index) -> tensor<4xf32> {
// CHECK:           template.branch 0
// CHECK:         } {
// CHECK:         ^bb0(%[[IMPL_ARG:.*]]: index):
// The unimplemented op should be replaced with the provided implementation.
// CHECK-NOT:       template.unimplemented
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<4xf32>
// CHECK-NEXT:      template.return %[[EMPTY]] : tensor<4xf32>
// CHECK:         }

// -----

// Test 3: Conversion with type bindings.
// !template.type<N> should be replaced with concrete types.
module {
  template.func @with_types(%arg0: index) -> !template.type<0> {
    %result = template.branch 0(%arg0) : (index) -> (!template.type<0>)
    template.return %result : !template.type<0>
  } {
  ^bb0(%idx: index):
    template.unimplemented -> !template.type<0>
  }

  func.func @test_type_binding(%idx: index) -> tensor<8xf32> {
    %result = template.call @with_types<tensor<8xf32>>(%idx : index) -> tensor<8xf32> {
    ^bb0(%inner_idx: index):
      %empty = tensor.empty() : tensor<8xf32>
      template.return %empty : tensor<8xf32>
    }
    return %result : tensor<8xf32>
  }
}

// CHECK-LABEL: func.func @test_type_binding
// CHECK:         template.instance ins(%{{.*}} : index) -> tensor<8xf32> {
// The template.type<0> should be replaced with tensor<8xf32>.
// CHECK:           %[[BR:.*]] = template.branch 0(%{{.*}}) : (index) -> tensor<8xf32>
// CHECK:           template.return %[[BR]] : tensor<8xf32>
// CHECK:         } {
// CHECK:         ^bb0(%{{.*}}: index):
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<8xf32>
// CHECK:           template.return %[[EMPTY]] : tensor<8xf32>
// CHECK:         }

// -----

// Test 4: Conversion with multiple implementation blocks.
// Multiple unimplemented blocks should all be replaced.
module {
  template.func @multi_impl(%arg0: index) -> tensor<4xf32> {
    %a = template.branch 0(%arg0) : (index) -> (tensor<4xf32>)
    %b = template.branch 1(%a) : (tensor<4xf32>) -> (tensor<4xf32>)
    template.return %b : tensor<4xf32>
  } {
  ^bb0(%idx: index):
    template.unimplemented -> tensor<4xf32>
  ^bb1(%input: tensor<4xf32>):
    template.unimplemented -> tensor<4xf32>
  }

  func.func @test_multi_impl(%idx: index) -> tensor<4xf32> {
    %result = template.call @multi_impl<>(%idx : index) -> tensor<4xf32> {
    ^bb0(%inner_idx: index):
      %empty = tensor.empty() : tensor<4xf32>
      template.return %empty : tensor<4xf32>
    ^bb1(%input: tensor<4xf32>):
      %cst = arith.constant dense<1.0> : tensor<4xf32>
      %added = arith.addf %input, %cst : tensor<4xf32>
      template.return %added : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @test_multi_impl
// CHECK:         template.instance ins(%{{.*}} : index) -> tensor<4xf32> {
// CHECK:           template.branch 0
// CHECK:           template.branch 1
// CHECK:         } {
// First impl block: tensor.empty
// CHECK:         ^bb0(%{{.*}}: index):
// CHECK-NOT:       template.unimplemented
// CHECK:           tensor.empty() : tensor<4xf32>
// CHECK:           template.return
// Second impl block: arith.addf
// CHECK:         ^bb1(%[[INPUT:.*]]: tensor<4xf32>):
// CHECK-NOT:       template.unimplemented
// CHECK:           %[[CST:.*]] = arith.constant dense<1.{{.*}}> : tensor<4xf32>
// CHECK:           %[[ADD:.*]] = arith.addf %[[INPUT]], %[[CST]] : tensor<4xf32>
// CHECK:           template.return %[[ADD]] : tensor<4xf32>
// CHECK:         }

// -----

// Test 5: Call with no results.
module {
  template.func @no_results(%arg0: index) {
    template.return
  }

  func.func @test_no_results(%idx: index) {
    template.call @no_results<>(%idx : index)
    return
  }
}

// CHECK-LABEL: func.func @test_no_results
// CHECK-SAME:    (%[[ARG0:.*]]: index)
// Instance should have no result types.
// CHECK:         template.instance ins(%[[ARG0]] : index) {
// CHECK-NEXT:      template.return
// CHECK-NEXT:    }
// CHECK-NEXT:    return

// -----

// Test 6: Call with multiple type bindings.
module {
  template.func @multi_types(%arg0: index) -> !template.type<1> {
    %loaded = template.branch 0(%arg0) : (index) -> (!template.type<0>)
    %result = template.branch 1(%loaded) : (!template.type<0>) -> (!template.type<1>)
    template.return %result : !template.type<1>
  } {
  ^bb0(%idx: index):
    template.unimplemented -> !template.type<0>
  ^bb1(%input: !template.type<0>):
    template.unimplemented -> !template.type<1>
  }

  func.func @test_multi_types(%idx: index) -> tensor<8xf32> {
    %result = template.call @multi_types<tensor<4xf16>, tensor<8xf32>>(%idx : index) -> tensor<8xf32> {
    ^bb0(%inner_idx: index):
      %empty = tensor.empty() : tensor<4xf16>
      template.return %empty : tensor<4xf16>
    ^bb1(%input: tensor<4xf16>):
      %out = tensor.empty() : tensor<8xf32>
      template.return %out : tensor<8xf32>
    }
    return %result : tensor<8xf32>
  }
}

// CHECK-LABEL: func.func @test_multi_types
// CHECK:         template.instance ins(%{{.*}} : index) -> tensor<8xf32> {
// type<0> = tensor<4xf16>, type<1> = tensor<8xf32>
// CHECK:           %[[BR0:.*]] = template.branch 0(%{{.*}}) : (index) -> tensor<4xf16>
// CHECK:           %[[BR1:.*]] = template.branch 1(%[[BR0]]) : (tensor<4xf16>) -> tensor<8xf32>
// CHECK:           template.return %[[BR1]] : tensor<8xf32>
// CHECK:         } {
// CHECK:         ^bb0(%{{.*}}: index):
// CHECK:           tensor.empty() : tensor<4xf16>
// CHECK:         ^bb1(%{{.*}}: tensor<4xf16>):
// CHECK:           tensor.empty() : tensor<8xf32>
// CHECK:         }

// -----

// Test 7: Templated function arguments.
// Template type parameters in function signature should be replaced.
module {
  template.func @with_templated_args(%arg0: !template.type<0>, %arg1: index) -> !template.type<0> {
    template.return %arg0 : !template.type<0>
  }

  func.func @test_templated_args(%t: tensor<4xf32>, %idx: index) -> tensor<4xf32> {
    %result = template.call @with_templated_args<tensor<4xf32>>(%t, %idx : tensor<4xf32>, index) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @test_templated_args
// CHECK-SAME:    (%[[T:.*]]: tensor<4xf32>, %[[IDX:.*]]: index)
// Instance inputs should have converted types.
// CHECK:         %[[RESULT:.*]] = template.instance ins(%[[T]], %[[IDX]] : tensor<4xf32>, index) -> tensor<4xf32> {
// CHECK-NEXT:      template.return %[[T]] : tensor<4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xf32>

// -----

// Test 8: Multiple calls to same func with different bindings.
// Same template function called multiple times with different type bindings.
module {
  template.func @generic(%arg0: !template.type<0>) -> !template.type<0> {
    template.return %arg0 : !template.type<0>
  }

  func.func @test_multiple_calls(%a: i32, %b: f32) -> (i32, f32) {
    %r1 = template.call @generic<i32>(%a : i32) -> i32
    %r2 = template.call @generic<f32>(%b : f32) -> f32
    return %r1, %r2 : i32, f32
  }
}

// CHECK-LABEL: func.func @test_multiple_calls
// CHECK-SAME:    (%[[A:.*]]: i32, %[[B:.*]]: f32)
// First instance with i32.
// CHECK:         %[[R1:.*]] = template.instance ins(%[[A]] : i32) -> i32 {
// CHECK-NEXT:      template.return %[[A]] : i32
// CHECK-NEXT:    }
// Second instance with f32.
// CHECK:         %[[R2:.*]] = template.instance ins(%[[B]] : f32) -> f32 {
// CHECK-NEXT:      template.return %[[B]] : f32
// CHECK-NEXT:    }
// CHECK:         return %[[R1]], %[[R2]] : i32, f32

// -----

// Test 9: Chained template func calls.
// One template func calls another with concrete types.
module {
  template.func @inner(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    template.return %arg0 : tensor<4xf32>
  }

  template.func @outer(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %result = template.call @inner<>(%arg0 : tensor<4xf32>) -> tensor<4xf32>
    template.return %result : tensor<4xf32>
  }

  func.func @test_chained(%t: tensor<4xf32>) -> tensor<4xf32> {
    %result = template.call @outer<>(%t : tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @test_chained
// CHECK-SAME:    (%[[T:.*]]: tensor<4xf32>)
// The outer call should be converted, and the nested inner call should also be converted.
// CHECK:         %[[RESULT:.*]] = template.instance ins(%[[T]] : tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           %[[INNER:.*]] = template.instance ins(%[[T]] : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:        template.return %[[T]] : tensor<4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      template.return %[[INNER]] : tensor<4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xf32>

// -----

// Test 10: Templated iter_args in scf.for.
// Template types in scf.for iter_args should be converted.
module {
  template.func @with_scf_for(%arg0: !template.type<0>, %n: index) -> !template.type<0> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %arg0) -> (!template.type<0>) {
      scf.yield %acc : !template.type<0>
    }
    template.return %result : !template.type<0>
  }

  func.func @test_scf_for(%t: tensor<4xf32>, %n: index) -> tensor<4xf32> {
    %result = template.call @with_scf_for<tensor<4xf32>>(%t, %n : tensor<4xf32>, index) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @test_scf_for
// CHECK-SAME:    (%[[T:.*]]: tensor<4xf32>, %[[N:.*]]: index)
// CHECK:         template.instance ins(%[[T]], %[[N]] : tensor<4xf32>, index) -> tensor<4xf32> {
// The scf.for iter_args type should be tensor<4xf32>, not !template.type<0>.
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[FOR:.*]] = scf.for %{{.*}} = %[[C0]] to %[[N]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[T]]) -> (tensor<4xf32>) {
// CHECK-NEXT:        scf.yield %[[ACC]] : tensor<4xf32>
// CHECK-NEXT:      }
// CHECK:           template.return %[[FOR]] : tensor<4xf32>
// CHECK:         }

// -----

// Test 11: Templated result type in scf.if.
// Template types in scf.if results should be converted.
module {
  template.func @with_scf_if(%cond: i1, %a: !template.type<0>, %b: !template.type<0>) -> !template.type<0> {
    %result = scf.if %cond -> (!template.type<0>) {
      scf.yield %a : !template.type<0>
    } else {
      scf.yield %b : !template.type<0>
    }
    template.return %result : !template.type<0>
  }

  func.func @test_scf_if(%cond: i1, %a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
    %result = template.call @with_scf_if<tensor<8xf32>>(%cond, %a, %b : i1, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}

// CHECK-LABEL: func.func @test_scf_if
// CHECK-SAME:    (%[[COND:.*]]: i1, %[[A:.*]]: tensor<8xf32>, %[[B:.*]]: tensor<8xf32>)
// CHECK:         template.instance ins(%[[COND]], %[[A]], %[[B]] : i1, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32> {
// The scf.if result type should be tensor<8xf32>, not !template.type<0>.
// CHECK:           %[[IF:.*]] = scf.if %[[COND]] -> (tensor<8xf32>) {
// CHECK-NEXT:        scf.yield %[[A]] : tensor<8xf32>
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %[[B]] : tensor<8xf32>
// CHECK-NEXT:      }
// CHECK:           template.return %[[IF]] : tensor<8xf32>
// CHECK:         }

// -----

// Test 12: Templated iter_args in scf.while.
// Template types in scf.while iter_args should be converted.
module {
  template.func @with_scf_while(%init: !template.type<0>, %limit: index) -> !template.type<0> {
    %c0 = arith.constant 0 : index
    %result = scf.while (%arg = %init, %i = %c0) : (!template.type<0>, index) -> (!template.type<0>) {
      %cond = arith.cmpi slt, %i, %limit : index
      scf.condition(%cond) %arg : !template.type<0>
    } do {
    ^bb0(%arg: !template.type<0>):
      scf.yield %arg, %c0 : !template.type<0>, index
    }
    template.return %result : !template.type<0>
  }

  func.func @test_scf_while(%init: tensor<4xf32>, %limit: index) -> tensor<4xf32> {
    %result = template.call @with_scf_while<tensor<4xf32>>(%init, %limit : tensor<4xf32>, index) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @test_scf_while
// CHECK-SAME:    (%[[INIT:.*]]: tensor<4xf32>, %[[LIMIT:.*]]: index)
// CHECK:         template.instance ins(%[[INIT]], %[[LIMIT]] : tensor<4xf32>, index) -> tensor<4xf32> {
// The scf.while iter_args types should be tensor<4xf32>, not !template.type<0>.
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[WHILE:.*]] = scf.while (%[[ARG:.*]] = %[[INIT]], %[[I:.*]] = %[[C0]]) : (tensor<4xf32>, index) -> tensor<4xf32> {
// CHECK:             %[[COND:.*]] = arith.cmpi slt, %[[I]], %[[LIMIT]] : index
// CHECK-NEXT:        scf.condition(%[[COND]]) %[[ARG]] : tensor<4xf32>
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%[[ARG2:.*]]: tensor<4xf32>):
// CHECK-NEXT:        scf.yield %[[ARG2]], %[[C0]] : tensor<4xf32>, index
// CHECK-NEXT:      }
// CHECK:           template.return %[[WHILE]] : tensor<4xf32>
// CHECK:         }

// -----

// Test 13: Multi-type template expansion.
// A single type parameter binding expands to multiple types.
module {
  template.func @multi_type_expand(%arg0: index, %arg1: !template.type<0>) -> !template.type<0> {
    template.return %arg1 : !template.type<0>
  }

  func.func @test_multi_type_expand(%idx: index, %a: i32, %b: f32) -> (i32, f32) {
    %r1, %r2 = template.call @multi_type_expand<[i32, f32]>(%idx, %a, %b : index, i32, f32) -> (i32, f32)
    return %r1, %r2 : i32, f32
  }
}

// CHECK-LABEL: func.func @test_multi_type_expand
// CHECK-SAME:    (%[[IDX:.*]]: index, %[[A:.*]]: i32, %[[B:.*]]: f32)
// Instance should have two results from 1:N expansion.
// CHECK:         %[[R:.*]]:2 = template.instance ins(%[[IDX]], %[[A]], %[[B]] : index, i32, f32) -> i32, f32 {
// The return should have the unpacked values.
// CHECK:           template.return %[[A]], %[[B]] : i32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[R]]#0, %[[R]]#1 : i32, f32

// -----

// Test 14: Chained calls with template type propagation.
// Inner function is generic, outer calls it with its own template type parameter.
// The type should propagate through both levels.
module {
  template.func @inner_generic(%arg0: !template.type<0>) -> !template.type<0> {
    template.return %arg0 : !template.type<0>
  }

  template.func @outer_generic(%arg0: !template.type<0>) -> !template.type<0> {
    // Call inner with the same template type - should propagate.
    %result = template.call @inner_generic<!template.type<0>>(%arg0 : !template.type<0>) -> !template.type<0>
    template.return %result : !template.type<0>
  }

  func.func @test_chained_generic(%t: tensor<4xf32>) -> tensor<4xf32> {
    %result = template.call @outer_generic<tensor<4xf32>>(%t : tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @test_chained_generic
// CHECK-SAME:    (%[[T:.*]]: tensor<4xf32>)
// Both outer and inner should have tensor<4xf32> substituted.
// CHECK:         %[[RESULT:.*]] = template.instance ins(%[[T]] : tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           %[[INNER:.*]] = template.instance ins(%[[T]] : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:        template.return %[[T]] : tensor<4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      template.return %[[INNER]] : tensor<4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xf32>

// -----

// Test 15: Chained calls with type expansion that appends to template type.
// Inner function takes a 1:N expanded type, outer appends a concrete type to its template param.
module {
  template.func @inner_multi(%arg0: index, %arg1: !template.type<0>) -> !template.type<0> {
    template.return %arg1 : !template.type<0>
  }

  template.func @outer_append(%arg0: index, %arg1: !template.type<0>, %arg2: i32) -> (!template.type<0>, i32) {
    // Call inner with [!template.type<0>, i32] - appends i32 to whatever type<0> is.
    %r1, %r2 = template.call @inner_multi<[!template.type<0>, i32]>(%arg0, %arg1, %arg2 : index, !template.type<0>, i32) -> (!template.type<0>, i32)
    template.return %r1, %r2 : !template.type<0>, i32
  }

  func.func @test_chained_append(%idx: index, %t: tensor<4xf32>, %i: i32) -> (tensor<4xf32>, i32) {
    // Bind type<0> = tensor<4xf32>, so inner gets [tensor<4xf32>, i32].
    %r1, %r2 = template.call @outer_append<tensor<4xf32>>(%idx, %t, %i : index, tensor<4xf32>, i32) -> (tensor<4xf32>, i32)
    return %r1, %r2 : tensor<4xf32>, i32
  }
}

// CHECK-LABEL: func.func @test_chained_append
// CHECK-SAME:    (%[[IDX:.*]]: index, %[[T:.*]]: tensor<4xf32>, %[[I:.*]]: i32)
// Outer should have tensor<4xf32> for type<0>.
// CHECK:         %[[R:.*]]:2 = template.instance ins(%[[IDX]], %[[T]], %[[I]] : index, tensor<4xf32>, i32) -> tensor<4xf32>, i32 {
// Inner should have [tensor<4xf32>, i32] expansion - two results.
// CHECK:           %[[INNER:.*]]:2 = template.instance ins(%[[IDX]], %[[T]], %[[I]] : index, tensor<4xf32>, i32) -> tensor<4xf32>, i32 {
// CHECK:             template.return %[[T]], %[[I]] : tensor<4xf32>, i32
// CHECK-NEXT:      }
// CHECK:           template.return %[[INNER]]#0, %[[INNER]]#1 : tensor<4xf32>, i32
// CHECK-NEXT:    }
// CHECK:         return %[[R]]#0, %[[R]]#1 : tensor<4xf32>, i32

