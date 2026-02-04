// RUN: iree-opt --iree-template-concretize-calls --split-input-file --verify-diagnostics %s

// Test 1: Missing callee error.
// Calling a non-existent template function should error.
module {
  func.func @test_missing_callee(%idx: index) -> index {
    // expected-error @below {{cannot find template function 'nonexistent'}}
    %result = template.call @nonexistent<>(%idx : index) -> index
    return %result : index
  }
}

// -----

// Test 2: Insufficient implementations provided.
// If the func has more unimplemented blocks than the call provides, error.
module {
  template.func @needs_two_impls(%arg0: index) -> tensor<4xf32> {
    %a = template.branch 0(%arg0) : (index) -> (tensor<4xf32>)
    %b = template.branch 1(%a) : (tensor<4xf32>) -> (tensor<4xf32>)
    template.return %b : tensor<4xf32>
  } {
  ^bb0(%idx: index):
    template.unimplemented -> tensor<4xf32>
  ^bb1(%input: tensor<4xf32>):
    template.unimplemented -> tensor<4xf32>
  }

  func.func @test_insufficient_impls(%idx: index) -> tensor<4xf32> {
    // Only providing one implementation when two are needed.
    // expected-error @below {{'template.instance' op implementation blocks must be terminated with template.return}}
    %result = template.call @needs_two_impls<>(%idx : index) -> tensor<4xf32> {
    ^bb0(%inner_idx: index):
      %empty = tensor.empty() : tensor<4xf32>
      template.return %empty : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}

// -----

// Test 3: Direct recursion.
// A template func that calls itself should be detected.
module {
  template.func @recursive(%arg0: index) -> index {
    %result = template.call @recursive<>(%arg0 : index) -> index
    template.return %result : index
  }

  func.func @test_direct_recursion(%idx: index) -> index {
    // expected-error @below {{recursive template call detected}}
    %result = template.call @recursive<>(%idx : index) -> index
    return %result : index
  }
}

// -----

// Test 4: Indirect recursion cycle (A -> B -> C -> A).
// A chain of template func calls that forms a cycle.
module {
  template.func @func_a(%arg0: index) -> index {
    %result = template.call @func_b<>(%arg0 : index) -> index
    template.return %result : index
  }

  template.func @func_b(%arg0: index) -> index {
    %result = template.call @func_c<>(%arg0 : index) -> index
    template.return %result : index
  }

  template.func @func_c(%arg0: index) -> index {
    %result = template.call @func_a<>(%arg0 : index) -> index
    template.return %result : index
  }

  func.func @test_indirect_recursion(%idx: index) -> index {
    // expected-error @below {{recursive template call detected}}
    %result = template.call @func_a<>(%idx : index) -> index
    return %result : index
  }
}

// -----

// Test 5: Multiple calls where second recurses.
// First call is safe, but second call creates recursion.
module {
  template.func @safe(%arg0: index) -> index {
    template.return %arg0 : index
  }

  template.func @outer(%arg0: index) -> index {
    %r1 = template.call @safe<>(%arg0 : index) -> index
    %r2 = template.call @outer<>(%r1 : index) -> index
    template.return %r2 : index
  }

  func.func @test_second_recurses(%idx: index) -> index {
    // expected-error @below {{recursive template call detected}}
    %result = template.call @outer<>(%idx : index) -> index
    return %result : index
  }
}
