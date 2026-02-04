// RUN: iree-opt --iree-template-inline-instances --split-input-file %s | FileCheck %s

// Test 1: Simple instance with no branches - just inline the main region.
module {
  func.func @test_simple(%arg0: index) -> index {
    %result = template.instance ins(%arg0 : index) -> index {
      %c1 = arith.constant 1 : index
      %sum = arith.addi %arg0, %c1 : index
      template.return %sum : index
    }
    return %result : index
  }
}

// CHECK-LABEL: func.func @test_simple
// CHECK-SAME:    (%[[ARG0:.*]]: index)
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[SUM:.*]] = arith.addi %[[ARG0]], %[[C1]] : index
// CHECK-NEXT:    return %[[SUM]] : index

// -----

// Test 2: Instance with a single branch.
module {
  func.func @test_single_branch(%arg0: index) -> tensor<4xf32> {
    %result = template.instance ins(%arg0 : index) -> tensor<4xf32> {
      %loaded = template.branch 0(%arg0) : (index) -> (tensor<4xf32>)
      template.return %loaded : tensor<4xf32>
    } {
    ^bb0(%idx: index):
      %empty = tensor.empty() : tensor<4xf32>
      template.return %empty : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @test_single_branch
// CHECK-SAME:    (%[[ARG0:.*]]: index)
// The branch should be inlined - tensor.empty appears directly.
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<4xf32>
// CHECK-NEXT:    return %[[EMPTY]] : tensor<4xf32>

// -----

// Test 3: Instance with multiple branches.
module {
  func.func @test_multiple_branches(%arg0: index) -> tensor<4xf32> {
    %result = template.instance ins(%arg0 : index) -> tensor<4xf32> {
      %a = template.branch 0(%arg0) : (index) -> (tensor<4xf32>)
      %b = template.branch 1(%a) : (tensor<4xf32>) -> (tensor<4xf32>)
      template.return %b : tensor<4xf32>
    } {
    ^bb0(%idx: index):
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

// CHECK-LABEL: func.func @test_multiple_branches
// CHECK-SAME:    (%[[ARG0:.*]]: index)
// First branch inlined.
// CHECK-NEXT:    %[[EMPTY:.*]] = tensor.empty() : tensor<4xf32>
// Second branch inlined.
// CHECK-NEXT:    %[[CST:.*]] = arith.constant dense<1.{{.*}}> : tensor<4xf32>
// CHECK-NEXT:    %[[ADDED:.*]] = arith.addf %[[EMPTY]], %[[CST]] : tensor<4xf32>
// CHECK-NEXT:    return %[[ADDED]] : tensor<4xf32>

// -----

// Test 4: Nested instances - inner should be inlined first.
module {
  func.func @test_nested(%arg0: index) -> index {
    %result = template.instance ins(%arg0 : index) -> index {
      %inner = template.instance ins(%arg0 : index) -> index {
        %c1 = arith.constant 1 : index
        %sum = arith.addi %arg0, %c1 : index
        template.return %sum : index
      }
      %c2 = arith.constant 2 : index
      %sum2 = arith.addi %inner, %c2 : index
      template.return %sum2 : index
    }
    return %result : index
  }
}

// CHECK-LABEL: func.func @test_nested
// CHECK-SAME:    (%[[ARG0:.*]]: index)
// Both instances should be fully inlined.
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[SUM1:.*]] = arith.addi %[[ARG0]], %[[C1]] : index
// CHECK-NEXT:    %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[SUM2:.*]] = arith.addi %[[SUM1]], %[[C2]] : index
// CHECK-NEXT:    return %[[SUM2]] : index

// -----

// Test 5: Instance with no results.
module {
  func.func @test_no_results(%arg0: index) {
    template.instance ins(%arg0 : index) {
      template.return
    }
    return
  }
}

// CHECK-LABEL: func.func @test_no_results
// CHECK-SAME:    (%[[ARG0:.*]]: index)
// The instance should be gone, just return remains.
// CHECK-NEXT:    return

// -----

// Test 6: Instance with multiple results.
module {
  func.func @test_multiple_results(%arg0: index) -> (index, index) {
    %r0, %r1 = template.instance ins(%arg0 : index) -> index, index {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %sum1 = arith.addi %arg0, %c1 : index
      %sum2 = arith.addi %arg0, %c2 : index
      template.return %sum1, %sum2 : index, index
    }
    return %r0, %r1 : index, index
  }
}

// CHECK-LABEL: func.func @test_multiple_results
// CHECK-SAME:    (%[[ARG0:.*]]: index)
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[SUM1:.*]] = arith.addi %[[ARG0]], %[[C1]] : index
// CHECK-NEXT:    %[[SUM2:.*]] = arith.addi %[[ARG0]], %[[C2]] : index
// CHECK-NEXT:    return %[[SUM1]], %[[SUM2]] : index, index

// -----

// Test 7: Branch with multiple arguments.
module {
  func.func @test_branch_multi_args(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %result = template.instance ins(%a, %b : tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> {
      %sum = template.branch 0(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
      template.return %sum : tensor<4xf32>
    } {
    ^bb0(%x: tensor<4xf32>, %y: tensor<4xf32>):
      %added = arith.addf %x, %y : tensor<4xf32>
      template.return %added : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @test_branch_multi_args
// CHECK-SAME:    (%[[A:.*]]: tensor<4xf32>, %[[B:.*]]: tensor<4xf32>)
// CHECK-NEXT:    %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : tensor<4xf32>
// CHECK-NEXT:    return %[[SUM]] : tensor<4xf32>

// -----

// Test 8: Same block branched to multiple times.
module {
  func.func @test_same_block_twice(%arg0: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %r0, %r1 = template.instance ins(%arg0 : index) -> tensor<4xf32>, tensor<4xf32> {
      %first = template.branch 0(%arg0) : (index) -> (tensor<4xf32>)
      %c1 = arith.constant 1 : index
      %next = arith.addi %arg0, %c1 : index
      %second = template.branch 0(%next) : (index) -> (tensor<4xf32>)
      template.return %first, %second : tensor<4xf32>, tensor<4xf32>
    } {
    ^bb0(%idx: index):
      %empty = tensor.empty() : tensor<4xf32>
      template.return %empty : tensor<4xf32>
    }
    return %r0, %r1 : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @test_same_block_twice
// CHECK-SAME:    (%[[ARG0:.*]]: index)
// Both branches to block 0 should be inlined separately.
// CHECK-NEXT:    %[[EMPTY1:.*]] = tensor.empty() : tensor<4xf32>
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[NEXT:.*]] = arith.addi %[[ARG0]], %[[C1]] : index
// CHECK-NEXT:    %[[EMPTY2:.*]] = tensor.empty() : tensor<4xf32>
// CHECK-NEXT:    return %[[EMPTY1]], %[[EMPTY2]] : tensor<4xf32>, tensor<4xf32>
