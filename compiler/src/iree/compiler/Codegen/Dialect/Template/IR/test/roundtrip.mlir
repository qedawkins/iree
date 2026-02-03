// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

func.func @test_instance_simple() -> tensor<256x256xf32> {
  %c0 = arith.constant 0 : index
  %result = template.instance -> tensor<256x256xf32> {
    %init = template.branch 0(%c0) : (index) -> (tensor<256x256xf32>)
    template.return %init : tensor<256x256xf32>
  } {
  ^bb0(%arg0: index):
    %empty = tensor.empty() : tensor<256x256xf32>
    template.return %empty : tensor<256x256xf32>
  }
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func.func @test_instance_simple() -> tensor<256x256xf32> {
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[RESULT:.+]] = template.instance -> tensor<256x256xf32> {
// CHECK:           %[[INIT:.+]] = template.branch 0(%[[C0]]) : (index) -> tensor<256x256xf32>
// CHECK:           template.return %[[INIT]] : tensor<256x256xf32>
// CHECK:         } {
// CHECK:         ^bb0(%[[ARG0:.+]]: index):
// CHECK:           %[[EMPTY:.+]] = tensor.empty() : tensor<256x256xf32>
// CHECK:           template.return %[[EMPTY]] : tensor<256x256xf32>
// CHECK:         }
// CHECK:         return %[[RESULT]] : tensor<256x256xf32>
// CHECK:       }

// -----

func.func @test_instance_multiple_blocks() -> tensor<4x8xf32> {
  %c0 = arith.constant 0 : index
  %result = template.instance -> tensor<4x8xf32> {
    %init = template.branch 0(%c0, %c0) : (index, index) -> (tensor<4x8xf32>)
    %lhs = tensor.empty() : tensor<4x4xf16>
    %rhs = tensor.empty() : tensor<4x8xf16>
    %computed = template.branch 1(%lhs, %rhs, %init) : (tensor<4x4xf16>, tensor<4x8xf16>, tensor<4x8xf32>) -> (tensor<4x8xf32>)
    template.return %computed : tensor<4x8xf32>
  } {
  ^bb0(%sg_id: index, %lane_id: index):
    %empty = tensor.empty() : tensor<4x8xf32>
    template.return %empty : tensor<4x8xf32>
  ^bb1(%lhs: tensor<4x4xf16>, %rhs: tensor<4x8xf16>, %acc: tensor<4x8xf32>):
    %result_inner = tensor.empty() : tensor<4x8xf32>
    template.return %result_inner : tensor<4x8xf32>
  }
  return %result : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_instance_multiple_blocks() -> tensor<4x8xf32> {
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[RESULT:.+]] = template.instance -> tensor<4x8xf32> {
// CHECK:           %[[INIT:.+]] = template.branch 0(%[[C0]], %[[C0]]) : (index, index) -> tensor<4x8xf32>
// CHECK:           %[[LHS2:.+]] = tensor.empty() : tensor<4x4xf16>
// CHECK:           %[[RHS2:.+]] = tensor.empty() : tensor<4x8xf16>
// CHECK:           %[[COMPUTED:.+]] = template.branch 1(%[[LHS2]], %[[RHS2]], %[[INIT]]) : (tensor<4x4xf16>, tensor<4x8xf16>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK:           template.return %[[COMPUTED]] : tensor<4x8xf32>
// CHECK:         } {
// CHECK:         ^bb0(%[[SG_ID:.+]]: index, %[[LANE_ID:.+]]: index):
// CHECK:           %[[EMPTY0:.+]] = tensor.empty() : tensor<4x8xf32>
// CHECK:           template.return %[[EMPTY0]] : tensor<4x8xf32>
// CHECK:         ^bb1(%[[LHS:.+]]: tensor<4x4xf16>, %[[RHS:.+]]: tensor<4x8xf16>, %[[ACC:.+]]: tensor<4x8xf32>):
// CHECK:           %[[EMPTY1:.+]] = tensor.empty() : tensor<4x8xf32>
// CHECK:           template.return %[[EMPTY1]] : tensor<4x8xf32>
// CHECK:         }
// CHECK:         return %[[RESULT]] : tensor<4x8xf32>
// CHECK:       }

// -----

module {
  template.func @test_func_with_unimplemented(%dest: tensor<256x256xf32>, %k_size: index) -> tensor<4x8xf32> {
    %c0 = arith.constant 0 : index
    %init = template.branch 0(%c0, %c0) : (index, index) -> (tensor<4x8xf32>)
    %lhs = tensor.empty() : tensor<4x4xf16>
    %rhs = tensor.empty() : tensor<4x8xf16>
    %computed = template.branch 1(%lhs, %rhs, %init) : (tensor<4x4xf16>, tensor<4x8xf16>, tensor<4x8xf32>) -> (tensor<4x8xf32>)
    template.return %computed : tensor<4x8xf32>
  } {
  ^bb0(%sg_id: index, %lane_id: index):
    template.unimplemented -> tensor<4x8xf32>
  ^bb1(%lhs: tensor<4x4xf16>, %rhs: tensor<4x8xf16>, %acc: tensor<4x8xf32>):
    template.unimplemented -> tensor<4x8xf32>
  }
}

// CHECK-LABEL: module {
// CHECK:         template.func @test_func_with_unimplemented (%[[DEST:.+]]: tensor<256x256xf32>, %[[K_SIZE:.+]]: index) -> tensor<4x8xf32> {
// CHECK:           %[[C0:.+]] = arith.constant 0 : index
// CHECK:           %[[INIT:.+]] = template.branch 0(%[[C0]], %[[C0]]) : (index, index) -> tensor<4x8xf32>
// CHECK:           %[[LHS2:.+]] = tensor.empty() : tensor<4x4xf16>
// CHECK:           %[[RHS2:.+]] = tensor.empty() : tensor<4x8xf16>
// CHECK:           %[[COMPUTED:.+]] = template.branch 1(%[[LHS2]], %[[RHS2]], %[[INIT]]) : (tensor<4x4xf16>, tensor<4x8xf16>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK:           template.return %[[COMPUTED]] : tensor<4x8xf32>
// CHECK:         } {
// CHECK:         ^bb0(%[[SG_ID:.+]]: index, %[[LANE_ID:.+]]: index):
// CHECK:           template.unimplemented -> tensor<4x8xf32>
// CHECK:         ^bb1(%[[LHS:.+]]: tensor<4x4xf16>, %[[RHS:.+]]: tensor<4x8xf16>, %[[ACC:.+]]: tensor<4x8xf32>):
// CHECK:           template.unimplemented -> tensor<4x8xf32>
// CHECK:         }
// CHECK:       }

// -----

module {
  template.func @test_func_with_mixed_blocks -> tensor<4x4xf32> {
    %c0 = arith.constant 0 : index
    %loaded = template.branch 0(%c0) : (index) -> (tensor<4x4xf32>)
    %result = template.branch 1(%loaded, %loaded) : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>)
    template.return %result : tensor<4x4xf32>
  } {
  ^bb0(%offset: index):
    // Concrete implementation provided.
    %tile = tensor.empty() : tensor<4x4xf32>
    template.return %tile : tensor<4x4xf32>
  ^bb1(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>):
    // Abstract - caller provides.
    template.unimplemented -> tensor<4x4xf32>
  }
}

// CHECK-LABEL: module {
// CHECK:         template.func @test_func_with_mixed_blocks  -> tensor<4x4xf32> {
// CHECK:           %[[C0:.+]] = arith.constant 0 : index
// CHECK:           %[[LOADED:.+]] = template.branch 0(%[[C0]]) : (index) -> tensor<4x4xf32>
// CHECK:           %[[RESULT:.+]] = template.branch 1(%[[LOADED]], %[[LOADED]]) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:           template.return %[[RESULT]] : tensor<4x4xf32>
// CHECK:         } {
// CHECK:         ^bb0(%[[OFFSET:.+]]: index):
// CHECK:           %[[TILE:.+]] = tensor.empty() : tensor<4x4xf32>
// CHECK:           template.return %[[TILE]] : tensor<4x4xf32>
// CHECK:         ^bb1(%[[A:.+]]: tensor<4x4xf32>, %[[B:.+]]: tensor<4x4xf32>):
// CHECK:           template.unimplemented -> tensor<4x4xf32>
// CHECK:         }
// CHECK:       }

// -----

func.func @test_instance_with_inputs(%arg0: tensor<?x?xf16>, %arg1: index) -> tensor<256x256xf32> {
  %result = template.instance ins(%arg0, %arg1 : tensor<?x?xf16>, index) -> tensor<256x256xf32> {
    %init = template.branch 0(%arg0, %arg1) : (tensor<?x?xf16>, index) -> (tensor<256x256xf32>)
    template.return %init : tensor<256x256xf32>
  } {
  ^bb0(%src: tensor<?x?xf16>, %offset: index):
    %empty = tensor.empty() : tensor<256x256xf32>
    template.return %empty : tensor<256x256xf32>
  }
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func.func @test_instance_with_inputs(
// CHECK-SAME:      %[[ARG0:.+]]: tensor<?x?xf16>, %[[ARG1:.+]]: index) -> tensor<256x256xf32> {
// CHECK:         %[[RESULT:.+]] = template.instance ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf16>, index) -> tensor<256x256xf32> {
// CHECK:           %[[INIT:.+]] = template.branch 0(%[[ARG0]], %[[ARG1]]) : (tensor<?x?xf16>, index) -> tensor<256x256xf32>
// CHECK:           template.return %[[INIT]] : tensor<256x256xf32>
// CHECK:         } {
// CHECK:         ^bb0(%[[SRC:.+]]: tensor<?x?xf16>, %[[OFFSET:.+]]: index):
// CHECK:           %[[EMPTY:.+]] = tensor.empty() : tensor<256x256xf32>
// CHECK:           template.return %[[EMPTY]] : tensor<256x256xf32>
// CHECK:         }
// CHECK:         return %[[RESULT]] : tensor<256x256xf32>
// CHECK:       }

// -----

func.func @test_branch_no_args_no_results() {
  template.instance {
    template.branch 0
    template.return
  } {
  ^bb0:
    template.return
  }
  return
}

// CHECK-LABEL: func.func @test_branch_no_args_no_results() {
// CHECK:         template.instance {
// CHECK:           template.branch 0
// CHECK:           template.return
// CHECK:         } {
// CHECK:           template.return
// CHECK:         }
// CHECK:         return
// CHECK:       }

// -----

func.func @test_instance_no_implementations() -> tensor<4x4xf32> {
  %result = template.instance -> tensor<4x4xf32> {
    %empty = tensor.empty() : tensor<4x4xf32>
    template.return %empty : tensor<4x4xf32>
  }
  return %result : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_instance_no_implementations() -> tensor<4x4xf32> {
// CHECK:         %[[RESULT:.+]] = template.instance -> tensor<4x4xf32> {
// CHECK:           %[[EMPTY:.+]] = tensor.empty() : tensor<4x4xf32>
// CHECK:           template.return %[[EMPTY]] : tensor<4x4xf32>
// CHECK:         }
// CHECK:         return %[[RESULT]] : tensor<4x4xf32>
// CHECK:       }

// -----

module {
  template.func @test_func_with_template_types(%m: index, %n: index) -> !template.type<2> {
    %c0 = arith.constant 0 : index
    %loaded = template.branch 0(%c0) : (index) -> (!template.type<0>)
    %loaded2 = template.branch 1(%c0) : (index) -> (!template.type<1>)
    %result = template.branch 2(%loaded, %loaded2) : (!template.type<0>, !template.type<1>) -> (!template.type<2>)
    template.return %result : !template.type<2>
  } {
  ^bb0(%idx: index):
    template.unimplemented -> !template.type<0>
  ^bb1(%idx2: index):
    template.unimplemented -> !template.type<1>
  ^bb2(%a: !template.type<0>, %b: !template.type<1>):
    template.unimplemented -> !template.type<2>
  }
}

// CHECK-LABEL: module {
// CHECK:         template.func @test_func_with_template_types (%[[M:.+]]: index, %[[N:.+]]: index) -> !template.type<2> {
// CHECK:           %[[C0:.+]] = arith.constant 0 : index
// CHECK:           %[[LOADED:.+]] = template.branch 0(%[[C0]]) : (index) -> !template.type<0>
// CHECK:           %[[LOADED2:.+]] = template.branch 1(%[[C0]]) : (index) -> !template.type<1>
// CHECK:           %[[RESULT:.+]] = template.branch 2(%[[LOADED]], %[[LOADED2]]) : (!template.type<0>, !template.type<1>) -> !template.type<2>
// CHECK:           template.return %[[RESULT]] : !template.type<2>
// CHECK:         } {
// CHECK:         ^bb0(%[[IDX:.+]]: index):
// CHECK:           template.unimplemented -> !template.type<0>
// CHECK:         ^bb1(%[[IDX2:.+]]: index):
// CHECK:           template.unimplemented -> !template.type<1>
// CHECK:         ^bb2(%[[A:.+]]: !template.type<0>, %[[B:.+]]: !template.type<1>):
// CHECK:           template.unimplemented -> !template.type<2>
// CHECK:         }
// CHECK:       }
