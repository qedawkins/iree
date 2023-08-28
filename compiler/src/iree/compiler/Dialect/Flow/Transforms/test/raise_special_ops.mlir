// RUN: iree-opt --iree-flow-raise-special-ops -canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: @softmax
//  CHECK-SAME: %[[ARG:.+]]: tensor<?x?x?xf32>
//       CHECK:   %[[E:.+]] = tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x?x?xf32>
//       CHECK:   %[[S:.+]] = iree_linalg_ext.softmax dimension(2) ins(%[[ARG]] : tensor<?x?x?xf32>) outs(%[[E]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
//       CHECK:   return %[[S]] : tensor<?x?x?xf32>

func.func @softmax(%src : tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -3.40282347E+38 : f32
  %c_0_index = arith.constant 0 : index
  %c_1_index = arith.constant 1 : index
  %c_2_index = arith.constant 2 : index
  %dim_0 = tensor.dim %src, %c_0_index : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %src, %c_1_index : tensor<?x?x?xf32>
  %dim_2 = tensor.dim %src, %c_2_index : tensor<?x?x?xf32>
  %1 = tensor.empty(%dim_0, %dim_1) : tensor<?x?xf32>
  %2 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<?x?x?xf32>) outs(%2 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.maxf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  %4 = tensor.empty(%dim_0, %dim_1, %dim_2) : tensor<?x?x?xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%src, %3 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %11 = arith.subf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?x?xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<?x?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = math.exp %arg0 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?x?xf32>
  %7 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : tensor<?x?x?xf32>) outs(%7 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.addf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<?x?xf32>) outs(%1 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.divf %cst, %arg0 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %9 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %11 = arith.mulf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?x?xf32>
  return %10 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @softmax_no_rcp
//  CHECK-SAME: %[[ARG:.+]]: tensor<10x4096x4096xf16>
//       CHECK:   %[[E:.+]] = tensor.empty() : tensor<10x4096x4096xf16>
//       CHECK:   %[[S:.+]] = iree_linalg_ext.softmax dimension(2) ins(%[[ARG]] : tensor<10x4096x4096xf16>) outs(%[[E]] : tensor<10x4096x4096xf16>) -> tensor<10x4096x4096xf16>
//       CHECK:   return %[[S]] : tensor<10x4096x4096xf16>
func.func @softmax_no_rcp(%src : tensor<10x4096x4096xf16>) -> (tensor<10x4096x4096xf16>) {
  %cst_158 = arith.constant -6.550400e+04 : f16
  %cst_121 = arith.constant 0.000000e+00 : f16
  %224 = tensor.empty() : tensor<10x4096xf16>
  %216 = tensor.empty() : tensor<10x4096x4096xf16>
  %225 = linalg.fill ins(%cst_158 : f16) outs(%224 : tensor<10x4096xf16>) -> tensor<10x4096xf16>
  %226 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<10x4096x4096xf16>) outs(%225 : tensor<10x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5290 = arith.maxf %in, %out : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096xf16>
  %227 = linalg.generic
  {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%src, %226 : tensor<10x4096x4096xf16>, tensor<10x4096xf16>) outs(%216 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %in_1572: f16, %out: f16):
    %5290 = arith.subf %in, %in_1572 : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096x4096xf16>
  %228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%227 : tensor<10x4096x4096xf16>) outs(%216 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5290 = math.exp %in : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096x4096xf16>
  %229 = tensor.empty() : tensor<10x4096xf16>
  %230 = linalg.fill ins(%cst_121 : f16) outs(%229 : tensor<10x4096xf16>) -> tensor<10x4096xf16>
  %231 = linalg.generic 
  {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel", "reduction"]}
  ins(%228 : tensor<10x4096x4096xf16>) outs(%230 : tensor<10x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5290 = arith.addf %in, %out : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096xf16>
  %232 = linalg.generic 
  {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%228, %231 : tensor<10x4096x4096xf16>, tensor<10x4096xf16>) outs(%216 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %in_1572: f16, %out: f16):
    %5290 = arith.divf %in, %in_1572 : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096x4096xf16>
  return %232 : tensor<10x4096x4096xf16>
}


// CHECK-LABEL: @softmax_broadcast
//  CHECK-SAME: %[[ARG:.+]]: tensor<12x128x128xf32>
//       CHECK:   %[[E:.+]] = tensor.empty() : tensor<12x128x128xf32>
//       CHECK:   %[[S:.+]] = iree_linalg_ext.softmax dimension(2) ins(%[[ARG]] : tensor<12x128x128xf32>) outs(%[[E]] : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
//       CHECK:   return %[[S]] : tensor<12x128x128xf32>
func.func @softmax_broadcast(%93 : tensor<12x128x128xf32>) -> (tensor<12x128x128xf32>) {
  %cst_16 = arith.constant 0xFF800000 : f32
  %cst_18 = arith.constant -0.000000e+00 : f32
  %94 = tensor.empty() : tensor<12x128xf32>
  %95 = linalg.fill ins(%cst_16 : f32) outs(%94 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%93 : tensor<12x128x128xf32>) outs(%95 : tensor<12x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2460 = arith.maxf %out, %in : f32
    linalg.yield %2460 : f32
  } -> tensor<12x128xf32>
  %97 = tensor.empty() : tensor<12x128x128xf32>
  %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%96 : tensor<12x128xf32>) outs(%97 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<12x128x128xf32>
  %99 = tensor.empty() : tensor<12x128x128xf32>
  %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93, %98 : tensor<12x128x128xf32>, tensor<12x128x128xf32>) outs(%99 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %in_261: f32, %out: f32):
    %2460 = arith.subf %in, %in_261 : f32
    linalg.yield %2460 : f32
  } -> tensor<12x128x128xf32>
  %101 = tensor.empty() : tensor<12x128x128xf32>
  %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100 : tensor<12x128x128xf32>) outs(%101 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2460 = math.exp %in : f32
    linalg.yield %2460 : f32
  } -> tensor<12x128x128xf32>
  %103 = tensor.empty() : tensor<12x128xf32>
  %104 = linalg.fill ins(%cst_18 : f32) outs(%103 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%102 : tensor<12x128x128xf32>) outs(%104 : tensor<12x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2460 = arith.addf %out, %in : f32
    linalg.yield %2460 : f32
  } -> tensor<12x128xf32>
  %106 = tensor.empty() : tensor<12x128x128xf32>
  %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%105 : tensor<12x128xf32>) outs(%106 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<12x128x128xf32>
  %108 = tensor.empty() : tensor<12x128x128xf32>
  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102, %107 : tensor<12x128x128xf32>, tensor<12x128x128xf32>) outs(%108 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %in_261: f32, %out: f32):
    %2460 = arith.divf %in, %in_261 : f32
    linalg.yield %2460 : f32
  } -> tensor<12x128x128xf32>
  return %109 : tensor<12x128x128xf32>
}

func.func @aTransposeBMatmul(%arg0 : tensor<10x20xf32>,
    %arg1 : tensor<40x20xf32>) -> tensor<10x40xf32> {
  %0 = tensor.empty() : tensor<20x40xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<40x20xf32>) outs(%0 : tensor<20x40xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      linalg.yield %b0 : f32
  } -> tensor<20x40xf32>
  %2 = tensor.empty() : tensor<10x40xf32>
  %3 = arith.constant 0.0 : f32
  %4 = linalg.fill ins(%3 : f32) outs(%2 : tensor<10x40xf32>) -> tensor<10x40xf32>
  %5 = linalg.matmul ins(%arg0, %1 : tensor<10x20xf32>, tensor<20x40xf32>)
      outs(%4 : tensor<10x40xf32>) -> tensor<10x40xf32>
  return %5 : tensor<10x40xf32>
}
// CHECK-LABEL: func @aTransposeBMatmul
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<40x20xf32>
//       CHECK:   %[[RESULT:.+]] = linalg.matmul_transpose_b
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
//       CHECK:   return %[[RESULT]]

func.func @aTransposeBBatchMatmul(%arg0 : tensor<5x10x20xf32>,
    %arg1 : tensor<5x40x20xf32>) -> tensor<5x10x40xf32> {
  %0 = tensor.empty() : tensor<5x20x40xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg1 : tensor<5x40x20xf32>) outs(%0 : tensor<5x20x40xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      linalg.yield %b0 : f32
  } -> tensor<5x20x40xf32>
  %2 = tensor.empty() : tensor<5x10x40xf32>
  %3 = arith.constant 0.0 : f32
  %4 = linalg.fill ins(%3 : f32) outs(%2 : tensor<5x10x40xf32>) -> tensor<5x10x40xf32>
  %5 = linalg.batch_matmul ins(%arg0, %1 : tensor<5x10x20xf32>, tensor<5x20x40xf32>)
      outs(%4 : tensor<5x10x40xf32>) -> tensor<5x10x40xf32>
  return %5 : tensor<5x10x40xf32>
}
// CHECK-LABEL: func @aTransposeBBatchMatmul
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<5x10x20xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<5x40x20xf32>
//       CHECK:   %[[RESULT:.+]] = linalg.batch_matmul_transpose_b
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
//       CHECK:   return %[[RESULT]]

func.func @generic_fill(%arg0: tensor<?x?xf32>) -> tensor<1x1x?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<1x1x?x?xf32>
    %1 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        outs(%0 : tensor<1x1x?x?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
    } -> tensor<1x1x?x?xf32>
    return %1 : tensor<1x1x?x?xf32>
}
// CHECK-LABEL: func @generic_fill
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[EMPTY:.+]] = tensor.empty
//  CHECK-SAME:   : tensor<1x1x?x?xf32>
//       CHECK:   %[[RESULT:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[CST]] : f32)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<1x1x?x?xf32>)
//       CHECK:   return %[[RESULT]]

// -----

#map = affine_map<(d0) -> (d0)>
func.func @test_rank_reduce(%A : tensor<1x1x5120xf32>, %B : tensor<5120xf32>) -> tensor<5120xf32> {
  %c0 = arith.constant 0 : index
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%B : tensor<5120xf32>) {
  ^bb0(%out: f32):
    %12 = linalg.index 0 : index
    %extracted = tensor.extract %A[%c0, %c0, %12] : tensor<1x1x5120xf32>
    linalg.yield %extracted : f32
  } -> tensor<5120xf32>
  return %0 : tensor<5120xf32>
}

// CHECK-LABEL: func @test_rank_reduce
//       CHECK:   tensor.extract_slice %{{.*}}[0, 0, 0] [1, 1, 5120] [1, 1, 1]
//  CHECK-SAME:     tensor<1x1x5120xf32> to tensor<5120xf32>

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_slice_middle(%A : tensor<64x64x64xf32>, %B : tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%B : tensor<64x64xf32>) {
  ^bb0(%out: f32):
    %i1 = linalg.index 0 : index
    %i2 = linalg.index 1 : index
    %extracted = tensor.extract %A[%i1, %c0, %i2] : tensor<64x64x64xf32>
    linalg.yield %extracted : f32
  } -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: func @test_slice_middle
//       CHECK:   tensor.extract_slice %{{.*}}[0, 0, 0] [64, 1, 64] [1, 1, 1]
//  CHECK-SAME:     tensor<64x64x64xf32> to tensor<64x64xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @test_unit_identity_slice(%A : tensor<1x1x64xf32>, %B : tensor<1x1x64xf32>) -> tensor<1x1x64xf32> {
  %c0 = arith.constant 0 : index
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%B : tensor<1x1x64xf32>) {
  ^bb0(%out: f32):
    %i2 = linalg.index 2 : index
    %extracted = tensor.extract %A[%c0, %c0, %i2] : tensor<1x1x64xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x1x64xf32>
  return %0 : tensor<1x1x64xf32>
}

// CHECK-LABEL: func @test_unit_identity_slice
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<1x1x64xf32>,
//       CHECK:   return %[[ARG0]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_dynamic_slice(%A : tensor<1x128x?xf32>) -> tensor<128x?xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %dim = tensor.dim %A, %c2 : tensor<1x128x?xf32>
  %empty = tensor.empty(%dim) : tensor<128x?xf32>
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%empty : tensor<128x?xf32>) {
  ^bb0(%out: f32):
    %i0 = linalg.index 0 : index
    %i1 = linalg.index 1 : index
    %extracted = tensor.extract %A[%c0, %i0, %i1] : tensor<1x128x?xf32>
    linalg.yield %extracted : f32
  } -> tensor<128x?xf32>
  return %0 : tensor<128x?xf32>
}

// CHECK-LABEL: func @test_dynamic_slice
//       CHECK:   %[[DIM:.+]] = tensor.dim {{.*}} : tensor<1x128x?xf32>
//       CHECK:   tensor.extract_slice %{{.*}}[0, 0, 0] [1, 128, %[[DIM]]] [1, 1, 1]
//  CHECK-SAME:     tensor<1x128x?xf32> to tensor<128x?xf32>

// -----

// This currently should not be raised as the operation does not remain
// elementwise after raising the tensor.extract to input.
#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @test_non_slice
func.func @test_non_slice(%A : tensor<128x128x128xf32>, %B : tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: linalg.generic
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%B : tensor<64x64xf32>) {
  ^bb0(%out: f32):
    %i1 = linalg.index 0 : index
    %i2 = linalg.index 1 : index
    %extracted = tensor.extract %A[%i1, %c0, %i2] : tensor<128x128x128xf32>
    linalg.yield %extracted : f32
  } -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
