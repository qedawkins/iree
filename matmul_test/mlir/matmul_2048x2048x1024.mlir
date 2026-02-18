#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_2048x2048x1024(%lhs: tensor<2048x1024xf16>, %rhs: tensor<2048x1024xf16>) -> tensor<2048x2048xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2048x2048xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : tensor<2048x1024xf16>, tensor<2048x1024xf16>) outs(%fill : tensor<2048x2048xf32>) {
  ^bb0(%in0: f16, %in1: f16, %out: f32):
    %0 = arith.extf %in0 : f16 to f32
    %1 = arith.extf %in1 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %2, %out : f32
    linalg.yield %3 : f32
  } -> tensor<2048x2048xf32>
  return %result : tensor<2048x2048xf32>
}
