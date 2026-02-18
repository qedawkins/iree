#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_256x256x256(%lhs: tensor<256x256xf16>, %rhs: tensor<256x256xf16>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<256x256xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : tensor<256x256xf16>, tensor<256x256xf16>) outs(%fill : tensor<256x256xf32>) {
  ^bb0(%in0: f16, %in1: f16, %out: f32):
    %0 = arith.extf %in0 : f16 to f32
    %1 = arith.extf %in1 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %2, %out : f32
    linalg.yield %3 : f32
  } -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}
