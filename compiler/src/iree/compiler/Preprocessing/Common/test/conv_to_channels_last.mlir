// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-to-channels-last))" %s | \
// RUN:   FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-to-channels-last{tile-size=16}))" %s | \
// RUN:   FileCheck %s --check-prefix=TILE16

util.func @conv_nhwc_hwcf_no_transpose(%arg0: tensor<1x16x16x256xf32>, %arg1: tensor<3x3x256x128xf32>, %arg2: tensor<1x14x14x128xf32>) -> tensor<1x14x14x128xf32> {
    %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<1x16x16x256xf32>, tensor<3x3x256x128xf32>)
      outs(%arg2: tensor<1x14x14x128xf32>) -> tensor<1x14x14x128xf32>
    util.return %0 : tensor<1x14x14x128xf32>
}
// CHECK-LABEL: @conv_nhwc_hwcf_no_transpose
// CHECK: linalg.conv_2d_nhwc_hwcf

// TILE16-LABEL: @conv_nhwc_hwcf_no_transpose
// TILE16: linalg.conv_2d_nhwc_hwcf

// -----

util.func @conv_nchw_nhwc(%arg0: tensor<8x256x16x16xf32>, %arg1: tensor<16x256x3x3xf32>, %arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32> {
    %0 = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<8x256x16x16xf32>, tensor<16x256x3x3xf32>)
      outs(%arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32>
    util.return %0 : tensor<8x16x14x14xf32>
}

// CHECK-LABEL: util.func public @conv_nchw_nhwc
// CHECK:         %[[IMG_TRANSP:.+]] = linalg.transpose ins(%{{.*}} : tensor<8x256x16x16xf32>)
// CHECK:         %[[IMG_SLICE:.+]] = tensor.insert_slice %[[IMG_TRANSP]]
// CHECK:         %[[IMG:.+]] = tensor.collapse_shape %[[IMG_SLICE]] {{.*}} into tensor<8x16x16x256xf32>
// CHECK:         %[[FILTER_TRANSP:.+]] = linalg.transpose ins(%{{.*}} : tensor<16x256x3x3xf32>)
// CHECK:         %[[FILTER_SLICE:.+]] = tensor.insert_slice %[[FILTER_TRANSP]]
// CHECK:         %[[FILTER:.+]] = tensor.collapse_shape %[[FILTER_SLICE]] {{.*}} into tensor<3x3x256x16xf32>
// CHECK:         %[[OUT_TRANSP:.+]] = linalg.transpose ins(%{{.*}} : tensor<8x16x14x14xf32>)
// CHECK:         %[[OUT_SLICE:.+]] = tensor.insert_slice %[[OUT_TRANSP]]
// CHECK:         %[[OUT:.+]] = tensor.collapse_shape %[[OUT_SLICE]] {{.*}} into tensor<8x14x14x16xf32>
// CHECK:         %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:      ins(%[[IMG]], %[[FILTER]] : tensor<8x16x16x256xf32>, tensor<3x3x256x16xf32>) outs(%[[OUT]] : tensor<8x14x14x16xf32>) -> tensor<8x14x14x16xf32>
// CHECK:         %[[CONV_EXPAND:.+]] = tensor.expand_shape %[[CONV]]
// CHECK:         %[[CONV_SLICE:.+]] = tensor.extract_slice %[[CONV_EXPAND]]
// CHECK:         linalg.transpose ins(%[[CONV_SLICE]] : tensor<8x14x14x16xf32>) outs(%{{.*}} : tensor<8x16x14x14xf32>)

// TILE16-LABEL: util.func public @conv_nchw_nhwc

// TILE16:      %[[IMG:.+]] = linalg.transpose ins(%{{[A-Za-z0-9]+}} : tensor<8x16x16x16x16xf32>)
// TILE16-SAME:   outs(%{{.*}} : tensor<8x16x16x16x16xf32>) permutation = [0, 1, 3, 4, 2]
// TILE16:      %[[FILTER:.+]] = linalg.transpose ins(%{{.*}} : tensor<1x16x16x16x3x3xf32>)
// TILE16-SAME:   outs(%{{.*}} : tensor<1x16x3x3x16x16xf32>) permutation = [0, 2, 4, 5, 3, 1]
// TILE16:      %[[OUT:.+]] = linalg.transpose ins(%{{.*}} : tensor<8x16x14x14xf32>)
// TILE16-SAME:   outs(%{{.*}} : tensor<8x14x14x16xf32>) permutation = [0, 2, 3, 1]
// TILE16:      %[[OUT_SLICE:.+]] = tensor.insert_slice %[[OUT]]
// TILE16:      %[[TILED_CONV:.+]] = linalg.generic {indexing_maps = [#map, #map1, #map2]
// TILE16-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction", "parallel"]}
// TILE16:         ins(%[[IMG]], %[[FILTER]] : tensor<8x16x16x16x16xf32>, tensor<1x16x3x3x16x16xf32>)
// TILE16:         outs(%[[OUT_SLICE]] : tensor<8x1x14x14x16xf32>) {
// TILE16:        ^bb0
// TILE16:          arith.mulf
// TILE16:          arith.addf
// TILE16:        } -> tensor<8x1x14x14x16xf32>
// TILE16:      %[[RES_SLICE:.+]] = tensor.extract_slice %[[TILED_CONV:.+]]
// TILE16:      linalg.transpose ins(%[[RES_SLICE]] : tensor<8x14x14x16xf32>)
// TILE16-SAME:   outs(%{{.*}} : tensor<8x16x14x14xf32>) permutation = [0, 3, 1, 2]
