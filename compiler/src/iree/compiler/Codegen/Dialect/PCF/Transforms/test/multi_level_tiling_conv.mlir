// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-test-multi-level-tiling{subgroup-tile-sizes=0,4,4,16,0,0,0 lane-tile-sizes=0,2,2,4,0,0,0 reduction-tile-sizes=0,0,0,0,1,1,8})" --split-input-file | FileCheck %s

// Conv2D with multiple reduction dimensions (KH, KW, IC).
// 7 iteration dims: d0(N), d1(OH), d2(OW), d3(OC), d4(KH), d5(KW), d6(IC).
// Subgroup tiles [0,4,4,16,0,0,0], Lane tiles [0,2,2,4,0,0,0],
// Reduction tiles [0,0,0,0,1,1,8] to tile filter spatial dims to unit.
// Expected: nested scf.for loops for KH, KW, IC inside the pcf nesting.

func.func @conv_2d_nhwc_hwcf(
    %input: tensor<1x14x14x32xf16>,
    %filter: tensor<3x3x32x64xf16>,
    %init: tensor<1x12x12x64xf32>) -> tensor<1x12x12x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>,
       strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x14x14x32xf16>, tensor<3x3x32x64xf16>)
      outs(%init : tensor<1x12x12x64xf32>) -> tensor<1x12x12x64xf32>
  return %result : tensor<1x12x12x64xf32>
}

// CHECK-LABEL: func @conv_2d_nhwc_hwcf
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9_]+]]: tensor<1x14x14x32xf16>
//  CHECK-SAME:   %[[FILTER:[A-Za-z0-9_]+]]: tensor<3x3x32x64xf16>
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<1x12x12x64xf32>
//       CHECK:   pcf.generic scope(#pcf.sequential)
//       CHECK:     execute(%{{.+}} <- %[[INPUT]], %{{.+}} <- %[[FILTER]], %{{.+}} = %[[INIT]])
//       CHECK:       pcf.generic scope(#pcf.sequential)
//       CHECK:         execute
//       CHECK:           pcf.read_slice
//       CHECK:           scf.for
//       CHECK:             scf.for
//       CHECK:               scf.for
//       CHECK:                 pcf.read_slice
//       CHECK:                 pcf.read_slice
//       CHECK:                 linalg.conv_2d_nhwc_hwcf
//       CHECK:                 scf.yield
//       CHECK:               scf.yield
//       CHECK:             scf.yield
//       CHECK:           pcf.write_slice
//       CHECK:           pcf.return
//       CHECK:       pcf.return
