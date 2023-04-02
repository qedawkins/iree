// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule -cse -split-input-file --verify-diagnostics | FileCheck %s

builtin.module {
  func.func @matmul_dispatch_0_matmul_16x8x16() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %1, 64 : memref<16x8xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %4 = vector.transfer_read %1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x8xf16>, vector<16x8xf16>
    %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %cst : vector<16x16xf16>, vector<16x8xf16> into vector<16x8xf16>
    vector.transfer_write %5, %2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !pdl.operation):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!pdl.operation) -> (!pdl.operation)
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_dispatch_0_matmul_16x8x16() {
// CHECK:        %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<16x8xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK:        %[[D3:.+]] = gpu.thread_id  x
// CHECK:        %[[D4:.+]] = gpu.thread_id  y
// CHECK:        %[[D5:.+]] = gpu.thread_id  z
// CHECK:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D6:.+]] = affine.apply #[[MAP]](%[[D3]], %[[D4]], %[[D5]])
// CHECK-DAG:      %[[D7:.+]] = affine.apply #[[MAP1]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D8:.+]] = memref.load %[[D0]][%[[D6]], %[[D7]]] : memref<16x16xf16>
// CHECK:        %[[D9:.+]] = vector.broadcast %[[D8]] : f16 to vector<1xf16>
// CHECK:        %[[D10:.+]] = vector.insert_strided_slice %[[D9]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D11:.+]] = affine.apply #[[MAP2]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D12:.+]] = memref.load %[[D0]][%[[D6]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D13:.+]] = vector.broadcast %[[D12]] : f16 to vector<1xf16>
// CHECK:        %[[D14:.+]] = vector.insert_strided_slice %[[D13]], %[[D10]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D15:.+]] = affine.apply #[[MAP3]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D16:.+]] = memref.load %[[D0]][%[[D6]], %[[D15]]] : memref<16x16xf16>
// CHECK:        %[[D17:.+]] = vector.broadcast %[[D16]] : f16 to vector<1xf16>
// CHECK:        %[[D18:.+]] = vector.insert_strided_slice %[[D17]], %[[D14]] {offsets = [0, 0, 2, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D19:.+]] = affine.apply #[[MAP4]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D20:.+]] = memref.load %[[D0]][%[[D6]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D21:.+]] = vector.broadcast %[[D20]] : f16 to vector<1xf16>
// CHECK:        %[[D22:.+]] = vector.insert_strided_slice %[[D21]], %[[D18]] {offsets = [0, 0, 2, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D23:.+]] = affine.apply #[[MAP5]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D24:.+]] = memref.load %[[D0]][%[[D23]], %[[D7]]] : memref<16x16xf16>
// CHECK:        %[[D25:.+]] = vector.broadcast %[[D24]] : f16 to vector<1xf16>
// CHECK:        %[[D26:.+]] = vector.insert_strided_slice %[[D25]], %[[D22]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D27:.+]] = memref.load %[[D0]][%[[D23]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D28:.+]] = vector.broadcast %[[D27]] : f16 to vector<1xf16>
// CHECK:        %[[D29:.+]] = vector.insert_strided_slice %[[D28]], %[[D26]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D30:.+]] = memref.load %[[D0]][%[[D23]], %[[D15]]] : memref<16x16xf16>
// CHECK:        %[[D31:.+]] = vector.broadcast %[[D30]] : f16 to vector<1xf16>
// CHECK:        %[[D32:.+]] = vector.insert_strided_slice %[[D31]], %[[D29]] {offsets = [0, 0, 3, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D33:.+]] = memref.load %[[D0]][%[[D23]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D34:.+]] = vector.broadcast %[[D33]] : f16 to vector<1xf16>
// CHECK:        %[[D35:.+]] = vector.insert_strided_slice %[[D34]], %[[D32]] {offsets = [0, 0, 3, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK-DAG:      %[[D36:.+]] = affine.apply #[[MAP6]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D37:.+]] = memref.load %[[D1]][%[[D7]], %[[D36]]] : memref<16x8xf16>
// CHECK:        %[[D38:.+]] = vector.broadcast %[[D37]] : f16 to vector<1xf16>
// CHECK:        %[[D39:.+]] = vector.insert_strided_slice %[[D38]], %[[CST_0]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D40:.+]] = memref.load %[[D1]][%[[D11]], %[[D36]]] : memref<16x8xf16>
// CHECK:        %[[D41:.+]] = vector.broadcast %[[D40]] : f16 to vector<1xf16>
// CHECK:        %[[D42:.+]] = vector.insert_strided_slice %[[D41]], %[[D39]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D43:.+]] = memref.load %[[D1]][%[[D15]], %[[D36]]] : memref<16x8xf16>
// CHECK:        %[[D44:.+]] = vector.broadcast %[[D43]] : f16 to vector<1xf16>
// CHECK:        %[[D45:.+]] = vector.insert_strided_slice %[[D44]], %[[D42]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D46:.+]] = memref.load %[[D1]][%[[D19]], %[[D36]]] : memref<16x8xf16>
// CHECK:        %[[D47:.+]] = vector.broadcast %[[D46]] : f16 to vector<1xf16>
// CHECK:        %[[D48:.+]] = vector.insert_strided_slice %[[D47]], %[[D45]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf16>
// CHECK:        %[[D49:.+]] = vector.extract %[[D35]][0, 0] : vector<1x1x4x2xf16>
// CHECK:        %[[D50:.+]] = vector.extract %[[D48]][0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D51:.+]] = nvgpu.mma.sync(%[[D49]], %[[D50]], %[[CST_1]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:     (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D52:.+]] = vector.insert %[[D51]], %[[CST_0]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D53:.+]] = vector.extract %[[D52]][0, 0, 0, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D53]], %[[D2]][%[[D6]], %[[D7]]] : memref<16x8xf16>
// CHECK:        %[[D54:.+]] = vector.extract %[[D52]][0, 0, 0, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D54]], %[[D2]][%[[D6]], %[[D11]]] : memref<16x8xf16>
// CHECK:        %[[D55:.+]] = vector.extract %[[D52]][0, 0, 1, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D55]], %[[D2]][%[[D23]], %[[D7]]] : memref<16x8xf16>
// CHECK:        %[[D56:.+]] = vector.extract %[[D52]][0, 0, 1, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D56]], %[[D2]][%[[D23]], %[[D11]]] : memref<16x8xf16>
// CHECK:        return
// CHECK:      }

// -----

builtin.module {
  func.func @matmul_reduction() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %init = arith.constant dense<-1.000000e+04> : vector<16xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %1, 64 : memref<16x8xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %4 = vector.transfer_read %1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x8xf16>, vector<16x8xf16>
    %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %cst : vector<16x16xf16>, vector<16x8xf16> into vector<16x8xf16>
    %6 = vector.multi_reduction <maxf>, %5, %init [1] : vector<16x8xf16> to vector<16xf16>
    %7 = vector.broadcast %6 : vector<16xf16> to vector<8x16xf16>
    %8 = vector.transpose %7, [1, 0] : vector<8x16xf16> to vector<16x8xf16>
    vector.transfer_write %8, %2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !pdl.operation):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!pdl.operation) -> (!pdl.operation)
  }
}

// CHECK-DAG:#[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:#[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:#[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:#[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:#[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:#[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:#[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_reduction() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<16x8xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK-DAG:    %[[D3:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D4:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D6:.+]] = affine.apply #[[MAP]](%[[D3]], %[[D4]], %[[D5]])
// CHECK-DAG:    %[[D7:.+]] = affine.apply #[[MAP1]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D8:.+]] = memref.load %[[D0]][%[[D6]], %[[D7]]] : memref<16x16xf16>
// CHECK:        %[[D9:.+]] = vector.broadcast %[[D8]] : f16 to vector<1xf16>
// CHECK:        %[[D10:.+]] = vector.insert_strided_slice %[[D9]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D11:.+]] = affine.apply #[[MAP2]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D12:.+]] = memref.load %[[D0]][%[[D6]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D13:.+]] = vector.broadcast %[[D12]] : f16 to vector<1xf16>
// CHECK:        %[[D14:.+]] = vector.insert_strided_slice %[[D13]], %[[D10]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D15:.+]] = affine.apply #[[MAP3]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D16:.+]] = memref.load %[[D0]][%[[D6]], %[[D15]]] : memref<16x16xf16>
// CHECK:        %[[D17:.+]] = vector.broadcast %[[D16]] : f16 to vector<1xf16>
// CHECK:        %[[D18:.+]] = vector.insert_strided_slice %[[D17]], %[[D14]] {offsets = [0, 0, 2, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D19:.+]] = affine.apply #[[MAP4]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D20:.+]] = memref.load %[[D0]][%[[D6]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D21:.+]] = vector.broadcast %[[D20]] : f16 to vector<1xf16>
// CHECK:        %[[D22:.+]] = vector.insert_strided_slice %[[D21]], %[[D18]] {offsets = [0, 0, 2, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D23:.+]] = affine.apply #[[MAP5]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D24:.+]] = memref.load %[[D0]][%[[D23]], %[[D7]]] : memref<16x16xf16>
// CHECK:        %[[D25:.+]] = vector.broadcast %[[D24]] : f16 to vector<1xf16>
// CHECK:        %[[D26:.+]] = vector.insert_strided_slice %[[D25]], %[[D22]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D27:.+]] = memref.load %[[D0]][%[[D23]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D28:.+]] = vector.broadcast %[[D27]] : f16 to vector<1xf16>
// CHECK:        %[[D29:.+]] = vector.insert_strided_slice %[[D28]], %[[D26]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D30:.+]] = memref.load %[[D0]][%[[D23]], %[[D15]]] : memref<16x16xf16>
// CHECK:        %[[D31:.+]] = vector.broadcast %[[D30]] : f16 to vector<1xf16>
// CHECK:        %[[D32:.+]] = vector.insert_strided_slice %[[D31]], %[[D29]] {offsets = [0, 0, 3, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D33:.+]] = memref.load %[[D0]][%[[D23]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D34:.+]] = vector.broadcast %[[D33]] : f16 to vector<1xf16>
// CHECK:        %[[D35:.+]] = vector.insert_strided_slice %[[D34]], %[[D32]] {offsets = [0, 0, 3, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D36:.+]] = affine.apply #[[MAP6]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D37:.+]] = memref.load %[[D1]][%[[D7]], %[[D36]]] : memref<16x8xf16>
// CHECK:        %[[D38:.+]] = vector.broadcast %[[D37]] : f16 to vector<1xf16>
// CHECK:        %[[D39:.+]] = vector.insert_strided_slice %[[D38]], %[[CST_0]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D40:.+]] = memref.load %[[D1]][%[[D11]], %[[D36]]] : memref<16x8xf16>
// CHECK:        %[[D41:.+]] = vector.broadcast %[[D40]] : f16 to vector<1xf16>
// CHECK:        %[[D42:.+]] = vector.insert_strided_slice %[[D41]], %[[D39]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D43:.+]] = memref.load %[[D1]][%[[D15]], %[[D36]]] : memref<16x8xf16>
// CHECK:        %[[D44:.+]] = vector.broadcast %[[D43]] : f16 to vector<1xf16>
// CHECK:        %[[D45:.+]] = vector.insert_strided_slice %[[D44]], %[[D42]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D46:.+]] = memref.load %[[D1]][%[[D19]], %[[D36]]] : memref<16x8xf16>
// CHECK:        %[[D47:.+]] = vector.broadcast %[[D46]] : f16 to vector<1xf16>
// CHECK:        %[[D48:.+]] = vector.insert_strided_slice %[[D47]], %[[D45]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf16>
// CHECK:        %[[D49:.+]] = vector.extract %[[D35]][0, 0] : vector<1x1x4x2xf16>
// CHECK:        %[[D50:.+]] = vector.extract %[[D48]][0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D51:.+]] = nvgpu.mma.sync(%[[D49]], %[[D50]], %[[CST_1]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:     (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D52:.+]] = vector.insert %[[D51]], %[[CST_0]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant -1.000000e+04 : f16
// CHECK:        %[[D53:.+]] = vector.extract %[[D52]][0, 0, 0] : vector<1x1x2x2xf16>
// CHECK:        %[[D54:.+]] = vector.bitcast %[[D53]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D55:.+]] = vector.extract %[[D54]][0] : vector<1xi32>
// CHECK-DAG:    %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-DAG:    %[[C32_I32:.+]] = arith.constant 32 : i32
// CHECK:        %[[SHUFFLERESULT:.+]], %[[VALID:.+]] = gpu.shuffle  xor %[[D55]], %[[C1_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D56:.+]] = vector.broadcast %[[SHUFFLERESULT]] : i32 to vector<1xi32>
// CHECK:        %[[D57:.+]] = vector.bitcast %[[D56]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D58:.+]] = arith.maxf %[[D57]], %[[D53]] : vector<2xf16>
// CHECK:        %[[D59:.+]] = vector.bitcast %[[D58]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D60:.+]] = vector.extract %[[D59]][0] : vector<1xi32>
// CHECK-DAG:    %[[C2_I32:.+]] = arith.constant 2 : i32
// CHECK:        %[[SHUFFLERESULT_3:.+]], %[[VALID_4:.+]] = gpu.shuffle  xor %[[D60]], %[[C2_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D61:.+]] = vector.broadcast %[[SHUFFLERESULT_3]] : i32 to vector<1xi32>
// CHECK:        %[[D62:.+]] = vector.bitcast %[[D61]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D63:.+]] = arith.maxf %[[D62]], %[[D58]] : vector<2xf16>
// CHECK:        %[[D64:.+]] = vector.extract %[[D63]][0] : vector<2xf16>
// CHECK:        %[[D65:.+]] = arith.maxf %[[CST_2]], %[[D64]] : f16
// CHECK:        %[[D66:.+]] = vector.extract %[[D63]][1] : vector<2xf16>
// CHECK:        %[[D67:.+]] = arith.maxf %[[D65]], %[[D66]] : f16
// CHECK:        %[[D68:.+]] = vector.insert %[[D67]], %[[CST_0]] [0, 0, 0, 0] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D69:.+]] = vector.insert %[[D67]], %[[D68]] [0, 0, 0, 1] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D70:.+]] = vector.extract %[[D52]][0, 0, 1] : vector<1x1x2x2xf16>
// CHECK:        %[[D71:.+]] = vector.bitcast %[[D70]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D72:.+]] = vector.extract %[[D71]][0] : vector<1xi32>
// CHECK:        %[[SHUFFLERESULT_5:.+]], %[[VALID_6:.+]] = gpu.shuffle  xor %[[D72]], %[[C1_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D73:.+]] = vector.broadcast %[[SHUFFLERESULT_5]] : i32 to vector<1xi32>
// CHECK:        %[[D74:.+]] = vector.bitcast %[[D73]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D75:.+]] = arith.maxf %[[D74]], %[[D70]] : vector<2xf16>
// CHECK:        %[[D76:.+]] = vector.bitcast %[[D75]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D77:.+]] = vector.extract %[[D76]][0] : vector<1xi32>
// CHECK:        %[[SHUFFLERESULT_7:.+]], %[[VALID_8:.+]] = gpu.shuffle  xor %[[D77]], %[[C2_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D78:.+]] = vector.broadcast %[[SHUFFLERESULT_7]] : i32 to vector<1xi32>
// CHECK:        %[[D79:.+]] = vector.bitcast %[[D78]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D80:.+]] = arith.maxf %[[D79]], %[[D75]] : vector<2xf16>
// CHECK:        %[[D81:.+]] = vector.extract %[[D80]][0] : vector<2xf16>
// CHECK:        %[[D82:.+]] = arith.maxf %[[CST_2]], %[[D81]] : f16
// CHECK:        %[[D83:.+]] = vector.extract %[[D80]][1] : vector<2xf16>
// CHECK:        %[[D84:.+]] = arith.maxf %[[D82]], %[[D83]] : f16
// CHECK:        %[[D85:.+]] = vector.insert %[[D84]], %[[D69]] [0, 0, 1, 0] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D86:.+]] = vector.insert %[[D84]], %[[D85]] [0, 0, 1, 1] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D87:.+]] = vector.extract %[[D86]][0, 0, 0, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D87]], %[[D2]][%[[D6]], %[[D7]]] : memref<16x8xf16>
// CHECK:        %[[D88:.+]] = vector.extract %[[D86]][0, 0, 0, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D88]], %[[D2]][%[[D6]], %[[D11]]] : memref<16x8xf16>
// CHECK:        %[[D89:.+]] = vector.extract %[[D86]][0, 0, 1, 0] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D89]], %[[D2]][%[[D23]], %[[D7]]] : memref<16x8xf16>
// CHECK:        %[[D90:.+]] = vector.extract %[[D86]][0, 0, 1, 1] : vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D90]], %[[D2]][%[[D23]], %[[D11]]] : memref<16x8xf16>
// CHECK:        return
// CHECK:      }
