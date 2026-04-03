// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-pcf, iree-codegen-gpu-apply-multi-level-tiling, cse))" --mlir-print-local-scope --split-input-file | FileCheck %s

// Basic matmul: workgroup=[64,64,0], subgroup=[32,32,0], thread=[8,8,0], reduction=[0,0,16].
// Expected: pcf.loop(wg) { fill fused; pcf.generic(sg) { pcf.generic(lane) { init; for { matmul } wb } } }

func.func @matmul_pipeline(%lhs: tensor<256x512xf16>, %rhs: tensor<512x256xf16>,
                           %init: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
  %result = linalg.matmul {
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [64, 64, 0],
      reduction = [0, 0, 16],
      subgroup = [32, 32, 0],
      thread = [8, 8, 0]
    }>
  } ins(%lhs, %rhs : tensor<256x512xf16>, tensor<512x256xf16>)
    outs(%fill : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func @matmul_pipeline
// Workgroup loop.
// CHECK: pcf.loop scope(#iree_codegen.workgroup_scope<linearize>)
// Fill fused into workgroup loop.
// CHECK:   linalg.fill
// Subgroup generic.
// CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
// Lane generic.
// CHECK:     pcf.generic scope(#iree_gpu.lane_scope)
// Init read.
// CHECK:       pcf.read_slice
// Reduction loop.
// CHECK:       scf.for
// Read LHS and RHS.
// CHECK:         pcf.read_slice
// CHECK:         pcf.read_slice
// Tiled matmul.
// CHECK:         linalg.matmul
// CHECK:         scf.yield
// Writeback.
// CHECK:       pcf.write_slice

// -----

// Matmul with promotion: same as above but with promote_operands.

func.func @matmul_promoted(%lhs: tensor<256x512xf16>, %rhs: tensor<512x256xf16>,
                           %init: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
  %result = linalg.matmul {
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [64, 64, 0],
      reduction = [0, 0, 16],
      subgroup = [32, 32, 0],
      thread = [8, 8, 0],
      promote_operands = [0, 1]
    }>
  } ins(%lhs, %rhs : tensor<256x512xf16>, tensor<512x256xf16>)
    outs(%fill : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func @matmul_promoted
// CHECK: pcf.loop
// Symbols for promoted operand tile sizes defined in initializer.
// CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope) initialize {
// CHECK:     pcf.index_symbol "n0.d0"
// CHECK:     pcf.index_symbol "n0.d1"
// CHECK:     pcf.index_symbol "n1.d0"
// CHECK:     pcf.index_symbol "n1.d1"
// CHECK:     pcf.yield
// CHECK:   }
// CHECK:     pcf.generic scope(#iree_gpu.lane_scope)
// CHECK:       scf.for
// Promotion ops inside reduction loop.
// CHECK:         iree_gpu.promote_operand
// CHECK:         iree_gpu.promote_operand
// CHECK:         linalg.matmul

// -----

// Matmul with MMA kind: subgroup tiling + MMA conversion.

func.func @matmul_mma(%lhs: tensor<256x512xf16>, %rhs: tensor<512x256xf16>,
                      %init: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
  %result = linalg.matmul {
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [64, 64, 0],
      reduction = [0, 0, 16],
      subgroup = [32, 32, 0],
      mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
      promote_operands = [0, 1]
    }>
  } ins(%lhs, %rhs : tensor<256x512xf16>, tensor<512x256xf16>)
    outs(%fill : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// CHECK-LABEL: func @matmul_mma
// CHECK: pcf.loop
// CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
// CHECK:     pcf.generic scope(#iree_gpu.lane_scope)
// CHECK:       scf.for
// MMA kind carried as verified lowering_config attribute.
// CHECK:         linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>}>
