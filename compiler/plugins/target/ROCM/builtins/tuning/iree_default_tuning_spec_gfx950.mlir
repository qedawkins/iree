// RUN: iree-opt %s

// Element types

!lhs = f4E2M1FN
!rhs = f4E2M1FN

!scale_ty = f8E8M0FNU

// Inputs

!lhs_ty = tensor<64x?x32x!lhs>
!rhs_ty = tensor<64x?x32x!rhs>

!lhs_byte_ty = tensor<64x?x16xi8>
!rhs_byte_ty = tensor<64x?x16xi8>

!lhs_global_ty = memref<64x?x16xi8, strided<[?, 16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
!rhs_global_ty = memref<64x?x16xi8, strided<[?, 16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>

!lhs_buffer_ty = memref<64x?x16xi8, strided<[?, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_buffer_ty = memref<64x?x16xi8, strided<[?, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

// 2 = double buffer
// 64 = m tile
// 64 = n tile
// 8 = k split factor
// 64 = 128 / 2 = k tile / 2
!lhs_shared_ty = memref<2x64x8x64xi8, #gpu.address_space<workgroup>>
!rhs_shared_ty = memref<2x64x8x64xi8, #gpu.address_space<workgroup>>

!lhs_shared_expand_ty = memref<2x4x16x8x64xi8, #gpu.address_space<workgroup>>
!rhs_shared_expand_ty = memref<2x4x16x8x64xi8, #gpu.address_space<workgroup>>

!lhs_copy_vec_ty = vector<16xi8>
!rhs_copy_vec_ty = vector<16xi8>

!lhs_byte_vec_ty = vector<2x1x8x16xi8>
!lhs_vec_ty = vector<2x1x8x32xf4E2M1FN>

!rhs_byte_vec_ty = vector<1x1x8x16xi8>
!rhs_vec_ty = vector<1x1x8x32xf4E2M1FN>

// Scales

!lhs_scale_ty = tensor<64x?x!scale_ty>
!rhs_scale_ty = tensor<64x?x!scale_ty>

!lhs_scale_byte_ty = tensor<64x?xi8>
!rhs_scale_byte_ty = tensor<64x?xi8>

!lhs_scale_global_ty = memref<64x?xi8, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
!rhs_scale_global_ty = memref<64x?xi8, strided<[?, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>

!lhs_scale_buffer_ty = memref<64x?xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!rhs_scale_buffer_ty = memref<64x?xi8, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

!lhs_scale_copy_vec_ty = vector<4xi8>
!rhs_scale_copy_vec_ty = vector<4xi8>

// 2 = double buffer
// 64 = m tile
// 64 = n tile
// 8 = k split factor
// 4 = k tile / k block
!lhs_scale_shared_ty = memref<2x64x8x4xi8, #gpu.address_space<workgroup>>
!rhs_scale_shared_ty = memref<2x64x8x4xi8, #gpu.address_space<workgroup>>

!lhs_scale_shared_expand_ty = memref<2x4x16x8x4xi8, #gpu.address_space<workgroup>>
!rhs_scale_shared_expand_ty = memref<2x4x16x8x4xi8, #gpu.address_space<workgroup>>

!lhs_scale_byte_vec_ty = vector<2x1x8x1xi8>
!lhs_scale_vec_ty = vector<2x1x8x1xf8E8M0FNU>

!rhs_scale_byte_vec_ty = vector<1x1x8x1xi8>
!rhs_scale_vec_ty = vector<1x1x8x1xf8E8M0FNU>


!res_ty = tensor<4x16x4x16xf32>
!return_ty = tensor<64x64xf32>

// subgroup_m = 2 waves subgroup_n = 4 waves
!acc_ty = vector<2x1x8x4x1xf32>

!reduce_ty = vector<2x1x4x1xf32>
!store_ty = vector<2x4x1x1xf32>
!tensor_store_ty = tensor<2x4x1x1xf32>

#contraction_accesses = [
  affine_map<(i, j, k, d) -> (i, d, k)>,
  affine_map<(i, j, k, d) -> (i, d, k)>,
  affine_map<(i, j, k, d) -> (j, d, k)>,
  affine_map<(i, j, k, d) -> (j, d, k)>,
  affine_map<(i, j, k, d) -> (i, j, k)>
]

#iterator_types = [
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<parallel>
]

#mfma_type = #iree_gpu.scaled_mma_layout<
  intrinsic = MFMA_SCALE_F32_16x16x128_B32,
  lhs_elem_type = f4E2M1FN,
  rhs_elem_type = f4E2M1FN,
  acc_elem_type = f32>

#result_reassoc = [[0, 1], [2, 3]]

#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>

module @iree_default_tuning_spec_gfx942 attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {

// ============================================================
// * Handwritten Codegen Schedules *
// ============================================================

// ------------------------------------------------------------
// * Large Pingpong f4 *
// ------------------------------------------------------------

util.func private @pingpong_f4(
    %lhs_base: !lhs_ty,
    %rhs_base: !rhs_ty,
    %lhs_scale_base: !lhs_scale_ty,
    %rhs_scale_base: !rhs_scale_ty,
    %unused_acc: !return_ty) -> !return_ty {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index

  %cst_lhs = arith.constant 0 : i8
  %cst_rhs = arith.constant 0 : i8
  %cst_scale = arith.constant 0 : i8
  %lhs_shared_base = memref.alloc() : !lhs_shared_ty
  %rhs_shared_base = memref.alloc() : !rhs_shared_ty
  %lhs_scale_shared = memref.alloc() : !lhs_scale_shared_ty
  %rhs_scale_shared = memref.alloc() : !rhs_scale_shared_ty

  %lhs_shared_expand = memref.expand_shape %lhs_shared_base [[0], [1, 2], [3], [4]]
    output_shape [2, 4, 16, 8, 64] : !lhs_shared_ty into !lhs_shared_expand_ty
  %rhs_shared_expand = memref.expand_shape %rhs_shared_base [[0], [1, 2], [3], [4]]
    output_shape [2, 4, 16, 8, 64] : !rhs_shared_ty into !rhs_shared_expand_ty

  %lhs_scale_shared_expand = memref.expand_shape %lhs_scale_shared [[0], [1, 2], [3], [4]]
    output_shape [2, 4, 16, 8, 4] : !lhs_scale_shared_ty into !lhs_scale_shared_expand_ty
  %rhs_scale_shared_expand = memref.expand_shape %rhs_scale_shared [[0], [1, 2], [3], [4]]
    output_shape [2, 4, 16, 8, 4] : !rhs_scale_shared_ty into !rhs_scale_shared_expand_ty

  %k = tensor.dim %lhs_base, %c1 : !lhs_ty

  %lhs_byte = iree_tensor_ext.bitcast %lhs_base : !lhs_ty{%k} -> !lhs_byte_ty{%k}
  %rhs_byte = iree_tensor_ext.bitcast %rhs_base : !rhs_ty{%k} -> !rhs_byte_ty{%k}

  %lhs_scale_byte = iree_tensor_ext.bitcast %lhs_scale_base : !lhs_scale_ty{%k} -> !lhs_scale_byte_ty{%k}
  %rhs_scale_byte = iree_tensor_ext.bitcast %rhs_scale_base : !rhs_scale_ty{%k} -> !rhs_scale_byte_ty{%k}

  %lhs = bufferization.to_buffer %lhs_byte {read_only} : !lhs_byte_ty to !lhs_buffer_ty
  %rhs = bufferization.to_buffer %rhs_byte {read_only} : !rhs_byte_ty to !rhs_buffer_ty

  %lhs_scale = bufferization.to_buffer %lhs_scale_byte {read_only} : !lhs_scale_byte_ty to !lhs_scale_buffer_ty
  %rhs_scale = bufferization.to_buffer %rhs_scale_byte {read_only} : !rhs_scale_byte_ty to !rhs_scale_buffer_ty

  // Use of thread/subgroup id here is spooky, the pipelines assume you don't.
  // It is required to ensure that the LDS offset analyzes as uniform.
  %idx = gpu.thread_id x
  %sg = gpu.subgroup_id : index

  // Input load offsets.
  %ids:2 = affine.delinearize_index %idx into (16, 32) : index, index
  %inner = arith.muli %ids#1, %c1 : index
  %outer_base = arith.muli %ids#0, %c1 : index

  %shared_inner_base = arith.muli %sg, %c2 : index

  // Scale load offsets.
  %scale_ids:2 = affine.delinearize_index %idx into (64, 8) : index, index
  %scale_inner_base = arith.muli %scale_ids#1, %c4 : index
  %scale_inner = arith.addi %scale_inner_base, %c0 : index
  %scale_outer = arith.muli %scale_ids#0, %c1 : index
  %scale_shared_inner = arith.muli %sg, %c8 : index

  scf.forall (%unused_id) in (512) {
    // Copy inputs.
    scf.for %i = %c0 to %c64 step %c16 {
      %shared_inner = arith.addi %shared_inner_base, %i : index
      %outer = arith.addi %outer_base, %i : index
      amdgpu.gather_to_lds %lhs[%outer, %inner, %c0], %lhs_shared_base[%c0, %shared_inner, %c0, %c0]
        : !lhs_copy_vec_ty, !lhs_buffer_ty, !lhs_shared_ty
      amdgpu.gather_to_lds %rhs[%outer, %inner, %c0], %rhs_shared_base[%c0, %shared_inner, %c0, %c0]
        : !rhs_copy_vec_ty, !rhs_buffer_ty, !rhs_shared_ty
    }

    // Copy scales.

    amdgpu.gather_to_lds %lhs_scale[%scale_outer, %scale_inner], %lhs_scale_shared[%c0, %scale_shared_inner, %c0, %c0]
      : !lhs_scale_copy_vec_ty, !lhs_scale_buffer_ty, !lhs_scale_shared_ty
    amdgpu.gather_to_lds %rhs_scale[%scale_outer, %scale_inner], %rhs_scale_shared[%c0, %scale_shared_inner, %c0, %c0]
      : !rhs_scale_copy_vec_ty, !rhs_scale_buffer_ty, !rhs_scale_shared_ty
  } {mapping = [#gpu.thread<linear_dim_0>]}

  %0 = tensor.empty() : !res_ty
  %1 = scf.forall (%id) in (512) shared_outs(%out = %0) -> !res_ty {
    %init = arith.constant dense<0.0> : !acc_ty

    %mfma_ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %m_id = arith.muli %mfma_ids#0, %c2 : index
    %n_id = arith.muli %mfma_ids#1, %c1 : index
    %inner_lane_offset = arith.muli %mfma_ids#3, %c1 : index
    %outer_lane_offset = arith.muli %mfma_ids#2, %c16 : index

    // cond barrier.
    %cmp0 = arith.cmpi slt, %id, %c256 : index
    %cmp1 = arith.cmpi sge, %id, %c256 : index
    scf.if %cmp0 {
      rocdl.s.barrier
    }
    %loop = scf.for %i = %c32 to %k step %c32 iter_args(%iter = %init) -> !acc_ty {
      %shift = arith.shrui %i, %c5 : index
      %buffer_num = arith.andi %shift, %c1 : index
      %curr_buffer_num = arith.xori %buffer_num, %c1 : index

      rocdl.s.waitcnt 0
      rocdl.s.barrier

      // Copy inputs / scales.
      %loop_inner = arith.addi %inner, %i : index
      scf.for %j = %c0 to %c64 step %c16 {
        %shared_inner = arith.addi %shared_inner_base, %j : index
        %outer = arith.addi %outer_base, %j : index
        amdgpu.gather_to_lds %lhs[%outer, %loop_inner, %c0], %lhs_shared_base[%buffer_num, %shared_inner, %c0, %c0]
          : !lhs_copy_vec_ty, !lhs_buffer_ty, !lhs_shared_ty
        amdgpu.gather_to_lds %rhs[%outer, %loop_inner, %c0], %rhs_shared_base[%buffer_num, %shared_inner, %c0, %c0]
          : !rhs_copy_vec_ty, !rhs_buffer_ty, !rhs_shared_ty
      }

      %loop_scale_inner = arith.addi %scale_inner_base, %i : index

      amdgpu.gather_to_lds %lhs_scale[%scale_outer, %loop_scale_inner], %lhs_scale_shared[%buffer_num, %scale_shared_inner, %c0, %c0]
        : !lhs_scale_copy_vec_ty, !lhs_scale_buffer_ty, !lhs_scale_shared_ty
      amdgpu.gather_to_lds %rhs_scale[%scale_outer, %loop_scale_inner], %rhs_scale_shared[%buffer_num, %scale_shared_inner, %c0, %c0]
        : !rhs_scale_copy_vec_ty, !rhs_scale_buffer_ty, !rhs_scale_shared_ty

      rocdl.s.barrier
      amdgpu.lds_barrier
      rocdl.s.waitcnt 0
      rocdl.sched.barrier 0
      rocdl.s.setprio 1

      // Load inputs/scales from LDS.
      %lhs_byte_vec = vector.transfer_read %lhs_shared_expand[%curr_buffer_num, %m_id, %inner_lane_offset, %c0, %outer_lane_offset],
        %cst_lhs {in_bounds = [true, true, true, true]} : !lhs_shared_expand_ty, !lhs_byte_vec_ty
      %lhs_vec = vector.bitcast %lhs_byte_vec : !lhs_byte_vec_ty to !lhs_vec_ty
      %rhs_byte_vec = vector.transfer_read %rhs_shared_expand[%curr_buffer_num, %n_id, %inner_lane_offset, %c0, %outer_lane_offset],
        %cst_rhs {in_bounds = [true, true, true, true]} : !rhs_shared_expand_ty, !rhs_byte_vec_ty
      %rhs_vec = vector.bitcast %rhs_byte_vec : !rhs_byte_vec_ty to !rhs_vec_ty

      %lhs_scale_byte_vec = vector.transfer_read %lhs_scale_shared_expand[%curr_buffer_num, %m_id, %inner_lane_offset, %c0, %mfma_ids#2],
        %cst_scale {in_bounds = [true, true, true, true]} : !lhs_scale_shared_expand_ty, !lhs_scale_byte_vec_ty
      %lhs_scale_vec = vector.bitcast %lhs_scale_byte_vec : !lhs_scale_byte_vec_ty to !lhs_scale_vec_ty
      %rhs_scale_byte_vec = vector.transfer_read %rhs_scale_shared_expand[%curr_buffer_num, %n_id, %inner_lane_offset, %c0, %mfma_ids#2],
        %cst_scale {in_bounds = [true, true, true, true]} : !rhs_scale_shared_expand_ty, !rhs_scale_byte_vec_ty
      %rhs_scale_vec = vector.bitcast %rhs_scale_byte_vec : !rhs_scale_byte_vec_ty to !rhs_scale_vec_ty

      %dot = iree_codegen.inner_tiled ins(%lhs_vec, %lhs_scale_vec, %rhs_vec, %rhs_scale_vec) outs(%iter) {
        indexing_maps = #contraction_accesses,
        iterator_types = #iterator_types,
        kind = #mfma_type
      } : !lhs_vec_ty, !lhs_scale_vec_ty, !rhs_vec_ty, !rhs_scale_vec_ty into !acc_ty

      rocdl.s.setprio 0
      amdgpu.lds_barrier
      rocdl.sched.barrier 0
      rocdl.s.waitcnt 0

      scf.yield %dot : !acc_ty
    }
    scf.if %cmp1 {
      rocdl.s.barrier
    }

    amdgpu.lds_barrier
    rocdl.s.waitcnt 0
    rocdl.s.barrier

    %last_shift = arith.shrui %k, %c5 : index
    %prev_buffer_num = arith.andi %last_shift, %c1 : index
    %final_buffer_num = arith.xori %prev_buffer_num, %c1 : index

    // Load inputs/scales from LDS.
    %lhs_byte_vec = vector.transfer_read %lhs_shared_expand[%final_buffer_num, %m_id, %inner_lane_offset, %c0, %outer_lane_offset],
      %cst_lhs {in_bounds = [true, true, true, true]} : !lhs_shared_expand_ty, !lhs_byte_vec_ty
    %lhs_vec = vector.bitcast %lhs_byte_vec : !lhs_byte_vec_ty to !lhs_vec_ty
    %rhs_byte_vec = vector.transfer_read %rhs_shared_expand[%final_buffer_num, %n_id, %inner_lane_offset, %c0, %outer_lane_offset],
      %cst_rhs {in_bounds = [true, true, true, true]} : !rhs_shared_expand_ty, !rhs_byte_vec_ty
    %rhs_vec = vector.bitcast %rhs_byte_vec : !rhs_byte_vec_ty to !rhs_vec_ty

    %lhs_scale_byte_vec = vector.transfer_read %lhs_scale_shared_expand[%final_buffer_num, %m_id, %inner_lane_offset, %c0, %mfma_ids#2],
      %cst_scale {in_bounds = [true, true, true, true]} : !lhs_scale_shared_expand_ty, !lhs_scale_byte_vec_ty
    %lhs_scale_vec = vector.bitcast %lhs_scale_byte_vec : !lhs_scale_byte_vec_ty to !lhs_scale_vec_ty
    %rhs_scale_byte_vec = vector.transfer_read %rhs_scale_shared_expand[%final_buffer_num, %n_id, %inner_lane_offset, %c0, %mfma_ids#2],
      %cst_scale {in_bounds = [true, true, true, true]} : !rhs_scale_shared_expand_ty, !rhs_scale_byte_vec_ty
    %rhs_scale_vec = vector.bitcast %rhs_scale_byte_vec : !rhs_scale_byte_vec_ty to !rhs_scale_vec_ty

    %epilogue_dot = iree_codegen.inner_tiled ins(%lhs_vec, %lhs_scale_vec, %rhs_vec, %rhs_scale_vec) outs(%loop) {
      indexing_maps = #contraction_accesses,
      iterator_types = #iterator_types,
      kind = #mfma_type
    } : !lhs_vec_ty, !lhs_scale_vec_ty, !rhs_vec_ty, !rhs_scale_vec_ty into !acc_ty

    %reduce_init = arith.constant dense<0.0> : !reduce_ty
    %reduce = vector.multi_reduction <add>, %epilogue_dot, %reduce_init [2] :
      !acc_ty to !reduce_ty

    %t = vector.transpose %reduce, [0, 2, 1, 3] : !reduce_ty to !store_ty

    %empty = tensor.empty() : !tensor_store_ty
    %to_tensor = vector.transfer_write %t, %empty[%c0, %c0, %c0, %c0]
      {in_bounds = [true, true, true, true]} : !store_ty, !tensor_store_ty

    %store_mfma_ids:4 = affine.delinearize_index %id into (2, 4, 4, 16) : index, index, index, index
    %store_outer_offset = arith.muli %store_mfma_ids#2, %c4 : index
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %to_tensor into %out[%m_id, %store_outer_offset, %n_id, %store_mfma_ids#3] [2, 4, 1, 1] [1, 1, 1, 1]
        : !tensor_store_ty into !res_ty
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  %collapse = "tensor.collapse_shape"(%1) {reassociation = #result_reassoc} : (!res_ty) -> (!return_ty)
  util.return %collapse : !return_ty
}

/// Entry point for custom lowering strategy for pingpong. Rewrites a matmul
/// as follows:
///
///  %acc = linalg.fill 0.0 : tensor<256x256xf32>
///  %0 = linalg.mmt {lowering_strategy = "cast_and_call_pingpong_matmul"}
///    ins(%lhs, %rhs: tensor<256x?xf16>, tensor<256x?xf16>)
///    outs(%acc: tensor<256x256xf32>)
///
/// to
///
///  %0 = call @pingpong_f4(%lhs, %rhs) -> tensor<256x256xf32>
///
/// And then inlines the call.
transform.named_sequence @cast_and_call_pingpong_matmul(%mm: !transform.any_op {transform.readonly}) {
  %module = transform.util.get_nearest_symbol_table %mm : (!transform.any_op) -> !transform.any_op
  %func = transform.util.lookup_nearest_symbol_from_self @pingpong_f4 : !transform.any_op
  %ins = transform.get_operand %mm[all] : (!transform.any_op) -> !transform.any_value
  %out = transform.get_result %mm[all] : (!transform.any_op) -> !transform.any_value
  // Replace
  transform.util.cast_and_call inline_call %func(%ins) -> %out after %mm {
        transform.type_conversion.tensor.cast_shape_dynamic_dims ignore_dynamic_info
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> ()
  transform.yield
}

transform.named_sequence @match_smmt_f4_f4_f32_impl(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%A: tensor<?x?x32x!lhs>,
         %B: tensor<?x?x32x!rhs>,
         %A_scales: tensor<?x?x!scale_ty>,
         %B_scales: tensor<?x?x!scale_ty>,
         %empty: tensor<?x?xf32>):
    %cst = arith.constant 0.0 : f32
    %C = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %0 = linalg.generic {
      indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
    } ins(%A, %B, %A_scales, %B_scales : tensor<?x?x32x!lhs>, tensor<?x?x32x!rhs>, tensor<?x?x!scale_ty>, tensor<?x?x!scale_ty>)
      outs(%C : tensor<?x?xf32>) {
    ^bb0(%a: !lhs, %b: !rhs, %a_scale: !scale_ty, %b_scale: !scale_ty, %out: f32):
      %1 = arith.scaling_extf %a, %a_scale : !lhs, !scale_ty to f32 
      %2 = arith.scaling_extf %b, %b_scale : !rhs, !scale_ty to f32 
      %3 = arith.mulf %1, %2 : f32 
      %4 = arith.addf %out, %3 : f32 
      linalg.yield %4 : f32 
    } -> tensor<?x?xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  transform.yield %root : !transform.any_op
}

transform.named_sequence
@match_smmt_f4_f4_f32(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_smmt_f4_f4_f32_impl failures(propagate) (%matmul)
    : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value

  // M % 256 == 0, K % 64 == 0, N % 256 == 0
  transform.iree.match.dim_is_multiple_of  %lhs[0], 64 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %lhs[1], 8 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[0], 64 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[1], 8 : !transform.any_value

  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{
      workgroup = [64, 64, 0],
      lowering_strategy = "cast_and_call_pingpong_matmul"}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
      workgroup_size = [512, 1, 1] subgroup_size = 64,
      {gpu_pipeline_options =
        #iree_gpu.pipeline_options<
          prefetch_shared_memory = false,
          no_reduce_shared_memory_bank_conflicts = true>,
        llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
  > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

/// Applies the op config for pingpong_large. This requires importing external
/// symbols needed for the custom lowering (in this case inline + replace).
transform.named_sequence @apply_pingpong_op_config(%op: !transform.any_op {transform.readonly},
                                        %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %op "__tuning_spec_applied__" : !transform.any_op
  %module = transform.util.get_nearest_symbol_table %op : (!transform.any_op) -> !transform.any_op

  // Create and serialize a module with the needed symbols.
  %syms = transform.util.create_serialized_module {
    ^bb0(%m: !transform.any_op):
      transform.util.import_symbol @cast_and_call_pingpong_matmul into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.util.import_symbol @pingpong_f4 into %m if undefined : (!transform.any_op) -> !transform.any_op
      transform.annotate %m "transform.with_named_sequence" : !transform.any_op
  } -> !transform.any_param

  // Annotate the parent function with the serialized module.
  %func = transform.get_parent_op %op {isolated_from_above} : (!transform.any_op) -> !transform.any_op
  transform.annotate %func "iree_codegen_external_symbols" = %syms : !transform.any_op, !transform.any_param
  transform.yield
}

transform.named_sequence
@__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
  attributes { iree_codegen.tuning_spec_entrypoint } {
  %res = transform.foreach_match in %variant_op
    // Match pingpong variants.
    @match_smmt_f4_f4_f32 -> @apply_pingpong_op_config
    : (!transform.any_op) -> !transform.any_op
  transform.yield %res : !transform.any_op
}

}
