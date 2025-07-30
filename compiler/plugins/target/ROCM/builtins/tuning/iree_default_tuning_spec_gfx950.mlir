// RUN: iree-opt %s

// Element types

!lhs = f4E2M1FN
!rhs = f4E2M1FN

!scale_ty = f8E8M0FNU

#contraction_accesses = [
  affine_map<(i, j, k, d) -> (i, d, k)>,
  affine_map<(i, j, k, d) -> (i, d, k)>,
  affine_map<(i, j, k, d) -> (j, d, k)>,
  affine_map<(i, j, k, d) -> (j, d, k)>,
  affine_map<(i, j, k, d) -> (i, j, k)>
]

#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>

module @iree_default_tuning_spec_gfx942 attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {

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
@match_smmt_64x256_f4f4f32(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param, !transform.any_param) {
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
    lowering_config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
      workgroup_size = [512, 1, 1] subgroup_size = 64,
      {gpu_pipeline_options =
        #iree_gpu.pipeline_options<
          prefetch_shared_memory = false,
          no_reduce_shared_memory_bank_conflicts = true>,
        llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
  > -> !transform.any_param
  %ukernel = transform.param.constant #iree_codegen.ukernel_descriptor<"mmt_64x256_f4f4f32", tensor> -> !transform.any_param
  transform.yield %matmul, %config, %ukernel : !transform.any_op, !transform.any_param, !transform.any_param
}

transform.named_sequence
@match_smmt_8x128_f4f4f32(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param, !transform.any_param) {
  %mmt = transform.include @match_smmt_f4_f4_f32_impl failures(propagate) (%matmul)
    : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value

  // M % 256 == 0, K % 64 == 0, N % 256 == 0
  transform.iree.match.dim_is_multiple_of  %lhs[0], 8 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %lhs[1], 32 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[0], 128 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %rhs[1], 32 : !transform.any_value

  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{workgroup = [8, 128, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
      workgroup_size = [512, 1, 1] subgroup_size = 64,
      {gpu_pipeline_options =
        #iree_gpu.pipeline_options<
          prefetch_shared_memory = false,
          no_reduce_shared_memory_bank_conflicts = true>,
        llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
  > -> !transform.any_param
  %ukernel = transform.param.constant #iree_codegen.ukernel_descriptor<"mmt_8x128_f4f4f32", tensor> -> !transform.any_param
  transform.yield %matmul, %config, %ukernel : !transform.any_op, !transform.any_param, !transform.any_param
}

/// Applies the op config for pingpong_large. This requires importing external
/// symbols needed for the custom lowering (in this case inline + replace).
transform.named_sequence @apply_ukernel_op_config(
    %op: !transform.any_op {transform.readonly},
    %config: !transform.any_param {transform.readonly},
    %ukernel: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %op "iree_codegen.ukernel" = %ukernel : !transform.any_op, !transform.any_param
  transform.annotate %op "__tuning_spec_applied__" : !transform.any_op
  %module = transform.util.get_nearest_symbol_table %op : (!transform.any_op) -> !transform.any_op

  transform.yield
}

transform.named_sequence
@__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
  attributes { iree_codegen.tuning_spec_entrypoint } {
  %res = transform.foreach_match in %variant_op
    // Match pingpong variants.
    @match_smmt_8x128_f4f4f32 -> @apply_ukernel_op_config,
    @match_smmt_64x256_f4f4f32 -> @apply_ukernel_op_config
    : (!transform.any_op) -> !transform.any_op
  transform.yield %res : !transform.any_op
}

}
