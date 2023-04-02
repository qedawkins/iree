// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  // Step 1. Split the reduction to get meatier (size(red) / 2)-way parallelism.
  // ===========================================================================
  %0 = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %init_or_alloc_op, %more_parallel_fill_op, %more_parallel_op, %combiner_op =
    transform.structured.split_reduction %0
      { split_factor = 2, insert_split_dimension = 1 }

  // Step 2. First level of tiling + fusion parallelizes to blocks.
  // ===========================================================================
  %forall_grid, %grid_combiner_op =
    transform.iree.tile_to_forall_and_workgroup_count_region %combiner_op tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
  %not_combiner = transform.merge_handles %fill, %more_parallel_fill_op, %more_parallel_op : !pdl.operation
  transform.structured.fuse_into_containing_op %not_combiner into %forall_grid

  // Step 3. Second level of tiling + fusion parallelizes to threads.
  // ===========================================================================
  %fill_1d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1xf32> in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  %forall_block_combiner_op, %block_combiner_op =
    transform.structured.tile_to_forall_op %grid_combiner_op tile_sizes [1] 
    ( mapping = [#gpu.thread<z>] )
  transform.structured.fuse_into_containing_op %fill_1d into %forall_block_combiner_op

  // Canonicalizations.
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation) -> ()

  %fill_2d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1x2xf32> in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  %grid_more_parallel_op = transform.structured.match ops{["linalg.generic"]}
    attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op 
      : (!pdl.operation) -> !pdl.operation
  %forall_block_more_parallel_op, %block_more_parallel_op =
    transform.structured.tile_to_forall_op %grid_more_parallel_op tile_sizes [1, 1] 
    ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
  transform.structured.fuse_into_containing_op %fill_2d into %forall_block_more_parallel_op

  // Step 4. Rank-reduce and vectorize.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
  %func_3 = transform.structured.vectorize %func

  // Step 5. Bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  transform.iree.apply_patterns %func_3 { fold_reassociative_reshapes } : (!pdl.operation) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!pdl.operation) -> !pdl.operation
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()

  // Step 6. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_5 = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.forall_to_workgroup %func_5 : (!pdl.operation) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_5
      workgroup_dims = [32, 2, 1] : (!pdl.operation) -> ()

  // Step 7. Post-bufferization vector distribution with rank-reduction.
  // ===========================================================================
  transform.iree.apply_patterns %func_5 { rank_reducing_linalg, rank_reducing_vector, fold_memref_aliases } : (!pdl.operation) -> ()
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  // Don't complain about unsupported if (threadIdx.x == 0 && threadIdx.y == 0)
  // at this point.
  transform.sequence %variant_op_3 : !pdl.operation failures(suppress) {
  ^bb0(%arg0: !pdl.operation):
    transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
  }
  transform.iree.vector.warp_distribute %func_5 : (!pdl.operation) -> ()


  // Late Canonicalizations.
  transform.iree.apply_patterns %variant_op_3
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation) -> ()
}
