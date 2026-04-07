// RUN: iree-opt --split-input-file --verify-diagnostics %s

func.func @mma_inner_tiled_invalid_num_inputs(%lhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{number of inputs (1) doesn't match expected number from kind (2)}}
  %0 = iree_codegen.inner_tiled ins(%lhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_num_outputs(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> (tensor<?x?x4xf32>, tensor<?x?x4xf32>) {
  // expected-error @+1 {{number of outputs (2) doesn't match expected number from kind (1)}}
  %0:2 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc, %acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>, tensor<?x?x4xf32>
  return %0#0, %0#1 : tensor<?x?x4xf32>, tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_num_indexing_maps(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{expected an indexing map for each operand}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_indexing_map_num_dims(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{expected indexing map 0 to have 3 input dims}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k, x) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_indexing_map_num_results(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{expected indexing map 0 to have fewer than 3 results}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, j, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_indexing_map_non_permutation(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{expected indexing map 0 to be a projected permutation}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, j + k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_outer_shape(%lhs: tensor<2x2x4xf16>, %rhs: tensor<2x3x4xf16>, %acc: tensor<2x2x4xf32>) -> tensor<2x2x4xf32> {
  // expected-error @+1 {{shape does not match iteration bounds}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<2x2x4xf16>, tensor<2x3x4xf16> into tensor<2x2x4xf32>
  return %0 : tensor<2x2x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_dynamic_inner_dim(%lhs: tensor<?x?x?xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{Unexpected dynamic inner dim for operand 0 of type 'tensor<?x?x?xf16>'}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x?xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_element_type(%lhs: tensor<?x?x4xf32>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{op operand element type f32 does not match expected MMA tile element type f16}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf32>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_inner_types_distributed_opaque(%lhs: tensor<?x?x3xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{op operand type tensor<?x?x3xf16>, implying tile type vector<3xf16>, is incompatible with permuted InnerTiledDescAttr tile type vector<4xf16> under semantics #iree_gpu.mma_semantics<distributed = true, opaque = true>}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
  } : tensor<?x?x3xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_inner_types_undistributed_nonopaque(%lhs: tensor<?x?x4x16x4xf16>, %rhs: tensor<?x?x4x16x4xf16>, %acc: tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> {
  // expected-error @+1 {{op operand type tensor<?x?x16x16xf32>, implying tile type vector<16x16xf32>, is incompatible with permuted InnerTiledDescAttr tile type vector<4x16x4xf32> under semantics #iree_gpu.mma_semantics<distributed = false, opaque = false>}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>
  } : tensor<?x?x4x16x4xf16>, tensor<?x?x4x16x4xf16> into tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}

// -----

func.func @mma_inner_tiled_invalid_inner_types_distributed_nonopaque(%lhs: tensor<?x?x1x1x4xf16>, %rhs: tensor<?x?x1x1x4xf16>, %acc: tensor<?x?x1x2x2xf32>) -> tensor<?x?x1x2x2xf32> {
  // expected-error @+1 {{op operand type tensor<?x?x1x2x2xf32>, implying tile type vector<1x2x2xf32>, is incompatible with permuted InnerTiledDescAttr tile type vector<1x1x4xf32> under semantics #iree_gpu.mma_semantics<distributed = true, opaque = false>}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : tensor<?x?x1x1x4xf16>, tensor<?x?x1x1x4xf16> into tensor<?x?x1x2x2xf32>
  return %0 : tensor<?x?x1x2x2xf32>
}

// -----

func.func @vector_multi_mma_with_wrong_number_of_permutations(%lhs: vector<2x3x4xf16>, %rhs: vector<3x5x4xf16>, %acc: vector<2x5x4xf32>) -> vector<2x5x4xf32> {
  // expected-error @+1 {{op mismatch between the number of permutations (2) and the number of operands (3)}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [
      affine_map<(i, j, k) -> (i, k)>,
      affine_map<(i, j, k) -> (k, j)>,
      affine_map<(i, j, k) -> (i, j)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>,
    permutations = [array<i64: 0, 1>, array<i64: 1, 0>]
  } : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
  return %0 : vector<2x5x4xf32>
}

// -----

func.func @vector_multi_mma_with_permutation_of_wrong_size(%lhs: vector<2x3x4xf16>, %rhs: vector<3x5x4xf16>, %acc: vector<2x5x4xf32>) -> vector<2x5x4xf32> {
  // expected-error @+1 {{op permutation #0 length 2 does not match the inner rank 1 of the corresponding operand of type vector<2x3x4xf16>}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [
      affine_map<(i, j, k) -> (i, k)>,
      affine_map<(i, j, k) -> (k, j)>,
      affine_map<(i, j, k) -> (i, j)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>,
    permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
  } : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
  return %0 : vector<2x5x4xf32>
}

// -----

func.func @coalesced_gather_dma_source_kind_mismatch_tensor_init(
    %source: memref<4x32xf32>, %dest: tensor<4x32xf32>, %lane: index) -> tensor<4x32xf32> {
  // expected-error @+1 {{source must be tensor when init is tensor}}
  %0 = iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    : memref<4x32xf32>, tensor<4x32xf32>, index -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

func.func @coalesced_gather_dma_source_kind_mismatch_memref_init(
    %source: tensor<4x32xf32>, %dest: memref<4x32xf32>, %lane: index) {
  // expected-error @+1 {{source must be memref when init is memref}}
  iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    : tensor<4x32xf32>, memref<4x32xf32>, index
  return
}

// -----

func.func @coalesced_gather_dma_memref_form_vector_index(
    %idx0: vector<4xi32>, %source: memref<64x32xf32>, %dest: memref<4x32xf32>, %lane: index) {
  // expected-error @+1 {{expected memref index operand 0 when init is memref}}
  iree_gpu.coalesced_gather_dma %source[%idx0] into %dest lane(%lane)
    : memref<64x32xf32>, vector<4xi32>, memref<4x32xf32>, index
  return
}

// -----

func.func @coalesced_gather_dma_too_many_indices(
    %idx0: vector<4xi32>, %idx1: vector<4xi32>, %idx2: vector<4xi32>,
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>, %lane: index) -> tensor<4x4xf32> {
  // expected-error @+1 {{number of indices (3) cannot exceed destination rank (2)}}
  %0 = iree_gpu.coalesced_gather_dma %source[%idx0, %idx1, %idx2] into %dest lane(%lane)
    : tensor<4x4xf32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, tensor<4x4xf32>, index -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

func.func @coalesced_gather_dma_indices_exceed_source_rank(
    %idx0: vector<4xi32>, %idx1: vector<4xi32>,
    %source: tensor<64xf32>, %dest: tensor<4x8xf32>, %lane: index) -> tensor<4x8xf32> {
  // expected-error @+1 {{number of indices (2) cannot exceed source rank (1)}}
  %0 = iree_gpu.coalesced_gather_dma %source[%idx0, %idx1] into %dest lane(%lane)
    : tensor<64xf32>, vector<4xi32>, vector<4xi32>, tensor<4x8xf32>, index -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

func.func @coalesced_gather_dma_dynamic_index_shape(
    %idx0: tensor<?xi32>, %source: tensor<4x32xf32>, %dest: tensor<4x32xf32>, %lane: index) -> tensor<4x32xf32> {
  // expected-error @+1 {{expected index 0 to have static shape}}
  %0 = iree_gpu.coalesced_gather_dma %source[%idx0] into %dest lane(%lane)
    : tensor<4x32xf32>, tensor<?xi32>, tensor<4x32xf32>, index -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

func.func @coalesced_gather_dma_non_1d_index(
    %idx0: vector<2x4xi32>, %source: tensor<4x32xf32>, %dest: tensor<4x32xf32>, %lane: index) -> tensor<4x32xf32> {
  // expected-error @+1 {{expected index 0 to be a 1-D tensor or vector}}
  %0 = iree_gpu.coalesced_gather_dma %source[%idx0] into %dest lane(%lane)
    : tensor<4x32xf32>, vector<2x4xi32>, tensor<4x32xf32>, index -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

func.func @coalesced_gather_dma_non_1d_index_1(
    %idx0: vector<4xi32>, %idx1: vector<2x4xi32>,
    %source: tensor<64x32xf32>, %dest: tensor<4x32xf32>, %lane: index) -> tensor<4x32xf32> {
  // expected-error @+1 {{expected index 1 to be a 1-D tensor or vector}}
  %0 = iree_gpu.coalesced_gather_dma %source[%idx0, %idx1] into %dest lane(%lane)
    : tensor<64x32xf32>, vector<4xi32>, vector<2x4xi32>, tensor<4x32xf32>, index -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

func.func @coalesced_gather_dma_index_length_mismatch(
    %idx0: vector<4xi32>, %idx1: vector<5xi32>,
    %source: tensor<64x128xf32>, %dest: tensor<4x128xf32>, %lane: index) -> tensor<4x128xf32> {
  // expected-error @+1 {{expected all index vectors to have the same length; index 1 has length 5 but expected 4}}
  %0 = iree_gpu.coalesced_gather_dma %source[%idx0, %idx1] into %dest lane(%lane)
    : tensor<64x128xf32>, vector<4xi32>, vector<5xi32>, tensor<4x128xf32>, index -> tensor<4x128xf32>
  return %0 : tensor<4x128xf32>
}

// -----

func.func @coalesced_gather_dma_batch_size_mismatch(
    %idx0: vector<5xi32>, %source: tensor<64x128xf32>, %dest: tensor<4x128xf32>, %lane: index) -> tensor<4x128xf32> {
  // expected-error @+1 {{expected batch size (length of index vectors: 5) to match first destination dimension (4)}}
  %0 = iree_gpu.coalesced_gather_dma %source[%idx0] into %dest lane(%lane)
    : tensor<64x128xf32>, vector<5xi32>, tensor<4x128xf32>, index -> tensor<4x128xf32>
  return %0 : tensor<4x128xf32>
}

// -----

func.func @coalesced_gather_dma_unindexed_dim_mismatch(
    %source: tensor<4x16xf32>, %dest: tensor<4x32xf32>, %lane: index) -> tensor<4x32xf32> {
  // expected-error @+1 {{expected unindexed dimension 1 to have same length in source (16) and destination (32)}}
  %0 = iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    : tensor<4x16xf32>, tensor<4x32xf32>, index -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

func.func @coalesced_gather_dma_source_rank_too_small(
    %source: tensor<4xf32>, %dest: tensor<4x32xf32>, %lane: index) -> tensor<4x32xf32> {
  // expected-error @+1 {{expected source to have at least 2 dimensions when destination has rank 2}}
  %0 = iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    : tensor<4xf32>, tensor<4x32xf32>, index -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

func.func @coalesced_gather_dma_in_bounds_size_mismatch(
    %source: tensor<4x32xf32>, %dest: tensor<4x32xf32>, %lane: index) -> tensor<4x32xf32> {
  // expected-error @+1 {{in_bounds array size (1) must match init rank (2)}}
  %0 = iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    in_bounds [true]
    : tensor<4x32xf32>, tensor<4x32xf32>, index -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

func.func @coalesced_gather_dma_in_bounds_size_mismatch_precedes_dim_check(
    %source: tensor<4x16xf32>, %dest: tensor<4x32xf32>, %lane: index) -> tensor<4x32xf32> {
  // expected-error @+1 {{in_bounds array size (1) must match init rank (2)}}
  %0 = iree_gpu.coalesced_gather_dma %source into %dest lane(%lane)
    in_bounds [true]
    : tensor<4x16xf32>, tensor<4x32xf32>, index -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

func.func @dma_copy_source_not_sref(
    %source: tensor<4x4xf16>, %dest: !pcf.sref<4x4xf16, #pcf.sequential>) {
  // expected-error @+1 {{source and dest must be pcf.sref types}}
  iree_gpu.dma_copy %source[0, 0] [4, 4] [1, 1]
                    to %dest[0, 0] [4, 4] [1, 1]
                    : tensor<4x4xf16>
                    -> !pcf.sref<4x4xf16, #pcf.sequential>
  return
}

// -----

func.func @dma_copy_element_type_mismatch(
    %source: !pcf.sref<4x4xf16, #pcf.test_scope>,
    %dest: !pcf.sref<4x4xf32, #pcf.sequential>) {
  // expected-error @+1 {{source element type 'f16' does not match dest element type 'f32'}}
  iree_gpu.dma_copy %source[0, 0] [4, 4] [1, 1]
                    to %dest[0, 0] [4, 4] [1, 1]
                    : !pcf.sref<4x4xf16, #pcf.test_scope>
                    -> !pcf.sref<4x4xf32, #pcf.sequential>
  return
}
