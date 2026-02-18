// RUN: iree-opt %s
//
// Warp-specialized matmul ukernel for gfx950 (CDNA4/MI350).
// Uses amdgpu.gather_to_lds for async global-to-LDS DMA copies with true
// producer/consumer warp specialization (separate thread groups for copy
// and compute, following the dt_scaled pattern).
//
// Tile: 128x128, K-block: 64, Warps: 8 (512 threads), Subgroup size: 64.
// Subgroups: 2x2 (M=2, N=2) for compute, each handles 64x64 output.
// Per-subgroup: 4x4 = 16 MFMA-16x16 intrinsics, K-loop steps by 64.
// MFMA: MFMA_F32_16x16x32_F16 (K=32 per intrinsic, 2 K-steps per 64-K block).
//
// Schedule: True warp specialization with separate thread groups.
//   - Producer warps (threads 256-511, 4 subgroups): ONLY do DMA copies
//     via amdgpu.gather_to_lds. No compute at all.
//   - Consumer warps (threads 0-255, 4 subgroups): ONLY do LDS reads and
//     MFMA compute. No global loads at all.
//   - Double-buffered LDS (2 x 128x64 per operand = 64KB total).
//   - Synchronization via rocdl.s.barrier between producer and consumer.
//
// DMA scheme: 256 copy threads = 4 subgroups x 64 lanes.
// Each gather_to_lds: 64 lanes x vector<8xf16> = 512 f16.
// Per K-block per operand: 128x64 = 8192 f16 -> 16 gathers -> 4 per subgroup.
// Total per K-iteration: 8 gathers per subgroup (4 LHS + 4 RHS).

!in_ty = tensor<128x?xf16>
!in_buf_ty = memref<128x?xf16, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
// Double-buffered: buffer 0 = rows [0,128), buffer 1 = rows [128,256).
!shared_ty = memref<256x64xf16, #gpu.address_space<workgroup>>
// Expanded view for MFMA reads: 16 M-tiles x 16 lanes x 2 K-steps x 32 K-elems.
!shared_exp_ty = memref<16x16x2x32xf16, #gpu.address_space<workgroup>>

// Data-tiled output: [sg_m, sg_n, m_intr, n_intr, inner, lane, vec].
// Physical M = sg_m(2) * m_intr(4) * 16 = 128.
// Physical N = sg_n(2) * n_intr(4) * inner(4) * vec(4) = 128.
!return_ty = tensor<2x2x4x4x4x16x4xf32>

!store_ty = vector<1x1x4x4x1x1x4xf32>
!tensor_store_ty = tensor<1x1x4x4x1x1x4xf32>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

util.func private @warp_pipeline_f16(
    %lhs_base: !in_ty, %rhs_base: !in_ty,
    %unused_acc: !return_ty) -> !return_ty {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  // Double-buffered LDS for LHS and RHS.
  %lhs_shared = memref.alloc() : !shared_ty
  %rhs_shared = memref.alloc() : !shared_ty

  // K dimension and cache swizzle setup.
  %dim = tensor.dim %lhs_base, %c1 : !in_ty
  %dim_bytes = arith.muli %dim, %c2 overflow<nsw, nuw> : index
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim_bytes) : !in_ty
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim_bytes) : !in_ty

  // Bufferize inputs for DMA access.
  %lhs_buf = bufferization.to_buffer %lhs {read_only}
      : !in_ty to !in_buf_ty
  %rhs_buf = bufferization.to_buffer %rhs {read_only}
      : !in_ty to !in_buf_ty

  // Number of K-blocks.
  %num_k = arith.divui %dim, %c64 : index

  // Expanded views for MFMA LDS reads (covers both double-buffer halves).
  %lhs_exp = memref.expand_shape %lhs_shared [[0, 1], [2, 3]]
      output_shape [16, 16, 2, 32] : !shared_ty into !shared_exp_ty
  %rhs_exp = memref.expand_shape %rhs_shared [[0, 1], [2, 3]]
      output_shape [16, 16, 2, 32] : !shared_ty into !shared_exp_ty

  // =====================================================================
  // COPY THREADS (256-511): DMA global -> LDS via gather_to_lds.
  // 4 subgroups of 64 threads. Each subgroup handles 32 rows.
  // Double-buffered: iteration i writes to buffer (i % 2).
  // =====================================================================
  scf.forall (%base_id) in (512) {
    %cmp = arith.cmpi sge, %base_id, %c256 : index
    scf.if %cmp {
      %id = arith.subi %base_id, %c256 : index

      // Copy thread decomposition: 256 = 4 subgroups x 64 lanes.
      %sg = arith.divui %id, %c64 : index
      %lane = arith.remui %id, %c64 : index
      %lane_row = arith.divui %lane, %c8 : index
      %lane_col_idx = arith.remui %lane, %c8 : index
      %lane_col_off = arith.muli %lane_col_idx, %c8 overflow<nsw, nuw> : index
      %sg_base = arith.muli %sg, %c32 overflow<nsw, nuw> : index

      // K-loop: copy each K-block into the appropriate LDS buffer.
      scf.for %i = %c0 to %num_k step %c1 {
        %buffer_num = arith.andi %i, %c1 : index
        %write_off = arith.muli %buffer_num, %c128 overflow<nsw, nuw> : index
        %k_off = arith.muli %i, %c64 overflow<nsw, nuw> : index

        // LHS: 4 gathers per subgroup (32 rows / 8 rows per gather).
        scf.for %r = %c0 to %c4 step %c1 {
          %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
          %data_row = arith.addi %sg_base, %r_off overflow<nsw, nuw> : index
          %src_row = arith.addi %data_row, %lane_row overflow<nsw, nuw> : index
          %src_col = arith.addi %k_off, %lane_col_off overflow<nsw, nuw> : index
          %dst_row = arith.addi %write_off, %data_row overflow<nsw, nuw> : index
          amdgpu.gather_to_lds %lhs_buf[%src_row, %src_col],
              %lhs_shared[%dst_row, %c0]
              : vector<8xf16>, !in_buf_ty, !shared_ty
        }
        // RHS: 4 gathers per subgroup.
        scf.for %r = %c0 to %c4 step %c1 {
          %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
          %data_row = arith.addi %sg_base, %r_off overflow<nsw, nuw> : index
          %src_row = arith.addi %data_row, %lane_row overflow<nsw, nuw> : index
          %src_col = arith.addi %k_off, %lane_col_off overflow<nsw, nuw> : index
          %dst_row = arith.addi %write_off, %data_row overflow<nsw, nuw> : index
          amdgpu.gather_to_lds %rhs_buf[%src_row, %src_col],
              %rhs_shared[%dst_row, %c0]
              : vector<8xf16>, !in_buf_ty, !shared_ty
        }

        // Wait for all 8 gathers (4 LHS + 4 RHS) to complete.
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
      }

      // Final sync: ensure compute threads see the last barrier.
      rocdl.s.waitcnt 0
      rocdl.s.barrier
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}

  // =====================================================================
  // COMPUTE THREADS (0-255): LDS reads + MFMA.
  // 4 subgroups of 64 threads in 2x2 layout.
  // Each subgroup handles 64x64 output via 4x4 MFMA-16x16 intrinsics.
  // =====================================================================
  %empty = tensor.empty() : !return_ty
  %result = scf.forall (%cid) in (256)
      shared_outs(%out = %empty) -> !return_ty {

    // --- MFMA thread decomposition: 256 = 2x2x4x16 ---
    %ids:4 = affine.delinearize_index %cid into (2, 2, 4, 16)
        : index, index, index, index
    // K-dimension stride for MFMA K=32: ids#2 * 8.
    %inner_id_k = arith.muli %ids#2, %c8 overflow<nsw, nuw> : index
    // Subgroup tile offsets (4 MFMA-16 blocks per subgroup in M and N).
    %m_outer_id = arith.muli %ids#0, %c4 overflow<nsw, nuw> : index
    %n_outer_id = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index

    // Zero-initialized accumulator: 4M x 4N x 1K x 4 f32/thread.
    %zero_acc = arith.constant dense<0.0> : vector<4x4x1x4xf32>

    // Initial barrier: wait for copy threads to finish first K-block.
    rocdl.s.barrier

    // =================================================================
    // K-LOOP with double buffering.
    // =================================================================
    %loop = scf.for %i = %c0 to %num_k step %c1
        iter_args(%acc = %zero_acc) -> vector<4x4x1x4xf32> {

      // Buffer selection: read from buffer (i % 2).
      %buffer_num = arith.andi %i, %c1 : index
      %read_m_off = arith.muli %buffer_num, %c8 overflow<nsw, nuw> : index
      %m_base = arith.addi %m_outer_id, %read_m_off overflow<nsw, nuw> : index
      %n_base = arith.addi %n_outer_id, %read_m_off overflow<nsw, nuw> : index

      // LDS reads: K-step 0 (K[0:32]) and K-step 1 (K[32:64]).
      %lhs_vec_0 = vector.transfer_read
          %lhs_exp[%m_base, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<4x1x1x8xf16>
      %rhs_vec_0 = vector.transfer_read
          %rhs_exp[%n_base, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<4x1x1x8xf16>
      %lhs_vec_1 = vector.transfer_read
          %lhs_exp[%m_base, %ids#3, %c1, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<4x1x1x8xf16>
      %rhs_vec_1 = vector.transfer_read
          %rhs_exp[%n_base, %ids#3, %c1, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<4x1x1x8xf16>

      amdgpu.lds_barrier
      rocdl.sched.barrier 0

      // K-step 0: MFMA on K[0:32].
      %dot0 = iree_codegen.inner_tiled
          ins(%lhs_vec_0, %rhs_vec_0) outs(%acc) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      // K-step 1: MFMA on K[32:64].
      %dot1 = iree_codegen.inner_tiled
          ins(%lhs_vec_1, %rhs_vec_1) outs(%dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      // Barrier: release current LDS buffer, wait for next copy.
      rocdl.s.barrier

      scf.yield %dot1 : vector<4x4x1x4xf32>
    }

    // =================================================================
    // RESULT WRITEBACK via tensor.parallel_insert_slice.
    // Shape-cast vector<4x4x1x4xf32> -> vector<1x1x4x4x1x1x4xf32>
    // to match the data-tiled output layout.
    // =================================================================
    %t = vector.shape_cast %loop : vector<4x4x1x4xf32> to !store_ty

    %store_empty = tensor.empty() : !tensor_store_ty
    %to_tensor = vector.transfer_write %t,
        %store_empty[%c0, %c0, %c0, %c0, %c0, %c0, %c0]
        {in_bounds = [true, true, true, true, true, true, true]}
        : !store_ty, !tensor_store_ty

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %to_tensor into %out
          [%ids#0, %ids#1, %c0, %c0, %ids#2, %ids#3, %c0]
          [1, 1, 4, 4, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1]
          : !tensor_store_ty into !return_ty
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  util.return %result : !return_ty
}
