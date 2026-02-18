// RUN: iree-opt %s
//
// Triton-inspired 1-cluster pingpong matmul ukernel for gfx950 (CDNA4/MI350).
// Uses amdgpu.gather_to_lds for async global-to-LDS DMA copies.
//
// Tile: 128x128, K-block: 64, Warps: 4 (256 threads), Stages: 2.
// Double-buffered LDS (2 × 128×64 per operand = 64KB total).
// Subgroups: 2x2 (M=2, N=2), each handles 64x64 output.
// Per-subgroup: 4x4 = 16 MFMA-16x16 intrinsics, K-loop steps by 64.
// Schedule: Single memory cluster + single dot cluster.
//   - DMA gathers issued first (non-blocking).
//   - LDS reads and MFMAs grouped together (dot cluster).
//   - s_setprio 1 around dot cluster, s_setprio 0 elsewhere.
// MFMA: MFMA_F32_16x16x32_F16 (K=32 per intrinsic, 2 K-steps per 64-K block).
//
// DMA scheme: 256 threads = 4 subgroups × 64 lanes.
// Each gather_to_lds: 64 lanes × vector<8xf16> = 512 f16.
// Per K-block: 128×64 = 8192 f16 → 16 gathers → 4 per subgroup.

!in_ty = tensor<128x?xf16>
!in_buf_ty = memref<128x?xf16, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
// Double-buffered: buffer 0 = rows [0,128), buffer 1 = rows [128,256).
!shared_ty = memref<256x64xf16, #gpu.address_space<workgroup>>
// Expanded view for MFMA reads: 16 M-tiles × 16 lanes × 2 K-steps × 32 K-elems.
!shared_exp_ty = memref<16x16x2x32xf16, #gpu.address_space<workgroup>>

!out_sref = !pcf.sref<128x128xf32, sync(#iree_gpu.subgroup_scope)>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

util.func private @pingpong_1cluster_f16(
    %lhs_base: !in_ty, %rhs_base: !in_ty,
    %unused_acc: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.0 : f16

  // Double-buffered LDS for LHS and RHS.
  %lhs_shared = memref.alloc() : !shared_ty
  %rhs_shared = memref.alloc() : !shared_ty

  // K dimension and cache swizzle setup.
  %dim = tensor.dim %lhs_base, %c1 : !in_ty
  %c2 = arith.constant 2 : index
  %dim_bytes = arith.muli %dim, %c2 overflow<nsw, nuw> : index
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim_bytes) : !in_ty
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim_bytes) : !in_ty

  // Bufferize inputs for DMA access.
  %lhs_buf = bufferization.to_buffer %lhs {read_only}
      : !in_ty to !in_buf_ty
  %rhs_buf = bufferization.to_buffer %rhs {read_only}
      : !in_ty to !in_buf_ty

  // Expanded views for MFMA LDS reads (covers both double-buffer halves).
  %lhs_exp = memref.expand_shape %lhs_shared [[0, 1], [2, 3]]
      output_shape [16, 16, 2, 32] : !shared_ty into !shared_exp_ty
  %rhs_exp = memref.expand_shape %rhs_shared [[0, 1], [2, 3]]
      output_shape [16, 16, 2, 32] : !shared_ty into !shared_exp_ty

  // =====================================================================
  // PROLOGUE: DMA first K-block (k=0) into buffer 0 (rows [0,128)).
  // =====================================================================
  scf.forall (%tid) in (256) {
    %sg = arith.divui %tid, %c64 : index
    %lane = arith.remui %tid, %c64 : index
    %lane_row = arith.divui %lane, %c8 : index
    %lane_col_idx = arith.remui %lane, %c8 : index
    %lane_col_off = arith.muli %lane_col_idx, %c8 overflow<nsw, nuw> : index
    %sg_base = arith.muli %sg, %c32 overflow<nsw, nuw> : index

    // LHS: 4 gathers per subgroup (128 rows / 4 sg / 8 rows per gather).
    scf.for %r = %c0 to %c4 step %c1 {
      %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
      %row_base = arith.addi %sg_base, %r_off overflow<nsw, nuw> : index
      %src_row = arith.addi %row_base, %lane_row overflow<nsw, nuw> : index
      amdgpu.gather_to_lds %lhs_buf[%src_row, %lane_col_off],
          %lhs_shared[%row_base, %c0]
          : vector<8xf16>, !in_buf_ty, !shared_ty
    }
    // RHS: 4 gathers per subgroup.
    scf.for %r = %c0 to %c4 step %c1 {
      %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
      %row_base = arith.addi %sg_base, %r_off overflow<nsw, nuw> : index
      %src_row = arith.addi %row_base, %lane_row overflow<nsw, nuw> : index
      amdgpu.gather_to_lds %rhs_buf[%src_row, %lane_col_off],
          %rhs_shared[%row_base, %c0]
          : vector<8xf16>, !in_buf_ty, !shared_ty
    }

    // Wait for all 8 gathers (4 LHS + 4 RHS) to complete.
    amdgpu.memory_counter_wait load(0)
    rocdl.s.barrier
  } {mapping = [#gpu.thread<linear_dim_0>]}

  // =====================================================================
  // MAIN COMPUTE: pcf.generic with pingpong K-loop.
  // =====================================================================
  %result = pcf.generic scope(#iree_gpu.subgroup_scope)
    execute(%out_ref = %unused_acc)[%sg_id: index, %num_sg: index]
         : (!out_sref) -> (tensor<128x128xf32>) {
    pcf.generic scope(#iree_gpu.lane_scope)
      execute[%lane_id: index, %sg_size: index] {

      // --- MFMA thread decomposition: 256 = 2×2×4×16 ---
      %id = affine.linearize_index disjoint [%sg_id, %lane_id]
          by (%num_sg, %sg_size) : index
      %ids:4 = affine.delinearize_index %id into (2, 2, 4, 16)
          : index, index, index, index
      // K-dimension stride for MFMA K=32: ids#2 * 8.
      %inner_id_k = arith.muli %ids#2, %c8 overflow<nsw, nuw> : index
      // N-dimension stride for MFMA 16x16: ids#2 * 4.
      %inner_id_n = arith.muli %ids#2, %c4 overflow<nsw, nuw> : index
      // Subgroup tile offsets (4 MFMA-16 blocks per subgroup in M and N).
      %m_outer_id = arith.muli %ids#0, %c4 overflow<nsw, nuw> : index
      %n_outer_id = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index

      // --- DMA thread decomposition ---
      %lane_row = arith.divui %lane_id, %c8 : index
      %lane_col_idx = arith.remui %lane_id, %c8 : index
      %lane_col_off = arith.muli %lane_col_idx, %c8 overflow<nsw, nuw> : index
      %sg_row_base = arith.muli %sg_id, %c32 overflow<nsw, nuw> : index

      // Zero-initialized accumulator: 4M × 4N × 1K × 4 f32/thread.
      %zero_acc = arith.constant dense<0.0> : vector<4x4x1x4xf32>
      %c0_idx = arith.constant 0 : index

      // =================================================================
      // K-LOOP with double buffering.
      // cur_buf: index of buffer containing current (ready) data.
      // DMA writes to (1 - cur_buf), MFMA reads from cur_buf.
      // =================================================================
      %loop:2 = scf.for %k = %c64 to %dim step %c64
          iter_args(%acc = %zero_acc, %cur_buf = %c0_idx)
          -> (vector<4x4x1x4xf32>, index) {

        // Buffer selection.
        %next_buf = arith.subi %c1, %cur_buf : index
        %write_off = arith.muli %next_buf, %c128 overflow<nsw, nuw> : index
        %read_m_off = arith.muli %cur_buf, %c8 overflow<nsw, nuw> : index

        // === MEMORY CLUSTER: DMA next K-block to next_buf ===
        // LHS: 4 gathers.
        scf.for %r = %c0 to %c4 step %c1 {
          %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
          %data_row = arith.addi %sg_row_base, %r_off overflow<nsw, nuw> : index
          %src_row = arith.addi %data_row, %lane_row overflow<nsw, nuw> : index
          %src_col = arith.addi %k, %lane_col_off overflow<nsw, nuw> : index
          %dst_row = arith.addi %write_off, %data_row overflow<nsw, nuw> : index
          amdgpu.gather_to_lds %lhs_buf[%src_row, %src_col],
              %lhs_shared[%dst_row, %c0]
              : vector<8xf16>, !in_buf_ty, !shared_ty
        }
        // RHS: 4 gathers.
        scf.for %r = %c0 to %c4 step %c1 {
          %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
          %data_row = arith.addi %sg_row_base, %r_off overflow<nsw, nuw> : index
          %src_row = arith.addi %data_row, %lane_row overflow<nsw, nuw> : index
          %src_col = arith.addi %k, %lane_col_off overflow<nsw, nuw> : index
          %dst_row = arith.addi %write_off, %data_row overflow<nsw, nuw> : index
          amdgpu.gather_to_lds %rhs_buf[%src_row, %src_col],
              %rhs_shared[%dst_row, %c0]
              : vector<8xf16>, !in_buf_ty, !shared_ty
        }

        // === DOT CLUSTER: MFMA on current K-block from cur_buf ===
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

        rocdl.sched.barrier 0
        rocdl.s.setprio 1 { iree_gpu.swap_mfma = 1 }

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

        rocdl.s.setprio 0
        rocdl.sched.barrier 0

        // === SYNC: wait for DMA completion ===
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
        rocdl.sched.barrier 0

        scf.yield %dot1, %next_buf : vector<4x4x1x4xf32>, index
      }

      // =================================================================
      // EPILOGUE: compute last K-block (already in LDS from last DMA).
      // =================================================================
      %epi_m_off = arith.muli %loop#1, %c8 overflow<nsw, nuw> : index
      %epi_m = arith.addi %m_outer_id, %epi_m_off overflow<nsw, nuw> : index
      %epi_n = arith.addi %n_outer_id, %epi_m_off overflow<nsw, nuw> : index

      %lhs_epi_0 = vector.transfer_read
          %lhs_exp[%epi_m, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<4x1x1x8xf16>
      %rhs_epi_0 = vector.transfer_read
          %rhs_exp[%epi_n, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<4x1x1x8xf16>
      %epi_dot0 = iree_codegen.inner_tiled
          ins(%lhs_epi_0, %rhs_epi_0) outs(%loop#0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      %lhs_epi_1 = vector.transfer_read
          %lhs_exp[%epi_m, %ids#3, %c1, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<4x1x1x8xf16>
      %rhs_epi_1 = vector.transfer_read
          %rhs_exp[%epi_n, %ids#3, %c1, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<4x1x1x8xf16>
      %epi_dot1 = iree_codegen.inner_tiled
          ins(%lhs_epi_1, %rhs_epi_1) outs(%epi_dot0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      // =================================================================
      // RESULT WRITEBACK via pcf.write_slice.
      // =================================================================
      %tp = vector.transpose %epi_dot1, [0, 2, 1, 3]
          : vector<4x4x1x4xf32> to vector<4x1x4x4xf32>
      %empty = tensor.empty() : tensor<4x1x4x4xf32>
      %result_tensor = vector.transfer_write %tp, %empty[%c0, %c0, %c0, %c0]
          {in_bounds = [true, true, true, true]}
          : vector<4x1x4x4xf32>, tensor<4x1x4x4xf32>
      scf.for %a = %c0 to %c4 step %c1 {
        scf.for %b = %c0 to %c4 step %c1 {
          %row = affine.apply affine_map<(d0, d1, d2) -> ((d0 + d1) * 16 + d2)>
              (%m_outer_id, %a, %ids#3)
          %col = affine.apply affine_map<(d0, d1, d2) -> ((d0 + d1) * 16 + d2)>
              (%n_outer_id, %b, %inner_id_n)
          %tile = tensor.extract_slice %result_tensor[%a, 0, %b, 0] [1, 1, 1, 4] [1, 1, 1, 1]
              : tensor<4x1x4x4xf32> to tensor<1x4xf32>
          pcf.write_slice %tile into %out_ref[%row, %col] [1, 4] [1, 1]
              : tensor<1x4xf32> into !out_sref
        } {iree_codegen.unroll}
      } {iree_codegen.unroll}
      pcf.return
    }
    pcf.return
  }
  util.return %result : tensor<128x128xf32>
}
