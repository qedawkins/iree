// RUN: iree-opt %s
//
// Triton-inspired chained-dot pingpong ukernel for gfx950 (CDNA4/MI350).
// Uses amdgpu.gather_to_lds for async global-to-LDS DMA copies.
//
// Demonstrates the FlashAttention v3 scheduling pattern where two matmuls
// are chained: dot1's output (cast f32->f16) feeds dot2's LHS input.
//   C1 = A1 @ B1^T          (dot1: standard matmul)
//   C2 = trunc(C1) @ B2^T   (dot2: chained, LHS from dot1 result)
//
// Tile: 128x128 per dot, K-block: 64, Warps: 4 (256 threads), Stages: 2.
// Subgroups: 2x2 (M=2, N=2), each handles 64x64 output.
// Per-subgroup: 4x4 = 16 MFMA-16x16 intrinsics, K-loop steps by 64.
// MFMA: MFMA_F32_16x16x32_F16 (K=32 per intrinsic, 2 K-steps per 64-K block).
//
// DMA scheme: 256 threads = 4 subgroups x 64 lanes.
// Each gather_to_lds: 64 lanes x vector<8xf16> = 512 f16.
// Per K-block: 128x64 = 8192 f16 -> 16 gathers -> 4 per subgroup.
//
// Schedule: Reversed priority compared to standard 1-cluster pingpong.
//   - Memory operations get HIGH priority (s_setprio 1).
//   - Compute operations get LOW priority (s_setprio 0).
//   This overlaps dot2 memory prefetch with dot1 compute.
//
// LDS layout: 5 buffers, 96KB total (fits in gfx950's 128KB LDS).
//   - q_shared:  128x64  f16 (16KB) - A1/Q, single buffer, loaded once.
//   - k_shared:  256x64  f16 (32KB) - B1/K, double-buffered.
//   - v_shared:  128x64  f16 (16KB) - B2/V, single buffer per dot2 K-block.
//   - s0_shared: 128x64  f16 (16KB) - Intermediate S columns 0-63.
//   - s1_shared: 128x64  f16 (16KB) - Intermediate S columns 64-127.
//
// Between dot1 and dot2:
//   1. Truncate dot1 accumulator f32 -> f16.
//   2. Store columns 0-63 of result to s0_shared.
//   3. Store columns 64-127 of result to s1_shared.
//   4. Barrier.
//   5. Read back as MFMA-A inputs for dot2 K-loop.
//
// Dot2 K-loop has exactly 2 iterations (K=128, K-block=64):
//   Iteration 0: LHS from s0_shared (dot1 columns 0-63), RHS from v_shared.
//   Iteration 1: LHS from s1_shared (dot1 columns 64-127), RHS from v_shared.

!in_ty = tensor<128x?xf16>
!in_buf_ty = memref<128x?xf16, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

// Q: single buffer 128x64.
!q_shared_ty = memref<128x64xf16, #gpu.address_space<workgroup>>
!q_exp_ty = memref<8x16x2x32xf16, #gpu.address_space<workgroup>>

// K: double-buffered 256x64 (buffer 0 = rows [0,128), buffer 1 = rows [128,256)).
!k_shared_ty = memref<256x64xf16, #gpu.address_space<workgroup>>
!k_exp_ty = memref<16x16x2x32xf16, #gpu.address_space<workgroup>>

// V: single buffer 128x64 (reloaded for each dot2 K-block via DMA).
!v_shared_ty = memref<128x64xf16, #gpu.address_space<workgroup>>
!v_exp_ty = memref<8x16x2x32xf16, #gpu.address_space<workgroup>>

// Intermediate S: two 128x64 buffers for dot1 result columns 0-63 and 64-127.
!s_shared_ty = memref<128x64xf16, #gpu.address_space<workgroup>>
!s_exp_ty = memref<8x16x2x32xf16, #gpu.address_space<workgroup>>

!out_sref = !pcf.sref<128x128xf32, sync(#iree_gpu.subgroup_scope)>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

util.func private @pingpong_chained_f16(
    %a1_base: !in_ty,
    %b1_base: !in_ty,
    %b2_base: !in_ty,
    %unused_acc: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.0 : f16

  // === LDS ALLOCATION: 5 buffers, 96KB total ===
  %q_shared = memref.alloc() : !q_shared_ty
  %k_shared = memref.alloc() : !k_shared_ty
  %v_shared = memref.alloc() : !v_shared_ty
  %s0_shared = memref.alloc() : !s_shared_ty
  %s1_shared = memref.alloc() : !s_shared_ty

  // Dynamic K dimension (shared by A1 and B1).
  %dim = tensor.dim %a1_base, %c1 : !in_ty
  %dim_bytes = arith.muli %dim, %c2 overflow<nsw, nuw> : index
  %a1 = iree_gpu.buffer_resource_cast %a1_base cacheSwizzleStride(%dim_bytes) : !in_ty
  %b1 = iree_gpu.buffer_resource_cast %b1_base cacheSwizzleStride(%dim_bytes) : !in_ty

  // B2 has its own K dimension (same as dot1 output N = 128, but dynamic arg).
  %dim_b2 = tensor.dim %b2_base, %c1 : !in_ty
  %dim_b2_bytes = arith.muli %dim_b2, %c2 overflow<nsw, nuw> : index
  %b2 = iree_gpu.buffer_resource_cast %b2_base cacheSwizzleStride(%dim_b2_bytes) : !in_ty

  // Bufferize inputs for DMA access.
  %a1_buf = bufferization.to_buffer %a1 {read_only}
      : !in_ty to !in_buf_ty
  %b1_buf = bufferization.to_buffer %b1 {read_only}
      : !in_ty to !in_buf_ty
  %b2_buf = bufferization.to_buffer %b2 {read_only}
      : !in_ty to !in_buf_ty

  // Expanded views for MFMA LDS reads.
  %q_exp = memref.expand_shape %q_shared [[0, 1], [2, 3]]
      output_shape [8, 16, 2, 32] : !q_shared_ty into !q_exp_ty
  %k_exp = memref.expand_shape %k_shared [[0, 1], [2, 3]]
      output_shape [16, 16, 2, 32] : !k_shared_ty into !k_exp_ty
  %v_exp = memref.expand_shape %v_shared [[0, 1], [2, 3]]
      output_shape [8, 16, 2, 32] : !v_shared_ty into !v_exp_ty
  %s0_exp = memref.expand_shape %s0_shared [[0, 1], [2, 3]]
      output_shape [8, 16, 2, 32] : !s_shared_ty into !s_exp_ty
  %s1_exp = memref.expand_shape %s1_shared [[0, 1], [2, 3]]
      output_shape [8, 16, 2, 32] : !s_shared_ty into !s_exp_ty

  // =========================================================================
  // PROLOGUE: DMA first K-block of A1 (Q) and B1 (K) into LDS.
  // Q -> q_shared, K[0] -> k_shared rows [0,128).
  // =========================================================================
  scf.forall (%tid) in (256) {
    %sg = arith.divui %tid, %c64 : index
    %lane = arith.remui %tid, %c64 : index
    %lane_row = arith.divui %lane, %c8 : index
    %lane_col_idx = arith.remui %lane, %c8 : index
    %lane_col_off = arith.muli %lane_col_idx, %c8 overflow<nsw, nuw> : index
    %sg_base = arith.muli %sg, %c32 overflow<nsw, nuw> : index

    // Q (A1): 4 gathers per subgroup into q_shared.
    scf.for %r = %c0 to %c4 step %c1 {
      %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
      %row_base = arith.addi %sg_base, %r_off overflow<nsw, nuw> : index
      %src_row = arith.addi %row_base, %lane_row overflow<nsw, nuw> : index
      amdgpu.gather_to_lds %a1_buf[%src_row, %lane_col_off],
          %q_shared[%row_base, %c0]
          : vector<8xf16>, !in_buf_ty, !q_shared_ty
    }
    // K (B1) block 0: 4 gathers per subgroup into k_shared rows [0,128).
    scf.for %r = %c0 to %c4 step %c1 {
      %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
      %row_base = arith.addi %sg_base, %r_off overflow<nsw, nuw> : index
      %src_row = arith.addi %row_base, %lane_row overflow<nsw, nuw> : index
      amdgpu.gather_to_lds %b1_buf[%src_row, %lane_col_off],
          %k_shared[%row_base, %c0]
          : vector<8xf16>, !in_buf_ty, !k_shared_ty
    }

    // Wait for all 8 gathers (4 Q + 4 K) to complete.
    amdgpu.memory_counter_wait load(0)
    rocdl.s.barrier
  } {mapping = [#gpu.thread<linear_dim_0>]}

  // =========================================================================
  // MAIN COMPUTE: pcf.generic with dot1 K-loop, intermediate, and dot2.
  // =========================================================================
  %result = pcf.generic scope(#iree_gpu.subgroup_scope)
    execute(%out_ref = %unused_acc)[%sg_id: index, %num_sg: index]
         : (!out_sref) -> (tensor<128x128xf32>) {
    pcf.generic scope(#iree_gpu.lane_scope)
      execute[%lane_id: index, %sg_size: index] {

      // --- MFMA thread decomposition: 256 = 2x2x4x16 ---
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

      // Zero-initialized accumulator for dot1: 4M x 4N x 1K x 4 f32/thread.
      %zero_acc = arith.constant dense<0.0> : vector<4x4x1x4xf32>

      // ===================================================================
      // DOT1 K-LOOP: C1 = A1 @ B1^T, with double-buffered DMA for B1.
      // Q (A1) is read from q_shared (single buffer, loaded in prologue).
      // K (B1) is double-buffered in k_shared.
      // ===================================================================
      %dot1_loop:2 = scf.for %k = %c64 to %dim step %c64
          iter_args(%dot1_acc = %zero_acc, %cur_buf = %c0)
          -> (vector<4x4x1x4xf32>, index) {

        // Buffer selection for K double-buffering.
        %next_buf = arith.subi %c1, %cur_buf : index
        %write_off = arith.muli %next_buf, %c128 overflow<nsw, nuw> : index
        %read_k_off = arith.muli %cur_buf, %c8 overflow<nsw, nuw> : index

        // === MEMORY CLUSTER: DMA next K-block of B1 to next_buf ===
        scf.for %r = %c0 to %c4 step %c1 {
          %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
          %data_row = arith.addi %sg_row_base, %r_off overflow<nsw, nuw> : index
          %src_row = arith.addi %data_row, %lane_row overflow<nsw, nuw> : index
          %src_col = arith.addi %k, %lane_col_off overflow<nsw, nuw> : index
          %dst_row = arith.addi %write_off, %data_row overflow<nsw, nuw> : index
          amdgpu.gather_to_lds %b1_buf[%src_row, %src_col],
              %k_shared[%dst_row, %c0]
              : vector<8xf16>, !in_buf_ty, !k_shared_ty
        }

        // === DOT1 CLUSTER: MFMA on Q and K[cur_buf] ===
        // Q reads from q_exp (single buffer, no offset).
        // K reads from k_exp with cur_buf offset.
        %k_base = arith.addi %n_outer_id, %read_k_off overflow<nsw, nuw> : index

        // LDS reads: K-step 0 (K[0:32]) and K-step 1 (K[32:64]).
        %q_vec_0 = vector.transfer_read
            %q_exp[%m_outer_id, %ids#3, %c0, %inner_id_k], %cst
            {in_bounds = [true, true, true, true]}
            : !q_exp_ty, vector<4x1x1x8xf16>
        %k_vec_0 = vector.transfer_read
            %k_exp[%k_base, %ids#3, %c0, %inner_id_k], %cst
            {in_bounds = [true, true, true, true]}
            : !k_exp_ty, vector<4x1x1x8xf16>
        %q_vec_1 = vector.transfer_read
            %q_exp[%m_outer_id, %ids#3, %c1, %inner_id_k], %cst
            {in_bounds = [true, true, true, true]}
            : !q_exp_ty, vector<4x1x1x8xf16>
        %k_vec_1 = vector.transfer_read
            %k_exp[%k_base, %ids#3, %c1, %inner_id_k], %cst
            {in_bounds = [true, true, true, true]}
            : !k_exp_ty, vector<4x1x1x8xf16>

        rocdl.sched.barrier 0
        rocdl.s.setprio 1

        // K-step 0: MFMA on K[0:32].
        %d1s0 = iree_codegen.inner_tiled
            ins(%q_vec_0, %k_vec_0) outs(%dot1_acc) {
          indexing_maps = #contraction_accesses,
          iterator_types = [#linalg.iterator_type<parallel>,
                            #linalg.iterator_type<parallel>,
                            #linalg.iterator_type<reduction>],
          kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
          semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
        } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

        // K-step 1: MFMA on K[32:64].
        %d1s1 = iree_codegen.inner_tiled
            ins(%q_vec_1, %k_vec_1) outs(%d1s0) {
          indexing_maps = #contraction_accesses,
          iterator_types = [#linalg.iterator_type<parallel>,
                            #linalg.iterator_type<parallel>,
                            #linalg.iterator_type<reduction>],
          kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
          semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
        } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

        rocdl.s.setprio 0

        // === SYNC: wait for DMA completion ===
        amdgpu.memory_counter_wait load(0)
        rocdl.s.barrier
        rocdl.sched.barrier 0

        scf.yield %d1s1, %next_buf : vector<4x4x1x4xf32>, index
      }

      // ===================================================================
      // DOT1 EPILOGUE: compute last K-block from LDS.
      // ===================================================================
      %epi_k_off = arith.muli %dot1_loop#1, %c8 overflow<nsw, nuw> : index
      %epi_k = arith.addi %n_outer_id, %epi_k_off overflow<nsw, nuw> : index

      %q_epi_0 = vector.transfer_read
          %q_exp[%m_outer_id, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !q_exp_ty, vector<4x1x1x8xf16>
      %k_epi_0 = vector.transfer_read
          %k_exp[%epi_k, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !k_exp_ty, vector<4x1x1x8xf16>
      %d1e0 = iree_codegen.inner_tiled
          ins(%q_epi_0, %k_epi_0) outs(%dot1_loop#0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      %q_epi_1 = vector.transfer_read
          %q_exp[%m_outer_id, %ids#3, %c1, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !q_exp_ty, vector<4x1x1x8xf16>
      %k_epi_1 = vector.transfer_read
          %k_exp[%epi_k, %ids#3, %c1, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !k_exp_ty, vector<4x1x1x8xf16>
      %d1e1 = iree_codegen.inner_tiled
          ins(%q_epi_1, %k_epi_1) outs(%d1e0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      // ===================================================================
      // INTERMEDIATE: cast dot1 result f32 -> f16 and store to LDS.
      // Columns 0-63 go to s0_shared, columns 64-127 go to s1_shared.
      //
      // The dot1 accumulator vector<4x4x1x4xf32> is distributed as:
      //   dim0: 4 M-MFMA blocks, row = (m_outer_id + a) * 16 + ids#3.
      //   dim1: 4 N-MFMA blocks, col = (n_outer_id + b) * 16 + inner_id_n.
      //   dim2: 1 (always).
      //   dim3: 4 contiguous column values.
      //
      // For dot2, dot1's N-dimension (128) becomes the K-dimension.
      // We write the 128x128 result to two 128x64 LDS buffers so dot2 can
      // read them as standard MFMA-A inputs through the K-loop.
      // ===================================================================
      %dot1_f16 = arith.truncf %d1e1 : vector<4x4x1x4xf32> to vector<4x4x1x4xf16>

      // Transpose to [M-blocks, K=1, N-blocks, 4-values] for element-wise store.
      %dot1_tp = vector.transpose %dot1_f16, [0, 2, 1, 3]
          : vector<4x4x1x4xf16> to vector<4x1x4x4xf16>
      %dot1_empty = tensor.empty() : tensor<4x1x4x4xf16>
      %dot1_tensor = vector.transfer_write %dot1_tp, %dot1_empty[%c0, %c0, %c0, %c0]
          {in_bounds = [true, true, true, true]}
          : vector<4x1x4x4xf16>, tensor<4x1x4x4xf16>

      // Write dot1 result to LDS. Each thread writes its 4x4 MFMA tiles.
      // N-subgroup 0 (n_outer_id=0): columns 0-63 -> s0_shared.
      // N-subgroup 1 (n_outer_id=4): columns 64-127 -> s1_shared (offset by 64).
      scf.for %a = %c0 to %c4 step %c1 {
        %row = affine.apply affine_map<(d0, d1, d2) -> ((d0 + d1) * 16 + d2)>
            (%m_outer_id, %a, %ids#3)
        scf.for %b = %c0 to %c4 step %c1 {
          %col_abs = affine.apply affine_map<(d0, d1, d2) -> ((d0 + d1) * 16 + d2)>
              (%n_outer_id, %b, %inner_id_n)
          %tile_f16 = tensor.extract_slice %dot1_tensor[%a, 0, %b, 0] [1, 1, 1, 4] [1, 1, 1, 1]
              : tensor<4x1x4x4xf16> to tensor<1x4xf16>
          %vec_f16 = vector.transfer_read %tile_f16 [%c0, %c0], %cst
              {in_bounds = [true, true]} : tensor<1x4xf16>, vector<1x4xf16>
          // Determine which LDS buffer and local column offset.
          // If col_abs < 64, write to s0_shared at col_abs.
          // If col_abs >= 64, write to s1_shared at col_abs - 64.
          %is_upper = arith.cmpi sge, %col_abs, %c64 : index
          %col_local = arith.subi %col_abs, %c64 : index
          scf.if %is_upper {
            vector.transfer_write %vec_f16, %s1_shared[%row, %col_local]
                {in_bounds = [true, true]} : vector<1x4xf16>, !s_shared_ty
          } else {
            vector.transfer_write %vec_f16, %s0_shared[%row, %col_abs]
                {in_bounds = [true, true]} : vector<1x4xf16>, !s_shared_ty
          }
        } {iree_codegen.unroll}
      } {iree_codegen.unroll}

      // ===================================================================
      // DOT2 PROLOGUE: DMA first K-block of B2 (V columns 0-63) into LDS.
      // ===================================================================
      // DMA V[:, 0:64] into v_shared.
      scf.for %r = %c0 to %c4 step %c1 {
        %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
        %data_row = arith.addi %sg_row_base, %r_off overflow<nsw, nuw> : index
        %src_row = arith.addi %data_row, %lane_row overflow<nsw, nuw> : index
        amdgpu.gather_to_lds %b2_buf[%src_row, %lane_col_off],
            %v_shared[%data_row, %c0]
            : vector<8xf16>, !in_buf_ty, !v_shared_ty
      }

      amdgpu.memory_counter_wait load(0)
      rocdl.s.barrier

      // ===================================================================
      // DOT2 ITERATION 0: LHS from s0_shared, RHS from v_shared.
      //
      // Dot2's K dimension is 128 (the N dimension of dot1's output).
      // K-block size is 64, so exactly 2 iterations.
      // Iteration 0: LHS from s0_shared (dot1 columns 0-63), RHS from v_shared.
      // ===================================================================

      %zero_acc_d2 = arith.constant dense<0.0> : vector<4x4x1x4xf32>

      // Read dot2 LHS K-step 0 and K-step 1 from s0_shared.
      %d2_lhs_0_k0 = vector.transfer_read
          %s0_exp[%m_outer_id, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !s_exp_ty, vector<4x1x1x8xf16>
      %d2_rhs_0_k0 = vector.transfer_read
          %v_exp[%n_outer_id, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !v_exp_ty, vector<4x1x1x8xf16>

      %d2_lhs_0_k1 = vector.transfer_read
          %s0_exp[%m_outer_id, %ids#3, %c1, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !s_exp_ty, vector<4x1x1x8xf16>
      %d2_rhs_0_k1 = vector.transfer_read
          %v_exp[%n_outer_id, %ids#3, %c1, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !v_exp_ty, vector<4x1x1x8xf16>

      // DMA V[:, 64:128] (next K-block) into v_shared (non-blocking).
      scf.for %r = %c0 to %c4 step %c1 {
        %r_off = arith.muli %r, %c8 overflow<nsw, nuw> : index
        %data_row = arith.addi %sg_row_base, %r_off overflow<nsw, nuw> : index
        %src_row = arith.addi %data_row, %lane_row overflow<nsw, nuw> : index
        %src_col = arith.addi %c64, %lane_col_off overflow<nsw, nuw> : index
        amdgpu.gather_to_lds %b2_buf[%src_row, %src_col],
            %v_shared[%data_row, %c0]
            : vector<8xf16>, !in_buf_ty, !v_shared_ty
      }

      // Memory cluster: HIGH priority for DMA completion visibility.
      rocdl.sched.barrier 0
      rocdl.s.setprio 1

      // K-step 0: MFMA on first 32 K-elements.
      %d2i0s0 = iree_codegen.inner_tiled
          ins(%d2_lhs_0_k0, %d2_rhs_0_k0) outs(%zero_acc_d2) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      // K-step 1: MFMA on second 32 K-elements.
      %d2i0s1 = iree_codegen.inner_tiled
          ins(%d2_lhs_0_k1, %d2_rhs_0_k1) outs(%d2i0s0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      rocdl.s.setprio 0

      // Wait for V DMA to complete before reading v_shared for iteration 1.
      amdgpu.memory_counter_wait load(0)
      rocdl.s.barrier
      rocdl.sched.barrier 0

      // ===================================================================
      // DOT2 ITERATION 1: LHS from s1_shared, RHS from v_shared (updated).
      // ===================================================================

      // Read dot2 LHS K-step 0 and K-step 1 from s1_shared.
      %d2_lhs_1_k0 = vector.transfer_read
          %s1_exp[%m_outer_id, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !s_exp_ty, vector<4x1x1x8xf16>
      %d2_rhs_1_k0 = vector.transfer_read
          %v_exp[%n_outer_id, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !v_exp_ty, vector<4x1x1x8xf16>

      %d2_lhs_1_k1 = vector.transfer_read
          %s1_exp[%m_outer_id, %ids#3, %c1, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !s_exp_ty, vector<4x1x1x8xf16>
      %d2_rhs_1_k1 = vector.transfer_read
          %v_exp[%n_outer_id, %ids#3, %c1, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !v_exp_ty, vector<4x1x1x8xf16>

      // K-step 0: MFMA on first 32 K-elements of second K-block.
      %d2i1s0 = iree_codegen.inner_tiled
          ins(%d2_lhs_1_k0, %d2_rhs_1_k0) outs(%d2i0s1) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      // K-step 1: MFMA on second 32 K-elements of second K-block.
      %d2i1s1 = iree_codegen.inner_tiled
          ins(%d2_lhs_1_k1, %d2_rhs_1_k1) outs(%d2i1s0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      // ===================================================================
      // DOT2 RESULT WRITEBACK via pcf.write_slice.
      // ===================================================================
      %tp = vector.transpose %d2i1s1, [0, 2, 1, 3]
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
