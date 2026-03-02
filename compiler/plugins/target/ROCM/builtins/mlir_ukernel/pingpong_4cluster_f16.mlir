// RUN: iree-opt %s
//
// Triton gfx950 async matmul ukernel — exact transformTwoClusterWithLocalLoadAndAll.
// Uses amdgpu.gather_to_lds for async global-to-LDS DMA copies.
//
// Tile: 256x256, K-block: 32, Warps: 8 (512 threads), Stages: 3.
// Triple-buffered LDS (3x256x32 per operand = 48KB each, 96KB total).
// Subgroups: 2x4 (M=2, N=4), each handles 128x64 output.
// Per-subgroup: 8x4 = 32 MFMA-16x16 intrinsics, K-loop steps by 32.
// Single MFMA per iteration (K_block = MFMA_K = 32, no K-slicing).
//
// Schedule (per loop iteration, exact Triton gfx950 pattern):
//   1. LDS reads from current buffer (safe: 3-stage headroom).
//   2. sched_barrier 0.
//   3. DMA LHS gathers to write buffer.
//   4. sched_barrier 0.
//   5. memory_counter_wait load(0) — catches previous iteration's DMAs.
//   6. sched_barrier 0.
//   7. sched_group_barrier hints (MFMA/SALU interleaving).
//   8. DMA RHS gathers to write buffer.
//   9. MFMA (single dot, RHS DMA overlaps with compute).
//  10. sched_barrier 0 + s_barrier + sched_barrier 0.
//
// MFMA: MFMA_F32_16x16x32_F16 (K=32 per intrinsic, 1 per K-block).
//
// DMA scheme: 512 threads = 8 subgroups x 64 lanes.
// K=32: 4 lanes per row, vector<8xf16>, 16 rows per gather.
// Per operand (256 rows): 256 / 16 = 16 gathers / 8 sg = 2 per sg.

!in_ty = tensor<256x?xf16>
!in_buf_ty = memref<256x?xf16, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
// Triple-buffered: buf 0 = rows [0,256), buf 1 = [256,512), buf 2 = [512,768).
!shared_ty = memref<768x32xf16, #gpu.address_space<workgroup>>
// Expanded: 48 M-groups x 16 lanes x 1 K-step x 32 K-elems.
!shared_exp_ty = memref<48x16x1x32xf16, #gpu.address_space<workgroup>>

!out_sref = !pcf.sref<256x256xf32, sync(#iree_gpu.subgroup_scope)>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

util.func private @pingpong_4cluster_f16(
    %lhs_base: !in_ty, %rhs_base: !in_ty,
    %unused_acc: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  // Triple-buffered LDS for LHS and RHS.
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

  // Expanded views for MFMA LDS reads.
  %lhs_exp = memref.expand_shape %lhs_shared [[0, 1], [2, 3]]
      output_shape [48, 16, 1, 32] : !shared_ty into !shared_exp_ty
  %rhs_exp = memref.expand_shape %rhs_shared [[0, 1], [2, 3]]
      output_shape [48, 16, 1, 32] : !shared_ty into !shared_exp_ty

  // =====================================================================
  // PROLOGUE: Fill first 2 buffers (k=0 into buf 0, k=32 into buf 1).
  // =====================================================================
  scf.forall (%tid) in (512) {
    %sg = arith.divui %tid, %c64 : index
    %lane = arith.remui %tid, %c64 : index
    // K=32: 4 lanes per row, vector<8xf16>, 16 rows per gather.
    %lane_row = arith.divui %lane, %c4 : index
    %lane_col_idx = arith.remui %lane, %c4 : index
    %lane_col_off = arith.muli %lane_col_idx, %c8 overflow<nsw, nuw> : index
    %sg_base = arith.muli %sg, %c32 overflow<nsw, nuw> : index

    scf.for %stage = %c0 to %c2 step %c1 {
      %k_off = arith.muli %stage, %c32 overflow<nsw, nuw> : index
      %buf_off = arith.muli %stage, %c256 overflow<nsw, nuw> : index

      // LHS: 2 gathers per subgroup (32 rows / 16 per gather).
      scf.for %r = %c0 to %c2 step %c1 {
        %r_off = arith.muli %r, %c16 overflow<nsw, nuw> : index
        %row_base = arith.addi %sg_base, %r_off overflow<nsw, nuw> : index
        %src_row = arith.addi %row_base, %lane_row overflow<nsw, nuw> : index
        %src_col = arith.addi %k_off, %lane_col_off overflow<nsw, nuw> : index
        %dst_row = arith.addi %buf_off, %row_base overflow<nsw, nuw> : index
        amdgpu.gather_to_lds %lhs_buf[%src_row, %src_col],
            %lhs_shared[%dst_row, %c0]
            : vector<8xf16>, !in_buf_ty, !shared_ty
      }
      // RHS: 2 gathers per subgroup.
      scf.for %r = %c0 to %c2 step %c1 {
        %r_off = arith.muli %r, %c16 overflow<nsw, nuw> : index
        %row_base = arith.addi %sg_base, %r_off overflow<nsw, nuw> : index
        %src_row = arith.addi %row_base, %lane_row overflow<nsw, nuw> : index
        %src_col = arith.addi %k_off, %lane_col_off overflow<nsw, nuw> : index
        %dst_row = arith.addi %buf_off, %row_base overflow<nsw, nuw> : index
        amdgpu.gather_to_lds %rhs_buf[%src_row, %src_col],
            %rhs_shared[%dst_row, %c0]
            : vector<8xf16>, !in_buf_ty, !shared_ty
      }

      amdgpu.memory_counter_wait load(0)
      rocdl.s.barrier
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}

  // =====================================================================
  // MAIN COMPUTE: pcf.generic with 3-stage gfx950 async scheduling.
  // =====================================================================
  %result = pcf.generic scope(#iree_gpu.subgroup_scope)
    execute(%out_ref = %unused_acc)[%sg_id: index, %num_sg: index]
         : (!out_sref) -> (tensor<256x256xf32>) {
    pcf.generic scope(#iree_gpu.lane_scope)
      execute[%lane_id: index, %sg_size: index] {

      // --- MFMA thread decomposition: 512 = 2x4x4x16 ---
      %id = affine.linearize_index disjoint [%sg_id, %lane_id]
          by (%num_sg, %sg_size) : index
      %ids:4 = affine.delinearize_index %id into (2, 4, 4, 16)
          : index, index, index, index
      %inner_id_k = arith.muli %ids#2, %c8 overflow<nsw, nuw> : index
      %inner_id_n = arith.muli %ids#2, %c4 overflow<nsw, nuw> : index
      // Each M-subgroup handles 8 MFMA-16 blocks (128/16 = 8).
      %m_outer_id = arith.muli %ids#0, %c8 overflow<nsw, nuw> : index
      // Each N-subgroup handles 4 MFMA-16 blocks (64/16 = 4).
      %n_outer_id = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index

      // --- DMA thread decomposition (K=32: 4 lanes/row) ---
      %lane_row = arith.divui %lane_id, %c4 : index
      %lane_col_idx = arith.remui %lane_id, %c4 : index
      %lane_col_off = arith.muli %lane_col_idx, %c8 overflow<nsw, nuw> : index
      %sg_row_base = arith.muli %sg_id, %c32 overflow<nsw, nuw> : index

      %zero_acc = arith.constant dense<0.0> : vector<8x4x1x4xf32>

      // =================================================================
      // K-LOOP: 3-stage pipeline, step by 32.
      // read_buf cycles 0->1->2->0->..., write_buf = (read_buf+2)%3.
      // =================================================================
      %loop:2 = scf.for %k = %c64 to %dim step %c32
          iter_args(%acc = %zero_acc, %read_buf = %c0)
          -> (vector<8x4x1x4xf32>, index) {

        // Buffer management.
        %read_off = arith.muli %read_buf, %c16 overflow<nsw, nuw> : index
        %wb_raw = arith.addi %read_buf, %c2 : index
        %wb_ge3 = arith.cmpi sge, %wb_raw, %c3 : index
        %wb_sub = arith.subi %wb_raw, %c3 : index
        %write_buf = arith.select %wb_ge3, %wb_sub, %wb_raw : index
        %write_off = arith.muli %write_buf, %c256 overflow<nsw, nuw> : index

        // === 1. LDS READS from current buffer ===
        %m_base = arith.addi %m_outer_id, %read_off overflow<nsw, nuw> : index
        %n_base = arith.addi %n_outer_id, %read_off overflow<nsw, nuw> : index

        %lhs_vec = vector.transfer_read
            %lhs_exp[%m_base, %ids#3, %c0, %inner_id_k], %cst
            {in_bounds = [true, true, true, true]}
            : !shared_exp_ty, vector<8x1x1x8xf16>
        %rhs_vec = vector.transfer_read
            %rhs_exp[%n_base, %ids#3, %c0, %inner_id_k], %cst
            {in_bounds = [true, true, true, true]}
            : !shared_exp_ty, vector<4x1x1x8xf16>

        // === 2. DMA LHS to write buffer ===
        rocdl.sched.barrier 0
        scf.for %r = %c0 to %c2 step %c1 {
          %r_off = arith.muli %r, %c16 overflow<nsw, nuw> : index
          %data_row = arith.addi %sg_row_base, %r_off overflow<nsw, nuw> : index
          %src_row = arith.addi %data_row, %lane_row overflow<nsw, nuw> : index
          %src_col = arith.addi %k, %lane_col_off overflow<nsw, nuw> : index
          %dst_row = arith.addi %write_off, %data_row overflow<nsw, nuw> : index
          amdgpu.gather_to_lds %lhs_buf[%src_row, %src_col],
              %lhs_shared[%dst_row, %c0]
              : vector<8xf16>, !in_buf_ty, !shared_ty
        }

        // === 3. WAIT for previous iteration's DMAs ===
        rocdl.sched.barrier 0
        amdgpu.memory_counter_wait load(0)
        rocdl.sched.barrier 0

        // === 4. SCHEDULING HINTS ===
        rocdl.sched.group.barrier 8, 1, 0
        rocdl.sched.group.barrier 4, 3, 0
        rocdl.sched.group.barrier 8, 1, 0
        rocdl.sched.group.barrier 4, 3, 0
        rocdl.sched.group.barrier 8, 1, 0

        // === 5. DMA RHS to write buffer ===
        scf.for %r = %c0 to %c2 step %c1 {
          %r_off = arith.muli %r, %c16 overflow<nsw, nuw> : index
          %data_row = arith.addi %sg_row_base, %r_off overflow<nsw, nuw> : index
          %src_row = arith.addi %data_row, %lane_row overflow<nsw, nuw> : index
          %src_col = arith.addi %k, %lane_col_off overflow<nsw, nuw> : index
          %dst_row = arith.addi %write_off, %data_row overflow<nsw, nuw> : index
          amdgpu.gather_to_lds %rhs_buf[%src_row, %src_col],
              %rhs_shared[%dst_row, %c0]
              : vector<8xf16>, !in_buf_ty, !shared_ty
        }

        // === 6. MFMA: single dot (RHS DMA overlaps with compute) ===
        %dot = iree_codegen.inner_tiled
            ins(%lhs_vec, %rhs_vec) outs(%acc) {
          indexing_maps = #contraction_accesses,
          iterator_types = [#linalg.iterator_type<parallel>,
                            #linalg.iterator_type<parallel>,
                            #linalg.iterator_type<reduction>],
          kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
          semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
        } : vector<8x1x1x8xf16>, vector<4x1x1x8xf16> into vector<8x4x1x4xf32>

        // === 7. BARRIER ===
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0

        // Advance read buffer: (read_buf + 1) % 3.
        %nb_raw = arith.addi %read_buf, %c1 : index
        %nb_eq3 = arith.cmpi eq, %nb_raw, %c3 : index
        %next_buf = arith.select %nb_eq3, %c0, %nb_raw : index

        scf.yield %dot, %next_buf : vector<8x4x1x4xf32>, index
      }

      // =================================================================
      // EPILOGUE: process last 2 K-blocks remaining in pipeline.
      // =================================================================
      amdgpu.memory_counter_wait load(0)

      // Epilogue K-block 1: in read_buf.
      %epi_off_0 = arith.muli %loop#1, %c16 overflow<nsw, nuw> : index
      %epi_m_0 = arith.addi %m_outer_id, %epi_off_0 overflow<nsw, nuw> : index
      %epi_n_0 = arith.addi %n_outer_id, %epi_off_0 overflow<nsw, nuw> : index

      %lhs_epi_0 = vector.transfer_read
          %lhs_exp[%epi_m_0, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<8x1x1x8xf16>
      %rhs_epi_0 = vector.transfer_read
          %rhs_exp[%epi_n_0, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<4x1x1x8xf16>
      %epi_dot_0 = iree_codegen.inner_tiled
          ins(%lhs_epi_0, %rhs_epi_0) outs(%loop#0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x8xf16>, vector<4x1x1x8xf16> into vector<8x4x1x4xf32>

      // Epilogue K-block 2: in (read_buf + 1) % 3.
      %epi_nb_raw = arith.addi %loop#1, %c1 : index
      %epi_nb_eq3 = arith.cmpi eq, %epi_nb_raw, %c3 : index
      %epi_nb = arith.select %epi_nb_eq3, %c0, %epi_nb_raw : index
      %epi_off_1 = arith.muli %epi_nb, %c16 overflow<nsw, nuw> : index
      %epi_m_1 = arith.addi %m_outer_id, %epi_off_1 overflow<nsw, nuw> : index
      %epi_n_1 = arith.addi %n_outer_id, %epi_off_1 overflow<nsw, nuw> : index

      %lhs_epi_1 = vector.transfer_read
          %lhs_exp[%epi_m_1, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<8x1x1x8xf16>
      %rhs_epi_1 = vector.transfer_read
          %rhs_exp[%epi_n_1, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_ty, vector<4x1x1x8xf16>
      %epi_dot_1 = iree_codegen.inner_tiled
          ins(%lhs_epi_1, %rhs_epi_1) outs(%epi_dot_0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<8x1x1x8xf16>, vector<4x1x1x8xf16> into vector<8x4x1x4xf32>

      // =================================================================
      // RESULT WRITEBACK via pcf.write_slice.
      // =================================================================
      %tp = vector.transpose %epi_dot_1, [0, 2, 1, 3]
          : vector<8x4x1x4xf32> to vector<8x1x4x4xf32>
      %empty = tensor.empty() : tensor<8x1x4x4xf32>
      %result_tensor = vector.transfer_write %tp, %empty[%c0, %c0, %c0, %c0]
          {in_bounds = [true, true, true, true]}
          : vector<8x1x4x4xf32>, tensor<8x1x4x4xf32>
      scf.for %a = %c0 to %c8 step %c1 {
        scf.for %b = %c0 to %c4 step %c1 {
          %row = affine.apply affine_map<(d0, d1, d2) -> ((d0 + d1) * 16 + d2)>
              (%m_outer_id, %a, %ids#3)
          %col = affine.apply affine_map<(d0, d1, d2) -> ((d0 + d1) * 16 + d2)>
              (%n_outer_id, %b, %inner_id_n)
          %tile = tensor.extract_slice %result_tensor[%a, 0, %b, 0] [1, 1, 1, 4] [1, 1, 1, 1]
              : tensor<8x1x4x4xf32> to tensor<1x4xf32>
          pcf.write_slice %tile into %out_ref[%row, %col] [1, 4] [1, 1]
              : tensor<1x4xf32> into !out_sref
        } {iree_codegen.unroll}
      } {iree_codegen.unroll}
      pcf.return
    }
    pcf.return
  }
  util.return %result : tensor<256x256xf32>
}
