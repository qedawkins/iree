// RUN: iree-opt %s
//
// Triton gfx950 async matmul ukernel — exact transformTwoClusterWithLocalLoadAndAll.
// Uses amdgpu.gather_to_lds for async global-to-LDS DMA copies.
//
// Tile: 256x128, K-block: 32, Warps: 8 (512 threads), Stages: 3.
// Triple-buffered LDS: LHS = 768x32 (3x256x32 = 48KB), RHS = 384x32 (3x128x32 = 24KB).
// Total LDS: 72KB.
// Subgroups: 4x2 (M=4, N=2), each handles 64x64 output.
// Per-subgroup: 4x4 = 16 MFMA-16x16 intrinsics, K-loop steps by 32.
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
// LHS (256 rows): 256 / 16 = 16 gathers / 8 sg = 2 per sg.
// RHS (128 rows): 128 / 16 = 8 gathers / 8 sg = 1 per sg.

!in_ty_lhs = tensor<256x?xf16>
!in_ty_rhs = tensor<128x?xf16>
!in_buf_lhs = memref<256x?xf16, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
!in_buf_rhs = memref<128x?xf16, strided<[?, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>

// Triple-buffered: buf 0 = rows [0,N), buf 1 = [N,2N), buf 2 = [2N,3N).
!shared_lhs_ty = memref<768x32xf16, #gpu.address_space<workgroup>>
!shared_rhs_ty = memref<384x32xf16, #gpu.address_space<workgroup>>
// Expanded: LHS 48 M-groups x 16 lanes x 1 K-step x 32 K-elems.
!shared_exp_lhs = memref<48x16x1x32xf16, #gpu.address_space<workgroup>>
// Expanded: RHS 24 N-groups x 16 lanes x 1 K-step x 32 K-elems.
!shared_exp_rhs = memref<24x16x1x32xf16, #gpu.address_space<workgroup>>

!out_sref = !pcf.sref<256x128xf32, sync(#iree_gpu.subgroup_scope)>

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]

util.func private @pingpong_2cluster_f16(
    %lhs_base: !in_ty_lhs, %rhs_base: !in_ty_rhs,
    %unused_acc: tensor<256x128xf32>) -> tensor<256x128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.0 : f16

  // Triple-buffered LDS for LHS and RHS.
  %lhs_shared = memref.alloc() : !shared_lhs_ty
  %rhs_shared = memref.alloc() : !shared_rhs_ty

  // K dimension and cache swizzle setup.
  %dim = tensor.dim %lhs_base, %c1 : !in_ty_lhs
  %dim_bytes = arith.muli %dim, %c2 overflow<nsw, nuw> : index
  %lhs = iree_gpu.buffer_resource_cast %lhs_base cacheSwizzleStride(%dim_bytes) : !in_ty_lhs
  %rhs = iree_gpu.buffer_resource_cast %rhs_base cacheSwizzleStride(%dim_bytes) : !in_ty_rhs

  // Bufferize inputs for DMA access.
  %lhs_buf = bufferization.to_buffer %lhs {read_only}
      : !in_ty_lhs to !in_buf_lhs
  %rhs_buf = bufferization.to_buffer %rhs {read_only}
      : !in_ty_rhs to !in_buf_rhs

  // Expanded views for MFMA LDS reads.
  %lhs_exp = memref.expand_shape %lhs_shared [[0, 1], [2, 3]]
      output_shape [48, 16, 1, 32] : !shared_lhs_ty into !shared_exp_lhs
  %rhs_exp = memref.expand_shape %rhs_shared [[0, 1], [2, 3]]
      output_shape [24, 16, 1, 32] : !shared_rhs_ty into !shared_exp_rhs

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
    // LHS: 8 subgroups each handle 32 rows (256/8).
    %sg_base_lhs = arith.muli %sg, %c32 overflow<nsw, nuw> : index
    // RHS: 8 subgroups each handle 16 rows (128/8).
    %sg_base_rhs = arith.muli %sg, %c16 overflow<nsw, nuw> : index

    scf.for %stage = %c0 to %c2 step %c1 {
      %k_off = arith.muli %stage, %c32 overflow<nsw, nuw> : index
      %lhs_buf_off = arith.muli %stage, %c256 overflow<nsw, nuw> : index
      %rhs_buf_off = arith.muli %stage, %c128 overflow<nsw, nuw> : index

      // LHS: 2 gathers per subgroup (32 rows / 16 per gather).
      scf.for %r = %c0 to %c2 step %c1 {
        %r_off = arith.muli %r, %c16 overflow<nsw, nuw> : index
        %row_base = arith.addi %sg_base_lhs, %r_off overflow<nsw, nuw> : index
        %src_row = arith.addi %row_base, %lane_row overflow<nsw, nuw> : index
        %src_col = arith.addi %k_off, %lane_col_off overflow<nsw, nuw> : index
        %dst_row = arith.addi %lhs_buf_off, %row_base overflow<nsw, nuw> : index
        amdgpu.gather_to_lds %lhs_buf[%src_row, %src_col],
            %lhs_shared[%dst_row, %c0]
            : vector<8xf16>, !in_buf_lhs, !shared_lhs_ty
      }
      // RHS: 1 gather per subgroup (16 rows / 16 per gather).
      %rhs_src_row = arith.addi %sg_base_rhs, %lane_row overflow<nsw, nuw> : index
      %rhs_src_col = arith.addi %k_off, %lane_col_off overflow<nsw, nuw> : index
      %rhs_dst_row = arith.addi %rhs_buf_off, %sg_base_rhs overflow<nsw, nuw> : index
      amdgpu.gather_to_lds %rhs_buf[%rhs_src_row, %rhs_src_col],
          %rhs_shared[%rhs_dst_row, %c0]
          : vector<8xf16>, !in_buf_rhs, !shared_rhs_ty

      amdgpu.memory_counter_wait load(0)
      rocdl.s.barrier
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}

  // =====================================================================
  // MAIN COMPUTE: pcf.generic with 3-stage gfx950 async scheduling.
  // =====================================================================
  %result = pcf.generic scope(#iree_gpu.subgroup_scope)
    execute(%out_ref = %unused_acc)[%sg_id: index, %num_sg: index]
         : (!out_sref) -> (tensor<256x128xf32>) {
    pcf.generic scope(#iree_gpu.lane_scope)
      execute[%lane_id: index, %sg_size: index] {

      // --- MFMA thread decomposition: 512 = 4x2x4x16 ---
      %id = affine.linearize_index disjoint [%sg_id, %lane_id]
          by (%num_sg, %sg_size) : index
      %ids:4 = affine.delinearize_index %id into (4, 2, 4, 16)
          : index, index, index, index
      %inner_id_k = arith.muli %ids#2, %c8 overflow<nsw, nuw> : index
      %inner_id_n = arith.muli %ids#2, %c4 overflow<nsw, nuw> : index
      %m_outer_id = arith.muli %ids#0, %c4 overflow<nsw, nuw> : index
      %n_outer_id = arith.muli %ids#1, %c4 overflow<nsw, nuw> : index

      // --- DMA thread decomposition (K=32: 4 lanes/row) ---
      %lane_row = arith.divui %lane_id, %c4 : index
      %lane_col_idx = arith.remui %lane_id, %c4 : index
      %lane_col_off = arith.muli %lane_col_idx, %c8 overflow<nsw, nuw> : index
      // LHS: 8 subgroups each handle 32 rows.
      %sg_row_base_lhs = arith.muli %sg_id, %c32 overflow<nsw, nuw> : index
      // RHS: 8 subgroups each handle 16 rows.
      %sg_row_base_rhs = arith.muli %sg_id, %c16 overflow<nsw, nuw> : index

      %zero_acc = arith.constant dense<0.0> : vector<4x4x1x4xf32>

      // =================================================================
      // K-LOOP: 3-stage pipeline, step by 32.
      // read_buf cycles 0->1->2->0->..., write_buf = (read_buf+2)%3.
      // =================================================================
      %loop:2 = scf.for %k = %c64 to %dim step %c32
          iter_args(%acc = %zero_acc, %read_buf = %c0)
          -> (vector<4x4x1x4xf32>, index) {

        // Buffer management.
        // LHS read offset in expanded view: read_buf * 16 groups.
        %read_off_m = arith.muli %read_buf, %c16 overflow<nsw, nuw> : index
        // RHS read offset in expanded view: read_buf * 8 groups.
        %read_off_n = arith.muli %read_buf, %c8 overflow<nsw, nuw> : index
        // Write buffer = (read_buf + 2) % 3.
        %wb_raw = arith.addi %read_buf, %c2 : index
        %wb_ge3 = arith.cmpi sge, %wb_raw, %c3 : index
        %wb_sub = arith.subi %wb_raw, %c3 : index
        %write_buf = arith.select %wb_ge3, %wb_sub, %wb_raw : index
        // LHS write offset: write_buf * 256 rows.
        %lhs_write_off = arith.muli %write_buf, %c256 overflow<nsw, nuw> : index
        // RHS write offset: write_buf * 128 rows.
        %rhs_write_off = arith.muli %write_buf, %c128 overflow<nsw, nuw> : index

        // === 1. LDS READS from current buffer ===
        %m_base = arith.addi %m_outer_id, %read_off_m overflow<nsw, nuw> : index
        %n_base = arith.addi %n_outer_id, %read_off_n overflow<nsw, nuw> : index

        %lhs_vec = vector.transfer_read
            %lhs_exp[%m_base, %ids#3, %c0, %inner_id_k], %cst
            {in_bounds = [true, true, true, true]}
            : !shared_exp_lhs, vector<4x1x1x8xf16>
        %rhs_vec = vector.transfer_read
            %rhs_exp[%n_base, %ids#3, %c0, %inner_id_k], %cst
            {in_bounds = [true, true, true, true]}
            : !shared_exp_rhs, vector<4x1x1x8xf16>

        // === 2. DMA LHS to write buffer ===
        rocdl.sched.barrier 0
        scf.for %r = %c0 to %c2 step %c1 {
          %r_off = arith.muli %r, %c16 overflow<nsw, nuw> : index
          %data_row = arith.addi %sg_row_base_lhs, %r_off overflow<nsw, nuw> : index
          %src_row = arith.addi %data_row, %lane_row overflow<nsw, nuw> : index
          %src_col = arith.addi %k, %lane_col_off overflow<nsw, nuw> : index
          %dst_row = arith.addi %lhs_write_off, %data_row overflow<nsw, nuw> : index
          amdgpu.gather_to_lds %lhs_buf[%src_row, %src_col],
              %lhs_shared[%dst_row, %c0]
              : vector<8xf16>, !in_buf_lhs, !shared_lhs_ty
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
        %rhs_src_row = arith.addi %sg_row_base_rhs, %lane_row overflow<nsw, nuw> : index
        %rhs_src_col = arith.addi %k, %lane_col_off overflow<nsw, nuw> : index
        %rhs_dst_row = arith.addi %rhs_write_off, %sg_row_base_rhs overflow<nsw, nuw> : index
        amdgpu.gather_to_lds %rhs_buf[%rhs_src_row, %rhs_src_col],
            %rhs_shared[%rhs_dst_row, %c0]
            : vector<8xf16>, !in_buf_rhs, !shared_rhs_ty

        // === 6. MFMA: single dot (RHS DMA overlaps with compute) ===
        %dot = iree_codegen.inner_tiled
            ins(%lhs_vec, %rhs_vec) outs(%acc) {
          indexing_maps = #contraction_accesses,
          iterator_types = [#linalg.iterator_type<parallel>,
                            #linalg.iterator_type<parallel>,
                            #linalg.iterator_type<reduction>],
          kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
          semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
        } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

        // === 7. BARRIER ===
        rocdl.sched.barrier 0
        rocdl.s.barrier
        rocdl.sched.barrier 0

        // Advance read buffer: (read_buf + 1) % 3.
        %nb_raw = arith.addi %read_buf, %c1 : index
        %nb_eq3 = arith.cmpi eq, %nb_raw, %c3 : index
        %next_buf = arith.select %nb_eq3, %c0, %nb_raw : index

        scf.yield %dot, %next_buf : vector<4x4x1x4xf32>, index
      }

      // =================================================================
      // EPILOGUE: process last 2 K-blocks remaining in pipeline.
      // =================================================================
      amdgpu.memory_counter_wait load(0)

      // Epilogue K-block 1: in read_buf.
      %epi_off_m_0 = arith.muli %loop#1, %c16 overflow<nsw, nuw> : index
      %epi_off_n_0 = arith.muli %loop#1, %c8 overflow<nsw, nuw> : index
      %epi_m_0 = arith.addi %m_outer_id, %epi_off_m_0 overflow<nsw, nuw> : index
      %epi_n_0 = arith.addi %n_outer_id, %epi_off_n_0 overflow<nsw, nuw> : index

      %lhs_epi_0 = vector.transfer_read
          %lhs_exp[%epi_m_0, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_lhs, vector<4x1x1x8xf16>
      %rhs_epi_0 = vector.transfer_read
          %rhs_exp[%epi_n_0, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_rhs, vector<4x1x1x8xf16>
      %epi_dot_0 = iree_codegen.inner_tiled
          ins(%lhs_epi_0, %rhs_epi_0) outs(%loop#0) {
        indexing_maps = #contraction_accesses,
        iterator_types = [#linalg.iterator_type<parallel>,
                          #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16, col_major = true>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : vector<4x1x1x8xf16>, vector<4x1x1x8xf16> into vector<4x4x1x4xf32>

      // Epilogue K-block 2: in (read_buf + 1) % 3.
      %epi_nb_raw = arith.addi %loop#1, %c1 : index
      %epi_nb_eq3 = arith.cmpi eq, %epi_nb_raw, %c3 : index
      %epi_nb = arith.select %epi_nb_eq3, %c0, %epi_nb_raw : index
      %epi_off_m_1 = arith.muli %epi_nb, %c16 overflow<nsw, nuw> : index
      %epi_off_n_1 = arith.muli %epi_nb, %c8 overflow<nsw, nuw> : index
      %epi_m_1 = arith.addi %m_outer_id, %epi_off_m_1 overflow<nsw, nuw> : index
      %epi_n_1 = arith.addi %n_outer_id, %epi_off_n_1 overflow<nsw, nuw> : index

      %lhs_epi_1 = vector.transfer_read
          %lhs_exp[%epi_m_1, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_lhs, vector<4x1x1x8xf16>
      %rhs_epi_1 = vector.transfer_read
          %rhs_exp[%epi_n_1, %ids#3, %c0, %inner_id_k], %cst
          {in_bounds = [true, true, true, true]}
          : !shared_exp_rhs, vector<4x1x1x8xf16>
      %epi_dot_1 = iree_codegen.inner_tiled
          ins(%lhs_epi_1, %rhs_epi_1) outs(%epi_dot_0) {
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
      %tp = vector.transpose %epi_dot_1, [0, 2, 1, 3]
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
  util.return %result : tensor<256x128xf32>
}
