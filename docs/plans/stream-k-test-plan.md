# Stream-K Testing Plan

**Date**: 2026-02-12
**Owner**: Fuzzer
**Status**: Active
**Hub Issue**: iree-stream-k-3to
**Testing Issue**: iree-stream-k-f5k

## Overview

This document defines the complete test matrix for the Stream-K distribution
strategy. It covers both the **codegen worktree** (`iree-stream-k/iree/`,
branch `stream-k`) and the **host worktree** (`iree-stream-k/iree-host/`,
branch `stream-k-host`).

---

## 1. Codegen Worktree Tests

### 1.1 `pcf.stream_k_recombine` Op — Parser/Printer Roundtrip

**File**: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/stream_k_ops.mlir`

| Test Case | Description | Key Checks |
|-----------|-------------|------------|
| `basic_recombine_f32` | Basic f32 matmul-like recombine with addf combiner | All operands parsed, combiner region present, writeback region present |
| `basic_recombine_i32` | Integer partial tile with addi combiner | Element type propagation to combiner args |
| `dynamic_offsets_sizes` | All offsets/sizes are dynamic (%vars) | Dynamic SSA values survive roundtrip |
| `static_offsets_sizes` | All offsets/sizes are constants | Static values printed correctly |
| `mixed_offsets_sizes` | Mix of static and dynamic | Mixed representation correct |
| `writeback_with_epilogue` | Writeback region contains bias add + relu | Multi-op writeback region survives |
| `writeback_empty_body` | Minimal writeback (just pcf.write_slice + pcf.yield) | Minimal valid writeback |
| `f16_element_type` | f16 partial tile and combiner | Half precision types handled |
| `bf16_element_type` | bf16 partial tile and combiner | BFloat16 types handled |
| `3d_tile_shape` | tensor<4x8x16xf32> partial tile | Non-2D tiles supported |
| `1d_tile_shape` | tensor<128xf32> partial tile (reduction-only) | 1D tiles for pure reduction |
| `large_static_shape` | tensor<256x256xf32> typical workload size | Large tiles parse correctly |

### 1.2 `pcf.stream_k_recombine` Op — Verifier Error Cases

**File**: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/stream_k_invalid.mlir`

| Test Case | Expected Error | Description |
|-----------|---------------|-------------|
| `combiner_wrong_num_args` | Combiner must take exactly 2 arguments | Combiner with 1 or 3 args |
| `combiner_wrong_arg_type` | Combiner arg type must match tile element type | f32 tile but i32 combiner args |
| `combiner_wrong_result_type` | Combiner must yield single result matching element type | Yields wrong type |
| `combiner_missing_yield` | Combiner region must terminate with pcf.yield | Missing terminator |
| `writeback_wrong_num_args` | Writeback must take exactly 1 tensor argument | Wrong arg count |
| `writeback_wrong_arg_type` | Writeback arg must match partial_tile type | Type mismatch |
| `writeback_missing_yield` | Writeback must terminate with pcf.yield | Missing terminator |
| `output_sref_rank_mismatch` | Output sref rank must match partial tile rank | 2D tile into 1D sref |
| `output_sref_eltype_mismatch` | Output sref element type must match tile | f32 tile into i32 sref |
| `scratch_sref_wrong_eltype` | Scratch sref element type must match tile | f32 tile, i32 scratch |
| `counter_sref_wrong_type` | Counter sref must be scalar i32 or i64 | Counter with wrong shape/type |
| `num_in_group_not_index` | num_in_group must be index type | Wrong type |
| `offsets_count_mismatch` | Number of offsets must match tile rank | 3 offsets for 2D tile |
| `sizes_count_mismatch` | Number of sizes must match tile rank | 1 size for 2D tile |
| `strides_count_mismatch` | Number of strides must match tile rank | Wrong stride count |

### 1.3 Stream-K Tiling Transformation

**File**: `compiler/src/iree/compiler/Codegen/LLVMGPU/test/stream_k_tiling.mlir`

| Test Case | Description | Key FileCheck Patterns |
|-----------|-------------|----------------------|
| `matmul_128x128x256` | Standard matmul, tiles 64x64x64 | pcf.loop with correct count, divmod decode, recombine placement |
| `matmul_256x64x1024` | Tall-skinny output, large K | Correct total_work = 4*1*16=64, items_per_wg distribution |
| `matmul_64x256x1024` | Wide output, large K | Same as above with swapped dims |
| `batch_matmul` | 3D problem: B x M x N x K | 3 parallel dims + 1 reduction, correct linearization |
| `single_reduction_dim` | Only 1 reduction dim (standard matmul) | k_tiles_per_out computed correctly |
| `multiple_reduction_dims` | 2+ reduction dims | Multiple reduction dims linearized innermost |
| `tile_size_not_dividing` | Dims not divisible by tile sizes | ceildiv arithmetic present |
| `num_workgroups_1` | Single workgroup (degenerate case) | items_per_wg == total_work, all num_in_group == 1 |
| `num_workgroups_exceeds_work` | More WGs than work items | Bounded correctly, no OOB |
| `dynamic_shapes` | Dynamic M, N, K | Dynamic ceildiv, dynamic divmod chain |
| `group_size_arithmetic` | Verify first_wg/last_wg/num_in_group | Exact arithmetic sequence checked |
| `decode_divmod_chain` | Verify out_idx decomposition | arith.divui/remui chain for parallel dims |
| `reduction_innermost` | K tiles are innermost in linearization | out_idx = linear / k_per_out, k_idx = linear % k_per_out |
| `input_slice_offsets` | TilingInterface input slicing | Correct offset computation from decoded coords |
| `output_sref_slicing` | Output sref slice per output tile | Correct offset/size from parallel coords |
| `scratch_and_counter_slicing` | Scratch/counter sref per output tile | Indexed by out_idx |

### 1.4 Recombine Lowering

**File**: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/lower_stream_k_recombine.mlir`
(or `compiler/src/iree/compiler/Codegen/LLVMGPU/test/lower_stream_k_recombine.mlir`)

| Test Case | Description | Key FileCheck Patterns |
|-----------|-------------|----------------------|
| `atomic_counter_increment` | Counter RMW with correct type | memref.atomic_rmw addi on counter_ref |
| `sole_contributor_bypass` | num_in_group == 1 fast path | arith.cmpi eq + scf.if with direct writeback |
| `staging_write` | Non-last contributor stores to scratch | Store to scratch_ref within scf.if |
| `last_contributor_accumulate` | Last contributor reads + combines all slots | Loop over scratch slots, inlined combiner |
| `combiner_inlined` | Combiner region inlined as element-wise ops | arith.addf (for add combiner) in accumulation loop |
| `writeback_inlined` | Writeback region inlined in final branch | pcf.write_slice (or lowered form) in writeback branch |
| `memory_fence_placement` | Fence between writes and reads | pcf.fence or gpu fence between scratch write and read |
| `f16_accumulation` | Half-precision accumulation path | Correct f16 ops in combiner |
| `epilogue_in_writeback` | Bias + ReLU in writeback region | Epilogue ops appear only in writeback branch |

### 1.5 `iree_codegen.alloc_scratch` Op

**File**: `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/test/alloc_scratch_ops.mlir`

| Test Case | Description |
|-----------|-------------|
| `basic_alloc_scratch` | Parser/printer roundtrip for basic alloc_scratch |
| `alloc_scratch_with_alignment` | Alignment attribute present |
| `alloc_scratch_dynamic_size` | Dynamic byte count operand |

### 1.6 Scratch Aggregation Pass

**File**: `compiler/src/iree/compiler/Codegen/LLVMGPU/test/aggregate_scratch_allocations.mlir`

| Test Case | Description | Key FileCheck Patterns |
|-----------|-------------|----------------------|
| `single_alloc_scratch` | One alloc_scratch in dispatch | Size copied to scratch_size region, replaced with subview at offset 0 |
| `multiple_alloc_scratch` | Two alloc_scratch ops | Sizes summed, second gets offset == first's size |
| `alignment_padding` | Alloc with alignment requirements | Padding between allocations |
| `no_alloc_scratch` | Dispatch without any scratch | Pass is a no-op, no scratch_size region created |

---

## 2. Host Worktree Tests

### 2.1 Flow Export scratch_size Region

**File**: `compiler/src/iree/compiler/Dialect/Flow/IR/test/executable_ops.mlir` (additions)

| Test Case | Description | Key Checks |
|-----------|-------------|------------|
| `export_with_scratch_size` | Export with scratch_size region | Keyword parsed, args/return correct |
| `export_with_both_regions` | workgroups + scratch_size on same export | Both regions coexist |
| `export_scratch_size_only` | scratch_size without workgroups | Valid standalone |
| `scratch_size_returns_index` | Region returns single index | Return type validated |
| `scratch_size_bad_return_count` | Returns 0 or 2+ values | expected-error |
| `scratch_size_bad_return_type` | Returns non-index type | expected-error |
| `scratch_size_workload_args` | Accepts workload arguments | Args parsed and usable |

### 2.2 Stream Export scratch_size Region

**File**: `compiler/src/iree/compiler/Dialect/Stream/IR/test/executable_ops.mlir` (additions)

| Test Case | Description |
|-----------|-------------|
| `stream_export_with_scratch_size` | Parser/printer roundtrip |
| `stream_export_both_regions` | workgroups + scratch_size coexist |
| `stream_scratch_size_bad_return` | Verifier: wrong return type/count |

### 2.3 HAL Export scratch_size Region

**File**: `compiler/src/iree/compiler/Dialect/HAL/IR/test/executable_ops.mlir` (additions)

| Test Case | Description |
|-----------|-------------|
| `hal_export_with_scratch_size` | scratch_size(%device, %workload...) -> index |
| `hal_export_both_count_and_scratch` | count + scratch_size on same export |
| `hal_export_all_three_regions` | count + condition + scratch_size |
| `hal_scratch_size_bad_return` | Verifier: wrong return type/count |

### 2.4 scratch_size Calling Ops

**File**: New test files per dialect

| Test Case | Dialect | Description |
|-----------|---------|-------------|
| `flow_scratch_size_call` | Flow | `flow.executable.scratch_size @ex::@entry [%wl...]` roundtrip |
| `stream_scratch_size_call` | Stream | `stream.executable.scratch_size @ex::@entry [%wl...]` roundtrip |
| `hal_scratch_size_call` | HAL | `hal.executable.scratch_size @ex::@entry [%wl...]` roundtrip |
| `call_without_region` | All | Error: calling scratch_size on export without region |

### 2.5 Dispatch Formation Pass

**File**: `compiler/src/iree/compiler/Dialect/Flow/Transforms/test/dispatch_formation_scratch.mlir`

| Test Case | Description | Key FileCheck Patterns |
|-----------|-------------|----------------------|
| `matmul_gets_scratch` | linalg.matmul dispatch gets scratch buffer added | flow.executable.scratch_size call, tensor.empty(%size), extra dispatch operand |
| `non_matmul_no_scratch` | linalg.generic without reduction: no scratch | Pass is no-op |
| `batch_matmul_gets_scratch` | Batch matmul also gets scratch | Same pattern as matmul |
| `multiple_dispatches` | Two matmul dispatches | Each gets independent scratch |

### 2.6 Flow -> Stream -> HAL Lowering

**File**: Additions to existing lowering test files

| Test Case | Description |
|-----------|-------------|
| `scratch_size_survives_flow_to_stream` | scratch_size region present after Flow->Stream |
| `scratch_size_survives_stream_to_hal` | scratch_size region present after Stream->HAL |
| `scratch_size_call_lowers_flow_to_stream` | flow.executable.scratch_size -> stream.executable.scratch_size |
| `scratch_size_call_lowers_stream_to_hal` | stream.executable.scratch_size -> hal.executable.scratch_size |

---

## 3. Edge Cases & Stress Tests

### 3.1 Numerical Edge Cases

| Case | Description | Risk |
|------|-------------|------|
| `num_in_group == 1` for all tiles | Small problem, many WGs | Entire fast path, scratch never touched |
| `num_in_group == num_workgroups` | Single output tile, all WGs contribute | Maximum contention on one counter |
| `total_work == 1` | Degenerate: 1 tile, 1 k-step | Only 1 WG does anything |
| `total_work == num_workgroups` | Exactly 1 item per WG | items_per_wg == 1, boundary conditions |
| `total_work < num_workgroups` | More WGs than work | Some WGs do nothing |
| `tile_size == dim_size` | No tiling needed for that dim | tile_count == 1 |
| `tile_size == 1` | Maximum tiling granularity | Maximum work items |
| `dim_size == 0` | Empty tensor | Should be rejected or handled gracefully |

### 3.2 Numerical Correctness

| Case | Baseline | Tolerance |
|------|----------|-----------|
| `i32 matmul` | Data-parallel | Bitwise identical |
| `f32 matmul` | Data-parallel | ULP tolerance (accumulation order differs) |
| `f16 matmul` | Data-parallel | Larger tolerance (reduced precision) |
| `bf16 matmul` | Data-parallel | Larger tolerance |
| `large K (4096)` | Data-parallel | Accumulation order most impactful here |

### 3.3 Fusion Edge Cases

| Case | Description |
|------|-------------|
| `bias_add_fusion` | Bias add fuses into writeback region |
| `relu_fusion` | ReLU fuses into writeback |
| `bias_plus_relu` | Both fuse together |
| `no_epilogue` | Pure matmul, empty writeback (just store) |
| `multi_consumer` | Multiple consumers of matmul output |

---

## 4. Benchmarks

### 4.1 Target Hardware

- **GPU**: RDNA4 (on this system)
- **Driver**: ROCM/HIP

### 4.2 Benchmark Matrix

| Problem Size | M | N | K | Category | Expected Benefit |
|-------------|---|---|---|----------|-----------------|
| Tall-skinny | 32 | 32 | 4096 | High K, small output | Strong: few output tiles, many WGs idle in DP |
| Tall-skinny | 64 | 64 | 8192 | High K, small output | Strong |
| Wide | 4096 | 4096 | 32 | Low K, large output | Weak: DP already saturates |
| Square | 1024 | 1024 | 1024 | Balanced | Moderate: depends on CU count |
| Non-power-of-2 | 127 | 311 | 4096 | Irregular | Strong: poor DP load balance |
| Batch | 8x128x128x1024 | Batched high K | Depends on batch parallelism |
| Tiny | 16 | 16 | 16 | Minimum viable | Baseline: overhead check |

### 4.3 Comparison Baselines

1. **Data-parallel**: Standard IREE matmul (current default)
2. **Split-K**: If available, split-K with fixed partition count
3. **Stream-K**: Our implementation

### 4.4 Metrics

- Wall-clock time (median of 100 runs, after 10 warmup)
- GPU occupancy (via rocprof if available)
- Scratch memory consumption

---

## 5. Test File Locations Summary

### Codegen Worktree (`iree-stream-k/iree/`)

| File | Purpose | Beads Issue |
|------|---------|-------------|
| `compiler/.../PCF/IR/test/stream_k_ops.mlir` | Recombine op roundtrip | iree-stream-k-297 |
| `compiler/.../PCF/IR/test/stream_k_invalid.mlir` | Recombine op verifier | iree-stream-k-297 |
| `compiler/.../LLVMGPU/test/stream_k_tiling.mlir` | Tiling transformation | iree-stream-k-ygd |
| `compiler/.../PCF/Transforms/test/lower_stream_k_recombine.mlir` | Recombine lowering | iree-stream-k-d06 |
| `compiler/.../Codegen/IR/test/alloc_scratch_ops.mlir` | alloc_scratch op | iree-stream-k-wq6 |
| `compiler/.../LLVMGPU/test/aggregate_scratch_allocations.mlir` | Aggregation pass | iree-stream-k-71y |

### Host Worktree (`iree-stream-k/iree-host/`)

| File | Purpose | Beads Issue |
|------|---------|-------------|
| `compiler/.../Flow/IR/test/executable_ops.mlir` | Flow scratch_size region | iree-stream-k-owg |
| `compiler/.../Stream/IR/test/executable_ops.mlir` | Stream scratch_size region | iree-stream-k-big |
| `compiler/.../HAL/IR/test/executable_ops.mlir` | HAL scratch_size region | iree-stream-k-nyy |
| `compiler/.../Flow/Transforms/test/dispatch_formation_scratch.mlir` | Dispatch formation | iree-stream-k-1u7 |
| `compiler/.../Flow/Transforms/test/scratch_size_lowering.mlir` | Flow->Stream lowering | iree-stream-k-aua |
| `compiler/.../Stream/Transforms/test/scratch_size_lowering.mlir` | Stream->HAL lowering | iree-stream-k-aua |

---

## 6. Testing Infrastructure Requirements

### 6.1 Build Targets

Each new test file needs entries in:
- `BUILD.bazel` (iree_lit_test_suite with enforce_glob)
- `CMakeLists.txt` (iree_lit_test_suite)

### 6.2 Running Tests

```bash
# Codegen: Run all PCF IR tests
cd iree-stream-k/iree && bazel test //compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test:lit

# Codegen: Run specific test
cd iree-stream-k/iree && bazel test //compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test:lit --test_filter=stream_k_ops

# Host: Run Flow IR tests
cd iree-stream-k/iree-host && bazel test //compiler/src/iree/compiler/Dialect/Flow/IR/test:lit

# Run all Stream-K related tests (script TBD)
./run_stream_k_tests.sh
```

### 6.3 Correctness Validation Script

A script to run numerical correctness checks once the pipeline is complete:

```bash
# Compare stream-k vs data-parallel for a given matmul size
./validate_stream_k.sh --m=128 --n=128 --k=256 --dtype=f32 --tolerance=1e-6
```

---

## 7. Review Checklist for Implementers

Before requesting review from Fuzzer, ensure:

- [ ] All roundtrip tests pass (`iree-opt | iree-opt | FileCheck`)
- [ ] All verifier tests pass (`iree-opt --verify-diagnostics`)
- [ ] All transformation tests pass (`iree-opt --pass-pipeline=... | FileCheck`)
- [ ] BUILD.bazel and CMakeLists.txt updated with new test files
- [ ] No hardcoded values that should be parameterized
- [ ] Edge cases from Section 3 considered
- [ ] Test names are descriptive and follow existing patterns
