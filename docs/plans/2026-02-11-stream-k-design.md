# Stream-K Distribution Strategy

**Date**: 2026-02-11
**Status**: Design Complete
**Target**: LLVMGPU backend, RDNA4 hardware

## Overview

Stream-K is a workgroup distribution strategy for tiling-interface ops with
reduction dimensions (e.g., matrix multiplication). Instead of assigning each
workgroup a fixed output tile (data-parallel) or splitting the reduction
dimension across a fixed set of workgroups (split-K), stream-K linearizes the
entire work space — `output_tiles * k_tiles` — and distributes contiguous
chunks to workgroups. This maximizes GPU occupancy by decoupling the workgroup
count from the problem shape.

## Workstream Summary

The implementation consists of two largely independent workstreams that
converge late in the codegen pipeline:

1. **Codegen Workstream** — The `pcf.stream_k_recombine` op, the tiling
   transformation, and the lowering of the recombine op to atomics + branching.
2. **Host Workstream** — The scratch memory allocation pipeline, plumbed
   through Flow, Stream, and HAL, including dispatch formation changes.

The two workstreams converge at the **aggregation pass** (near
ReconcileTranslationInfo), where `iree_codegen.alloc_scratch` ops from the
codegen side get resolved against the `scratch_size` region from the host side.

---

## 1. `pcf.stream_k_recombine` Op

A new PCF dialect op that handles partial tile recombination. It is a
non-terminating, side-effecting control flow op that writes through srefs and
returns nothing. This makes it composable with other PCF ops and enables
fusion of downstream operations.

### Operands

| Operand | Type | Description |
|---------|------|-------------|
| `partial_tile` | tensor | The computed partial tile |
| `output_sref` | `pcf.sref` | Output buffer with offsets/sizes/strides |
| `scratch_sref` | `pcf.sref` | Pre-sliced scratch slot for this output tile |
| `counter_sref` | `pcf.sref` | Pre-sliced atomic counter for this output tile |
| `num_in_group` | index | Number of workgroups contributing to this output tile |

### Regions

**Combiner region**: Takes two scalar elements (matching the tile element
type), yields one. Defines the accumulation semantics (e.g., addition for
matmul).

**Writeback region**: Takes the finished tile as a single block argument.
Performs the actual output write and any fused epilogue operations (bias add,
activation, etc.). This region is the fusion surface — consumer operations get
pulled into it.

### Runtime Semantics (3 Branches)

1. **`num_in_group == 1`**: Complete tile. Pass `partial_tile` directly to
   the writeback region. Skip scratch entirely.
2. **Atomic increment < `num_in_group`**: Not the last contributor. Write
   `partial_tile` to scratch. Do NOT execute writeback. Continue.
3. **Atomic increment == `num_in_group`**: Last contributor. Accumulate all
   scratch tiles using the combiner region. Pass the accumulated result to
   the writeback region.

### Assembly Format (Sketch)

```mlir
pcf.stream_k_recombine %partial
    into %out_ref[%off_m, %off_n] [%sz_m, %sz_n] [%s_m, %s_n]
    scratch %scratch_slot counter %ctr
    group(%num_in_group)
    combiner {
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        pcf.yield %sum : f32
    }
    writeback {
      ^bb0(%final: tensor<64x64xf32>):
        pcf.write_slice %final into %out_ref[%off_m, %off_n] [%sz_m, %sz_n] [%s_m, %s_n]
            : tensor<64x64xf32> into !pcf.sref<tensor<256x256xf32>>
        pcf.yield
    }
    : tensor<64x64xf32> into !pcf.sref<tensor<256x256xf32>>
```

---

## 2. Stream-K Tiling Transformation

A new tiling path in `LLVMGPUTileAndFuse`, triggered by a
`streamed_reduction` tiling level in the lowering config.

### Inputs

- A TilingInterface op with at least one reduction dimension.
- Tile sizes as a list mapping to the iteration space of the tilable op (not
  limited to standard matmul shapes).
- `num_workgroups` from the lowering config, typically chosen to saturate the
  GPU (CUs * waves-per-CU).

### Output IR Structure

```mlir
pcf.loop scope(#workgroup_scope) count(%total_work_items)
  execute(%out_ref = %output)[%work_idx: index]
      : (!pcf.sref<...>) -> (tensor<...>) {

    // 1. Decode: divmod chain decomposes %work_idx into
    //    parallel dimension coordinates + reduction dimension coordinates.
    //    Reduction dimensions are innermost in the linearization so that
    //    tiles for the same output position are adjacent.

    // 2. Group size: arithmetic on the output tile index + items_per_wg.

    // 3. Slice inputs using TilingInterface.

    // 4. Compute partial tile (the original op on tile-sized operands).

    // 5. Slice scratch + counter srefs for this output tile.

    // 6. pcf.stream_k_recombine with combiner + writeback.

    pcf.return
  }
```

### Work Distribution

For an op with iteration space `[d0, d1, ..., dN]` and tile sizes
`[t0, t1, ..., tN]`:

```
tile_counts[i] = ceil(d_i / t_i)

// Parallel dims form the "output tile" space.
// Reduction dims form the "k-tile" space.
output_tiles     = product(tile_counts[i] for parallel i)
k_tiles_per_out  = product(tile_counts[i] for reduction i)
total_work       = output_tiles * k_tiles_per_out

items_per_wg     = ceil(total_work / num_workgroups)
// Workgroup j handles [j * items_per_wg, min((j+1) * items_per_wg, total_work))
```

### Decode Arithmetic

Inside the loop, each work item's linear index is decomposed:

```
out_idx = linear_idx / k_tiles_per_out
k_idx   = linear_idx % k_tiles_per_out
```

The `out_idx` is further decomposed into per-parallel-dimension coordinates
using a divmod chain over the parallel tile counts.

### Group Size Arithmetic

For a given work item, the number of workgroups contributing to its output
tile:

```
first_linear = out_idx * k_tiles_per_out
last_linear  = first_linear + k_tiles_per_out - 1
first_wg     = first_linear / items_per_wg
last_wg      = last_linear  / items_per_wg
num_in_group = last_wg - first_wg + 1
```

This is pure index arithmetic — a few multiplies, floor divisions, and a
subtraction.

---

## 3. Scratch Memory Allocation Pipeline

A new mechanism for dispatches to request additional memory, plumbed through
all dialect levels (Flow, Stream, HAL).

### 3a. Export Region

An **optional** `scratch_size` region on executable export ops at all three
dialect levels (Flow, Stream, HAL).

- **Arguments**: `(%device, %workload...)` — same signature as the
  `workgroup_count` region.
- **Returns**: Single `index` value (byte count).
- **Initially**: Contains a placeholder op (analogous to the workgroup count
  placeholder). Populated by codegen later.
- **Absence**: The region is optional. Calling `scratch_size` on an export
  that does not have the region is a compiler error.

### 3b. Host-Side Ops

New ops at each dialect level:

| Dialect | Op | Description |
|---------|-----|-------------|
| Flow | `flow.executable.scratch_size` | Invokes the export's scratch_size region |
| Stream | `stream.executable.scratch_size` | Same, at Stream level |
| HAL | `hal.executable.scratch_size` | Same, at HAL level |

Each takes a reference to the export and the workload values, returns an
`index`. After executable translation, calls are resolved by inlining the
region (same mechanism as workgroup count resolution).

### 3c. Dispatch Formation

A pass during dispatch formation identifies matmul-like ops and:

1. Adds an empty `scratch_size` region (with placeholder) to the flow export.
2. Creates a `flow.executable.scratch_size` op before the dispatch.
3. Creates `tensor.empty(%size) : tensor<?xi8>` from the returned size.
4. Adds the tensor as an additional operand to the dispatch.

```mlir
%size = flow.executable.scratch_size @ex::@entry [%workload...]
%scratch = tensor.empty(%size) : tensor<?xi8>
%result = flow.dispatch @ex::@entry [%wl...] (%inputs..., %scratch)
```

This pass can be hacky initially — just pattern-match on matmul-like ops.

### 3d. Codegen Side

1. `pcf.alloc` with cross-workgroup scope lowers to
   `iree_codegen.alloc_scratch` (the current interface implementation for
   this scope is a failure; this replaces it).
2. An **aggregation pass** (near ReconcileTranslationInfo):
   - Collects all `iree_codegen.alloc_scratch` ops in the dispatch.
   - Sums their requested byte sizes.
   - Clones the total size computation into the export's `scratch_size`
     region.
   - Replaces each `alloc_scratch` with a subview of the scratch binding
     at the appropriate offset.

This is the convergence point of the two workstreams.

---

## 4. Lowering of `pcf.stream_k_recombine`

The recombine op lowers to concrete control flow, atomics, and inlined
regions during PCF lowering (before GPU codegen finalizes).

### Lowered Structure

```mlir
// 1. Atomic increment counter, get old value.
%old = memref.atomic_rmw addi %c1, %counter_ref[] : memref<i32>

// 2. Determine branch.
%is_last = arith.cmpi eq, %old, (%num_in_group - 1)
%is_only = arith.cmpi eq, %num_in_group, %c1
%needs_writeback = arith.ori %is_last, %is_only

// 3. Scratch write (unless sole contributor).
scf.if not %is_only {
  // Store partial tile to scratch slot.
  <store partial_tile to scratch_ref>

  scf.if %is_last {
    // Memory fence — ensure all other workgroups' writes are visible.
    // Load all tiles from scratch slots [0, num_in_group - 1).
    // Accumulate using combiner region (inlined as pointwise loop).
    // %accumulated = combine(slot[0], slot[1], ..., partial)
  }
}

// 4. Writeback (only for sole contributor or last in group).
scf.if %needs_writeback {
  // Inline writeback region with %final_tile.
}
```

### Key Lowering Concerns

- **Atomics**: The counter is in global memory. Must use global memory atomic
  RMW (not shared/local memory atomics).
- **Memory fence**: Between scratch writes (branch 2) and scratch reads
  (branch 3), the last workgroup needs to see all other workgroups' writes.
  Requires a device-scope memory fence.
- **Accumulation loop**: Iterates over `num_in_group - 1` scratch slots. The
  last workgroup already has its own partial tile, which participates in the
  accumulation.
- **Combiner inlining**: The combiner region is inlined as a pointwise loop
  over tile elements.

---

## 5. End-to-End Pipeline Integration

### Early Pipeline (Flow)

1. Dispatch formation identifies matmul-like ops.
2. New pass adds optional `scratch_size` region (placeholder) to the flow
   export.
3. Creates `flow.executable.scratch_size` op, `tensor.empty`, and additional
   dispatch operand.
4. Flow -> Stream -> HAL lowering carries the `scratch_size` region and
   calling op through unchanged.

### Codegen Pipeline (Inside Executable)

1. `LLVMGPUTileAndFuse` sees `streamed_reduction` tiling level.
2. Stream-K tiling transformation runs: produces `pcf.loop` + decode +
   compute + `pcf.stream_k_recombine`.
3. Inner tile lowers through normal PCF / thread / subgroup / vectorization
   pipeline.
4. `pcf.alloc` (cross-workgroup) -> `iree_codegen.alloc_scratch`.
5. `pcf.stream_k_recombine` lowers to atomics + branching + inlined
   combiner/writeback.
6. **Aggregation pass** (near ReconcileTranslationInfo):
   - Collects `alloc_scratch` ops, sums byte sizes.
   - Populates the `scratch_size` region with concrete arithmetic.
   - Replaces each `alloc_scratch` with subview of scratch binding.

### Post-Codegen (Host)

1. Executable translation finalizes the export regions.
2. `scratch_size` region calls get inlined (like workgroup count).
3. Runtime allocates scratch buffer, passes as binding.
4. Dispatch executes with scratch available as global memory.

---

## 6. Testing Strategy

### Unit Tests (Lit Tests)

- **`pcf.stream_k_recombine` op**: Parser/printer roundtrip, verifier tests
  (wrong combiner signature, missing writeback, type mismatches).
- **Stream-K tiling transformation**: Input a generic TilingInterface op,
  FileCheck the output structure (pcf.loop, decode arithmetic, recombine op
  placement).
- **Recombine lowering**: Input the recombine op, FileCheck the lowered
  atomics + branching + inlined regions.
- **`iree_codegen.alloc_scratch`**: Parser/printer, verifier.
- **Aggregation pass**: Multiple alloc_scratch ops -> single scratch_size
  region populated, subview replacements correct.
- **Export region plumbing**: scratch_size region survives Flow -> Stream ->
  HAL lowering.
- **Dispatch formation pass**: Matmul-like op gets scratch buffer argument
  added.

### Integration Tests

- Small matmul (e.g., 128x128x256 with tile 64x64x64) through the full
  pipeline. Verify the generated GPU kernel is valid and executes.
- Verify scratch buffer size is computed correctly for known problem sizes.
- Compare stream-K numerical output against data-parallel matmul (bitwise
  identical for integer types, within tolerance for float).

### Hardware Tests

- Direct testing on the RDNA4 card on this system.
- Run real matmul dispatches and verify correctness end-to-end.

### Benchmarks

- Matmul sizes known to benefit from stream-K: non-square, large-K,
  small-M/N where data-parallel leaves SMs idle.
- Compare against data-parallel and split-K baselines.
- Key metrics: GPU occupancy and wall-clock time.
- Target hardware: RDNA4 GPU on this system.

---

## 7. Success Criteria

1. **Correctness**: Stream-K matmul produces numerically correct results on
   RDNA4, verified against data-parallel baseline.
2. **Performance**: Measurable speedup on workloads where data-parallel
   leaves CUs idle (tall-skinny, large-K matmuls) on the RDNA4 card.
3. **Fusion**: Epilogue ops (bias add, activation) fuse into the writeback
   region and execute correctly.
4. **Pipeline completeness**: Full compilation from `linalg.matmul` to
   executable that runs on RDNA4, including scratch buffer allocation and
   plumbing.
5. **No regressions**: Existing data-parallel matmul paths are unaffected.
