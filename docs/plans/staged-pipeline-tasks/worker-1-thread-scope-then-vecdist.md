# Worker 1: Thread Scope + VectorDistribute Path

You are implementing two sequential tasks for the experimental staged pipeline. Complete task 1 fully (build, test, commit) before starting task 2.

**Branch:** Create a worktree from `shared-exec-2`
**Beads:** root-3v5 (task 1), root-3f5 (task 2)

---

## Task 1: Add Thread Scope and shared_executor Wrapping Pass

### Background

The experimental staged pipeline in `compiler/plugins/target/ROCM/ROCMTarget.cpp` has a "Subgroup + lane distribution" phase (see the TODO at ~line 621) that needs to wrap block-level PCF code in a `pcf.shared_executor`. This task creates the prerequisites.

After the workgroup distribution phase, the IR contains `pcf.generic`/`pcf.loop` ops with workgroup scope. These represent the per-workgroup computation. The next step is to distribute this work across subgroups and lanes (threads). To do that, we wrap the body in a `shared_executor` with a "thread scope" that represents the full set of threads in the workgroup.

### 1a. Add thread scope

Thread scope represents all threads in a workgroup — semantically a combination of subgroup and lane levels. It needs to implement `ScopeAttrInterface`.

Look at how scopes are defined:
- `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.td` — `WorkgroupScopeAttr` definition
- `compiler/src/iree/compiler/Codegen/Common/GPU/GPUScopeExternalModels.cpp` — GPU implementations of `getWorkerIDs`, `getWorkerCounts`, `addBarrier`, `getNativeNumIds`

Create a new scope (either as a new attr in the Codegen dialect or as a PCF scope) that:
- `getWorkerIDs(builder, loc, numIds)` — returns linearized thread ID (from `gpu.thread_id` x/y/z linearized)
- `getWorkerCounts(builder, loc, numIds)` — returns total thread count (workgroup size)
- `getNativeNumIds()` — returns 1 (linearized)
- `addBarrier(builder)` — creates a `gpu.barrier`
- `getAllocMemSpace(context)` — returns workgroup memory space (for shared memory allocation)

### 1b. Add wrapping pass

Create a pass (likely in `Codegen/Common/GPU/`) that:
1. Finds `pcf.generic`/`pcf.loop` ops with workgroup scope
2. Creates a `pcf.shared_executor` with thread scope wrapping the body
3. All tensor values used inside the body that are defined outside become readonly inputs — use `pcf.to_sref` inside the shared_executor to create sref views
4. Readwrite outputs (if any) become readwrite inits on the shared_executor
5. The pcf.generic/loop's region body moves into the shared_executor's execute region

### 1c. Wire into staged pipeline

In `ROCMTarget.cpp`, in the subgroup + lane distribution section, add this pass for both TileAndFuse and VectorDistribute paths.

### 1d. Test

- Roundtrip test for shared_executor with thread scope
- Pass test: pcf.generic with workgroup scope → shared_executor wrapping
- Wire into the staged pipeline and verify the pipeline constructs correctly (use `--mlir-print-ir-after-all` on a simple dispatch)

### 1e. Commit

```
[PCF/Staged] Add thread scope and shared_executor wrapping pass
```

Update beads: `br update root-3v5 --status=in_progress` when starting, `br close root-3v5` when done.

---

## Task 2: VectorDistribute Path for Staged Pipeline

### Background

After block-level code is wrapped in `pcf.shared_executor` (task 1), the VectorDistribute path needs to run pre-bufferization VectorDistribute passes within that shared_executor, then distribute using the PCF distribution interface.

Read the TODO at `compiler/plugins/target/ROCM/ROCMTarget.cpp` line 626 for full context.

### 2a. Extract pre-bufferization VectorDistribute passes

Look at `addGPUVectorDistributePassPipeline` in `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp` (lines 729-860). Extract the passes that run BEFORE bufferization:

- Reduction tiling (`TilingLevel::Reduction`)
- Partial reduction tiling (`TilingLevel::PartialReduction`)
- Serial tiling (`TilingLevel::Serial`)
- Pack to intrinsics, decompose attention
- Vectorization (with masking and decomposition)
- Allocate tensors for copies to shared memory

Create a helper function (e.g., `addGPUVectorDistributePreBufferizePasses`) that adds just these passes. This may need to be exposed in `Passes.h` as public API.

### 2b. Add read_slice folding patterns

Add folding patterns to compose `read_slice(tensor.extract_slice)` chains. The goal is to fold down to `read_slice(vector.transfer_read)` or `read_slice(vector.gather)`. Then "bufferize" the transfer by inserting `pcf.get_memref` and having the transfer read directly from the memref.

This is new pattern work. Look at how `pcf.read_slice` and `pcf.get_memref` work:
- `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.td` — ReadSliceOp, GetMemrefOp

### 2c. VectorAlloc + VectorDistribute via PCF interface

Run:
- `createGPUVectorAllocPass` — allocate shared memory for copies
- VectorDistribute using the PCF distribution interface

The PCF distribution interface is implemented in:
- `compiler/src/iree/compiler/Codegen/Common/GPU/GPUPCFDistribution.cpp`
- `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/DistributeAndLower.cpp`

Tested in:
- `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/distribute_vector.mlir`

### 2d. Wire into staged pipeline

In `ROCMTarget.cpp`, add a `MultiPipelineNest` case for VectorDistribute in the subgroup + lane distribution phase that runs steps 2a-2c.

### 2e. Test

- End-to-end test: simple matmul through the staged VectorDistribute path
- Verify vector ops get distributed to SIMT form
- Use reference IR from the monolithic pipeline for comparison

### 2f. Commit

```
[PCF/Staged] Add VectorDistribute path for staged pipeline
```

Update beads: `br update root-3f5 --status=in_progress` when starting, `br close root-3f5` when done.
