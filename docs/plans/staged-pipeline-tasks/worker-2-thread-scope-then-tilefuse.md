# Worker 2: TileAndFuse Path

You are implementing the TileAndFuse path for the experimental staged pipeline. This is fully independent of the VectorDistribute path (worker 1).

**Branch:** Create a worktree from `shared-exec-2`
**Beads:** root-v3n

---

## Task: TileAndFuse Path — Port GPUConvertThreadForallToSubgroupLanePCF

### Background

The experimental staged pipeline in `compiler/plugins/target/ROCM/ROCMTarget.cpp` has a "Subgroup + lane distribution" phase (see the TODO at ~line 621). For TileAndFuse, this phase runs the normal TileAndFuse pipeline up through FuseAndHoistParallelLoops, then converts the resulting scf.forall ops with subgroup/lane mappings to PCF ops.

This path does NOT use shared_executor wrapping — it has its own conversion from scf.forall → PCF via `GPUConvertThreadForallToSubgroupLanePCF`.

### Step 1: Extract pre-distribution TileAndFuse passes

Look at `addGPUTileAndFusePassPipeline` in `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp` (lines 421-570). Extract everything up through `createGPUFuseAndHoistParallelLoopsPass`:

- User annotated strategies (LowerTensorUKernels, LoweringConfigInterpreter)
- Promote matmul operands, pack to intrinsics
- Reduction tiling
- Thread/subgroup tiling
- Vectorization
- Fuse and hoist parallel loops

But NOT bufferization, vector distribution, or post-bufferization passes.

Create a helper (e.g., `addGPUTileAndFusePreDistributionPasses`) exposed in `Passes.h`.

### Step 2: Port GPUConvertThreadForallToSubgroupLanePCF

Port the WIP changes from these commits. Read them carefully:

```bash
cd /home/quinn/root/iree-shared-exec/iree
git show 76ac573970eb14fa  # Phase 2: barrier_region → PCF ops + new ops
git show 5621cb80ef373e56  # WIP refinements + reference IR dumps
```

**Commit 76ac573970** adds:
- `GPUConvertThreadForallToSubgroupLanePCF` pass in `Codegen/Common/GPU/`
- Phase 2: converts `bufferization.alloc_tensor` + `barrier_region` + consumer chains to PCF ops
- Consumer chain walking: `expand_shape → pcf.expand_shape`, `extract_slice → pcf.subview`, `transfer_read → pcf.read_slice`
- New PCF ops: `pcf.expand_shape`, `pcf.subview`
- Wider DMA init acceptance via `AnyNonVectorShaped` type constraint

**Commit 5621cb80ef** adds:
- Refinements to the conversion pass
- Reference IR dumps for a 256x256x32 matmul at different pipeline stages (in `refs/` directory)

### Step 3: Add pcf.expand_shape and pcf.subview ops

Port these op definitions from commit `76ac573970`:
- `pcf.expand_shape` — reshape an sref
- `pcf.subview` — subview of an sref

Include their ODS definitions, parse/print/verify, and tests.

### Step 4: Wire into staged pipeline

In `ROCMTarget.cpp`, in the subgroup + lane distribution phase, add a `MultiPipelineNest` case for TileAndFuse that:
1. Runs the pre-distribution TileAndFuse passes (step 1)
2. Runs `GPUConvertThreadForallToSubgroupLanePCF` (step 2)

### Step 5: Fusion cleanup

After the TileAndFuse conversion, add `createGPUFuseSubgroupConsumersPass` (or the appropriate fusion pass from `Codegen/Common/GPU/`) to fuse consumer ops into the subgroup-level PCF producers. This is a single pass insertion — see the comment at ROCMTarget.cpp ~line 647.

### Step 6: Test

- Port tests from commit `76ac573970` (`convert_thread_forall_to_subgroup_lane_pcf.mlir`)
- Use the reference IR dumps from `5621cb80ef` (`refs/` directory) as golden files
- End-to-end test through the staged pipeline with a simple matmul

### Commit

```
[PCF/Staged] Add TileAndFuse path with GPUConvertThreadForallToSubgroupLanePCF
```

Update beads: `br update root-v3n --status=in_progress` when starting, `br close root-v3n` when done.
Also close root-2na (fusion cleanup) since it's included in this task.
