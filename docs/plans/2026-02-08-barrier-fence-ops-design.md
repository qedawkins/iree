# Barrier and Fence Ops for Pingpong Double-Buffering

## Problem

The pingpong double-buffer schedule uses a skewed barrier placement where
even and odd subgroup paths have consecutive barriers (A+B in even, E+F in
odd). The `gpu.barrier` canonicalizer (`eraseRedundantGpuBarrierOps` in
`GPUDialect.cpp`) merges consecutive barriers, shifting barrier pairing
between if/else branches and causing shared memory races.

The fundamental issue: `gpu.barrier` bundles synchronization with memory
fencing, and the canonicalizer assumes consecutive barriers are redundant.
For the skewed pingpong schedule, consecutive barriers are intentional and
serve as distinct synchronization points.

## Solution

Separate synchronization from memory fencing into distinct ops:

1. **`iree_gpu.global_subgroup_barrier`** - Pure synchronization, no memory fence
2. **`iree_codegen.fence`** - Memory fence with release/acquire semantics
3. **`pcf.fence`** - Higher-level fence operating on sref values

## New Ops

### `iree_gpu.global_subgroup_barrier`

**Purpose**: Synchronization-only barrier. All subgroups in the workgroup
must reach any instance before any can proceed past it.

**Assembly format**:
```mlir
iree_gpu.global_subgroup_barrier
```

**Traits/Effects**: No `Pure` trait (has synchronization side effects). No
memory effects (memory ops can be freely reordered with respect to it).
No `hasCanonicalizer` - consecutive instances are preserved.

**Lowering to ROCDL** (based on LDSBarrierOp but without fences):
- Pre-gfx12: `rocdl.s.barrier`
- gfx12+: `rocdl.barrier.signal -1` + `rocdl.barrier.wait -1`

**Lowering to NVVM**: `nvvm.barrier0` with default flags.

### `iree_codegen.fence`

**Purpose**: Memory fence ensuring visibility of operations at a specific
memory space, with release or acquire semantics.

**Assembly format**:
```mlir
iree_codegen.fence release #gpu.address_space<workgroup>
iree_codegen.fence acquire #gpu.address_space<workgroup>
```

**Operands**:
- `memory_space`: Attribute specifying which memory space to fence
- `is_release`: Bool attribute (true = release, false = acquire)

**Side effects**: Has memory effects on the specified memory space.

**Default LLVM lowering** (via external dialect interface):
```
llvm.fence release "workgroup"
llvm.fence acquire "workgroup"
```

**ROCDL override** (higher benefit pattern): Adds MMRA attribute to restrict
fence to appropriate address space:
```
Attribute mmra = "amdgpu-synchronize-as" = "local"  // for workgroup
llvm.fence release "workgroup" {mmra = ...}
```

### `pcf.fence`

**Purpose**: Higher-level fence operating on sref values. Lowered once
memory spaces of srefs are known.

**Assembly format**:
```mlir
pcf.fence release %lhs_alloc, %rhs_alloc : !pcf.sref<...>, !pcf.sref<...>
pcf.fence acquire %lhs_alloc, %rhs_alloc : !pcf.sref<...>, !pcf.sref<...>
```

**Operands**:
- `srefs`: Variadic list of sref values to fence
- `is_release`: Bool attribute

**Lowering in ConvertSRefToMemRef**:
1. Collect memory space for each sref operand (from scope's `getAllocMemSpace()`)
2. Deduplicate: multiple srefs with same memory space -> one `iree_codegen.fence`
3. Emit fences in operand order

## Changes to Existing Code

### SubgroupScopeModel::addBarrier

In `GPUScopeExternalModels.cpp`, change:
```cpp
// OLD:
gpu::BarrierOp::create(builder, builder.getUnknownLoc());
// NEW:
GPU::GlobalSubgroupBarrierOp::create(builder, builder.getUnknownLoc());
```

### sync=true on pcf.loop/pcf.generic

In `LowerStructuralPCF.cpp`, when `getSyncOnReturn()` is true, emit:
```mlir
iree_codegen.fence release <alloc_memory_space>
<barrier from scope>
iree_codegen.fence acquire <alloc_memory_space>
```

The memory space comes from `getScope().getAllocMemSpace()`.

## Fence Placement in Pingpong Schedule

Fences are placed explicitly in PingpongConfig.cpp. Rule: release before
barrier when the current path wrote to LDS, acquire after barrier when
the current path will read from LDS.

### Even path (copy-first):
```
copy(buf0, k=0)               // WRITE to LDS
fence release
barrier A
barrier B                     // no fence (even hasn't read/written since A)
fence acquire                 // prepare for loop reads
for k=0..K-1:
    copy(buf0, k+1)           // WRITE to LDS
    read(buf1)                // READ from LDS
    fence release
    barrier C
    compute()                 // register-only
    barrier D
    fence acquire             // see odd's writes for next iter
write_results
```

### Odd path (compute-first):
```
copy(buf1, k=0)               // WRITE to LDS
fence release
barrier A
fence acquire                 // prepare to read buf0
read(buf0)                    // READ from LDS
barrier B                     // no fence (odd only read since A)
for k=1..K-1:
    compute()                 // register-only
    barrier C
    fence acquire             // see even's writes
    copy(buf1, k)             // WRITE to LDS
    read(buf0)                // READ from LDS
    fence release
    barrier D
epilogue: compute()
barrier E                     // structural, no fences
barrier F                     // structural, no fences
write_results
```

## Testing

### Lit tests
1. `iree_gpu.global_subgroup_barrier`: roundtrip, consecutive not canonicalized,
   ROCDL lowering (no fences), NVVM lowering
2. `iree_codegen.fence`: roundtrip, default LLVM lowering, ROCDL override with MMRA
3. `pcf.fence`: roundtrip, ConvertSRefToMemRef lowering with deduplication

### Integration test
- 2048x2048x2048 matmul with skewed pingpong schedule
- Must go from 50% correct (barrier bug) to 100% correct
- Verify IR: no `gpu.barrier`, only `iree_gpu.global_subgroup_barrier` + fences
