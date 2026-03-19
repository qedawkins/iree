# Telescoping Subscopes

## Goal

Enable multi-level distribution by allowing a distributed parent scope
to telescope into a child scope. After distributing at the subgroup
level, the resulting code is structurally identical to a lane-level
shared_executor. This design provides the ops and lowering mechanics
to express and lower that transition.

## Motivation

Distribution is a telescoping process. A workgroup-level
shared_executor gets distributed to produce subgroup-level code, which
itself needs distribution to produce lane-level code. Without explicit
scope transitions, each level would need to be a separate hand-written
shared_executor in the source IR. The telescoping ops make this
compositional: one shared_executor at the outermost scope contains the
entire computation, and the pipeline peels one scope level per
distribution phase.

Memory allocation complicates this because a child scope cannot
allocate memory apportioned for the parent. Shared memory for lanes
must be allocated at the subgroup level, but the code that uses it
lives inside the lane-scope execution. The solution is to allocate at
the parent level and attach the allocations to the threadgroup as
struct fields, then unpack them when telescoping into the child scope.

## New Ops

### `pcf.init_subscope`

Allocates memory for a child scope and attaches it as struct fields to
the threadgroup. The body contains allocation ops (`pcf.alloc`) that
produce srefs for the child scope. The yielded values become struct
fields on the returned threadgroup.

Semantics: run-once, uniform across all scope levels.

```mlir
%tg_with_allocs = pcf.init_subscope %tg {
  %alloc = pcf.alloc : !pcf.sref<128x64xf16, #lane_scope>
  pcf.yield %alloc
} -> !pcf.threadgroup<#subgroup_scope, {!pcf.sref<128x64xf16, #lane_scope>}>
```

The input threadgroup must have no struct fields (or the existing
fields are preserved and the new ones are appended — design choice
for later). The output threadgroup has the same scope but with struct
fields matching the yielded types.

### `pcf.telescope`

Takes a threadgroup with struct fields and a worker ID. Returns a
clean child-scope threadgroup (no struct fields) plus the struct fields
as separate SSA values. The worker ID is required as proof that the
caller is in a distributed context — telescoping only makes sense when
the parent scope has already been distributed.

```mlir
%child_tg, %sref = pcf.telescope %tg_with_allocs[%lane_id]
    : !pcf.threadgroup<#subgroup_scope, {!pcf.sref<128x64xf16, #lane_scope>}>
   -> (!pcf.threadgroup<#lane_scope>, !pcf.sref<128x64xf16, #lane_scope>)
```

The child scope is determined by the result threadgroup type. The
struct fields from the input threadgroup become the trailing result
values. The child threadgroup has no struct fields.

## Lowering

### `lowerSharedExecutor` with child scope

When `lowerSharedExecutor` is given a child scope, the lowering changes
from producing a bare `pcf.generic` to producing a `pcf.generic`
wrapping a new child-scope `shared_executor`.

**Input:**
```mlir
pcf.shared_executor scope(#parent) execute(...) [%tg] {
  %tg2 = pcf.init_subscope %tg {
    %alloc = pcf.alloc : !pcf.sref<..., #child>
    pcf.yield %alloc
  } -> !pcf.threadgroup<#parent, {!pcf.sref<..., #child>}>

  // ... body code ...

  %child_tg, %sref = pcf.telescope %tg2[%worker_id]

  // ... more body code using %child_tg and %sref ...

  pcf.return
}
```

**Output:**
```mlir
pcf.generic scope(#parent) execute(...) [...] {
  pcf.shared_executor scope(#child)
    initialize {
      %alloc = pcf.alloc : !pcf.sref<..., #child>
      pcf.yield %alloc
    } -> (%sref: !pcf.sref<..., #child>)
    execute(...) [%child_tg] {
      // The ENTIRE original body goes here.
      // init_subscope ops removed (bodies inlined to initializer).
      // telescope ops removed (values replaced with block args).
      pcf.return
    }
  pcf.return
}
```

The `pcf.generic` body contains exactly two things: the child
`shared_executor` and the terminator. There is no splitting of the
body between the generic and the child shared_executor — the entire
body moves into the child.

The lowering steps:

1. Collect all `init_subscope` ops targeting the child scope. Inline
   their bodies into the new child shared_executor's initializer
   region.
2. Find `telescope` ops. Replace the child threadgroup result with the
   new shared_executor's threadgroup block arg. Replace the sref
   results with the corresponding leading args from the initializer.
3. Move the entire body into the child shared_executor's execute
   region.
4. Remove the `init_subscope` and `telescope` ops (their values have
   been replaced).
5. Wrap the child shared_executor in a `pcf.generic` for the parent
   scope.

### Pipeline orchestration

Multi-level distribution is driven by repeated invocations of the
`DistributeAndLower` pass, each peeling one scope level:

**Workgroup → subgroup → lane (3 phases):**

Phase 1: Distribute at workgroup scope, child = subgroup.
```
shared_executor(workgroup) { ... }
→ pcf.generic(workgroup) { shared_executor(subgroup) { ... } }
```

Phase 2: Distribute at subgroup scope, child = lane.
```
pcf.generic(workgroup) { shared_executor(subgroup) { ... } }
→ pcf.generic(workgroup) { pcf.generic(subgroup) { shared_executor(lane) { ... } } }
```

Phase 3: Distribute at lane scope, no child.
```
→ pcf.generic(workgroup) { pcf.generic(subgroup) { pcf.generic(lane) { ... } } }
```

Each phase uses the same pass with a different child scope parameter.
The `DistributeAndLower` pass gains an option for the child scope. The
`lowerSharedExecutor` function uses it to decide between the current
behavior (no child → straight to pcf.generic) and the telescoping
behavior (child → pcf.generic wrapping child shared_executor).

## Design Decisions

**Why separate `init_subscope` and `telescope`?** Memory allocation
must happen before distribution (at the parent scope level, where all
workers can cooperate on allocation). The telescoping into the child
scope happens after distribution (inside a distributed region where
the worker ID is available). Separating these into two ops reflects
the temporal ordering: allocate first, then telescope.

**Why require a worker ID on `telescope`?** The ID serves as proof
that the caller is in a distributed context. Without it, telescoping
in undistributed code would be a silent semantic error. Making the ID
an explicit operand ensures the op can only appear after distribution
has assigned IDs.

**Why does the entire body move into the child shared_executor?** The
parent scope is fully distributed at this point — there is no
parent-scope work left to do. Everything remaining is child-scope
work. The pcf.generic just provides the structural wrapper for the
parent scope's distributed execution.

**Why run-once semantics for `init_subscope`?** The allocations are
logically global setup. The exact execution model is lowering-driven,
but from the IR's perspective the allocations are uniform across all
scope levels and happen before any distribution.

## Testing

- Roundtrip tests for `init_subscope` and `telescope` syntax.
- Verifier tests: telescope without matching struct fields, scope
  mismatches, init_subscope on threadgroup that already has struct
  fields.
- Lowering test: shared_executor with init_subscope + telescope
  lowered to pcf.generic wrapping child shared_executor.
- Two-level telescoping: workgroup → subgroup → lane pipeline.
- Integration with tile_group: init_subscope + telescope inside a
  tile_group body.
