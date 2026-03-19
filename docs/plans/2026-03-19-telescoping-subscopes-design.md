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

## Related Changes

**Drop `uniform` from ClusterType.** The `uniform` struct kind is
removed. ClusterType retains only `private` and `shared` struct kinds.
Threadgroup values flow through as regular private struct elements on
clusters.

**`run_thread` accepts threadgroup types.** The struct fields on a
cluster can include threadgroup-typed values. This allows the
threadgroup (enriched with child-scope allocations) to flow through
the cluster execution hierarchy into `run_thread` bodies where
`telescope` is called.

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

The input threadgroup must have no struct fields. The output
threadgroup has the same scope but with struct fields matching the
yielded types. Multiple `init_subscope` ops targeting the same scope
are not allowed — use a single `init_subscope` that yields all
allocations.

**ODS structure:**
- Operand: `!pcf.threadgroup<#scope>` (no struct fields)
- Region: single-block body terminated by `pcf.yield`
- Result: `!pcf.threadgroup<#scope, {yielded types...}>`

**Verifier rules:**
- Result threadgroup scope must equal input threadgroup scope.
- Input threadgroup must have no struct fields.
- Yielded types become the struct field types of the result.

### `pcf.telescope`

A pure type conversion op. Takes a threadgroup with struct fields and
an index operand (the thread ID). Returns a child-scope threadgroup
(no struct fields) plus the struct fields as separate SSA values.

The op is pure — it does not change execution context. It converts the
parent-scope threadgroup identity into a child-scope threadgroup
identity using the provided thread ID. The thread ID operand is the
parent scope's worker ID; passing a value that is not the actual
thread ID is undefined behavior.

In practice, `telescope` appears inside `run_thread` bodies where the
thread ID is available as a block argument. This is not enforced by
the verifier — the semantics are correct only when the thread ID
operand is the actual distributed worker ID.

```mlir
%child_tg, %sref = pcf.telescope %tg_with_allocs[%tid]
    : !pcf.threadgroup<#subgroup_scope, {!pcf.sref<128x64xf16, #lane_scope>}>
   -> (!pcf.threadgroup<#lane_scope>, !pcf.sref<128x64xf16, #lane_scope>)
```

The child scope is determined by the result threadgroup type. The
struct fields from the input threadgroup become the trailing result
values. The child threadgroup has no struct fields.

## Source IR Shape

The source IR contains a single shared_executor at the outermost
scope. `init_subscope` prepares child-scope allocations. `telescope`
appears inside `run_thread` bodies after distribution has provided
thread IDs. There is no child `shared_executor` in the source IR —
it is created by the lowering.

```mlir
pcf.shared_executor scope(#parent) execute(...) [%tg] {
  %tg2 = pcf.init_subscope %tg {
    %alloc = pcf.alloc : !pcf.sref<..., #child>
    pcf.yield %alloc
  } -> !pcf.threadgroup<#parent, {!pcf.sref<..., #child>}>

  pcf.shared_executor.tile_group %tg2 split [...] (%cluster: ...) {
    pcf.shared_executor.run_thread(%cluster)[...] ()[%tid: index] {
      %child_tg, %sref = pcf.telescope %tg2[%tid] : ... -> ...
      // child-scope code using %child_tg and %sref
      // no child shared_executor here — lowering creates it
      pcf.cluster_yield
    } : (...)
    pcf.return
  } : ...
  pcf.return
}
```

## Lowering

### `lowerSharedExecutor` with child scope

When `lowerSharedExecutor` is given a child scope, the lowering
produces a `pcf.generic` wrapping a new child-scope `shared_executor`
instead of a bare `pcf.generic`.

**Output:**
```mlir
pcf.generic scope(#parent) execute(...) [...] {
  pcf.shared_executor scope(#child)
    initialize {
      %alloc = pcf.alloc : !pcf.sref<..., #child>
      pcf.yield %alloc
    } -> (%sref: !pcf.sref<..., #child>)
    execute(...) [%child_tg] {
      // The entire original body.
      // init_subscope ops removed (bodies inlined to initializer).
      // telescope ops removed (values replaced with block args).
      pcf.return
    }
  pcf.return
}
```

The `pcf.generic` body contains exactly two things: the child
`shared_executor` and the terminator. The entire original body moves
into the child shared_executor's execute region.

**Lowering steps:**

1. Create the `pcf.generic` for the parent scope.
2. Create the child `shared_executor` inside the generic's body.
3. Collect all `init_subscope` ops targeting the child scope. Inline
   their bodies into the child shared_executor's initializer region.
4. Move the entire body into the child shared_executor's execute
   region.
5. Replace `telescope` results: child threadgroup maps to the new
   shared_executor's threadgroup block arg, sref values map to the
   corresponding leading args from the initializer.
6. Remove the `init_subscope` and `telescope` ops (their values have
   been replaced).

**Interaction with parent initializer:** If the parent shared_executor
has its own initializer region, it is handled as before (moved into
the pcf.generic's initializer). The parent's leading args are
available in the body and can be used by `init_subscope`. The child's
initializer is populated solely from `init_subscope` bodies. These are
independent.

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
workers can cooperate on allocation). Telescoping happens inside
`run_thread` bodies where thread IDs are available. The two ops
reflect this temporal separation: allocate collectively first, then
telescope per-thread.

**Why is `telescope` a pure op?** It is a type conversion, not an
execution context change. It converts a parent-scope threadgroup
identity into a child-scope identity using the thread ID. The
execution context change happens when the lowering wraps the body in a
child shared_executor. Making `telescope` pure means it composes
freely with other ops and can be moved/CSEd by standard passes.

**Why does the entire body move into the child shared_executor?** The
parent scope is fully distributed at this point. The `pcf.generic`
provides the structural wrapper for parent-scope distributed
execution. Everything inside is child-scope work that will be
distributed in the next phase.

**Why does `telescope` take a thread ID operand instead of getting it
from the scope?** The thread ID is proof that the caller is in a
distributed context. Requiring it as an explicit operand prevents
accidentally telescoping in undistributed code. Passing a value that
is not the actual thread ID is undefined behavior.

**Why drop `uniform` from ClusterType?** The semantics of uniform
struct fields are not yet well-defined. Removing them simplifies the
type system. They can be re-added when a clear use case emerges.

**Why must `init_subscope` input have no struct fields?** Multiple
chained `init_subscope` ops would complicate the lowering (which
init_subscope body goes into which initializer slot?). A single
`init_subscope` yielding all allocations is simpler and sufficient.

## Testing

**Roundtrip tests:**
- `init_subscope` with single and multiple allocations.
- `telescope` with single and multiple struct fields.

**Verifier tests:**
- `init_subscope` on threadgroup with existing struct fields (error).
- `telescope` where struct field types do not match result types.
- Scope mismatch between input and output of `init_subscope`.

**Lowering tests:**
- shared_executor with init_subscope + telescope lowered to
  pcf.generic wrapping child shared_executor.
- Two-level telescoping: workgroup → subgroup → lane pipeline.
- Integration with tile_group: init_subscope + telescope inside a
  tile_group body with run_thread.

**Negative lowering tests:**
- `telescope` without preceding `init_subscope`.
- `init_subscope` yielded types not matching `telescope` struct fields.

**Related change tests:**
- ClusterType without `uniform` field (update existing tests).
- `run_thread` with threadgroup-typed private struct field.
