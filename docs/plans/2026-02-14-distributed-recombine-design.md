# Distributed Stream-K Recombine Design

**Date**: 2026-02-14
**Status**: Design
**Source**: ~/root/stream_k_dump_comments.txt (user's original design)
**Priority**: This document is the authoritative design. Treat it like your
entire life.

## Problem

The current `LowerStreamKRecombine` pass lowers `pcf.stream_k_recombine` to
a single-threaded implementation where only thread-0 performs the scratch
read, accumulation, and writeback. This is unacceptable for performance. The
recombine must be distributed to all threads, matching the distribution of
the producer.

## Overview

Instead of a standalone lowering pass, `pcf.stream_k_recombine` should be
handled via a **fusion pattern** that integrates the recombine into the
producer's distributed compute. The key insight is that different parts of
the recombine have different distribution requirements:

1. **Scratch write** — distributed (all threads write their portion)
2. **Atomic counter** — single-invocation (once per workgroup)
3. **Recombine accumulation** — distributed (via configurable pcf scope nest)
4. **Writeback** — distributed (fused into producer or into recombine)

## Part 1: Control-Flow-Aware Consumer Fusion

Before tackling `stream_k_recombine`, consumer fusion in PCF must properly
handle control flow. Currently, fusing a consumer that lives inside an
`scf.if` runs the fused code unconditionally. The fix:

**Mirror the consumer's control flow inside the `pcf.generic/loop`**, then
mark where the result becomes "real" with `pcf.guarantee_value`:

```mlir
// BEFORE fusion:
%cond = i1
%0 = pcf.generic { ... } -> tensor
scf.if %cond {
    %1 = first_user %0
    %2 = second_user %1
}

// AFTER fusing first_user:
%cond = i1
%0 = pcf.generic {
  scf.if %cond {
    %local_val = ...
    %fused = fused_first_user %local_val
    pcf.write_slice %fused
  }
} -> tensor
scf.if %cond {
    %1 = pcf.guarantee_value %0  // Value is valid here
    %2 = second_user %1
}
```

`pcf.guarantee_value` tells us the value is guaranteed valid IF AND ONLY IF
control flow reaches that point. This means fusing `second_user` does NOT
require creating another `scf.if` for the same condition. This recurses for
nested control flow (another if, else branch, switch, etc.).

**Status**: Partially implemented in issue 696. The `pcf.guarantee_value` op
exists, `createMirroredControlFlow` exists in `FuseConsumers.cpp`. May need
verification that it works for the else branch and recursive cases.

## Part 2: Stream-K Recombine Fusion Pattern

### The Problem with Naive Fusion

If we naively fuse the atomic into the distributed producer, the atomic
repeats for each subgroup. This breaks group tracking, especially when
specialization makes the actual number of workers opaque.

So at the block level (the scope where the op is introduced),
`stream_k_recombine` makes sense. Once we fuse it, it must be decomposed.

### Before Fusion

```mlir
%scratch_ref = ...
%0 = pcf.generic execute(%subgroup_ref: !pcf.sref) {
  %local = ... // Some subgroup local value
  pcf.write_slice %local to %subgroup_ref
}
pcf.stream_k_recombine %0
    into %scratch_ref [<offsets>] [<sizes>]
    scratch %arg0 counter %arg1[%arg2]
    group(%42)
    combiner { ... }
    writeback { ... }
```

### After Fusion

```mlir
%completeness_cond = // condition that the tile is NOT complete
%scratch_ref = ...

// NOTE: As soon as we fuse stream_k_recombine, we MUST set
// sync_on_return = true on the generic. This sync is for WORKGROUP
// memory (scratch writes), not subgroup memory.
%0 = pcf.generic sync execute(%subgroup_ref: !pcf.sref) {
  %local = ...

  scf.if %completeness_cond {
    // Write to scratch. The atomic has NOT happened yet since if we
    // do the atomic BEFORE writing to scratch, another workgroup could
    // jump in and load data before it's ready.
    pcf.write_slice %local to %scratch_ref
  }

  // Keep writing back like normal. Do NOT put this in an else branch
  // because there could be other users. We're only distributing the
  // scratch case above.
  pcf.write_slice %local to %subgroup_ref
}

// At this point all subgroups have finished their writes. Now do the
// atomic and check if we need to recombine.
scf.if %completeness_cond {
    // Partial tile branch. Must happen once per workgroup! Predicate
    // with scf.forall or pcf.generic nest where lane/subgroup ids = 0,
    // then broadcast result using a single dword of shared memory.
    %old = atomic_rmw ...
    %last_tile_cond = ...  // Are we the last contributor?
    if %last_tile_cond {
        // Do the recombine. This work must be distributed. Since this
        // happens after consumer fusion (phase ordering), the fusion
        // pattern must create a pcf scope nest (pcf.generic) to
        // distribute the recombine work. This is configurable.
        %reduced = ... // distributed recombine via pcf.generic

        // FIRST copy of writeback region goes here.
        // Will get fused with its producer (the recombine).
        writeback(%reduced)
    }
    // If not last, nothing happens.
} else {
    // SECOND copy of writeback region. This needs control-flow-aware
    // fusion into the producer pcf ops (Part 1 above).
    writeback(%0)
}
```

### Key Requirements

1. **sync_on_return = true** on the producer generic after fusion.
2. **Scratch write BEFORE atomic** — never do atomic before scratch is written.
3. **Atomic once per workgroup** — predicate on lane_id == 0 && sg_id == 0.
4. **Recombine is distributed** — the fusion pattern creates a pcf scope nest.
5. **Two copies of writeback** — one for last-contributor path, one for
   non-split path. The non-split writeback gets fused via Part 1.
6. **Existing write_slice stays unconditional** — other users may exist.

## Part 3: Pipeline Integration

### Required Pass Ordering

1. `iree-gpu-convert-forall-to-generic-nest` — converts `scf.forall` with
   thread mapping to nested subgroup + lane `pcf.generic` ops. **Must run
   before fusion** so fusion targets exist.

2. Greedy consumer fusion pass — runs all consumer fusion patterns including:
   - Standard tilable consumer fusion into pcf.generic/loop
   - Stream-K recombine fusion pattern (Part 2)
   - Control-flow-aware fusion (Part 1)

3. **Delete** `LowerStreamKRecombine` standalone pass — it's replaced by the
   fusion pattern.

### Pass That Needs to Exist

A greedy consumer fusion pass for `pcf.generic/loop` ops at subgroup scope.
This pass:
- Finds `pcf.generic` ops with subgroup scope
- Looks for fusable consumers (including `stream_k_recombine`)
- Applies fusion patterns greedily
- Handles control flow mirroring via `pcf.guarantee_value`

## Implementation Tasks

### Task 1: Delete LowerStreamKRecombine (issue ux3)
- Remove `LowerStreamKRecombine.cpp`
- Remove the pass from `Passes.td`, `BUILD.bazel`, `CMakeLists.txt`
- Remove from pipeline in `Passes.cpp`
- Delete `lower_stream_k_recombine.mlir` test
- Keep `stream_k_recombine` op definition (it's still needed)

### Task 2: Implement stream_k_recombine fusion pattern (issue rmv)
- New pattern in `FuseConsumers.cpp` (or new file)
- Recognizes `stream_k_recombine` consuming a `pcf.generic` result
- Decomposes into: distributed scratch write + barrier + atomic + distributed
  recombine + two writeback copies
- Sets `sync_on_return` on producer generic
- Creates configurable pcf scope nest for recombine distribution
- Tests: new test file exercising the fusion

### Task 3: Wire pipeline (issue lqa)
- Add `iree-gpu-convert-forall-to-generic-nest` before fusion
- Add greedy consumer fusion pass after generic nest conversion

### Task 4: Greedy consumer fusion pass (issue rr2)
- New pass that runs fusion patterns on subgroup-scope pcf.generic/loop
- Applies stream_k_recombine fusion + standard fusion + control-flow fusion
- Tests: integration test showing full chain

## Critical Design Constraints

- The atomic MUST NOT happen before scratch writes complete.
- The atomic MUST happen exactly once per workgroup (not per thread/subgroup).
- The recombine accumulation MUST be distributed (not single-threaded).
- The writeback MUST appear in BOTH branches (split and non-split).
- The non-split writeback MUST be fused via control-flow-aware fusion.
- If an agent hits a fundamental design issue, **bubble it up to mastermind**
  who will escalate to the user. Do NOT take shortcuts.
