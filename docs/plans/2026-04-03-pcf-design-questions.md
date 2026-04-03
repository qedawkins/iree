# PCF Dialect Design Questions

Reviewer: Claude (semantics and design review)
Branch: `shared-exec-rebase`
Date: 2026-04-02

---

## 1. sref Synchronization Model

### Q1: Readonly srefs have no sync scope -- what prevents write-after-read hazards?

**Observed**: When `pcf.generic` is built with `readonlyInits`, the builder creates sref block arguments with **no sync scope** (`PCFOps.cpp:728-734`). Readwrite srefs get `SyncOnReturn` (`PCFOps.cpp:737-743`). The comment says "no sync scope needed since they are never written."

**Issue**: The claim "they are never written" is only true within this particular `pcf.generic` body. But the *backing tensor* can be aliased by a readwrite sref in a sibling or parent scope. Consider:

```
%t = ... : tensor<128xf32>
%r1 = pcf.generic scope(#sg)
  execute(%rw = %t)[...] : (!pcf.sref<..., sync(#sg)>) -> tensor<128xf32> {
    // Worker A writes to %rw...
    pcf.return
  }
// %r1 is the snapshot of %rw after all workers return.
// But what if another pcf.generic is nested or runs concurrently?
```

The no-sync semantics on readonly are safe **only if** the source tensor is immutable for the duration of the pcf.generic's execution. This is true for SSA tensors (value semantics), but what happens after bufferization when tensors become memrefs? A readonly sref backed by a memref could race with a concurrent writer.

**Decision needed**: Is the contract that readonly srefs always alias immutable data (guaranteed by the value-semantic tensor origin), and that bufferization must preserve this property (e.g., via aliasing analysis ensuring no concurrent write)? If so, this invariant needs to be documented. If not, readonly srefs need at least a read-fence on scope exit.

**ANSWER**:

Readonly is a contract with the body of the pcf.generic stating that we will never write to that memory. For bufferization, this just means setting read effects on the readonly inputs and readwrite side effects on the tied outs.

You are correct though, technically there does need to be a read fence on scope exit. Right now there is no suitable way to do this and we don't worry about read fencing anywhere else (typically reads are fenced if they need to wherever their results are used so we in practice never really have to). So in short, no need for the read fence nor anything special from bufferization.

### Q2: SyncOnReturn semantics are described but underspecified for nested scopes

**Observed**: `SyncOnReturnAttr` description says "fenced when the parent of the same scope returns. This is akin to memory order acquire on scope entry and __syncthreads followed by a memory order release fence on scope exit" (`PCFBase.td:346-354`). The `enqueueWrite` method is a no-op. The `getConcreteTypes` returns `TypeRange()`, meaning expansion drops the sync scope entirely.

**Issue**: Consider a nested case:
```
pcf.generic scope(#sg) execute(%rw_outer = %t) {
  pcf.generic scope(#lane) execute(%rw_inner = ...) {
    // Write to %rw_inner -- when does this become visible to outer scope?
    pcf.return  // <-- lane scope return: barrier only among lanes
  }
  // Is %rw_inner visible here? Only if lane-level barrier happened.
  // But %rw_outer is subgroup-scoped -- needs subgroup barrier.
  pcf.return  // <-- subgroup scope return
}
```

The SyncOnReturn is tied to "the parent of the **same** scope." But the inner generic has a *different* scope than the outer. What guarantees ordering between nested scopes? The code resolves tokens by simply dropping them (`getConcreteTypes` returns empty). Does lowering insert the right barriers, or must the user manually insert barriers between scope levels?

**Decision needed**: Clarify the interaction between SyncOnReturn at different scope levels. Specifically: when an inner-scope pcf.generic returns, does its SyncOnReturn guarantee visibility to the enclosing outer-scope body? Or must explicit sync be added between nesting levels?

**ANSWER**:

The lowering is responsible for inserting the barriers. There is nothing underspecified about nested scopes. In your example if %rw_inner has SyncOnReturn then when the lane generic returns all lanes in the subgroup have access to the value. %rw_outer and %rw_inner have absolutely nothing to do with one another in your example so I don't understand why you're talking about them both. Also %rw_inner is a block argument to the inner generic so the outer generic can't actually access it directly. You are missing the value that the nested pcf.generic returns that represents the collective value written to %rw_inner. TLDR; your question is ill formed.

### Q3: pcf.generic is not IsolatedFromAbove -- implicit capture of srefs across scope boundaries

**Observed**: `pcf.generic` does **not** have the `IsolatedFromAbove` trait. The code explicitly relies on this: inner generics "capture outer srefs directly" (`MultiLevelTiling.cpp:411`). The `DistributeAndLower.cpp:611` comment confirms this is intentional.

**Issue**: This means the inner (lane-scope) generic can freely reference sref block arguments of the outer (subgroup-scope) generic. This works because srefs are reference types, not value types. But it creates an implicit data dependency that is invisible in the op's operand list. Questions:

1. How does the verifier ensure that a captured sref's scope is compatible with the capturing scope? For instance, is it legal for a lane-scope body to capture a workgroup-scoped sref? The verifier only checks that the *body's own* block argument srefs match the body's scope (`PCFOps.cpp:100-111`), not captured values.

2. How does bufferization handle captured values? The bufferization model (`BufferizationExternalModels.cpp`) only analyzes the `inits` operands, not implicitly captured values.

3. How does the sref-to-memref conversion track the lifetime of an sref that is defined in an outer scope but used in an inner scope?

**Decision needed**: Should there be verifier constraints on cross-scope sref capture? Should captured srefs be required to appear as explicit operands (perhaps with a "capture" keyword) to make the data flow visible for analysis passes?

**ANSWER**:

Obviously it's valid for inner generics to access the srefs from outer ones???? An sref is just memory. There absolutely 1000% should not be any restrictions of cross-scope stuff.

So for 1 just completely forget you ever thought this. There is absolutely 0 verification of captures and never will be.

For 2 if I'm understanding your question, bufferization might be missing handling of readonly inputs. This is a big problem if so. Otherwise bufferization doesn't need to know anything special about captured values.

For 3 there is never lifetime tracking. Bad question.

---

## 2. pcf.generic vs pcf.shared_executor vs pcf.loop -- Abstraction Boundaries

### Q4: Three parallel constructs with overlapping semantics

**Observed**: The dialect has three parallel execution ops:
- `pcf.generic` -- spawns workers at native parallelism, SIMT, workers get IDs
- `pcf.loop` -- explicit iteration count, maps iterations to workers
- `pcf.shared_executor` -- collective/SIMD semantics, workers cooperate

**Issue**: The boundary between `pcf.generic` and `pcf.loop` is unclear. `pcf.generic` with `num_iterators=1` and a scope whose `getNativeNumProcessorIds()` returns 1 (e.g., `#pcf.sequential`) effectively becomes `pcf.loop count(1)`. More critically:

- `pcf.generic` creates both id **and count** block arguments (2 * numIterators). The count args are the `nproc` values from the scope. But `pcf.loop` only creates id args (one per count operand). This asymmetry is confusing.
- The MultiLevelTiling pass only uses `pcf.generic`, not `pcf.loop`. When would `pcf.loop` be preferred?

**Decision needed**: What is the intended use case for `pcf.loop` vs `pcf.generic`? Is `pcf.loop` for cases where the iteration count differs from the hardware parallelism (oversubscription / undersubscription)? If so, should `pcf.generic` be thought of as syntactic sugar for `pcf.loop count(scope.nproc)`? Clarify the design intent.

**ANSWER**:

Totally backwards. pcf.loop is the syntactic sugar for pcf.generic(scf.forall <spillover iterations>). The fundamental ops are `pcf.generic` and `pcf.shared_executor`.

### Q5: shared_executor has a threadgroup type but generic/loop do not

**Observed**: `pcf.shared_executor` introduces `!pcf.threadgroup<#scope>` as a block argument, giving the body a handle to the collective worker group. `pcf.generic` and `pcf.loop` do not have threadgroup arguments; they expose raw index IDs instead.

**Issue**: The threadgroup handle is then destructured via `pcf.shared_executor.tile_group` into `!pcf.cluster` types, which carry bounds, private/shared struct fields, and hierarchical IDs. This is a much richer abstraction than the flat ID model in `pcf.generic`. But the tiling pipeline (`MultiLevelTiling.cpp`) generates `pcf.generic` ops, not `pcf.shared_executor` ops.

Questions:
1. Is the intended flow: MultiLevelTiling produces pcf.generic, then DistributeAndLower converts to pcf.shared_executor? Or are they parallel paths for different architectures?
2. If pcf.shared_executor is the "real" target for GPU codegen, why does MultiLevelTiling generate pcf.generic instead?

**Decision needed**: Clarify the relationship between the two abstraction levels. Is pcf.generic an intermediate IR that gets lowered to pcf.shared_executor? Or do they coexist for different use cases?

**ANSWER**:

Wrong ordering. pcf.shared_executor converts to pcf.generic. MultiLevelTiling is a shortcut that skips shared_executor. This is a result of the way that TileAndFuse works with the way it uses tiling (and thus local transforms) as a way to distribute.

---

## 3. Namespace Symbol Mechanism

### Q6: IndexSymbolOp definitions are placed in the initializer region, but resolution semantics are not formalized

**Observed**: `pcf.index_symbol` defines a named symbol in the initializer region of a `pcf.generic`. `NamespaceOpInterface` provides `getDefinedSymbols()`. Symbols "resolve upward through the parent chain -- never sideways to siblings" (`PCFInterfaces.td:209-211`).

**Issue**: The resolution algorithm is described informally but has no formal specification:
1. What happens if two nested namespaces define the same symbol name? Does the inner definition shadow the outer?
2. Can a symbol defined in the initializer reference symbols from a parent namespace?
3. The `PromoteOperandOp` takes a `symbols` array attribute with string names like `"operand_0_dim_0"`. How are these resolved? By walking up the namespace chain? What if the resolution fails at lowering time?

**Decision needed**: Specify the symbol resolution algorithm formally. In particular: shadowing rules, error behavior for unresolved symbols, and whether resolution is purely syntactic (name matching) or has scope/type constraints.

**ANSWER**:

1 is a great question. If this is not properly documented let's make sure to do so. Symbol resolution happens via first match from inner namespace to outer. So if two namespaces define the same symbol then it will match with the inner namespace.
The way that symbols can further specify which definition they are using is by adding namespace qualifiers the same way we do in C++. So if the inner namespace's name is "foo" and the outer is "bar", we can reference the overloaded symbol of
specifically the outer namespace with "bar::<sym_name>".

2 yes, no reason why not.

3 I hope they aren't string attributes. We need a formal symbol attribute type complete with definition lookup infrastructure separate from any specific op implementations. Resolution failure is a compiler failure. That means the IR was ill formed.
They are resolved using the standard algorithm for finding namespace symbol definitions. The scope/type constraints can be imposed by the PromoteOperandOp which requires that its symbol definitions come from a parent with a scope matching the
input sref.

### Q7: Symbol names in MultiLevelTiling are generated mechanically with no collision avoidance

**Observed**: `MultiLevelTiling.cpp:144-146` generates symbol names like `"operand_" + std::to_string(i) + "_dim_" + std::to_string(d)`. The same naming scheme is used in `PromoteOperandOp` creation (`MultiLevelTiling.cpp:147`).

**Issue**: If multiple tiling levels or multiple promote ops use the same naming scheme, there could be collisions. The code does not check for uniqueness. Also, the names are human-readable but brittle -- they encode operand indices that may change after fusion or other transforms.

**Decision needed**: Should symbol names be gensym'd (guaranteed unique) or is the current scheme intentionally stable for debugging? If stable, what happens when operand indices change after fusion?

**ANSWER**:

Yes, on creation we should be taking steps to avoid collisions. Instead of `operand_i_dim_d` just do `d0`, `d1`, ..., `dn` incrementing dims. The easiest way to avoid collisions is to fully qualify the parent namespace scope and then add new symbols to the parent uniqued. Then when creating a new child scope ensure that you pick a different namespace name (for that we can just use incrementing `n0`, `n1`, ..., `nn`).

---

## 4. PCFTilingOpInterface Design

### Q8: getDistributedImplementation mixes analysis and mutation

**Observed**: `getDistributedImplementation` takes an `OpBuilder` and creates new ops (reads, writes, clones). The caller (MultiLevelTiling) builds up `DistributedOperandInfo` and `DistributedResultInfo` describing how operands/results should be handled.

**Issue**: The interface design couples the "what to do" decision (analysis) with the "do it" action (IR mutation) in a single call. This makes it impossible to:
1. Query the tiling behavior without modifying IR (e.g., for cost modeling).
2. Validate the tiling plan before applying it.
3. Roll back a failed tiling attempt (the builder has already created ops).

This matters because `MultiLevelTiling.cpp:230-235` calls `getDistributedImplementation` inside a reduction loop and checks `if (failed(...))` -- but by then, ops may have been created but not cleaned up.

**Decision needed**: Is the current design acceptable given that MLIR rewriters handle rollback? Or should getDistributedImplementation be split into a query phase (returns a plan) and an apply phase?

**ANSWER**:

Uh oh, that is a very good catch. Mutating IR then returning failure is an extremely egregious implementation mistake. MLIR rewriters do NOT handle rollback. That is only in a legacy implementation of dialect conversion. So yes, we need to
split into a match + rewrite.

### Q9: getIterArgTypes / emitInitTileLoad / emitResultTileStore are tightly coupled to MultiLevelTiling

**Observed**: These three interface methods (`PCFInterfaces.td:363-412`) exist specifically to support the reduction loop pattern in `MultiLevelTiling.cpp`. They take `MultiLevelTilingParams` as arguments.

**Issue**: This means PCFTilingOpInterface has a hard dependency on `MultiLevelTilingParams`, which is a struct specific to the multi-level tiling strategy. This makes the interface unusable for:
1. Single-level tiling
2. Alternative tiling strategies (e.g., wavefront tiling, software pipelining)
3. Ops that have non-standard reduction patterns (e.g., prefix sum)

The interface is supposed to be generic ("Interface for distributing ops into PCF parallel constructs"), but the methods leak the multi-level tiling strategy into the interface.

**Decision needed**: Should the methods be moved out of PCFTilingOpInterface into a separate `MultiLevelTilingInterface`? Or is the current design acceptable because multi-level tiling is the only intended consumer?

**ANSWER**:

This is a good question and you're example alternative paths is a good list. This might mostly be a naming thing, instead of saying `emitReductionWriteback` we can remove the notion of reductions from the names and just call them `emitInitTileLoad` and `emitResultTileStore`. Note that these methods could do a lot more than just write data back, they can do post-loop cleanup too if they want. For non-standard reduction patterns, implementations are always free to not have a loop carried
variable and do the reduction in-place (so `getIterArgTypes` would return empty). And for 1 and 2 the interface still functions perfectly fine for those.

### Q10: DistributedOperandInfo.isTile semantics when value is an sref

**Observed**: `DistributedOperandInfo` has `Value value` and `bool isTile`. In `LinalgOpDistributedTiling.cpp:131`, the implementation checks `if (info.isTile && isa<RankedTensorType>(operandType))` to decide whether to use the value directly.

**Issue**: What does `isTile=true` mean when `value` is a `pcf.sref`? The code at `MultiLevelTiling.cpp:128-129` sets `isTile=true` for DPS init operands from iter_args (which are tensors), and `isTile=false` for inputs from readonly srefs. But `isTile=false` with a tensor value is also used for non-tileable operands (`MultiLevelTiling.cpp:121-122`). The semantics are overloaded:

- `isTile=false, value=tensor` means "pass the full tensor"
- `isTile=false, value=sref` means "read a tile from the sref using indexing map"
- `isTile=true, value=tensor` means "this is already a tile-sized tensor"
- `isTile=true, value=sref` means... what? Never happens currently?

**Decision needed**: Should `DistributedOperandInfo` use an enum instead of a bool to disambiguate these cases? E.g., `{FullValue, TensorTile, SrefToSlice, SrefDirect}`.

**ANSWER**:

`isTile=true` and `value=sref` shouldn't happen at the moment. We can assume this never happens (with an assert if implementers want) for the time being.

---

## 5. Promotion Design

### Q11: PromoteOperandOp operates on srefs but is created during tiling (before bufferization)

**Observed**: `iree_gpu.promote_operand` takes a `pcf.sref` source and produces a `pcf.sref` result at a different scope. In `MultiLevelTiling.cpp:148-156`, it is created during multi-level tiling. But at tiling time, the operands are tensors, not srefs. The sref comes from the block argument of the enclosing `pcf.generic`.

**Issue**: The promotion op is created in the right place (inside the subgroup-scope generic), but it introduces a scope transition: `!pcf.sref<..., #wg_scope> -> !pcf.sref<..., #sg_scope>`. This scope change has implications:
1. The source sref has workgroup scope, meaning all subgroups can see it. The destination sref has subgroup scope, meaning only one subgroup sees it. This implies a copy from global/shared memory to LDS.
2. But the actual copy is deferred to the `emitDistributedCopy` method on `PromotionAttr`. When does this method get called? It's not called during tiling.
3. How does the lowering pipeline know to invoke `emitDistributedCopy`? Is there a separate pass?

**Decision needed**: Document the complete lifecycle of `PromoteOperandOp`: when is it created, when is it lowered (who calls `emitDistributedCopy`), and what IR does it produce. Is there a separate "LowerPromotion" pass, or is it folded into DistributeAndLower?

**ANSWER**:

We should not be documenting lifecycle in terms of specific patterns and passes on the ops. That is lazy documentation. The semantics of an operation should stand on its own without any reference to code.

To answer your question, yes, there is a pass for lowering promotions (or at least there should be. Maybe the agent responsible for that task took a massive shortcut).

### Q12: PromotionAttr interface only has promoteOperand and emitDistributedCopy -- missing analysis methods

**Observed**: `PromotionAttr` (`IREEGPUInterfaces.td:114-168`) has two methods:
- `promoteOperand`: returns the promoted value (for the forall-based pipeline)
- `emitDistributedCopy`: emits the copy for the PCF pipeline

**Issue**: There are no methods for:
1. Querying the required allocation size (for shared memory budgeting)
2. Querying the memory space of the promotion target
3. Querying whether promotion is legal for a given operand shape
4. Querying the expected copy bandwidth (for cost modeling)

This means the MultiLevelTiling pass cannot reason about whether promotion will fit in shared memory before committing to it.

**Decision needed**: Should `PromotionAttr` have analysis methods (`getAllocationSize`, `getTargetMemorySpace`, `isLegalForShape`)? Or is the intent that these checks happen before `MultiLevelTilingParams` is constructed?

**ANSWER**:

Yeah none of that is the responsibility of the promotion attr. If you tile with a tile size that doesn't fit within the resource constraints you're going to get a failure.

`getTargetMemorySpace` maybe but not needed yet.

---

## 6. Type System and Memory Model

### Q13: sref scope mismatch between pcf.generic and its contents is not verified

**Observed**: The verifier (`PCFOps.cpp:100-111`) checks that readonly and readwrite sref block arguments have the same scope as the `pcf.generic` op's scope attribute. But it does **not** verify:
1. That `pcf.read_slice` / `pcf.write_slice` sources have compatible scopes with the enclosing op.
2. That `pcf.alloc` results have a scope compatible with where they are used.
3. That a `pcf.to_sref` result scope matches the enclosing parallel scope.

**Issue**: Nothing prevents creating an sref at scope A and using it inside a `pcf.generic` at scope B. The IR would type-check but the sync semantics would be wrong.

**Decision needed**: Should scope compatibility be verified structurally (verifier checks that all sref ops inside a scope match that scope) or is it left to the analysis passes (ConvertSRefToMemRef)?

**ANSWER**:

Nope. In a few of these questions you seem to have the idea that ops can verify things about their contents. That's wrong. `read_slice` and `write_slice` are completely allowed to to access coarser or finer grain scopes. Verification of only the parent would be totally wrong. At most we could have a verifier on the alloc that the parent is the initializer (probably should add that actually) and has the same scope.

### Q14: ClusterType bounds use AffineMap but scope sizes are runtime values

**Observed**: `!pcf.cluster` bounds use `AffineExpr` with `s0, s1, ...` as "scope size symbols, implicit from the scope's getWorkerCounts" (`PCFBase.td:230-231`). The `d0, d1, ...` are "dependent variables" that add SSA index operand requirements.

**Issue**: The scope size symbols are conceptually constants for a given hardware target, but `getWorkerCounts` returns `SmallVector<Value>` -- runtime values. If the cluster's `boundsMap` references `s0` and `s0` is runtime-determined:
1. How does static analysis (e.g., tile size computation) work with symbolic bounds?
2. What happens if `s0` changes between two uses of the same cluster type? (It shouldn't, but the type system doesn't enforce this.)

**Decision needed**: Are scope size symbols always statically known at compile time (constant-folded)? If so, should they be encoded as integer parameters instead of symbolic affine expressions? If they can be dynamic, how does static analysis handle them?

**ANSWER**:

Not sure I follow what you mean by "how does static analysis handle them" but this sounds like you are imposing the logic of tiling onto clusters when they are separate concepts. The fact that tile sizes are static everywhere is a bug, there is
no reason they can't be dynamic and depend on the value of `s0`. Also `s0`'s value is enforced by the type system. The type carries the scope and the scope defines the value of `s0` so it is fully defined by the type itself. If there is an
implementation detail differing from what I'm saying that probably needs updating.

### Q15: pcf.get_memref breaks the sref abstraction

**Observed**: `pcf.get_memref` (`PCFOps.td:1095-1181`) "extracts a memref view from a slice of a sref, breaking the synchronization guarantees of the source."

**Issue**: This op is an explicit escape hatch from the sref sync model. Questions:
1. When is this op expected to be used? Only during lowering (ConvertSRefToMemRef), or also during tiling/codegen?
2. If used during codegen, how does the user ensure synchronization? The sync guarantees are "broken" per the description.
3. The returned memref has "maximally dynamic layout (all strides and offset dynamic) and no memory space." This means every subsequent memref operation must handle dynamic strides. Is this intentional for generality, or should the memref carry the layout determined by the sref's scope?

**Decision needed**: Is `pcf.get_memref` intended as a lowering-only op (used by ConvertSRefToMemRef) or as a user-facing escape hatch? If lowering-only, should it be restricted to appear only in the ConvertSRefToMemRef pass?

**ANSWER**:

It shows up during bufferization too if read_slice doesn't vectorize. No need to restrict it to one pass.

---

## 7. MultiLevelTiling Algorithm

### Q16: Only one reduction dimension is supported for non-MMA paths

**Observed**: `MultiLevelTiling.cpp:451-453`:
```cpp
if (!params.mmaKind && reductionDims.size() > 1) {
  return rewriter.notifyMatchFailure(
      target, "multiple reduction dimensions not yet supported");
}
```

**Issue**: Ops like `linalg.softmax`, batch-matmul variants, or custom reductions with multiple reduction dims will fail. Is this a known limitation or a design constraint?

**Decision needed**: Is multi-reduction-dimension support planned? If so, what is the tiling strategy -- nested `scf.for` loops? Or is this deliberately excluded because multi-reduction ops should be decomposed first?

**ANSWER**:

Oh, the implementer only supporting a single reduction dim is taking a massive shortcut. We absolutely need multi-dim support. There are helpers for creating scf.for loop nests, you should use that.

To test it we can test tiling of conv. We should include an mma test for conv where we tile the filter dims down to unit dims.

### Q17: Inner pcf.generic is created with 1 iterator even when numLaneIterators is 0

**Observed**: `MultiLevelTiling.cpp:416-418`:
```cpp
GenericOp innerGeneric =
    numLaneIterators > 0
        ? GenericOp::create(rewriter, loc, laneScope, numLaneIterators)
        : GenericOp::create(rewriter, loc, laneScope, /*numIterators=*/1);
```

When there are no lane-tiled dimensions, it still creates a generic with 1 iterator.

**Issue**: If no lane tiling is needed, why create a lane-scope generic at all? The inner generic with 1 iterator still spawns `nproc` workers, each getting an ID. But the body uses the full subgroup tile (`laneOffsets = sgOffsets`, `laneSizes = sgSizes`), meaning every worker executes the same computation on the same data. This is **redundant execution**.

**Decision needed**: Is this intentional (every lane does the same work, which is correct for SIMD semantics)? Or should the inner generic be omitted entirely when `numLaneIterators == 0`? If intentional, this needs a comment explaining why redundant execution is acceptable.

**ANSWER**:

Few things going on here. So first if there is subgroup tiling specified and no mechanism for lane tiling specified (either lane tile sizes or an mma_kind) we should just fail. That's an ill formed lowering config.

Second if we do for some reason need to create the lane scoped generic but it's doing redundant work, what this line looks like it's doing is wrong. We must mask off (via scf.if) the lanes not active rather than repeating work. Repeating work like
this indicates whoever implemented this code can't be trusted to have actually understood parallel executions semantics properly, well done asking the question.

Most likely what's missing is an `scf.forall` inside the pcf.generic + pcf.lane nest that handles spillover and masking automatically by mapping the native parallelism to the tile count. This should be generated in all cases.

### Q18: MMA path attaches mma_kind as a raw string attribute, not a verified type

**Observed**: `MultiLevelTiling.cpp:462-463`:
```cpp
tiledOp->setAttr("mma_kind", mmaKind);
```

The `mmaKind` attribute is set directly on the tiled op using `setAttr`.

**Issue**: This is a generic `Attribute` set with a bare string key. There is no verification that:
1. The tiled op actually supports the `mma_kind` attribute.
2. The attribute value is a valid `MmaInterfaceAttr`.
3. The attribute survives canonicalization or other transforms that clone the op.

**Decision needed**: Should `mma_kind` be a defined attribute on the tiled op (e.g., via an interface), rather than a duck-typed attribute? Or is this a temporary mechanism that gets consumed immediately by the next pass?

**ANSWER**:

Oh, looks like the agent implementing this took a shortcut. Not ok. Handling of `mma_kind` is NOT tiling. Let me repeat. It is NOT tiling. It is a completely different class of conversion, but it still MUST happen in one shot. Setting discardable attributes to do it in a separate step is wrong. Instead there should be separate handling for inner_tiled attribute descriptors. It has to compose between the type derived from the subgroup level though. No hard coding.

Also the fact that this file is in PCF/Transforms is insanely wrong. This logic is all GPU specific. Should be in Common/GPU like I originally said. No idea how it ever ended up where it is.

---

## 8. Bufferization and Lowering Path

### Q19: Two-phase lowering: bufferization converts tensors to memrefs, ConvertSRefToMemRef converts srefs to memrefs

**Observed**: The dialect doc (`PCFBase.td:88-96`) describes a three-phase lowering:
1. Token resolution (drop sync scopes)
2. sref -> memref conversion
3. PCF structural ops -> scf/cf

Bufferization (`BufferizationExternalModels.cpp`) handles tensor->memref for the *operands* of pcf.generic (the inits). But the *block arguments* are srefs, not tensors.

**Issue**: After bufferization, the inits are memrefs, but the block arguments are still srefs. The block argument types are set at construction time and contain the scope. So we have:
```
%result = pcf.generic scope(#sg) execute(%ref = %init_memref)
    : (!pcf.sref<..., #sg>) -> (memref<...>) { ... }
```

This means there's a semantic gap: the operand is a memref, the block argument is an sref. How does the sref block argument get connected to the memref? ConvertSRefToMemRef must bridge this gap. But the bufferization happens first.

**Decision needed**: Is the interaction between bufferization and ConvertSRefToMemRef fully specified? In particular: does bufferization update the block argument types, or does it leave them as srefs? If it leaves them, how does ConvertSRefToMemRef know the backing memref for each sref?

**ANSWER**:

Bufferization is only concerned with tensors. I don't understand how you are confused about this if the operand is a memref and the block arg is an sref. Those two are obviously tied together so ConvertSRefToMemRef can easily tie them together?

You seem fixated on bufferization doing everything. It just handles tensor -> memref. Plain and simple, nothing super fancy going on there.

### Q20: pcf.write_slice has MemoryEffectsOpInterface but pcf.read_slice does not

**Observed**: `pcf.write_slice` declares `DeclareOpInterfaceMethods<MemoryEffectsOpInterface>` (`PCFOps.td:914`). `pcf.read_slice` does **not** declare any memory effects interface (`PCFOps.td:1004`).

**Issue**: Without `MemoryEffectsOpInterface`, `pcf.read_slice` is assumed to be pure (no side effects). But reading from an sref is conceptually a memory read. If the compiler treats it as pure, it could:
1. CSE two reads from the same sref, even if a write happened between them.
2. Hoist a read out of a loop, missing updates from loop iterations.
3. Delete a "dead" read that actually has ordering significance.

**Decision needed**: Should `pcf.read_slice` declare memory read effects? If reads from srefs are intentionally pure (because srefs at the tensor level are value-semantic), this needs to be documented. If not, add the interface.

**ANSWER**:

Yes, it should declare memory read effects, good catch!

---

## 9. shared_executor and Cluster Subsystem

### Q21: ClusterType carries both private and shared struct elements but run_cluster/run_thread have asymmetric constraints

**Observed**:
- `run_cluster` "cannot have private elements (collective execution cannot produce per-thread data)" (`PCFOps.td:694-695`)
- `run_thread` "cannot have shared elements (per-thread execution cannot produce collective data)" (`PCFOps.td:765-767`)
- But `ClusterType` itself can carry both `privateTypes` and `sharedTypes` simultaneously.

**Issue**: A `!pcf.cluster` with both private and shared elements cannot be produced by either `run_cluster` or `run_thread`. It can only be consumed. So how does such a cluster get created? By composing a `run_cluster` result (shared-only) with a `run_thread` result (private-only)? There is no op that merges two clusters.

**Decision needed**: Can a cluster ever have both private and shared elements simultaneously? If yes, what op creates it? If no, should the type system enforce this (separate types for private-only and shared-only clusters)?

**ANSWER**:

They cannot have both. I specified in an earlier discussion that they should be mutually exclusive so we can reduce it to a single `structTypes` and a flag for if they have private or shared semantics.

### Q22: TileGroupOp requires source to have no struct elements

**Observed**: `pcf.shared_executor.tile_group` description states: "The source must not have struct elements (destructuring with struct is currently illegal)" (`PCFOps.td:585-586`).

**Issue**: This means you cannot partition a threadgroup that carries shared state. You must do `tile_group` first, then `init_subscope` on individual clusters to attach state. But `init_subscope` takes a `!pcf.threadgroup`, not a `!pcf.cluster`. So the flow must be: tile_group (splits threadgroup into clusters) -> but wait, init_subscope cannot take clusters.

This seems like a chicken-and-egg problem: you need struct elements on clusters, but you can only add struct elements to threadgroups, and you can only tile threadgroups without struct elements.

**Decision needed**: How does shared state get attached to individual clusters? Is there an equivalent of `init_subscope` for clusters, or is the intended pattern different?

**ANSWER**:

You got the wrong idea about `init_subscope`. That has a different purpose than this. To get a struct element on a cluster you just do `run_thread` or `run_cluster` on the cluster and return the new struct elements. Then the `run_cluster` and `run_thread` ops return new cluster types.

### Q23: TelescopeOp is "pure type conversion" but has semantic implications

**Observed**: `pcf.telescope` is marked `[Pure]` and described as "a pure type conversion op" (`PCFOps.td:1367-1368`). It converts a parent-scope threadgroup to a child-scope threadgroup, extracting struct fields.

**Issue**: The description says "passing a value that is not the actual thread ID is undefined behavior" (`PCFOps.td:1379-1380`). An op with UB depending on runtime values cannot truly be "pure" in the MLIR sense. "Pure" means no side effects and safe to CSE/hoist/eliminate. But:
1. Two `pcf.telescope` calls with the same threadgroup but different `threadId` values should NOT be CSE'd (they produce different child threadgroups).
2. Hoisting a telescope out of a loop that iterates over thread IDs would be incorrect.

The `Pure` trait is correct only if the threadId is guaranteed to be the same SSA value in a given scope (i.e., not a loop induction variable). But nothing in the type system prevents loop-variant thread IDs.

**Decision needed**: Should `pcf.telescope` be `Pure`? It seems like it should have a weaker purity guarantee (e.g., `NoMemoryEffect` but not `Speculatable`), or the UB clause should be removed and replaced with a verifier check.

**ANSWER**:

Yes, removing speculatable is correct, can go ahead and do that.

---

## 10. Integration and Coexistence

### Q24: Can PCF-tiled and forall-tiled code coexist in the same dispatch?

**Observed**: The branch adds PCF-based tiling as an alternative to the forall-based TileAndFuse pipeline. But the code review scope did not include the pipeline selection logic.

**Issue**: If two ops in the same dispatch use different tiling strategies (one PCF, one forall), the lowering pipeline must handle both. But `DistributeAndLower` expects PCF ops, while the existing pipeline expects `scf.forall`. Can they interleave? Or must the entire dispatch be committed to one pipeline?

**Decision needed**: Is mixed PCF/forall in the same dispatch supported? If not, what enforces homogeneity?

**ANSWER**:

The choice of lowering to use is determined by the pipeline. If the input includes an scf.forall then we can just handle it like normal, nothing crazy.

### Q25: DmaCopyOp has no verifier and no memory effects

**Observed**: `iree_gpu.dma_copy` (`IREEGPUOps.td:434-491`) has no `hasVerifier = 1` and does not declare `MemoryEffectsOpInterface`. It copies between srefs.

**Issue**: Without memory effects, the compiler may:
1. Dead-code-eliminate a DMA copy whose result sref is "never read" (because reads are also effectless per Q20).
2. Reorder DMA copies with respect to computation.

Also, without a verifier, there's no check that source and dest have compatible element types, or that the offsets/sizes/strides are within bounds.

**Decision needed**: Should `dma_copy` have a verifier and memory effects? At minimum, it should declare write effects on the destination sref.

**ANSWER**:

Yes, definitely needs read and write memory effects.

---

## Summary of Priority Decisions

| # | Topic | Priority |
|---|-------|----------|
| Q1 | Readonly sref safety after bufferization | Critical |
| Q2 | Nested scope sync ordering | Critical |
| Q3 | Implicit sref capture verification | High |
| Q10 | DistributedOperandInfo overloaded semantics | High |
| Q17 | Redundant execution when numLaneIterators=0 | High |
| Q19 | Bufferization <-> ConvertSRefToMemRef interaction | High |
| Q20 | read_slice missing memory effects | High |
| Q25 | dma_copy missing verifier and effects | High |
| Q4 | generic vs loop distinction | Medium |
| Q5 | generic vs shared_executor relationship | Medium |
| Q8 | Analysis/mutation coupling | Medium |
| Q9 | Reduction methods leak MultiLevelTilingParams | Medium |
| Q11 | PromoteOperandOp lifecycle | Medium |
| Q13 | Scope mismatch not verified | Medium |
| Q23 | TelescopeOp purity | Medium |
| Q6 | Symbol resolution formalization | Low |
| Q7 | Symbol name collision | Low |
| Q12 | PromotionAttr missing analysis methods | Low |
| Q14 | Cluster bounds static vs dynamic | Low |
| Q15 | get_memref escape hatch | Low |
| Q16 | Single reduction dim limitation | Low |
| Q18 | mma_kind as duck-typed attribute | Low |
| Q21 | Cluster private+shared coexistence | Low |
| Q22 | TileGroup struct element chicken-and-egg | Low |
| Q24 | PCF/forall coexistence | Low |
