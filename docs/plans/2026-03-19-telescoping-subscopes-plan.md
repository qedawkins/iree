# Telescoping Subscopes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable multi-level distribution (workgroup → subgroup → lane) through `init_subscope` and `telescope` ops, with lowering support for child-scope creation.

**Architecture:** Two new ops (`init_subscope`, `telescope`) enable scope transitions. The `DistributeAndLower` pass gains a child scope option: when present, `lowerSharedExecutor` wraps the body in a new child-scope `shared_executor` instead of a bare `pcf.generic`. Also removes the `uniform` struct kind from `ClusterType`.

**Tech Stack:** MLIR, C++, IREE PCF dialect, ODS/TableGen.

**Design spec:** `docs/plans/2026-03-19-telescoping-subscopes-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `PCF/IR/PCFBase.td` | Remove `uniformTypes` from ClusterType |
| `PCF/IR/PCFOps.td` | Add `InitSubscopeOp`, `TelescopeOp`; update `ClusterYieldOp` |
| `PCF/IR/PCFOps.cpp` | Parse/print/verify for new ops; remove uniform from ClusterYieldOp and run_cluster/run_thread verifiers |
| `PCF/IR/PCFTypes.cpp` | Remove uniform from ClusterType parse/print |
| `PCF/Transforms/DistributeAndLower.cpp` | Extend `lowerSharedExecutor` with child scope logic |
| `PCF/Transforms/Passes.td` | Add `childScope` pass option |
| `PCF/IR/test/*.mlir` | Remove all `uniform` references; add init_subscope/telescope roundtrip + verifier tests |
| `PCF/Transforms/test/*.mlir` | Add telescoping lowering tests |

---

## Chunk 1: Remove uniform (Task 1)

### Task 1: Remove uniform struct kind from ClusterType

This is a widespread change touching type definitions, op verifiers, yield parsing, and all test files.

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFBase.td`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.td`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.cpp`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/types.mlir`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/shared_executor.mlir`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/invalid.mlir`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/distribute_and_lower.mlir`

**Key changes:**

In `PCFBase.td`:
- Remove `OptionalArrayRefParameter<"Type">:$uniformTypes` from ClusterType parameters.
- Update `hasStructElements()` to remove `getUniformTypes()` check.

In `PCFOps.td` (ClusterYieldOp):
- Remove `Variadic<AnyType>:$uniformValues` operand.
- Remove `AttrSizedOperandSegments` trait (only one variadic left).
- Simplify description — no more `uniform(...)` syntax.

In `PCFOps.cpp`:
- ClusterYieldOp parse: remove `uniform(...)` keyword parsing. Just parse a flat type list.
- ClusterYieldOp print: remove `uniform(...)` printing.
- `verifyRunOp`: remove uniform type checking against result cluster. Only check private/shared types (for run_thread: private only; for run_cluster: shared only).

In `PCFTypes.cpp` (ClusterType parse/print):
- Remove `"uniform"` keyword handling from the struct group parser.
- Remove uniform group from printer.

In all test files:
- Remove all `uniform: {types}` from cluster type literals.
- Remove `uniform(...)` from cluster_yield ops.
- Update CHECK lines accordingly.

- [ ] **Step 1: Make all changes listed above**

Read each file carefully before modifying. The uniform type removal is mechanical but widespread. Search for `uniform` in each file and remove all references systematically.

- [ ] **Step 2: Build and test**

```bash
cmake --build /home/quinn/root/iree-shared-exec/iree-build --target iree-opt
ctest --test-dir /home/quinn/root/iree-shared-exec/iree-build -R "PCF|pcf" --output-on-failure
```

- [ ] **Step 3: Commit**

```
[PCF] Remove uniform struct kind from ClusterType
```

---

## Chunk 2: New ops (Tasks 2-3)

### Task 2: Add pcf.init_subscope op

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.td`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/shared_executor.mlir`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/invalid.mlir`

**Op definition (PCFOps.td):**

```tablegen
def InitSubscopeOp : PCF_Op<"init_subscope", [
    RecursiveMemoryEffects,
    SingleBlock,
    SingleBlockImplicitTerminator<"mlir::iree_compiler::IREE::PCF::YieldOp">,
  ]> {
  let summary = [{
    Allocate resources for a child scope and attach as struct fields.
  }];
  let description = [{
    Allocates memory for a child scope and attaches the allocations as
    struct fields to the threadgroup. The body contains allocation ops
    that produce values for the child scope. The yielded values become
    struct fields on the returned threadgroup.

    Semantics: run-once, uniform across all scope levels.

    The input threadgroup must have no struct fields. The output
    threadgroup has the same scope but with struct fields matching the
    yielded types.

    Example:
    ```mlir
    %tg2 = pcf.init_subscope %tg {
      %alloc = pcf.alloc() : !pcf.sref<128x64xf16, #child_scope>
      pcf.yield %alloc : !pcf.sref<128x64xf16, #child_scope>
    } -> !pcf.threadgroup<#parent_scope, {!pcf.sref<128x64xf16, #child_scope>}>
    ```
  }];

  let arguments = (ins PCF_AnyThreadGroup:$source);
  let results = (outs PCF_AnyThreadGroup:$result);
  let regions = (region MinSizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
```

Also update `PCF_YieldOp`'s `ParentOneOf` to include `InitSubscopeOp`.

**Parser/printer:** Follow the shared_executor initializer pattern. Parse the source threadgroup, a body region with `pcf.yield`, and `->` followed by the result threadgroup type.

**Verifier:**
- Input threadgroup must have no struct fields.
- Result threadgroup scope must equal input scope.
- Yielded types must match result threadgroup struct field types.

**Tests:**
- Roundtrip: single allocation, multiple allocations.
- Verifier errors: input with struct fields, scope mismatch, yield type mismatch.

- [ ] **Step 1: Add op definition to PCFOps.td and update YieldOp**
- [ ] **Step 2: Implement parse/print/verify in PCFOps.cpp**
- [ ] **Step 3: Add roundtrip tests to shared_executor.mlir**
- [ ] **Step 4: Add verifier tests to invalid.mlir**
- [ ] **Step 5: Build and test**
- [ ] **Step 6: Commit**

```
[PCF] Add pcf.init_subscope op
```

---

### Task 3: Add pcf.telescope op

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.td`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/shared_executor.mlir`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/invalid.mlir`

**Op definition (PCFOps.td):**

```tablegen
def TelescopeOp : PCF_Op<"telescope", [Pure]> {
  let summary = [{
    Convert a threadgroup identity from parent scope to child scope.
  }];
  let description = [{
    A pure type conversion op that takes a threadgroup with optional
    struct fields and an index operand (the thread ID). Returns a
    child-scope threadgroup (no struct fields) plus the struct fields
    as separate SSA values.

    The thread ID operand is the parent scope's worker ID. Passing a
    value that is not the actual thread ID is undefined behavior. In
    practice, this op appears inside `run_thread` bodies where the
    thread ID is available as a block argument.

    The child scope is determined by the result threadgroup type.

    Example with struct fields:
    ```mlir
    %child_tg, %sref = pcf.telescope %tg[%tid]
        : !pcf.threadgroup<#parent, {!pcf.sref<128x64xf16, #child>}>
       -> (!pcf.threadgroup<#child>, !pcf.sref<128x64xf16, #child>)
    ```

    Example without struct fields (pure scope conversion):
    ```mlir
    %child_tg = pcf.telescope %tg[%tid]
        : !pcf.threadgroup<#parent>
       -> !pcf.threadgroup<#child>
    ```
  }];

  let arguments = (ins PCF_AnyThreadGroup:$source, Index:$threadId);
  let results = (outs Variadic<AnyType>:$results);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
```

**Verifier:**
- First result must be a threadgroup with no struct fields.
- Remaining results must match the input threadgroup's struct field types (count and types).
- If input has no struct fields, there must be exactly one result (the child threadgroup).

**Tests:**
- Roundtrip: no struct fields, single struct field, multiple struct fields.
- Verifier errors: result count mismatch with struct fields, first result not threadgroup, struct field type mismatch.

- [ ] **Step 1: Add op definition to PCFOps.td**
- [ ] **Step 2: Implement parse/print/verify in PCFOps.cpp**
- [ ] **Step 3: Add roundtrip tests to shared_executor.mlir**
- [ ] **Step 4: Add verifier tests to invalid.mlir**
- [ ] **Step 5: Build and test**
- [ ] **Step 6: Commit**

```
[PCF] Add pcf.telescope op
```

---

## Chunk 3: Lowering (Tasks 4-5)

### Task 4: Extend lowerSharedExecutor with child scope

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/DistributeAndLower.cpp`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.td`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/distribute_and_lower.mlir`

**Changes to Passes.td:**
Add a child scope option:
```tablegen
Option<"childScope", "child-scope", "std::string",
       /*default=*/"\"\"",
       "Child scope for telescoping distribution.">,
```

**Changes to lowerSharedExecutor:**

Add a `ScopeAttrInterface childScope` parameter. When non-null:

1. Create `pcf.generic` for the parent scope (same as current).
2. Inside the generic body, create a new `SharedExecutorOp` with the child scope.
3. Walk the original body for `InitSubscopeOp` ops. For each:
   - Move the body ops into the child shared_executor's initializer region.
   - The yielded values become leading args in the child's execute region.
4. Move the entire original body into the child shared_executor's execute region.
5. Walk for `TelescopeOp` ops. For each:
   - Replace the child threadgroup result with the child shared_executor's threadgroup block arg.
   - Replace struct field results with the corresponding leading args.
   - Erase the telescope op.
6. Erase init_subscope ops (their values are replaced by the enriched threadgroup flowing through the body — but since the body is now in the child shared_executor, the init_subscope results are dead).

**Tests:**
- Basic: shared_executor with init_subscope + telescope → pcf.generic wrapping child shared_executor.
- Telescope without init_subscope (pure scope conversion).
- init_subscope without telescope (allocations attached but no scope transition yet).
- Two-level pipeline test (if feasible with test scopes).

- [ ] **Step 1: Add child-scope option to Passes.td**
- [ ] **Step 2: Implement child scope logic in lowerSharedExecutor**
- [ ] **Step 3: Add lowering tests to distribute_and_lower.mlir**
- [ ] **Step 4: Build and test**
- [ ] **Step 5: Commit**

```
[PCF] Add child scope support to DistributeAndLower pass
```

---

### Task 5: CMake regeneration and full verification

**Files:**
- Modify: BUILD.bazel files (if any new files added)
- Modify: CMakeLists.txt files

- [ ] **Step 1: Regenerate CMakeLists.txt**

```bash
cd /home/quinn/root/iree-shared-exec/iree
python3 build_tools/bazel_to_cmake/bazel_to_cmake.py \
  compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/BUILD.bazel \
  compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/BUILD.bazel \
  compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/BUILD.bazel
```

- [ ] **Step 2: Full build**

```bash
cmake --build /home/quinn/root/iree-shared-exec/iree-build
```

- [ ] **Step 3: Run full test suite**

```bash
ctest --test-dir /home/quinn/root/iree-shared-exec/iree-build -R "PCF|pcf|vector_distribut" --output-on-failure
```

- [ ] **Step 4: Run pre-submit**

```bash
/home/quinn/root/run_presubmit.sh
```

- [ ] **Step 5: Commit if needed**

```
[PCF] Regenerate CMakeLists.txt for telescoping subscopes
```
