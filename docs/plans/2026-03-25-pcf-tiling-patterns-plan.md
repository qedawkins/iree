# PCF Tiling and Fusion Patterns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement tile-to-PCF, consumer fusion, and producer fusion utilities using `PCFTilingOpInterface`.

**Architecture:** Three utility functions in `TileToPCF.cpp` with declarations in `Transforms.h`. Each uses `getDistributedImplementation` instead of the existing `TilingInterface` method combination. Shared template implementations over LoopOp/GenericOp.

**Tech Stack:** MLIR C++, PCF dialect, TilingInterface, PCFTilingOpInterface.

---

### Task 1: Add tile-to-PCF utility declarations to Transforms.h

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h`

- [ ] **Step 1: Add declarations**

Add after the existing `convertForallToGenericNest` declaration (~line 56):

```cpp
/// Tiles a PCFTilingOpInterface op into a pcf.loop with the given scope and
/// tile sizes. Tile size semantics match scf::tileUsingSCF: zero means don't
/// tile, non-zero is the tile size, iteration count = ceil(dim / tileSize).
/// Tileable input operands become readonly sref args, DPS inits become
/// readwrite sref args tied to results. Inside the body,
/// getDistributedImplementation is called with the tile offsets/sizes.
FailureOr<PCF::LoopOp> tileToPCFLoop(RewriterBase &rewriter,
                                      PCFTilingOpInterface target,
                                      ScopeAttrInterface scope,
                                      ArrayRef<OpFoldResult> tileSizes);

/// Same as tileToPCFLoop but creates a pcf.generic with nested scf.forall
/// for spillover iterations.
FailureOr<PCF::GenericOp> tileToPCFGeneric(RewriterBase &rewriter,
                                            PCFTilingOpInterface target,
                                            ScopeAttrInterface scope,
                                            ArrayRef<OpFoldResult> tileSizes);
```

Also add a forward declare for `PCFTilingOpInterface` near the top:

```cpp
class PCFTilingOpInterface;
```

- [ ] **Step 2: Commit**

### Task 2: Implement shared tile-to-PCF core

**Files:**
- Create: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/TileToPCF.cpp`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/BUILD.bazel` (add `TileToPCF.cpp` to srcs)
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/CMakeLists.txt` (add `TileToPCF.cpp` to SRCS)

The implementation follows the existing pattern: a template `tileToPCFImpl<OpTy>` that handles both LoopOp and GenericOp, with public functions `tileToPCFLoop` and `tileToPCFGeneric` delegating to it.

- [ ] **Step 1: Create TileToPCF.cpp with the shared implementation**

Key implementation steps within `tileToPCFImpl`:

1. Get iteration domain from `target.getIterationDomain()`.
2. Compute iteration counts: `count[i] = ceil(domain[i].size / tileSizes[i])` for non-zero tile sizes.
3. Classify operands using `target.getTileableOperandIndices()`:
   - Tileable non-DPS-init → readonly sref.
   - DPS init → readwrite sref (tied to result).
   - Non-tileable → passed through (not an sref arg).
4. Create the PCF op using the readonly+readwrite builder.
5. Inside the body, compute tile offsets/sizes:
   - `offset[i] = id[i] * tileSize[i]`
   - `size[i] = min(tileSize[i], domain[i].size - offset[i])`
6. Build `DistributedOperandInfo` per operand:
   - Tileable: `pcf.read_slice` from sref with computed offsets/sizes.
   - Non-tileable: original value, `isTile=false`.
7. Build `DistributedResultInfo` per result: readwrite sref as `destSref`.
8. Call `target.getDistributedImplementation(b, offsets, sizes, operandInfo, resultInfo)`.
9. For any non-null tiledValues, write via `pcf.write_slice`.
10. Replace original op results with the PCF op results.

Reference: The existing `convertForallToPCFLoop` in `ConvertForallToPCF.cpp` for how to create LoopOp/GenericOp and compute tile offsets. The existing `fuseIntoWriteSlices` in `FuseConsumers.cpp:355-441` for how to call tiling interface methods and create write_slice ops.

- [ ] **Step 2: Update BUILD.bazel**

Add `"TileToPCF.cpp"` to the `srcs` list in the `Transforms` library.

- [ ] **Step 3: Update CMakeLists.txt**

Add `"TileToPCF.cpp"` to the `SRCS` list.

- [ ] **Step 4: Build and verify compilation**

```bash
cd /home/quinn/root/iree-shared-exec/iree-build
cmake --build . --target iree_compiler_Codegen_Dialect_PCF_Transforms_Transforms.objects 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

### Task 3: Write tile-to-PCF lit tests

**Files:**
- Create: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/tile_to_pcf.mlir`

- [ ] **Step 1: Write test cases**

Test cases needed (using `#pcf.test_scope`):
1. Simple matmul-like linalg.generic tiled into pcf.loop — verify readonly srefs for inputs, readwrite sref for output, write_slice for result.
2. Same op tiled into pcf.generic — verify the generic+forall structure.
3. Op with non-tileable operand — verify it's passed through without sref.
4. Op with dynamic shapes — verify dynamic tile size handling.

Each test should use `iree-opt` with a test pass that calls `tileToPCFLoop`/`tileToPCFGeneric`. We may need a test pass; check if an existing test pass can be reused or if we need a new one.

Reference: Existing test files in `PCF/Transforms/test/` for the RUN line pattern and scope attributes.

- [ ] **Step 2: Run tests**

```bash
cd /home/quinn/root/iree-shared-exec/iree-build
ctest -R tile_to_pcf -j$(nproc) --output-on-failure
```

- [ ] **Step 3: Commit**

### Task 4: Implement consumer fusion with PCFTilingOpInterface

**Files:**
- Create: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/DistributedFuseConsumers.cpp`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h` (add declarations)
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/BUILD.bazel` (add source)
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/CMakeLists.txt` (add source)

- [ ] **Step 1: Add declarations to Transforms.h**

```cpp
// Distributed consumer fusion using PCFTilingOpInterface.
// Same match semantics as matchTilableConsumer but uses
// getDistributedImplementation and appends readonly sref args for
// non-fused input operands.
LogicalResult matchDistributedConsumer(RewriterBase &rewriter,
                                       PCF::GenericOp genericOp,
                                       PCFTilingOpInterface target,
                                       ConsumerFusionParams &params);
LogicalResult matchDistributedConsumer(RewriterBase &rewriter,
                                       PCF::LoopOp loopOp,
                                       PCFTilingOpInterface target,
                                       ConsumerFusionParams &params);

void fuseDistributedConsumer(RewriterBase &rewriter,
                              PCF::GenericOp genericOp,
                              PCFTilingOpInterface target,
                              ConsumerFusionParams &params);
void fuseDistributedConsumer(RewriterBase &rewriter,
                              PCF::LoopOp loopOp,
                              PCFTilingOpInterface target,
                              ConsumerFusionParams &params);
```

- [ ] **Step 2: Implement DistributedFuseConsumers.cpp**

Follow the structure of `FuseConsumers.cpp`. Key differences from `fuseIntoWriteSlices`:

1. For non-fused input operands of the consumer: append readonly sref args to the parent loop/generic using `addReadonlySrefArguments` (similar to `addSrefArguments` in FuseConsumers.cpp:443).
2. Build `DistributedOperandInfo`:
   - Fused-along operand: use write_slice source → `{value, isTile=true}`.
   - Non-fused tileable: `pcf.read_slice` from new readonly sref → `{value, isTile=true}`.
   - Non-tileable: original value → `{value, isTile=false}`.
3. Build `DistributedResultInfo`: new readwrite srefs.
4. Call `target.getDistributedImplementation(...)`.

- [ ] **Step 3: Update build files**

- [ ] **Step 4: Build and verify**

- [ ] **Step 5: Commit**

### Task 5: Write consumer fusion lit tests

**Files:**
- Create: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/distributed_fuse_consumers.mlir`

- [ ] **Step 1: Write test cases**

Mirror the existing `fuse_consumers.mlir` test cases but verify:
1. Readonly sref args appended for non-fused inputs.
2. `getDistributedImplementation` called (result written via write_slice).
3. Consumer's results become new results of the parent loop/generic.

- [ ] **Step 2: Run tests and verify**

- [ ] **Step 3: Commit**

### Task 6: Implement producer fusion with PCFTilingOpInterface

**Files:**
- Create: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/DistributedFuseProducers.cpp`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/BUILD.bazel`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/CMakeLists.txt`

- [ ] **Step 1: Add declarations to Transforms.h**

```cpp
struct DistributedProducerFusionParams {
  unsigned resultIndex;
  Operation *producer;
  SmallVector<PCF::ReadSliceOp> readSlices;
};

LogicalResult matchDistributedProducer(RewriterBase &rewriter,
                                        PCF::GenericOp genericOp,
                                        DistributedProducerFusionParams &params);
LogicalResult matchDistributedProducer(RewriterBase &rewriter,
                                        PCF::LoopOp loopOp,
                                        DistributedProducerFusionParams &params);

void fuseDistributedProducer(RewriterBase &rewriter,
                              PCF::GenericOp genericOp,
                              const DistributedProducerFusionParams &params);
void fuseDistributedProducer(RewriterBase &rewriter,
                              PCF::LoopOp loopOp,
                              const DistributedProducerFusionParams &params);
```

- [ ] **Step 2: Implement DistributedFuseProducers.cpp**

Follow the structure of `FuseProducers.cpp`. Key differences from `fuseTilableProducerImpl`:

1. For non-fused input operands of the producer: append readonly sref args to the parent.
2. At each read_slice site:
   - Build `DistributedOperandInfo` for producer's operands.
   - Build `DistributedResultInfo` with null destSref (return tile).
   - Call `producer.getDistributedImplementation(...)`.
   - Replace read_slice with the returned tile value.

- [ ] **Step 3: Update build files**

- [ ] **Step 4: Build and verify**

- [ ] **Step 5: Commit**

### Task 7: Write producer fusion lit tests

**Files:**
- Create: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/distributed_fuse_producers.mlir`

- [ ] **Step 1: Write test cases**

Mirror the existing `fuse_producers.mlir` test cases but verify:
1. Readonly sref args appended for non-fused producer inputs.
2. Producer's DPS init replaces parent's tied init.
3. Read_slice sites replaced with tiled producer computation.

- [ ] **Step 2: Run tests and verify**

- [ ] **Step 3: Commit**

### Task 8: Full integration build and test

- [ ] **Step 1: Full iree-opt build**

```bash
cd /home/quinn/root/iree-shared-exec/iree-build
cmake --build . --target iree-opt 2>&1 | tail -10
```

- [ ] **Step 2: Run all PCF tests**

```bash
cd /home/quinn/root/iree-shared-exec/iree-build
ctest -R PCF -j$(nproc) --output-on-failure
```

- [ ] **Step 3: Commit all remaining changes**
