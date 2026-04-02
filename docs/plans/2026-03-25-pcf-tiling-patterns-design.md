# PCF Tiling and Fusion Patterns Design

## Summary

Three utility functions that use `PCFTilingOpInterface` to tile ops into PCF
parallel constructs and fuse producers/consumers through readonly/readwrite
sref arguments. These replace the combination of `TilingInterface` methods
(`getTiledImplementation`, `generateResultTileValue`,
`getTiledImplementationFromOperandTiles`) with the single
`getDistributedImplementation` entry point.

## Tile-to-PCF

Two functions with a shared implementation:

```cpp
FailureOr<PCF::LoopOp> tileToPCFLoop(
    RewriterBase &rewriter, PCFTilingOpInterface target,
    ScopeAttrInterface scope, ArrayRef<OpFoldResult> tileSizes);

FailureOr<PCF::GenericOp> tileToPCFGeneric(
    RewriterBase &rewriter, PCFTilingOpInterface target,
    ScopeAttrInterface scope, ArrayRef<OpFoldResult> tileSizes);
```

Tile size semantics match `scf::tileUsingSCF`: zero means don't tile, non-zero
is the tile size, iteration count = `ceil(dim / tileSize)`.

### Steps

1. Query `getIterationDomain()`, compute iteration counts from tile sizes.
2. Query `getTileableOperandIndices()`.
3. Create outer construct (loop or generic + scf.forall):
   - Readonly sref args for tileable non-DPS-init operands.
   - Readwrite sref args for DPS inits (tied to results).
4. Inside body, compute tile offsets/sizes from iteration IDs.
5. Build `DistributedOperandInfo` per operand:
   - Tileable: `pcf.read_slice` from sref → `{value, isTile=true}`.
   - Non-tileable: pass original value → `{value, isTile=false}`.
6. Build `DistributedResultInfo` per result: `{destSref = readwrite sref}`.
7. Call `getDistributedImplementation(offsets, sizes, operandInfo, resultInfo)`.
8. Write any non-null `tiledValues` via `pcf.write_slice`.

## Consumer Fusion

Same match logic as existing `matchTilableConsumer` — find a
`PCFTilingOpInterface` user of a loop/generic result, verify dominance, find
write slices.

### Differences from existing pattern

1. Call `getDistributedImplementation` instead of
   `getTiledImplementationFromOperandTiles`.
2. Non-fused input operands: append readonly sref args to parent, read tiles.
3. Fused-along operand: use write_slice source directly (already-computed
   tile).
4. Consumer results: write to new readwrite sref args appended to parent.

## Producer Fusion

Same match logic as existing `matchTilableProducer` — find a DPS producer
feeding a tied init, verify dominance, find read_slice sites.

### Differences from existing pattern

1. Call `getDistributedImplementation` instead of `generateResultTileValue`.
2. Non-fused input operands: append readonly sref args to parent, read tiles.
3. Producer's DPS init: replace parent's tied init with producer's init.
4. At each read_slice site: replace with tiled producer result.

## File Organization

All three utilities go in a new file `PCF/Transforms/TileToPCF.cpp` with
declarations in `Transforms.h`. The shared implementation uses a template
over the PCF op type (LoopOp/GenericOp), matching the existing pattern in
`FuseConsumers.cpp` and `FuseProducers.cpp`.

## Registration of PCFTilingOpInterface

The tiling utilities call `registerAllDistributedTilingModels(registry)` to
ensure external models are attached before use. This is deferred from dialect
load time to avoid ordering issues with `TilingInterface` registration.
