# Scatter-like Ops: sref Destination Support Notes

## Overview

This document captures what changes would be needed for `iree_linalg_ext.scatter`
and `iree_linalg_ext.map_store` to write to a `pcf.sref` destination, for when
we implement the `PCFTilingOpInterface` external model on these ops.

## ScatterOp

**File**: `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td` (line 47)

**Current operands**:
- `updates` (AnyRankedTensorOrMemRef) - the source values to scatter.
- `indices` (AnyRankedTensorOrMemRef) - where to scatter.
- `original` (AnyRankedTensorOrMemRef) - the DPS init / destination.

**TilingInterface impl** (`TilingInterfaceImpl.cpp:124-260`):
- `getIterationDomain`: Defined by `updates` shape.
- `getTiledImplementation`: Slices `updates`, `indices`, and `original`, then
  clones the op with the sliced operands.
- `getResultTilePosition`: The result tile covers the full `original` in the
  scattered dimensions and is sliced in the update-slice dimensions.
- `generateScalarImplementation`: Loads from `updates`, computes destination
  indices from `indices`, loads from `original`, applies the combiner region,
  stores back to `original`.

**Changes needed for sref destination**:
1. The `PCFTilingOpInterface::getDistributedImplementation` would receive the
   destination sref via `DistributedResultInfo::destSref`. Instead of slicing
   `original` from the DPS init, the tiled op would:
   - Use `pcf.read_slice` to load the relevant tile from the sref.
   - Run the scatter on the loaded tile.
   - Use `pcf.write_slice` to write the result back.
2. The `updates` and `indices` operands are read-only inputs. They should be
   passed via `DistributedOperandInfo::value` (already sliced by the tiling
   framework).
3. When `unique_indices` is false, scatter has reduction semantics along the
   batch dimensions. This means multiple workers may write to overlapping
   regions of the sref. The distributed implementation must handle
   synchronization (e.g., atomic combine) or be restricted to
   `unique_indices=true` cases.
4. The combiner region (binary `(T, T) -> T`) must be preserved in the tiled
   clone. If atomic writes are needed, the combiner must map to an atomic RMW
   operation.

**Key challenge**: The `getResultTilePosition` returns a tile that covers
the *full* original in the scattered dimensions (offsets are zero, sizes are
full). This means the sref write would cover the entire destination in those
dimensions, which is correct for a scatter (you cannot know which indices
will be touched). The PCF write_slice would need to handle this full-extent
pattern efficiently.

## MapStoreOp

**File**: `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td` (line 349)

**Current operands**:
- `input` (AnyShaped) - the source values.
- `output` (AnyRankedTensorOrMemRef) - the destination to store into.

**TilingInterface impl** (`TilingInterfaceImpl.cpp:646-765`):
- `getIterationDomain`: Defined by `input` shape.
- `getTiledImplementation`: Slices `input`, clones the op with the sliced
  input but the *full* output. The transformation body indices are offset
  by the tile offset to maintain correct global addressing.
- `getResultTilePosition`: Returns the *full* output (offsets all zero, sizes
  are full output dimensions). This is because the index transformation
  is arbitrary -- any input element could map to any output location.
- `generateScalarImplementation`: Inlines the transformation body to compute
  destination indices, conditionally writes based on the mask.

**Changes needed for sref destination**:
1. The `PCFTilingOpInterface::getDistributedImplementation` would receive the
   sref via `DistributedResultInfo::destSref`. The implementation would:
   - Use the sliced `input` from `DistributedOperandInfo::value`.
   - Write directly to the sref using `pcf.write_slice` for each element
     (or use `generateScalarImplementation` which already does indexed
     stores).
2. The key insight is that `getResultTilePosition` returns the full output.
   This means the sref must cover the entire output tensor. The distributed
   tiling framework should provide the full sref, not a slice.
3. Since the transformation body provides arbitrary index mappings, multiple
   workers may write to the same output location. The mask value (`i1`)
   provides conditional execution, but overlapping writes from different
   tiles are possible. The distributed implementation needs either:
   - Guarantee that tiles produce non-overlapping writes (application-level).
   - Use atomic writes for the store operations.
4. For the scalar implementation path (lowered to loops + memref
   load/store), the sref would already be lowered to a memref by the time
   `generateScalarImplementation` runs. No changes needed for that path.

**Key challenge**: The arbitrary index transformation means we cannot
efficiently tile the output. The entire output sref must be available to
every worker. This is fundamentally different from linalg ops where the
output tile position is predictable from the iteration domain.

## Common Considerations

1. **Read-only inputs**: Both ops have read-only input operands (`updates`/
   `indices` for scatter, `input` for map_store). These should use the readonly
   sref mechanism (the `<-` binding) when enclosed in a pcf op, rather than
   being passed as readwrite refs.

2. **Full-output writes**: Both ops have the property that `getResultTilePosition`
   returns the full output extent. The `PCFTilingOpInterface` must handle this
   by providing the full destination sref rather than trying to slice it.

3. **Conflict handling**: Both ops can have write conflicts when distributed.
   The `unique_indices` flag on scatter provides opt-in safety. MapStore has
   the mask but no uniqueness guarantee. The distributed implementation should
   document these constraints clearly.

4. **Timing**: These changes will be implemented as part of the
   `PCFTilingOpInterface` external model registration, alongside similar
   models for tensor, linalg, and other LinalgExt ops.
