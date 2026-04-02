# PCF Tiling Interface Design

## Summary

Add a `PCFTilingOpInterface` that extends `TilingInterface` to provide a
unified entry point for distributing ops into PCF parallel constructs. The
interface replaces the combination of `getTiledImplementation`,
`generateResultTileValue`, and `getTiledImplementationFromOperandTiles` with
a single `getDistributedImplementation` method that accepts structured info
about how operands are provided and how results should be produced.

This also requires:
1. Adding readonly sref arguments to `pcf.generic` and `pcf.loop` (mirroring
   `pcf.shared_executor`).
2. Updating scatter-like ops (`MapStoreOp`, `ScatterOp`) to support
   `pcf.sref` destination types.
3. Implementing the interface for all upstream `TilingInterface` ops via
   external interfaces.

## Interface Definition

### Data Structures

```cpp
/// Info about how an operand is provided to the distributed op.
struct DistributedOperandInfo {
  Value value;   // The replacement value (tensor tile, sref, or full operand).
  bool isTile;   // Whether |value| is a tile or the full operand.
};

/// Info about how a result should be produced.
struct DistributedResultInfo {
  Value destSref;  // If non-null, write result to this sref instead of
                   // returning a tensor tile. Null means "return a tile".
};
```

### Interface Methods

```tablegen
def PCFTilingOpInterface : OpInterface<"PCFTilingOpInterface", [TilingInterface]> {
  let methods = [
    InterfaceMethod<
      "Returns indices of operands that are tileable.",
      "SmallVector<unsigned>", "getTileableOperandIndices",
      (ins), /*methodBody=*/[{}], /*defaultImplementation=*/[{}]
    >,
    InterfaceMethod<
      "Returns the distributed/tiled implementation of this op.",
      "FailureOr<TilingResult>", "getDistributedImplementation",
      (ins "OpBuilder &":$b,
           "ArrayRef<OpFoldResult>":$offsets,
           "ArrayRef<OpFoldResult>":$sizes,
           "ArrayRef<DistributedOperandInfo>":$operandInfo,
           "ArrayRef<DistributedResultInfo>":$resultInfo),
      /*methodBody=*/[{}], /*defaultImplementation=*/[{}]
    >,
  ];
}
```

### Semantics

- **Extends `TilingInterface`**: All implementations inherit iteration domain,
  loop iterator types, etc. from `TilingInterface`.
- **`getTileableOperandIndices`**: Returns indices of operands that can be
  tiled. Non-tileable operands are passed through as full values. All results
  are implicitly tileable (required for distribution into parallel constructs).
- **`getDistributedImplementation`**: Single entry point replacing the three
  separate `TilingInterface` methods. The `offsets` and `sizes` describe the
  current iteration tile (same as `getTiledImplementation`). Operand info
  and result info describe how inputs/outputs are provided.
- **Operand info covers all operands** (inputs + DPS inits), indexed by
  operand number. For DPS inits that are also results, the operand info and
  result info will typically point to the same sref.
- **Result behavior**: When `resultInfo[i].destSref` is non-null, the
  implementation writes to the sref and the corresponding `tiledValues` entry
  is empty. When null, the implementation returns a tensor tile.

## Readonly Sref Arguments for generic/loop

`pcf.generic` and `pcf.loop` currently only have readwrite (result-tied) sref
arguments. They need readonly sref arguments matching `pcf.shared_executor`.

### Changes

- Add `readonlyInits` operand list (separate from existing `inits`).
- Add `num_readonly_refs` property.
- Block argument order becomes: leading args -> readonly ref args -> readwrite
  ref args -> id/count args.
- Assembly format uses `<-` for readonly and `=` for readwrite (same as
  `pcf.shared_executor`).
- Readonly args do not produce results.

### Example

```mlir
%result = pcf.loop scope(#wg) count(%n)
  execute(%in_ref <- %input, %out_ref = %output)[%id: index]
       : (!pcf.sref<128x256xf16, #wg>,
          !pcf.sref<128x128xf32, #wg>)
      -> (tensor<128x128xf32>) {
    // %in_ref is readonly, %out_ref is readwrite.
    pcf.return
  }
```

## Sref Support for Scatter-like Ops

`MapStoreOp` and `ScatterOp` write to a destination rather than producing a
result tile. Their `getDistributedImplementation` will:
1. Accept a non-null `destSref` in `resultInfo`.
2. Generate a `pcf.write_slice` to the destination sref instead of returning a
   tensor tile.
3. Return empty `tiledValues` for that result.

These ops need to accept `pcf.sref` as valid destination types in their type
constraints or have the distributed implementation handle the sref-to-write
conversion externally.

## Directory Layout

```
PCF/IR/
  PCFInterfaces.td          // Add PCFTilingOpInterface definition
  PCFTilingInterface.h      // DistributedOperandInfo, DistributedResultInfo structs

PCF/TilingImplementations/
  BUILD.bazel
  CMakeLists.txt
  LinalgOpDistributedTiling.cpp      // LinalgOp external model
  LinalgExtDistributedTiling.cpp     // ScatterOp, MapStoreOp external models
  TensorOpDistributedTiling.cpp      // tensor.pad, tensor.pack, tensor.unpack
  RegisterAll.h / RegisterAll.cpp    // Single registration entry point
```

Registration is called from `PCF/ExternalInterfaces/Interfaces.cpp` alongside
existing bufferization registration.

For `LinalgOp`, a single external model covers all named ops + generic since
they share the same tiling implementation.

## Usage by Tiling/Fusion Patterns

The interface serves as the single entry point for three patterns:

### a) Tile to PCF

Creates `pcf.loop` or `pcf.generic` (+ nested `scf.forall` for spillover)
around a `PCFTilingOpInterface` op:
1. Readonly sref args for all tileable input operands.
2. Readwrite sref args for all result-tied DPS inits.
3. Inside body: construct `DistributedOperandInfo` per operand (read tiles
   from readonly srefs for inputs, pass readwrite sref for DPS inits).
4. Construct `DistributedResultInfo` with readwrite srefs as `destSref`.
5. Call `getDistributedImplementation`.
6. Returned tiles (if any) get written via `pcf.write_slice`.

### b) Consumer Fusion

Fusing a consumer into an existing `pcf.loop`/`pcf.generic`:
1. Append readonly sref args for the consumer's non-fused input operands.
2. Construct `DistributedOperandInfo`: use already-available tile value for
   fused-along operand, read from new srefs for others.
3. Call `getDistributedImplementation` on the consumer.

### c) Producer Fusion

Fusing a producer through a readonly sref:
1. At each `pcf.read_slice` site, replace with
   `getDistributedImplementation` on the producer.
2. Producer's non-fused operands get new readonly sref args.
3. Fused-along operand (producer's DPS init) gets its own sref.
