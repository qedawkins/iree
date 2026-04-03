# PCF Tiling Prerequisites Design

## Overview

Three prerequisite features needed before implementing the multi-level tiling
transform that recreates what TileAndFuse does:

1. Namespace symbol tables on `pcf.generic` and `pcf.shared_executor`.
2. `pcf.index_symbol` op for binding index values to symbols.
3. Multi-level tiling transform (subgroup + lane + reduction).

## 1. Namespace Symbol Tables

### NamespaceOpInterface Extension

Add `getSymbolRegion()` method to `NamespaceOpInterface`:

```tablegen
InterfaceMethod<
  "Returns the region that contains symbol definitions.",
  "Region &", "getSymbolRegion", (ins)
>,
```

Implementations:
- `pcf.generic`: returns the initializer region.
- `pcf.shared_executor`: returns the initializer region.
- `tile_group`: returns its body region (existing behavior).

### Add NamespaceOpInterface to pcf.generic and pcf.shared_executor

Both ops become anonymous namespaces (`getNamespaceName()` returns
`std::nullopt`). `getDefinedSymbols()` walks the initializer region for
`NamespaceSymbolOpInterface` ops.

### NamespaceSymbolOpInterface (new op interface)

Ops that define a symbol in a namespace:

```tablegen
def PCF_NamespaceSymbolOpInterface : OpInterface<"NamespaceSymbolOpInterface"> {
  let methods = [
    InterfaceMethod<"Returns the symbol name.",
                    "StringAttr", "getSymbolName", (ins)>,
    InterfaceMethod<"Returns the symbol definition as an OpFoldResult.",
                    "OpFoldResult", "getSymbolDefinition", (ins)>,
  ];
}
```

Verifier: the op's parent region must be the `getSymbolRegion()` of an
enclosing `NamespaceOpInterface` op.

## 2. pcf.index_symbol Op

Binds an SSA index value to a symbol name in the enclosing namespace:

```mlir
pcf.generic scope(#sg)
  initialize {
    %n = <some computation> : index
    pcf.index_symbol "tile_count" = %n
    pcf.yield
  }
  execute(...)[...] {
    // Ops that reference "tile_count" resolve it through the namespace.
  }
```

- Implements `NamespaceSymbolOpInterface`.
- Takes: `StringAttr` name, `Index` value operand.
- No results (pure binding).
- Must live in the `getSymbolRegion()` of a `NamespaceOpInterface` op.
- `getSymbolDefinition()` returns the index operand as an `OpFoldResult`.

## 3. Multi-Level Tiling Transform

### Input

A `PCFTilingOpInterface` op inside a workgroup-tiled `pcf.loop` body.

### Output

```mlir
pcf.generic #subgroup_scope
  initialize { ... symbols for promotion sizes ... }
  execute(...) {
    pcf.generic #lane_scope
      execute(...) {
        %init = // emitted by interface
        %result = scf.for iter_args(%acc = %init) {
          // promoted operand reads
          %tiled = // getDistributedImplementation(...)
          scf.yield %tiled
        }
        // writeback emitted by interface
      }
  }
```

### Transform Parameters

```cpp
struct TilingLevelParams {
  ScopeAttrInterface scope;
  ArrayRef<OpFoldResult> tileSizes;
};

struct DistributedTilingParams {
  TilingLevelParams subgroup;
  TilingLevelParams lane;
  ArrayRef<OpFoldResult> reductionTileSizes;
  SmallVector<unsigned> operandsToPromote;
};
```

### Transform API

```cpp
FailureOr<PCF::GenericOp> applyMultiLevelTiling(
    RewriterBase &rewriter,
    PCFTilingOpInterface target,
    const DistributedTilingParams &params);
```

### Responsibility Split

The **transform** controls:
- Creating the `pcf.generic` nest (subgroup + lane).
- Creating the `scf.for` for reduction.
- Defining symbols in the initializer for promotion sizes.
- Providing insertion points for init load and writeback.

The **interface** controls:
- Loop-carried value types (`getIterArgTypes`).
- Init load emission (`emitInitTileLoad`).
- Result tile store emission (`emitResultTileStore`).
- Tiled implementation (`getDistributedImplementation`).
- Promoted operand shapes.

### PCFTilingOpInterface Extensions Needed

```cpp
// Returns the types of loop-carried values.
SmallVector<Type> getIterArgTypes(
    OpBuilder &b, const DistributedTilingParams &params);

// Emits the initial tile load. Returns initial values for scf.for iter_args.
SmallVector<Value> emitInitTileLoad(
    OpBuilder &b, ArrayRef<DistributedOperandInfo> operandInfo,
    ArrayRef<DistributedResultInfo> resultInfo,
    const DistributedTilingParams &params);

// Emits the result tile store.
void emitResultTileStore(
    OpBuilder &b, ValueRange reductionResults,
    ArrayRef<DistributedResultInfo> resultInfo,
    const DistributedTilingParams &params);
```

### Three Configurations

All three produce the same structural template (subgroup generic + lane
generic + scf.for). The difference is in what tile sizes are used at each
level:

1. **Thread tiling**: subgroup tile sizes = thread tile sizes (product of
   subgroup * lane), lane tile sizes subdivide those.
2. **Separate subgroup + lane**: each level has independent tile sizes.
3. **Subgroup only + MMA conversion**: subgroup tile sizes are set, lane tile
   sizes are 1 (or identity), and the interface converts to inner_tiled form
   for MMA distribution.

The pass reads `#iree_gpu.lowering_config` to determine which configuration
and extract tile sizes, then calls `applyMultiLevelTiling` with the
appropriate `DistributedTilingParams`.
