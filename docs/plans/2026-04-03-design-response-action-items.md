# Action Items from Design Question Responses

## Code Changes Required

### 1. Q8: Split getDistributedImplementation into match + rewrite (CRITICAL)
- **Architect says**: "Mutating IR then returning failure is extremely egregious. MLIR rewriters do NOT handle rollback. Split into match + rewrite."
- **Files**: PCFInterfaces.td, LinalgOpDistributedTiling.cpp, LinalgExtDistributedTiling.cpp, TensorOpDistributedTiling.cpp, MultiLevelTiling.cpp, TileToPCF.cpp
- **Scope**: Large — interface change affects all implementations

### 2. Q16: Support multiple reduction dimensions (CRITICAL)
- **Architect says**: "Only supporting single reduction dim is a massive shortcut. Use scf.for loop nest helpers. Test with conv tiling filter dims to unit."
- **Files**: MultiLevelTiling.cpp
- **Scope**: Medium — replace single scf.for with nested loop construction

### 3. Q17: Fix redundant execution when numLaneIterators=0 (CRITICAL)
- **Architect says**: "Repeating work indicates the implementer didn't understand parallel execution semantics. Must mask off inactive lanes via scf.if. Most likely needs scf.forall inside pcf.generic+pcf.lane for spillover/masking."
- **Files**: MultiLevelTiling.cpp
- **Scope**: Medium — add masking/forall logic

### 4. Q18: Remove mma_kind duck-typed attribute, implement properly (CRITICAL)
- **Architect says**: "mma_kind handling is NOT tiling. Setting discardable attributes is wrong. Must have separate inner_tiled handling composing with subgroup level. Also this file should be in Common/GPU, not PCF/Transforms."
- **Files**: MultiLevelTiling.cpp -> move to Common/GPU, rework MMA handling
- **Scope**: Large — architectural change

### 5. Q20: Add memory read effects to pcf.read_slice
- **Architect says**: "Yes, good catch!"
- **Files**: PCFOps.td, PCFOps.cpp
- **Scope**: Small

### 6. Q25: Add memory effects and verifier to iree_gpu.dma_copy
- **Architect says**: "Definitely needs read and write memory effects."
- **Files**: IREEGPUOps.td, IREEGPUOps.cpp (or IREEGPUAttrs.cpp)
- **Scope**: Small

### 7. Q23: Remove Speculatable from pcf.telescope
- **Architect says**: "Removing speculatable is correct."
- **Files**: PCFOps.td
- **Scope**: Small

### 8. Q7: Fix symbol name collision avoidance
- **Architect says**: "Use d0,d1,...,dn for dims, n0,n1,...,nn for namespaces. Fully qualify parent namespace scope, unique within parent."
- **Files**: MultiLevelTiling.cpp
- **Scope**: Small

### 9. Q6: Formalize symbol resolution + create formal symbol attribute type
- **Architect says**: "First match inner-to-outer. C++-style namespace qualifiers (bar::sym_name). Need formal symbol attribute type with definition lookup infrastructure."
- **Files**: PCFInterfaces.td, PCFOps.td, new attribute type
- **Scope**: Medium — new attribute type + infrastructure

### 10. Q9: Rename reduction methods to be more generic -- DONE
- **Architect says**: "Rename emitReductionWriteback -> emitResultTileStore, emitReductionInit -> emitInitTileLoad. Remove notion of 'reductions' from names."
- **Completed**: Renamed getReductionIterArgTypes -> getIterArgTypes, emitReductionInit -> emitInitTileLoad, emitReductionWriteback -> emitResultTileStore across all files.

### 11. Q13: Add verifier on pcf.alloc that parent is initializer with matching scope
- **Architect says**: "At most we could have a verifier on the alloc... probably should add that actually."
- **Files**: PCFOps.cpp
- **Scope**: Small

### 12. Q21: ClusterType private/shared should be mutually exclusive
- **Architect says**: "Reduce to single structTypes + flag for private/shared semantics."
- **Files**: PCFBase.td, PCFTypes.cpp, related ops
- **Scope**: Medium

### 13. Q3.2: Check bufferization handles readonly inputs properly
- **Architect says**: "Bufferization might be missing handling of readonly inputs. This is a big problem if so."
- **Files**: BufferizationExternalModels.cpp
- **Scope**: Investigation + possible fix

## Documentation Only (no code changes)

- Q1: Document readonly=immutable contract. No read fence needed.
- Q2: Lowering inserts barriers. Nothing underspecified.
- Q4: pcf.loop is sugar for pcf.generic+scf.forall spillover.
- Q5: shared_executor converts to pcf.generic (not the other way).
- Q11: Document PromoteOperandOp semantics standalone (no lifecycle references to passes).
- Q14: s0 is defined by the type's scope. No implementation change needed.
- Q15: get_memref is not lowering-only. No restrictions needed.
- Q19: Bufferization only handles tensors. ConvertSRefToMemRef ties sref block args to memref operands.
- Q24: Pipeline determines lowering. scf.forall handled normally.
