# shared-exec-2 Branch Comprehensive Review

**Beads issue**: root-k6v

## Preamble

This review covers ALL work on the shared-exec-2 branch (74 commits). The
branch implements PCF-based tiling infrastructure as an alternative to the
forall-based TileAndFuse pipeline. This is production compiler infrastructure
and every change must meet the quality bar.

**We expect to find:**
- Holes in testing (missing edge cases, insufficient coverage)
- Implementation shortcuts that need fixing (TODOs, incomplete paths)
- Poor documentation (missing comments, unclear interfaces)
- Style violations (IREE coding standards)
- Architectural issues (wrong abstractions, leaky dependencies)
- Dead code, unused utilities, incomplete error handling

**ALL of these must be caught and addressed during review. There are no
shortcuts here.**

## Review Process

### Phase 1: Rebase and Clean Build
- [ ] Rebase shared-exec-2 on iree/main
- [ ] Resolve all merge conflicts
- [ ] Run full build (CMake)
- [ ] Run pre-submit linting checks
- [ ] Run Bazel build
- [ ] Fix all build/lint issues

### Phase 2: Peanut Gallery Review
Dispatch parallel subagent reviewers using personas:
- **Irene** (compiler expert): Core compiler changes, tiling/fusion
- **Merlin** (MLIR infra): Interface design, dialect structure
- **Petra** (style): Naming, organization, conventions
- **Vera** (testing): Test coverage, test quality
- **Soren** (minimalism): Scope, unnecessary complexity

Each reviewer gets the full diff and reviews independently.

### Phase 3: Pyre Review
Run pyre-review (~/root/pyre-review) for automated analysis.

### Phase 4: Address Findings
- Fix every issue found (Critical/Important/Minor)
- Document justification for any findings not addressed
- Re-review after fixes

### Phase 5: Final Verification
- Full build + test
- Pre-submit checks pass
- All review findings addressed

## Files Changed (summary)

### New files:
- PCF/IR/PCFTilingInterface.h — DistributedOperandInfo, DistributedResultInfo, MultiLevelTilingParams structs
- PCF/TilingImplementations/ — External model implementations for LinalgOp, LinalgExt, Tensor
- PCF/Transforms/TileToPCF.cpp — tileToPCFLoop utility
- PCF/Transforms/MultiLevelTiling.cpp — applyMultiLevelTiling (subgroup+lane+reduction)
- PCF/Transforms/DistributedFuseConsumers.cpp — Consumer fusion with PCFTilingOpInterface
- PCF/Transforms/DistributedFuseProducers.cpp — Producer fusion with PCFTilingOpInterface
- Common/TileDispatchUsingPCF.cpp — Workgroup tiling pass using PCF
- Common/GPU/GPUApplyMultiLevelTiling.cpp — GPU multi-level tiling pass
- GPU/Transforms/LowerPromoteOperand.cpp — Promotion lowering
- GPU/IR/ changes — promote_operand, dma_copy ops
- Codegen/IR/ changes — swizzle_hint sref support

### Modified files:
- PCF/IR/PCFInterfaces.td — PCFTilingOpInterface, NamespaceSymbolOpInterface
- PCF/IR/PCFOps.td — Readonly srefs on generic/loop, index_symbol, LoopOp builders
- PCF/IR/PCFOps.cpp — Parser/printer/verifier/builders for readonly srefs
- LinalgExt/IR/ — ScatterOp/MapStoreOp accept sref types
- GPU/IR/IREEGPUInterfaces.td — emitDistributedCopy on PromotionAttr
- GPU/IR/IREEGPUAttrs.td — Promotion type overrides
- Multiple test files

## Status
- [ ] Phase 1: Rebase and Clean Build
- [ ] Phase 2: Peanut Gallery Review
- [ ] Phase 3: Pyre Review
- [ ] Phase 4: Address Findings
- [ ] Phase 5: Final Verification
