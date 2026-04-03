# PCF Branch Review: Remaining Work

**Date**: 2026-04-02
**Branch**: shared-exec-rebase (77 commits + 4 review commits)
**Status**: 478/478 Codegen tests pass. Items below are unaddressed findings.

## Pyre-Review Findings (Gemini cross-validation)

### root-1h3 (Medium): TileToPCF resultInfo OOB
- TileToPCF.cpp:220-222 indexes readwriteRefs[i] directly
- Same issue as Irene Finding 5 in MultiLevelTiling (already fixed there)
- Needs same DPS-init-based mapping fix in TileToPCF

### root-5l2 (Medium): assert(succeeded()) in release builds
- DistributedFuseConsumers.cpp:369 uses `assert(succeeded(reifyOp.reifyResultShapes(...)))`
- assert is stripped in release builds, silently skipping reification
- Should use `if (failed(...)) return failure()` pattern

### root-3c2 (Medium): buildOperandInfo silent fallthrough
- MultiLevelTiling.cpp:128-132 falls back to original operand when rwIdx is out of range
- Could produce tensor operand where iter_arg is expected
- Should assert instead of silently falling through

## Completed (this session)

- Irene Critical 1: Lane offset computation
- Irene Critical 3/Finding 9: cast<AffineDimExpr> crash
- Irene Critical 4/Finding 14: Walk mutation UB
- Irene Finding 7: TileToPCF WriteSlice offsets
- Irene Finding 3: isZeroTileSize
- Irene Finding 4: Multiple reduction dims
- Irene Finding 5: operandToReadwriteIdx OOB
- Irene Finding 6: operand-building loop duplication -> buildOperandInfo
- All Petra findings (via agent + manual)
- Vera: DO NOT SUBMIT -> fixed checks
- Vera: Dynamic shapes, non-divisible tiles, matmul tests added
- Vera: Consumer fusion crash documented
- Vera: Elementwise producer fusion test added
- Pipeline fix: stray RematerializeParallelOps removed
- VectorExt: splat fold canonicalization loop removed
- Lint: pre-submit checks passed

## Remaining: Irene Findings

### Finding 8 (Important): TileToPCF padding logic fragile
- File: TileToPCF.cpp:258-267
- writeSizes uses getDimSize which returns kDynamic for dynamic dims
- Now partially addressed by getResultTilePosition fallback, but fallback itself has this issue
- **Action**: Verify the fallback path is unreachable for dynamic results, or fix padding

### Finding 10 (Important): computeOperandTilePosition returns nullopt with no diagnostic
- File: LinalgOpDistributedTiling.cpp:141-143
- Bare `return failure()` instead of notifyMatchFailure
- Cannot use notifyMatchFailure because interface takes OpBuilder not RewriterBase
- **Action**: Add LLVM_DEBUG message at minimum

### Finding 11 (Important): getDistributedImplementation takes OpBuilder instead of RewriterBase
- File: PCFInterfaces.td interface definition
- Plain OpBuilder bypasses rewriter notification; prevents notifyMatchFailure
- This is a deliberate workaround for greedy rewriter notification issues
- **Action**: Consider changing to RewriterBase in a follow-up; document the tradeoff

### Finding 12 (Minor): offsetIndices with non-identity maps
- Assessed as correct for standard cases
- **Action**: None needed

### Finding 13 (Minor): getIterArgTypes uses tiledShape.size() as index
- Correct but confusing; should use enumeration index
- **Action**: Change to llvm::enumerate

### Finding 15 (Important): Silently discards tiling failure
- GPUApplyMultiLevelTiling.cpp:132
- Petra agent added explanatory comment, but still silent on failure
- **Action**: Add emitWarning on failure

### Finding 16 (Important): No tile size validation
- GPUApplyMultiLevelTiling.cpp:79-100
- No check that thread tiles divide subgroup tiles
- **Action**: Add validation or at minimum documentation

### Finding 17 (Important): Stale Operation* after greedy rewriting
- TileDispatchUsingPCF.cpp user iteration during consumer fusion
- **Action**: Copy users to SmallVector before iterating

### Finding 18 (Minor): Single-trip loop elimination incomplete for dynamic shapes
- TileDispatchUsingPCF.cpp:83-106
- Safe but optimization silently skipped for dynamic shapes
- **Action**: None needed (optimization, not correctness)

## Remaining: Vera Findings

### Tile-to-PCF gaps
- No rank-1 tensor test
- No multiple-result op test
- **Action**: Add tests

### Multi-level tiling gaps
- No batch matmul test
- No asymmetric tile size test
- fill_matmul test missing CHECK-NOT for fill inside inner pcf.generic
- **Action**: Add tests

### Workgroup tiling FIXME tests (6 items)
1. @matmul_memrefs: crash on memref ops (0 results)
2. @matmul_fusion_test: extf generic fusion not supported
3. @matmul_consumer_fusion_test: same
4. @consumer_fuse_scatter: linalg.add missing PCFTilingOpInterface
5. @infusible_pack: generic+pack consumer failure
6. @arg_compare_fold_broadcast: arg_compare missing PCFTilingOpInterface
- **Action**: Create tracking issues for each

### Pipeline test gaps
- No dynamic shape pipeline test
- No non-matmul pipeline test
- @matmul_mma doesn't check promote_operand
- **Action**: Add tests

### Promote operand test gaps
- Only 2 tests (1 happy, 1 negative)
- No dynamic shapes, no rank-1/rank-3+
- **Action**: Add tests

### IR test gaps
- readonly_refs.mlir: no dynamic shapes, no negative mismatch tests
- namespace_symbols.mlir: no duplicate name test, no usage test
- **Action**: Add tests

## Remaining: Process Items

### Pyre-review (Phase 3)
- Never executed
- **Action**: Run cross-validated review or PAL codereview

### addReadonlyArgs helper extraction
- Duplicated across DistributedFuseConsumers.cpp and DistributedFuseProducers.cpp
- TODO added in Transforms.h but not implemented
- **Action**: Extract to shared utility

### MMA vs regular path duplication in MultiLevelTiling.cpp
- Irene Finding 6 partially addressed (operand building extracted)
- The reduction loop + writeback logic is still duplicated
- **Action**: Extract shared reduction loop helper

### Consumer fusion crash (dropUnusedResults)
- heap-buffer-overflow in Transforms.cpp:dropUnusedResults<GenericOp>
- Blocks all consumer fusion testing
- **Action**: Fix the crash
