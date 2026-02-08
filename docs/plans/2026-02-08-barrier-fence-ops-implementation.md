# Barrier and Fence Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Separate GPU synchronization from memory fencing to fix the barrier canonicalization bug in the pingpong double-buffer schedule.

**Architecture:** Three new ops: `iree_gpu.global_subgroup_barrier` (sync only), `iree_codegen.fence` (memory fence with address space + release/acquire), `pcf.fence` (high-level fence on sref values). The barrier canonicalizer only affects `gpu.barrier`, so `global_subgroup_barrier` is safe from merging. Fences are placed explicitly per the data hazard analysis in the design doc.

**Tech Stack:** MLIR TableGen (.td), C++ pattern rewrites, ROCDL/NVVM lowering, lit tests with FileCheck.

**Design doc:** `docs/plans/2026-02-08-barrier-fence-ops-design.md`

---

### Task 1: Define `iree_gpu.global_subgroup_barrier` op

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.td:425` (before `#endif`)
- Test: `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/test/iree_gpu_ops.mlir`

**Step 1: Add op definition to IREEGPUOps.td**

Insert before the `#endif // IREE_CODEGEN_DIALECT_IREEGPUOPS` line (line 427):

```tablegen
def IREEGPU_GlobalSubgroupBarrierOp : Op<IREEGPU_Dialect,
    "global_subgroup_barrier", []> {
  let summary = "Synchronization-only barrier across all subgroups.";
  let description = [{
    All subgroups in the workgroup must reach any instance of this op before
    any can proceed past it. Unlike `gpu.barrier`, this op has no memory
    fence semantics - memory operations can be freely reordered with respect
    to it. Use `iree_codegen.fence` for memory ordering.

    This op intentionally has no canonicalizer, so consecutive instances are
    preserved. This is critical for the pingpong double-buffer schedule where
    consecutive barriers serve as distinct synchronization points.
  }];

  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}
```

Key design choices:
- No `Pure` trait: the op has synchronization side effects and must not be DCE'd.
- No memory effects interface: MLIR will conservatively keep it.
- No `hasCanonicalizer`: consecutive instances are not merged.
- No arguments or results: pure synchronization point.

**Step 2: Write roundtrip test**

Append to `GPU/IR/test/iree_gpu_ops.mlir`:

```mlir
// -----

func.func @global_subgroup_barrier() {
  iree_gpu.global_subgroup_barrier
  iree_gpu.global_subgroup_barrier
  return
}

// CHECK-LABEL: func @global_subgroup_barrier
//       CHECK:   iree_gpu.global_subgroup_barrier
//       CHECK:   iree_gpu.global_subgroup_barrier
```

This test verifies: (a) roundtrip parsing/printing, (b) consecutive barriers survive.

**Step 3: Build and run test**

```bash
ninja -C /home/quinn/root/iree-pingpong/iree-build iree-opt
/home/quinn/root/iree-pingpong/iree-build/tools/iree-opt \
  --split-input-file --verify-diagnostics \
  compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/test/iree_gpu_ops.mlir \
  | /home/quinn/root/iree-pingpong/iree-build/llvm-project/bin/FileCheck \
  compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/test/iree_gpu_ops.mlir
```

Expected: PASS (both barriers survive roundtrip).

**Step 4: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.td \
        compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/test/iree_gpu_ops.mlir
git commit -m "[GPU] Add iree_gpu.global_subgroup_barrier op"
```

---

### Task 2: ROCDL lowering for `iree_gpu.global_subgroup_barrier`

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/LLVMGPU/ConvertToROCDL.cpp:61-75`
- Modify: `compiler/src/iree/compiler/Codegen/LLVMGPU/BUILD.bazel` (add IREEGPUDialect dep if missing)
- Test: `compiler/src/iree/compiler/Codegen/LLVMGPU/test/convert_to_rocdl.mlir`

**Step 1: Write the test**

Append to `LLVMGPU/test/convert_to_rocdl.mlir`:

```mlir
// -----

// Test that global_subgroup_barrier lowers to just the barrier, no fences.
builtin.module {
  func.func @global_subgroup_barrier() {
    iree_gpu.global_subgroup_barrier
    return
  }
}
// CHECK-LABEL: llvm.func @global_subgroup_barrier
//   CHECK-NOT: llvm.fence
//       CHECK: llvm.inline_asm has_side_effects asm_dialect = att ";;;WARNING: BREAKS DEBUG WATCHES{{.*}}s_barrier"
//   CHECK-NOT: llvm.fence
```

**Step 2: Add lowering pattern to ConvertToROCDL.cpp**

Add after the existing `ReplaceGPUBarrierWithLDSBarrier` struct (after line 71):

```cpp
struct LowerGlobalSubgroupBarrier
    : public OpRewritePattern<IREE::GPU::GlobalSubgroupBarrierOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::GPU::GlobalSubgroupBarrierOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Pure synchronization, no memory fences. The fences are handled
    // separately by iree_codegen.fence ops.
    //
    // Based on LDSBarrierOpLowering but without the release/acquire fences.
    // TODO: Detect chipset and use appropriate barrier intrinsic.
    // For now, use inline asm like LDSBarrierOp pre-gfx90a path.
    auto asmDialectAttr = LLVM::AsmDialectAttr::get(
        rewriter.getContext(), LLVM::AsmDialect::AD_ATT);
    const char *asmStr = ";;;WARNING: BREAKS DEBUG WATCHES\ns_barrier";
    const char *constraints = "";
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op, /*resultTypes=*/TypeRange(), /*operands=*/ValueRange(),
        /*asm_string=*/asmStr, constraints, /*has_side_effects=*/true,
        /*is_align_stack=*/false, LLVM::TailCallKind::None,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());
    return success();
  }
};
```

Add to `populateConvertGPUToAMDGPUPatterns` (line 73-75):

```cpp
static void populateConvertGPUToAMDGPUPatterns(RewritePatternSet &patterns) {
  patterns.add<ReplaceGPUBarrierWithLDSBarrier,
               LowerGlobalSubgroupBarrier>(patterns.getContext());
}
```

Add include at top of file:

```cpp
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
```

Check BUILD.bazel for `ConvertToROCDL` target: verify it has `//compiler/src/iree/compiler/Codegen/Dialect/GPU/IR:IREEGPUDialect` as a dep. If not, add it.

**Step 3: Build and run test**

```bash
ninja -C /home/quinn/root/iree-pingpong/iree-build iree-opt
/home/quinn/root/iree-pingpong/iree-build/tools/iree-opt \
  --split-input-file --iree-gpu-test-target=gfx908 --iree-convert-to-rocdl \
  compiler/src/iree/compiler/Codegen/LLVMGPU/test/convert_to_rocdl.mlir \
  | /home/quinn/root/iree-pingpong/iree-build/llvm-project/bin/FileCheck \
  compiler/src/iree/compiler/Codegen/LLVMGPU/test/convert_to_rocdl.mlir
```

Expected: PASS. The barrier emits `s_barrier` with no surrounding fences.

**Step 4: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/LLVMGPU/ConvertToROCDL.cpp \
        compiler/src/iree/compiler/Codegen/LLVMGPU/test/convert_to_rocdl.mlir \
        compiler/src/iree/compiler/Codegen/LLVMGPU/BUILD.bazel
git commit -m "[LLVMGPU] Add ROCDL lowering for global_subgroup_barrier"
```

---

### Task 3: Define `iree_codegen.fence` op

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.td:616` (before `#endif`)
- Test: `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/test/roundtrip.mlir`

**Step 1: Add op definition to IREECodegenOps.td**

Insert before the `#endif // IREE_CODEGEN_DIALECT_IREECODEGENOPS` line (line 618):

```tablegen
def IREECodegen_FenceOp : Op<IREECodegen_Dialect, "fence", []> {
  let summary = "Memory fence with release or acquire semantics.";
  let description = [{
    Ensures visibility of memory operations at a specific memory space.
    Release semantics flush prior writes; acquire semantics ensure subsequent
    reads see the latest values.

    This op is used together with `iree_gpu.global_subgroup_barrier` to
    separate memory fencing from synchronization. The barrier ensures all
    threads reach a common point; the fence ensures memory visibility.

    Examples:
    ```mlir
      iree_codegen.fence release #gpu.address_space<workgroup>
      iree_codegen.fence acquire #gpu.address_space<workgroup>
    ```
  }];

  let arguments = (ins
    UnitAttr:$is_release,
    AnyAttr:$memory_space
  );
  let results = (outs);
  let assemblyFormat = [{
    (`release` $is_release^)? (`acquire`)? $memory_space attr-dict
  }];

  let hasVerifier = 1;
}
```

**Step 2: Add verifier in IREECodegenOps.cpp**

Find `IREECodegenOps.cpp` and add:

```cpp
LogicalResult IREE::Codegen::FenceOp::verify() {
  // Must be either release or acquire (not both, not neither).
  // is_release=true means release, is_release absent means acquire.
  return success();
}
```

Actually, the assembly format handles this: `release` keyword sets `is_release` UnitAttr; absence means acquire. But we need to verify that it's a valid memory space attribute. Let me reconsider the assembly format.

A simpler approach - use a `BoolAttr` for `is_release`:

```tablegen
def IREECodegen_FenceOp : Op<IREECodegen_Dialect, "fence", []> {
  let summary = "Memory fence with release or acquire semantics.";
  let description = [{
    Ensures visibility of memory operations at a specific memory space.
    Release semantics flush prior writes; acquire semantics ensure subsequent
    reads see the latest values.

    Examples:
    ```mlir
      iree_codegen.fence release #gpu.address_space<workgroup>
      iree_codegen.fence acquire #gpu.address_space<workgroup>
    ```
  }];

  let arguments = (ins
    BoolAttr:$is_release,
    AnyAttr:$memory_space
  );
  let results = (outs);

  let hasCustomAssemblyFormat = 1;
}
```

With custom assembly format in `IREECodegenOps.cpp`:

```cpp
// Parse: iree_codegen.fence (release|acquire) <memory_space>
ParseResult IREE::Codegen::FenceOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  MLIRContext *context = parser.getContext();
  bool isRelease = false;
  if (succeeded(parser.parseOptionalKeyword("release"))) {
    isRelease = true;
  } else if (failed(parser.parseKeyword("acquire"))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'release' or 'acquire'");
  }
  result.addAttribute("is_release",
                       BoolAttr::get(context, isRelease));

  Attribute memorySpace;
  if (failed(parser.parseAttribute(memorySpace)))
    return failure();
  result.addAttribute("memory_space", memorySpace);

  return parser.parseOptionalAttrDict(result.attributes);
}

void IREE::Codegen::FenceOp::print(OpAsmPrinter &p) {
  p << (getIsRelease() ? " release" : " acquire");
  p << " " << getMemorySpace();
  SmallVector<StringRef> elidedAttrs = {"is_release", "memory_space"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}
```

**Step 3: Write roundtrip test**

Append to `Codegen/IR/test/roundtrip.mlir`:

```mlir
// -----

func.func @fence_release_acquire() {
  iree_codegen.fence release #gpu.address_space<workgroup>
  iree_codegen.fence acquire #gpu.address_space<workgroup>
  return
}
// CHECK-LABEL: func.func @fence_release_acquire
//       CHECK:   iree_codegen.fence release #gpu.address_space<workgroup>
//       CHECK:   iree_codegen.fence acquire #gpu.address_space<workgroup>
```

**Step 4: Build and run test**

```bash
ninja -C /home/quinn/root/iree-pingpong/iree-build iree-opt
/home/quinn/root/iree-pingpong/iree-build/tools/iree-opt \
  --split-input-file \
  compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/test/roundtrip.mlir \
  | /home/quinn/root/iree-pingpong/iree-build/llvm-project/bin/FileCheck \
  compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/test/roundtrip.mlir
```

Expected: PASS.

**Step 5: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.td \
        compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.cpp \
        compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/test/roundtrip.mlir
git commit -m "[Codegen] Add iree_codegen.fence op with release/acquire semantics"
```

---

### Task 4: ROCDL lowering for `iree_codegen.fence`

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/LLVMGPU/ConvertToROCDL.cpp`
- Test: `compiler/src/iree/compiler/Codegen/LLVMGPU/test/convert_to_rocdl.mlir`

**Step 1: Write the test**

Append to `LLVMGPU/test/convert_to_rocdl.mlir`:

```mlir
// -----

// Test that iree_codegen.fence lowers to llvm.fence with MMRA.
builtin.module {
  func.func @fence_workgroup() {
    iree_codegen.fence release #gpu.address_space<workgroup>
    iree_codegen.fence acquire #gpu.address_space<workgroup>
    return
  }
}
// CHECK: #[[$MMRA:.+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">
// CHECK-LABEL: llvm.func @fence_workgroup
//       CHECK: llvm.fence syncscope("workgroup") release {llvm.mmra = #[[$MMRA]]}
//       CHECK: llvm.fence syncscope("workgroup") acquire {llvm.mmra = #[[$MMRA]]}
```

**Step 2: Add lowering pattern to ConvertToROCDL.cpp**

Add after `LowerGlobalSubgroupBarrier`:

```cpp
struct LowerCodegenFenceToROCDL
    : public OpRewritePattern<IREE::Codegen::FenceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Codegen::FenceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Determine LLVM atomic ordering.
    LLVM::AtomicOrdering ordering = op.getIsRelease()
                                        ? LLVM::AtomicOrdering::release
                                        : LLVM::AtomicOrdering::acquire;

    // Map memory space to MMRA tag. For workgroup (LDS), use "local".
    StringRef mmraValue;
    if (auto gpuAddrSpace =
            dyn_cast<gpu::AddressSpaceAttr>(op.getMemorySpace())) {
      switch (gpuAddrSpace.getValue()) {
      case gpu::AddressSpace::Workgroup:
        mmraValue = "local";
        break;
      case gpu::AddressSpace::Global:
        mmraValue = "global";
        break;
      default:
        return op.emitOpError("unsupported address space for ROCDL fence");
      }
    } else {
      return op.emitOpError("expected gpu.address_space attribute");
    }

    Attribute mmra = rewriter.getAttr<LLVM::MMRATagAttr>(
        "amdgpu-synchronize-as", mmraValue);

    auto fence = LLVM::FenceOp::create(rewriter, loc, ordering, "workgroup");
    fence->setDiscardableAttr(LLVM::LLVMDialect::getMmraAttrName(), mmra);

    rewriter.eraseOp(op);
    return success();
  }
};
```

Add to `populateConvertGPUToAMDGPUPatterns`:

```cpp
static void populateConvertGPUToAMDGPUPatterns(RewritePatternSet &patterns) {
  patterns.add<ReplaceGPUBarrierWithLDSBarrier,
               LowerGlobalSubgroupBarrier,
               LowerCodegenFenceToROCDL>(patterns.getContext());
}
```

Add include at top:

```cpp
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
```

Add `IREECodegenDialect` dep to BUILD.bazel if missing.

**Step 3: Build and run test**

```bash
ninja -C /home/quinn/root/iree-pingpong/iree-build iree-opt
/home/quinn/root/iree-pingpong/iree-build/tools/iree-opt \
  --split-input-file --iree-gpu-test-target=gfx908 --iree-convert-to-rocdl \
  compiler/src/iree/compiler/Codegen/LLVMGPU/test/convert_to_rocdl.mlir \
  | /home/quinn/root/iree-pingpong/iree-build/llvm-project/bin/FileCheck \
  compiler/src/iree/compiler/Codegen/LLVMGPU/test/convert_to_rocdl.mlir
```

Expected: PASS. Fences have MMRA attribute restricting to LDS.

**Step 4: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/LLVMGPU/ConvertToROCDL.cpp \
        compiler/src/iree/compiler/Codegen/LLVMGPU/test/convert_to_rocdl.mlir \
        compiler/src/iree/compiler/Codegen/LLVMGPU/BUILD.bazel
git commit -m "[LLVMGPU] Add ROCDL lowering for iree_codegen.fence with MMRA"
```

---

### Task 5: Define `pcf.fence` op

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.td:471` (after BarrierOp)
- Test: `compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/control_flow_ops.mlir`

**Step 1: Add op definition to PCFOps.td**

Insert after the BarrierOp definition (after line 471):

```tablegen
def FenceOp : PCF_Op<"fence", []> {
  let summary = [{Memory fence on shared references.}];
  let description = [{
    Ensures visibility of memory operations on the given sref values.
    Release semantics flush prior writes; acquire semantics ensure subsequent
    reads see the latest values.

    Lowered to `iree_codegen.fence` in ConvertSRefToMemRef, using the memory
    space from each sref's scope. Multiple srefs with the same memory space
    produce a single `iree_codegen.fence`.

    Examples:
    ```mlir
      pcf.fence release %alloc : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
      pcf.fence acquire %alloc : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
    ```
  }];

  let arguments = (ins
    BoolAttr:$is_release,
    Variadic<PCF_ShapedRef>:$srefs
  );
  let results = (outs);

  let hasCustomAssemblyFormat = 1;
}
```

**Step 2: Add custom parser/printer in PCFOps.cpp**

```cpp
// Parse: pcf.fence (release|acquire) %v1, %v2 : type1, type2
ParseResult IREE::PCF::FenceOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  MLIRContext *context = parser.getContext();
  bool isRelease = false;
  if (succeeded(parser.parseOptionalKeyword("release"))) {
    isRelease = true;
  } else if (failed(parser.parseKeyword("acquire"))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'release' or 'acquire'");
  }
  result.addAttribute("is_release", BoolAttr::get(context, isRelease));

  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> types;
  if (failed(parser.parseOperandList(operands)))
    return failure();
  if (!operands.empty()) {
    if (failed(parser.parseColonTypeList(types)))
      return failure();
    if (failed(parser.resolveOperands(operands, types, parser.getNameLoc(),
                                      result.operands)))
      return failure();
  }
  return parser.parseOptionalAttrDict(result.attributes);
}

void IREE::PCF::FenceOp::print(OpAsmPrinter &p) {
  p << (getIsRelease() ? " release" : " acquire");
  if (!getSrefs().empty()) {
    p << " ";
    llvm::interleaveComma(getSrefs(), p);
    p << " : ";
    llvm::interleaveComma(getSrefs().getTypes(), p);
  }
  SmallVector<StringRef> elidedAttrs = {"is_release"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}
```

**Step 3: Write roundtrip test**

Append to `PCF/IR/test/control_flow_ops.mlir`:

```mlir
// -----

util.func private @fence(%alloc: !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>) {
  pcf.fence release %alloc : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
  pcf.fence acquire %alloc : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
  util.return
}

// CHECK-LABEL: @fence
//  CHECK-SAME:   %[[ALLOC:[A-Za-z0-9]+]]: !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:   pcf.fence release %[[ALLOC]] : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
//       CHECK:   pcf.fence acquire %[[ALLOC]] : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
```

**Step 4: Build and run test**

```bash
ninja -C /home/quinn/root/iree-pingpong/iree-build iree-opt
/home/quinn/root/iree-pingpong/iree-build/tools/iree-opt \
  --split-input-file \
  compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/control_flow_ops.mlir \
  | /home/quinn/root/iree-pingpong/iree-build/tools/iree-opt --split-input-file \
  | /home/quinn/root/iree-pingpong/iree-build/llvm-project/bin/FileCheck \
  compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/control_flow_ops.mlir
```

Expected: PASS.

**Step 5: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.td \
        compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp \
        compiler/src/iree/compiler/Codegen/Dialect/PCF/IR/test/control_flow_ops.mlir
git commit -m "[PCF] Add pcf.fence op for memory fencing on sref values"
```

---

### Task 6: Lower `pcf.fence` in ConvertSRefToMemRef

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/ConvertSRefToMemRef.cpp:1212`
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/BUILD.bazel` (add IREECodegenDialect dep)
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/CMakeLists.txt`
- Test: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/` (new or existing test file)

**Step 1: Add lowering pattern to ConvertSRefToMemRef.cpp**

Add a new pattern struct:

```cpp
struct ConvertFenceOp : public OpRewritePattern<PCF::FenceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(PCF::FenceOp fenceOp,
                                PatternRewriter &rewriter) const override {
    // Collect unique memory spaces from sref operands.
    llvm::SmallDenseSet<Attribute> seenSpaces;
    SmallVector<Attribute> uniqueSpaces;
    for (Value sref : fenceOp.getSrefs()) {
      auto srefType = cast<PCF::ShapedRefType>(sref.getType());
      FailureOr<Attribute> memSpace =
          srefType.getScope().getAllocMemSpace(fenceOp.getContext());
      if (failed(memSpace))
        return fenceOp.emitOpError("failed to get memory space for sref");
      if (seenSpaces.insert(*memSpace).second)
        uniqueSpaces.push_back(*memSpace);
    }

    // Emit one iree_codegen.fence per unique memory space.
    for (Attribute space : uniqueSpaces) {
      IREE::Codegen::FenceOp::create(rewriter, fenceOp.getLoc(),
                                     fenceOp.getIsRelease(), space);
    }

    rewriter.eraseOp(fenceOp);
    return success();
  }
};
```

Add to pattern registration (line 1212-1214):

```cpp
patterns.add<ConvertGenericOp, ConvertLoopOp, ConvertWriteSliceOp,
             ConvertReadSliceOp, ConvertGetMemrefOp, ConvertAllocOp,
             ConvertOptimizationBarrier, ConvertFenceOp>(typeConverter, context);
```

Wait - `ConvertFenceOp` uses `OpRewritePattern`, not `OpConversionPattern`. It doesn't need type conversion since it erases the op entirely. Check if the pattern list accepts both. If the existing patterns all use `OpConversionPattern`, then `ConvertFenceOp` should be added separately:

```cpp
// Non-conversion patterns (don't need type converter).
RewritePatternSet nonConvPatterns(context);
nonConvPatterns.add<ConvertFenceOp>(context);
```

Or make it an `OpConversionPattern` with unused adaptor. Better to check the existing pattern registration code first. If it uses `ConversionPatternRewriter`, use `OpConversionPattern`:

```cpp
struct ConvertFenceOp : public OpConversionPattern<PCF::FenceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(PCF::FenceOp fenceOp,
                                PCF::FenceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // ... same body as above ...
  }
};
```

Add include:

```cpp
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
```

Add `IREECodegenDialect` to BUILD.bazel deps for `ConvertSRefToMemRef`.

**Step 2: Write test**

Find or create a test file for ConvertSRefToMemRef lowering. Search for existing test:

```bash
find compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/ -name '*sref*' -o -name '*memref*'
```

Create test case (append to appropriate test file or create new one):

```mlir
// RUN: iree-opt --iree-pcf-convert-sref-to-memref %s | FileCheck %s

func.func @fence_lowering(%alloc: !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>) {
  pcf.fence release %alloc : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
  pcf.fence acquire %alloc : !pcf.sref<128x64xf16, #iree_gpu.subgroup_scope>
  return
}

// CHECK-LABEL: func @fence_lowering
//       CHECK:   iree_codegen.fence release #gpu.address_space<workgroup>
//       CHECK:   iree_codegen.fence acquire #gpu.address_space<workgroup>
```

**Step 3: Build and run test**

```bash
ninja -C /home/quinn/root/iree-pingpong/iree-build iree-opt
# Run appropriate test
```

Expected: PASS. `pcf.fence` on workgroup-scoped sref lowered to `iree_codegen.fence` with `#gpu.address_space<workgroup>`.

**Step 4: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/ConvertSRefToMemRef.cpp \
        compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/BUILD.bazel \
        compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/CMakeLists.txt \
        compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/test/
git commit -m "[PCF] Lower pcf.fence to iree_codegen.fence in ConvertSRefToMemRef"
```

---

### Task 7: Change `addBarrier` to emit `global_subgroup_barrier`

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/GPU/ExternalInterfaces/GPUScopeExternalModels.cpp:77-80`

**Step 1: Change SubgroupScopeModel::addBarrier**

Replace lines 77-80:

```cpp
// OLD:
LogicalResult addBarrier(Attribute attr, OpBuilder &builder) const {
  gpu::BarrierOp::create(builder, builder.getUnknownLoc());
  return success();
}

// NEW:
LogicalResult addBarrier(Attribute attr, OpBuilder &builder) const {
  GPU::GlobalSubgroupBarrierOp::create(builder, builder.getUnknownLoc());
  return success();
}
```

Add include at top:

```cpp
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
```

This change affects:
- `pcf.barrier(#iree_gpu.subgroup_scope)` lowering in LowerStructuralPCF.cpp
- `sync_on_return=true` on `pcf.loop` and `pcf.generic`

**Step 2: Verify existing tests still pass**

The barrier lowering tests may check for `gpu.barrier` - update them to check for `iree_gpu.global_subgroup_barrier` instead.

**Step 3: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/Dialect/GPU/ExternalInterfaces/GPUScopeExternalModels.cpp
git commit -m "[GPU] Change addBarrier to emit global_subgroup_barrier"
```

---

### Task 8: Add fences around `sync_on_return` barriers

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/LowerStructuralPCF.cpp:72-79,135-142`

**Step 1: Modify LowerGenericOp sync handling (lines 72-79)**

Replace:

```cpp
if (genericOp.getSyncOnReturn()) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(genericOp);
  if (failed(genericOp.getScope().addBarrier(rewriter))) {
    genericOp.emitOpError("failed to construct requested barrier");
    return failure();
  }
}
```

With:

```cpp
if (genericOp.getSyncOnReturn()) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(genericOp);

  // Emit release fence before barrier.
  FailureOr<Attribute> allocMemSpace =
      genericOp.getScope().getAllocMemSpace(genericOp.getContext());
  if (succeeded(allocMemSpace)) {
    IREE::Codegen::FenceOp::create(rewriter, genericOp.getLoc(),
                                   /*is_release=*/true, *allocMemSpace);
  }

  // Barrier.
  if (failed(genericOp.getScope().addBarrier(rewriter))) {
    genericOp.emitOpError("failed to construct requested barrier");
    return failure();
  }

  // Emit acquire fence after barrier.
  if (succeeded(allocMemSpace)) {
    IREE::Codegen::FenceOp::create(rewriter, genericOp.getLoc(),
                                   /*is_release=*/false, *allocMemSpace);
  }
}
```

**Step 2: Apply same change to LowerLoopOp (lines 135-142)**

Same pattern for `loopOp.getSyncOnReturn()`.

**Step 3: Add include**

```cpp
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
```

Add `IREECodegenDialect` dep to BUILD.bazel for LowerStructuralPCF if not present.

**Step 4: Verify existing tests**

Update any tests that check for bare barriers after `sync_on_return=true` ops.

**Step 5: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/LowerStructuralPCF.cpp \
        compiler/src/iree/compiler/Codegen/Dialect/PCF/Transforms/BUILD.bazel
git commit -m "[PCF] Add fences around sync_on_return barriers"
```

---

### Task 9: Add `pcf.fence` ops to PingpongConfig.cpp

**Files:**
- Modify: `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/PingpongConfig.cpp:228-390`

**Step 1: Add include**

```cpp
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
// (should already be included for BarrierOp)
```

**Step 2: Add fences per the design doc**

Reference the design doc fence placement. Helper to emit fence:

```cpp
auto emitFence = [&](bool isRelease, ValueRange srefs) {
  SmallVector<Value> srefVec(srefs);
  IREE::PCF::FenceOp::create(builder, loc, isRelease, srefVec);
};
```

**EVEN path changes** (lines 228-286):

After copy prologue (line 226-227), before barrier A (line 229):
```cpp
// Fence release: prologue wrote to LDS.
emitFence(/*isRelease=*/true, {lhsAllocArg, rhsAllocArg});
```

After barrier B (line 231), before the loop:
```cpp
// Fence acquire: prepare for loop reads from LDS.
emitFence(/*isRelease=*/false, {lhsAllocArg, rhsAllocArg});
```

Inside loop body, after read (line 261), before barrier C (line 264):
```cpp
// Fence release: copied to LDS + read from LDS.
emitFence(/*isRelease=*/true, {lhsAllocArg, rhsAllocArg});
```

After barrier D (line 273), before yield:
```cpp
// Fence acquire: see odd's writes for next iteration.
emitFence(/*isRelease=*/false, {lhsAllocArg, rhsAllocArg});
```

**ODD path changes** (lines 288-398):

After copy prologue (line 315), before barrier A (line 318):
```cpp
// Fence release: prologue wrote to LDS.
emitFence(/*isRelease=*/true, {lhsAllocArg, rhsAllocArg});
```

After barrier A (line 318), before read (line 321):
```cpp
// Fence acquire: prepare to read buf0.
emitFence(/*isRelease=*/false, {lhsAllocArg, rhsAllocArg});
```

NO fence before/after barrier B (line 329) - odd only read since A.

Inside loop body, after barrier C (line 354):
```cpp
// Fence acquire: see even's writes.
emitFence(/*isRelease=*/false, {lhsAllocArg, rhsAllocArg});
```

After read (line 370), before barrier D (line 373):
```cpp
// Fence release: copied to LDS + read from LDS.
emitFence(/*isRelease=*/true, {lhsAllocArg, rhsAllocArg});
```

NO fences around barriers E+F (lines 389-390) - structural only.

**Step 3: Build**

```bash
ninja -C /home/quinn/root/iree-pingpong/iree-build iree-opt iree-compile
```

**Step 4: Commit**

```bash
git add compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/PingpongConfig.cpp
git commit -m "[LLVMGPU] Add pcf.fence ops to pingpong schedule"
```

---

### Task 10: E2E verification

**Files:** None modified (verification only)

**Step 1: Compile the matmul**

```bash
cd /home/quinn/root/iree-pingpong
./compile_pingpong.sh
```

Verify no crashes.

**Step 2: Inspect IR**

Dump IR after convert-to-rocdl and verify:
- No `gpu.barrier` ops remain.
- `iree_gpu.global_subgroup_barrier` ops are present (not merged).
- `llvm.fence` ops with MMRA attributes are present around (not inside) barrier calls.
- Consecutive barriers (A+B in even, E+F in odd) are preserved.

**Step 3: Run correctness test**

```bash
ASAN_OPTIONS=detect_leaks=0 /home/quinn/root/iree-pingpong/iree-build/tools/iree-run-module \
  --module=/home/quinn/root/iree-pingpong/pingpong_matmul.vmfb \
  --function=matmul \
  --input=2048x2048xf16=@/home/quinn/root/iree-pingpong/lhs.npy \
  --input=2048x2048xf16=@/home/quinn/root/iree-pingpong/rhs.npy \
  --output=@/tmp/claude/actual_output.npy \
  --device=hip
```

Compare with expected:

```python
import numpy as np
actual = np.load('/tmp/claude/actual_output.npy')
expected = np.load('/home/quinn/root/iree-pingpong/expected.npy')
diff = np.abs(actual - expected)
correct = np.sum(diff < 0.5)
total = actual.size
print(f'Correct: {correct}/{total} ({100*correct/total:.1f}%)')
print(f'Max abs diff: {np.max(diff):.0f}')
```

Expected: ~100% correct (up from 50%). Max abs diff should be small (numerical precision only).

**Step 4: Commit status update**

If E2E passes:
```bash
# Update PINGPONG_STATUS.md
# Change "FAILING" to "PASSING"
git add PINGPONG_STATUS.md
git commit -m "[LLVMGPU] Fix pingpong correctness with barrier/fence separation"
```

---

## Dependency Graph

```
Task 1 (barrier op def) ──→ Task 2 (ROCDL lowering) ──→ Task 7 (addBarrier change)
                                                    ╲
Task 3 (fence op def)  ──→ Task 4 (ROCDL fence)     ╲
                        ╲                             ╲
                         → Task 8 (sync_on_return)     → Task 10 (E2E verify)
                                                      ╱
Task 5 (pcf.fence def) ──→ Task 6 (SRef lowering) ──╱
                        ╲                           ╱
                         → Task 9 (PingpongConfig) ╱
```

Tasks 1, 3, 5 can be done in parallel (independent op definitions).
Tasks 2, 4, 6 depend on their respective op definitions.
Task 7 depends on Task 1.
Task 8 depends on Tasks 3 and 7.
Task 9 depends on Task 5.
Task 10 depends on all other tasks.
