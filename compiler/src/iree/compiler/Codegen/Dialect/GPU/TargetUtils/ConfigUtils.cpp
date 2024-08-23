// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include <limits>

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-gpu-config-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static llvm::cl::opt<bool> clGPUPromoteCMatrix(
    "iree-gpu-promote-c-matrix",
    llvm::cl::desc(
        "control whether to promote the C matrix in matmul strategies"),
    llvm::cl::init(false));

namespace mlir::iree_compiler::IREE::GPU {

LogicalResult setMatmulLoweringConfig(IREE::GPU::TargetAttr target,
                                      mlir::FunctionOpInterface entryPoint,
                                      Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) {
    return failure();
  }

  if (target.getWgp().getMma().empty())
    return failure();

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  mlir::linalg::ContractionDimensions contractionDims =
      mlir::linalg::inferContractionDims(linalgOp).value();

  if (contractionDims.k.empty() || contractionDims.m.empty() ||
      contractionDims.n.empty()) {
    return failure();
  }

  // For now we are not being smart and trying to reshape dimensions to allow
  // for better usage of intrinsics, and instead are tiling all dimensions
  // except the inner most m, n, and k dimensions to 1.
  int64_t mDim = contractionDims.m.back();
  int64_t nDim = contractionDims.n.back();
  int64_t kDim = contractionDims.k.back();

  // Dynamic k dims are currently unsupported.
  if (ShapedType::isDynamic(bounds[kDim])) {
    return failure();
  }

  Value lhs = linalgOp.getDpsInputOperand(0)->get();
  Value rhs = linalgOp.getDpsInputOperand(1)->get();
  Value init = linalgOp.getDpsInitOperand(0)->get();

  Type lhsElemType = getElementTypeOrSelf(lhs);
  Type rhsElemType = getElementTypeOrSelf(rhs);
  Type initElemType = getElementTypeOrSelf(init);

  GPUMatmulShapeType problem{bounds[mDim], bounds[nDim], bounds[kDim],
                             lhsElemType,  rhsElemType,  initElemType};

  if (ShapedType::isDynamic(problem.mSize)) {
    problem.mSize = 1024;
  }
  if (ShapedType::isDynamic(problem.nSize)) {
    problem.nSize = 1024;
  }

  problem.mSize = llvm::divideCeil(problem.mSize, 32) * 32;
  problem.nSize = llvm::divideCeil(problem.nSize, 32) * 32;
  problem.kSize = llvm::divideCeil(problem.kSize, 32) * 32;

  bool needsPadding = problem.mSize != bounds[mDim] ||
                      problem.nSize != bounds[nDim] ||
                      problem.kSize != bounds[kDim];

  SmallVector<GPUMatmulShapeType> intrinsics;
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    if (mma.getSubgroupSize() != targetSubgroupSize)
      continue;
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType);
  }
  if (intrinsics.empty())
    return failure();

  GPUMMAHeuristicSeeds seeds;

  // Note that the following heuristic seeds are just placeholder values.
  // We need to clean it up and make it adjusting to different targets.
  // See https://github.com/iree-org/iree/issues/16341 for details.
  if (problem.mSize * problem.nSize <= 512 * 512) {
    // For matmuls with small M*N size, we want to distribute M*N onto more
    // workgroups to fill the GPU. Use a smaller bestMNTileCountPerSubgroup
    // and a larger bestKTileCountPerSubgroup.
    seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
             /*bestMNTileCountPerSubgroup=*/4,
             /*bestKTileCountPerSubgroup=*/8};
  } else {
    seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
             /*bestMNTileCountPerSubgroup=*/8,
             /*bestKTileCountPerSubgroup=*/4};
  }

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  LDBG("Matmul TileAndFuse Config");

  // Infer if lhs or rhs is transposed to help generate better schedule.
  // TODO: Drop this. This is only a consideration for other pipelines.
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  bool transposedLhs =
      kDim !=
      llvm::cast<AffineDimExpr>(maps[0].getResults().back()).getPosition();
  bool transposedRhs =
      nDim !=
      llvm::cast<AffineDimExpr>(maps[1].getResults().back()).getPosition();

  // First try to find a schedule with an exactly matching intrinsic.
  std::optional<GPUMMASchedule> schedule =
      deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                        targetSubgroupSize, transposedLhs, transposedRhs);
  if (!schedule) {
    // Then try again by allowing upcasting accumulator.
    schedule = deduceMMASchedule(
        problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize,
        transposedLhs, transposedRhs, /*canUpcastAcc=*/true);
  }

  if (!schedule) {
    LDBG("Failed to deduce TileAndFuse MMA schedule");
    return failure();
  }

  LDBG("Target Subgroup size: " << targetSubgroupSize);
  LDBG("Schedule: sizes [" << schedule->mSize << ", " << schedule->nSize << ", "
                           << schedule->kSize << "]");
  LDBG("Schedule: tile counts [" << schedule->mTileCount << ", "
                                 << schedule->nTileCount << ", "
                                 << schedule->kTileCount << "]");
  LDBG("Schedule: warp counts [" << schedule->mWarpCount << ", "
                                 << schedule->nWarpCount << "]");

  std::array<int64_t, 3> workgroupSize{
      schedule->nWarpCount * targetSubgroupSize, schedule->mWarpCount, 1};

  SmallVector<int64_t> workgroupTileSizes(linalgOp.getNumLoops(), 0);
  SmallVector<int64_t> reductionTileSizes(linalgOp.getNumLoops(), 0);
  SmallVector<int64_t> subgroupTileSizes(linalgOp.getNumLoops(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : contractionDims.batch) {
    workgroupTileSizes[batch] = 1;
  }

  // Tile all m, n, and k dimensions to 1 except the innermost. Unit dims
  // from this tiling are folded before vectorization.
  for (int64_t m : llvm::drop_end(contractionDims.m)) {
    workgroupTileSizes[m] = 1;
  }
  for (int64_t n : llvm::drop_end(contractionDims.n)) {
    workgroupTileSizes[n] = 1;
  }
  for (int64_t k : llvm::drop_end(contractionDims.k)) {
    reductionTileSizes[k] = 1;
  }

  // Compute the M/N dimension tile size by multiplying subgroup information.
  workgroupTileSizes[mDim] =
      schedule->mWarpCount * schedule->mTileCount * schedule->mSize;
  workgroupTileSizes[nDim] =
      schedule->nWarpCount * schedule->nTileCount * schedule->nSize;

  // Specify the subgroup tile sizes from the mma schedule. This is applied
  subgroupTileSizes[mDim] = schedule->mTileCount;
  subgroupTileSizes[nDim] = schedule->nTileCount;

  // Similarly the reduction tile size is just the post-packing tile count.
  reductionTileSizes[kDim] = schedule->kTileCount;

  IREE::GPU::MmaInterfaceAttr mmaKind =
      target.getWgp().getMma()[schedule->index];

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  MLIRContext *context = linalgOp.getContext();
  SmallVector<NamedAttribute, 1> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getIndexArrayAttr(workgroupTileSizes));
  attrs.emplace_back(StringAttr::get(context, "reduction"),
                     b.getIndexArrayAttr(reductionTileSizes));
  attrs.emplace_back(StringAttr::get(context, "subgroup"),
                     b.getIndexArrayAttr(subgroupTileSizes));
  attrs.emplace_back(StringAttr::get(context, "mma_kind"), mmaKind);
  if (clGPUPromoteCMatrix || needsPadding) {
    attrs.emplace_back(StringAttr::get(context, "promote_c"),
                       UnitAttr::get(context));
  }
  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, targetSubgroupSize);
}

LogicalResult setTileAndFuseLoweringConfig(IREE::GPU::TargetAttr target,
                                           mlir::FunctionOpInterface entryPoint,
                                           Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // Bail out on multi result cases as consumer fusion currently does not
  // support multi result ops.
  if (!linalgOp || linalgOp.getNumDpsInits() != 1) {
    return failure();
  }

  // This pipeline requires tensor semantics. Also fail for gather semantics
  // for now to simplify tile + fuse.
  if (!linalgOp.hasPureTensorSemantics() || linalgOp.hasIndexSemantics()) {
    return failure();
  }

  SmallVector<unsigned int> partitionableLoops;
  linalgOp.getParallelDims(partitionableLoops);

  // Bail out if op is not tilable.
  if (partitionableLoops.empty()) {
    return failure();
  }

  const int subgroupSize = target.getPreferredSubgroupSize();
  const unsigned loopDepth = linalgOp.getNumLoops();

  // Configurations we need to decide.
  std::array<int64_t, 3> workgroupSize;
  SmallVector<int64_t> workgroupTileSizes;
  SmallVector<int64_t> threadTileSizes;

  // Initialize the configuration.
  auto initConfiguration = [&]() {
    workgroupSize = {subgroupSize, 1, 1};
    workgroupTileSizes.resize(loopDepth, 0);
    threadTileSizes.resize(loopDepth, 0);

    // Initialize tiling along all partitioned loops with size 1.
    for (int64_t loopIndex : partitionableLoops) {
      workgroupTileSizes[loopIndex] = threadTileSizes[loopIndex] = 1;
    }
    // Override the innermost dimension to distribute to threads in a subgroup.
    workgroupTileSizes[partitionableLoops.back()] = subgroupSize;
  };

  // Common case for all linalg ops.

  // The core idea is to distribute the partitioned loops to the workgroup
  // dimensions. The goal is to fill up the GPU as much as possible, which means
  // 1) distributing to as many threads as possible, and 2) avoid assigning too
  // many threads to handle out-of-bound elements (thus idle).

  auto elementHasPowerOfTwoBitwidth = [](Value operand) {
    Type elementType = getElementTypeOrSelf(operand.getType());
    return isa<IntegerType, FloatType>(elementType) &&
           llvm::isPowerOf2_64(IREE::Util::getTypeBitWidth(elementType));
  };

  // Whether we can try to use the vectorization pipeline.
  SmallVector<int64_t> loopBounds = linalgOp.getStaticLoopRanges();
  bool projPerm =
      llvm::all_of(linalgOp.getIndexingMapsArray(),
                   [](AffineMap map) { return map.isProjectedPermutation(); });
  bool powTwo =
      llvm::all_of(linalgOp->getOperands(), elementHasPowerOfTwoBitwidth);
  bool staticShape = llvm::none_of(loopBounds, ShapedType::isDynamic);

  // Require all affine maps to be projected permutation so that we can
  // generate vector transfer ops.
  bool vectorizable = projPerm && powTwo && staticShape;

  const unsigned minBitwidth = getMinElementBitwidth(linalgOp);
  // Make sure we use a tile size that results in some integral number of bytes.
  const unsigned scaleToByte =
      std::max(8 / minBitwidth, static_cast<unsigned>(1));

  // Distribute workload to the given `numThreads` by allowing a potental loss.
  auto distributeToThreads = [&](int64_t numThreads,
                                 std::optional<int64_t> lossFactor =
                                     std::nullopt) {
    LDBG("Loss factor: " << lossFactor << "\n");
    initConfiguration();
    // If there are more than 3 parallel dim try to tile the extra higher level
    // dimensions to 1 for extra dimensions.
    if (isa<linalg::GenericOp>(linalgOp.getOperation())) {
      for (auto [i, tileSize] : llvm::enumerate(workgroupTileSizes)) {
        if (tileSize != 0)
          break;
        if (loopBounds[i] != 1)
          tileSize = 1;
      }
    }
    // Scan from the innermost shape dimension and try to deduce the
    // configuration for the corresponding GPU workgroup dimension.
    int64_t wgDim = 0;
    for (auto shapeDim : llvm::reverse(partitionableLoops)) {
      int64_t loopBound = loopBounds[shapeDim];
      // Skip dynamic dimensions.
      if (ShapedType::isDynamic(loopBound))
        continue;

      // Try to find some power of two that can devide the current shape dim
      // size. This vector keeps the candidate tile sizes.
      SmallVector<int64_t, 8> candidates;

      // For the inner most workgroup dim, try to see if we can have 4
      // elements per thread. This enables vectorization.
      if (vectorizable && wgDim == 0 && !lossFactor) {
        candidates.push_back(4 * numThreads);
      }
      // Try all power of two numbers up to the subgroup size.
      for (unsigned i = numThreads; i >= 1; i >>= 1) {
        candidates.push_back(i);
      }
      LLVM_DEBUG({
        llvm::dbgs() << "Base candidate tile sizes: [";
        llvm::interleaveComma(candidates, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      for (int64_t candidate : candidates) {
        int64_t scaledTileSize = candidate * scaleToByte;
        if (loopBound % scaledTileSize != 0) {
          if (!lossFactor)
            continue;
          // Skip this candidate if it causes many threads to be idle.
          int64_t idleThreads = candidate - (loopBound % scaledTileSize);
          if (idleThreads > candidate / *lossFactor)
            continue;
        }
        // If the workload is too small and we cannot distribute to more than 2
        // workgroups, try a smaller tile size to increase parallelism.
        if (partitionableLoops.size() == 1 && candidate > subgroupSize &&
            llvm::divideCeil(loopBound, scaledTileSize) <= 2) {
          continue;
        }

        // Found a suitable candidate. Try to let each thread handle 4
        // elements if this is the workgroup x dimension.
        // TODO: Try to take into account element type bit width to get
        // 4xdword reads instead of 4x{elements}.
        workgroupTileSizes[shapeDim] = scaledTileSize;
        LLVM_DEBUG(llvm::dbgs()
                   << "Chosen workgroup tile size: " << scaledTileSize << "\n");
        if (vectorizable && wgDim == 0 && !lossFactor && candidate % 4 == 0) {
          // Use size-1 vectors to increase parallelism if larger ones causes
          // idle threads in the subgroup.
          bool hasIdleThreads =
              partitionableLoops.size() == 1 && candidate <= subgroupSize;
          int vectorSize = hasIdleThreads ? 1 : 4;
          LLVM_DEBUG(llvm::dbgs() << "Use vector size: " << vectorSize << "\n");
          threadTileSizes[shapeDim] = vectorSize * scaleToByte;
          workgroupSize[wgDim] = candidate / vectorSize;
          assert(numThreads % (candidate / vectorSize) == 0);
          numThreads /= candidate / vectorSize;
        } else {
          if (wgDim == 0)
            vectorizable = false;
          threadTileSizes[shapeDim] = scaleToByte;
          workgroupSize[wgDim] = candidate;
          assert(numThreads % candidate == 0);
          numThreads /= candidate;
        }
        assert(numThreads >= 1);
        break;
      }

      // Stop if we have distributed all threads.
      if (numThreads == 1)
        break;
      wgDim++;
    }
    return numThreads;
  };

  // First try to see if we can use up all threads without any loss.
  if (distributeToThreads(subgroupSize) != 1) {
    // Otherwise, allow larger and larger loss factor.

    // Threads for distribution. Use 32 at least.
    int64_t numThreads = std::max(subgroupSize, 32);
    // We can tolerate (1 / lossFactor) of threads in the workgroup to be idle.
    int64_t lossFactor = 32;

    for (; lossFactor >= 1; lossFactor >>= 1) {
      if (distributeToThreads(numThreads, lossFactor) == 1)
        break;
    }
  }

  // TODO(qedawkins): Currently scf.forall resolution only supports static
  // trip counts, meaning the workgroup tile size must perfectly divide the
  // loop bound (and thread tile size must perfectly divide the workgroup tile)
  // so that the trip count won't be static. Remove this check once proper
  // dynamic trip count resolution support is added.
  for (auto [loopId, threadTile] : llvm::enumerate(threadTileSizes)) {
    if (threadTile == 0) {
      continue;
    }
    int64_t bound = loopBounds[loopId];
    int64_t wkgpTile = workgroupTileSizes[loopId];
    if (bound % wkgpTile != 0 || wkgpTile % threadTile != 0) {
      return failure();
    }
  }

  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(threadTileSizes);

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  MLIRContext *context = linalgOp.getContext();
  SmallVector<NamedAttribute, 1> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getIndexArrayAttr(workgroupTileSizes));

  attrs.emplace_back(StringAttr::get(context, "thread"),
                     b.getIndexArrayAttr(threadTileSizes));

  // Heuristic value chosen to limit maximum vector sizes when tiling below.
  const unsigned maxVectorSize = 32;

  // Try to tile all reductions by some small factor, preferrably 4, when
  // possible. This gives us a chance to perform vector4 load if an input has
  // its innnermost dimension being reduction. It also avoids generating too
  // many instructions when unrolling vector later. We limit the expected
  // vector size by estimating it from the size of the iteration space tile and
  // limit it to a reasonable value. We process the loops from inner most to
  // outer most to try to align loads along inner dimensions.
  int64_t vectorSize = 1;
  int64_t numLoops = linalgOp.getNumLoops();
  SmallVector<utils::IteratorType> iterTypes = linalgOp.getIteratorTypesArray();
  SmallVector<int64_t> loopTileSizes(numLoops, 0);
  for (auto [reverseIdx, iter] : llvm::enumerate(llvm::reverse(iterTypes))) {
    unsigned i = numLoops - reverseIdx - 1;
    if (linalg::isReductionIterator(iter) || i >= workgroupTileSizes.size() ||
        workgroupTileSizes[i] == 0) {
      int64_t tileSize = getReductionTilingFactor(loopBounds[i]);
      if (vectorSize * tileSize > maxVectorSize) {
        tileSize = 1;
      }
      vectorSize *= tileSize;
      loopTileSizes[i] = tileSize;
    }
  }
  if (llvm::any_of(loopTileSizes, [](int64_t s) { return s != 0; })) {
    attrs.emplace_back(StringAttr::get(context, "reduction"),
                       b.getIndexArrayAttr(loopTileSizes));
  }

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  LDBG("Selected tile and fuse lowering config: " << loweringConfig << "\n");

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, subgroupSize, DictionaryAttr());
}

} // namespace mlir::iree_compiler::IREE::GPU
