// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-llvmgpu-layout-analysis-and-distribution"

namespace mlir::iree_compiler {

namespace {

static constexpr int maxTensorDims = 2;
namespace DimType {
static constexpr int Batch0 = 0;  // Batch dimension for tensor dim 0
static constexpr int Batch1 = 1;  // Batch dimension for tensor dim 1
static constexpr int LaneIdZ = 2;
static constexpr int LaneIdY = 3;
static constexpr int LaneIdX = 4;
static constexpr int VecIdZ = 5;
static constexpr int VecIdY = 6;
static constexpr int VecIdX = 7;
static constexpr int NumDims = 8;
}  // namespace DimType

static std::string typeToString(int i) {
  switch (i) {
    case DimType::Batch0:
      return "Batch0";
    case DimType::Batch1:
      return "Batch1";
    case DimType::LaneIdZ:
      return "LaneIdZ";
    case DimType::LaneIdY:
      return "LaneIdY";
    case DimType::LaneIdX:
      return "LaneIdX";
    case DimType::VecIdZ:
      return "VecIdZ";
    case DimType::VecIdY:
      return "VecIdY";
    case DimType::VecIdX:
      return "VecIdX";
    default:
      return "";
  }
}

struct Dimension {
  int type;
  int value;
};

using DimOrderArray = std::array<std::array<Dimension, 3>, maxTensorDims>;
using OrderArray = std::array<std::array<int, 4>, maxTensorDims>;
using DimArray = std::array<int, DimType::NumDims>;

struct Layout {
  Layout(const DimOrderArray &orders,
         const std::array<int, maxTensorDims> &canonicalShape);
  // Updates the batch dims of the layout given the tensor dims
  void updateBatchDims(int dim0, int dim1);
  // Computes the ith dimension expression for a given state
  AffineExpr computeDim(int i, const DimArray &state, OpBuilder &builder);
  bool operator==(const Layout &layout) const { return shape == layout.shape; }
  bool operator!=(const Layout &layout) const { return shape != layout.shape; }
  void debugPrint(llvm::StringRef str) const;

  // Contains the shape of the layout
  DimArray shape;
  // Contains the order of layout dims for each of the tensor dims
  OrderArray order;
  // Shape of the tensor when used in mma
  std::array<int, maxTensorDims> canonicalShape;
  int rank;
};

// MMA Layout Utilities
enum class MMAType {
  M16N8K16,
  NONE,
};

enum class MMAMatrixType { AMatrix, BMatrix, CMatrix };

static std::array<Dimension, 3> getMMADimensions(MMAType mmaType,
                                                 MMAMatrixType matrixType,
                                                 int dim) {
  switch (mmaType) {
    case MMAType::M16N8K16:
      switch (matrixType) {
        case MMAMatrixType::AMatrix:
          if (dim == 0)
            return {{{DimType::LaneIdY, 8},
                     {DimType::VecIdZ, 2},
                     {DimType::LaneIdZ, 1}}};
          return {{{DimType::VecIdX, 2},
                   {DimType::LaneIdX, 4},
                   {DimType::VecIdY, 2}}};
        case MMAMatrixType::BMatrix:
          if (dim == 0)
            return {{{DimType::VecIdX, 2},
                     {DimType::LaneIdX, 4},
                     {DimType::VecIdY, 2}}};
          return {{{DimType::LaneIdY, 8},
                   {DimType::LaneIdZ, 1},
                   {DimType::VecIdZ, 1}}};
        case MMAMatrixType::CMatrix:
          if (dim == 0)
            return {{{DimType::LaneIdY, 8},
                     {DimType::VecIdY, 2},
                     {DimType::LaneIdZ, 1}}};
          return {{{DimType::VecIdX, 2},
                   {DimType::LaneIdX, 4},
                   {DimType::VecIdZ, 1}}};
      }
      break;
    default:
      return {};
  }
}

static std::array<int, 2> getMMACanonicalShape(MMAType mmaType,
                                               MMAMatrixType matrixType) {
  switch (mmaType) {
    case MMAType::M16N8K16:
      switch (matrixType) {
        case MMAMatrixType::AMatrix:
          return {16, 16};
        case MMAMatrixType::BMatrix:
        case MMAMatrixType::CMatrix:
          return {16, 8};
      }
      break;
    default:
      return {};
  }
}

Layout::Layout(const DimOrderArray &dimOrder,
               const std::array<int, maxTensorDims> &canonicalShape) {
  assert((dimOrder.size() > 0) && (dimOrder.size() <= maxTensorDims));
  for (int i = 0; i < maxTensorDims; i++) {
    int j;
    for (j = 0; j < dimOrder[i].size(); j++) {
      Dimension dim = dimOrder[i][j];
      order[i][j] = dim.type;
      shape[dim.type] = dim.value;
    }
    // Add batch dimension to the end
    if (i == 0) {
      order[i][j] = DimType::Batch0;
      shape[DimType::Batch0] = 1;
    } else {
      order[i][j] = DimType::Batch1;
      shape[DimType::Batch1] = 1;
    }
    this->canonicalShape[i] = canonicalShape[i];
  }
  rank = dimOrder.size();
}

void Layout::updateBatchDims(int dim0, int dim1) {
  shape[DimType::Batch0] = dim0 / canonicalShape[0];
  shape[DimType::Batch1] = dim1 / canonicalShape[1];
}

AffineExpr Layout::computeDim(int i, const DimArray &state,
                              OpBuilder &builder) {
  AffineExpr d0, d1, d2;
  bindDims(builder.getContext(), d0, d1, d2);
  AffineExpr dim = builder.getAffineConstantExpr(0);
  AffineExpr dimScale = builder.getAffineConstantExpr(1.0);
  for (const auto &dimType : order[i]) {
    switch (dimType) {
      case DimType::LaneIdX:
        dim = dim + dimScale * d0;
        break;
      case DimType::LaneIdY:
        dim = dim + dimScale * d1;
        break;
      case DimType::LaneIdZ:
        dim = dim + dimScale * d2;
        break;
      default:
        dim = dim + dimScale * builder.getAffineConstantExpr(state[dimType]);
        break;
    }
    dimScale = dimScale * builder.getAffineConstantExpr(shape[dimType]);
  }
  return dim;
}

void Layout::debugPrint(llvm::StringRef str) const {
  LLVM_DEBUG({
    llvm::dbgs() << str << " = \n";
    for (int i = 0; i < DimType::NumDims; i++) {
      llvm::dbgs() << "   " << typeToString(i) << ": " << shape[i] << " ";
      bool isRow{false};
      for (int k = 0; k < order[0].size(); k++) {
        if (order[0][k] == i) {
          isRow = true;
          break;
        }
      }
      if (isRow)
        llvm::dbgs() << "(R)";
      else
        llvm::dbgs() << "(C)";
      llvm::dbgs() << "  ";
    }
    llvm::dbgs() << "\n";
  });
}

static MMAType getMMAType(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                          ArrayRef<int64_t> cShape) {
  if ((aShape[0] % 16 == 0) && (aShape[1] % 16 == 0) && (cShape[0] % 16 == 0) &&
      (cShape[1] % 8 == 0)) {
    if ((bShape[0] % 16 == 0) && (bShape[1] % 8 == 0)) return MMAType::M16N8K16;
  }
  return MMAType::NONE;
}

void setMMALayout(Value aMatrix, Value bMatrix, Value cMatrix,
                  DenseMap<Value, Layout> &layoutMap) {
  // First determine which variant of MMA this op is most suitable for
  auto aType = aMatrix.getType().cast<ShapedType>();
  auto bType = aMatrix.getType().cast<ShapedType>();
  auto cType = aMatrix.getType().cast<ShapedType>();
  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  ArrayRef<int64_t> cShape = cType.getShape();
  MMAType mmaType = getMMAType(aShape, bShape, cShape);
  if (mmaType == MMAType::NONE) return;
  // Set layouts for A, B and C
  auto setLayout = [&](Value matrix, MMAMatrixType matrixType,
                       llvm::StringRef name) {
    DimOrderArray dimOrder;
    for (int i = 0; i < 2; i++) {
      dimOrder[i] = getMMADimensions(mmaType, matrixType, i);
    }
    std::array<int, 2> canonicalShape =
        getMMACanonicalShape(mmaType, matrixType);
    Layout layout(dimOrder, canonicalShape);
    ArrayRef<int64_t> shape = matrix.getType().cast<ShapedType>().getShape();
    layout.updateBatchDims(shape[0], shape[1]);
    layoutMap.try_emplace(matrix, layout);
    layout.debugPrint(name);
  };
  setLayout(aMatrix, MMAMatrixType::AMatrix, "aMatrix");
  setLayout(bMatrix, MMAMatrixType::BMatrix, "bMatrix");
  setLayout(cMatrix, MMAMatrixType::CMatrix, "cMatrix");
}

void propagateLayoutToReduceBroadcastTranspose(
    vector::MultiDimReductionOp reductionOp, vector::BroadcastOp broadcastOp,
    vector::TransposeOp transposeOp, DenseMap<Value, Layout> &layoutMap) {
  if (!broadcastOp) return;
  if (!transposeOp) return;
  Value reductionSrc = reductionOp.getSource();
  if (!layoutMap.count(reductionSrc)) return;
  // Get the reduction dims
  auto reductionDims = llvm::to_vector<4>(
      reductionOp.getReductionDims().getAsRange<IntegerAttr>());
  // Get the transpose permutation
  SmallVector<int64_t> perm;
  transposeOp.getTransp(perm);
  // Don't support dim-1 broadcasted dims
  llvm::SetVector<int64_t> dimOneBroadcastedDims =
      broadcastOp.computeBroadcastedUnitDims();
  if (dimOneBroadcastedDims.size() > 0) return;
  Value broadcastSource = broadcastOp.getSource();
  Value broadcastResult = broadcastOp.getResult();
  int64_t broadcastSourceRank =
      broadcastSource.getType().cast<VectorType>().getRank();
  int64_t broadcastResultRank =
      broadcastResult.getType().cast<VectorType>().getRank();
  int64_t rankDiff = broadcastResultRank - broadcastSourceRank;
  llvm::SetVector<int64_t> broadcastedDims;
  for (int64_t i = 0; i < rankDiff; i++) broadcastedDims.insert(i);
  ArrayRef<int64_t> broadcastShape =
      broadcastResult.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> srcShape =
      reductionSrc.getType().cast<ShapedType>().getShape();
  // Check that the same number of dims are reduced and broadcasted
  if (reductionDims.size() != broadcastedDims.size()) return;
  // Check that transpose(reductionDim) == broadcastDim
  // and that the shapes match
  for (IntegerAttr dimAttr : reductionDims) {
    int64_t dim = dimAttr.getInt();
    int64_t transposedDim = perm[dim];
    if (!broadcastedDims.contains(transposedDim)) return;
    if (srcShape[dim] != broadcastShape[transposedDim]) return;
  }
  Value transposedResult = transposeOp.getResult();
  layoutMap.try_emplace(transposedResult, layoutMap.at(reductionSrc));
  layoutMap.at(transposedResult).debugPrint("transposed");
}

std::tuple<vector::BroadcastOp, vector::TransposeOp>
checkForReduceBroadcastTranspose(vector::MultiDimReductionOp reductionOp) {
  vector::BroadcastOp broadcastOp{nullptr};
  vector::TransposeOp transposeOp{nullptr};
  for (Operation *user : reductionOp.getResult().getUsers()) {
    if (auto broadcast = dyn_cast<vector::BroadcastOp>(user)) {
      for (Operation *bUser : broadcast.getResult().getUsers()) {
        if (auto transpose = dyn_cast<vector::TransposeOp>(bUser)) {
          transposeOp = transpose;
          break;
        }
      }
      broadcastOp = broadcast;
      break;
    }
  }
  return std::make_tuple(broadcastOp, transposeOp);
}

void propagateLayout(Operation *op, DenseMap<Value, Layout> &layoutMap) {
  if (auto reductionOp = dyn_cast<vector::MultiDimReductionOp>(op)) {
    auto [broadcastOp, transposeOp] =
        checkForReduceBroadcastTranspose(reductionOp);
    propagateLayoutToReduceBroadcastTranspose(reductionOp, broadcastOp,
                                              transposeOp, layoutMap);
  }
}

void distributeTransferReads(vector::TransferReadOp readOp,
                             DenseMap<Value, Layout> &layoutMap,
                             DenseMap<Value, Value> &simdToSimtMap,
                             OpBuilder &rewriter,
                             SmallVectorImpl<Operation *> &ops) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(readOp);
  Value result = readOp.getResult();
  if (!layoutMap.count(result)) return;
  Value source = readOp.getSource();
  Location loc = readOp.getLoc();
  SmallVector<Value> indices = readOp.getIndices();
  Type elementType = source.getType().cast<ShapedType>().getElementType();
  Value threadIdX = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value threadIdY = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
  Value threadIdZ = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z);
  Layout layout = layoutMap.at(result);
  auto vecType = VectorType::get(
      {layout.shape[DimType::Batch0], layout.shape[DimType::Batch1],
       layout.shape[DimType::VecIdZ] * layout.shape[DimType::VecIdY],
       layout.shape[DimType::VecIdX]},
      elementType);
  Value vector = rewriter.create<arith::ConstantOp>(
      loc, vecType, rewriter.getZeroAttr(vecType));
  std::array<int, DimType::NumDims> state;
  for (int b0 = 0; b0 < layout.shape[DimType::Batch0]; b0++) {
    state[DimType::Batch0] = b0;
    for (int b1 = 0; b1 < layout.shape[DimType::Batch1]; b1++) {
      state[DimType::Batch1] = b1;
      for (int i = 0; i < layout.shape[DimType::VecIdZ]; i++) {
        state[DimType::VecIdZ] = i;
        for (int j = 0; j < layout.shape[DimType::VecIdY]; j++) {
          state[DimType::VecIdY] = j;
          for (int k = 0; k < layout.shape[DimType::VecIdX]; k++) {
            state[DimType::VecIdX] = k;
            AffineExpr row = layout.computeDim(0, state, rewriter);
            AffineMap rowMap = AffineMap::get(3, 0, row, rewriter.getContext());
            Value rowIndex = rewriter.create<AffineApplyOp>(
                loc, rowMap,
                SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
            AffineExpr col = layout.computeDim(1, state, rewriter);
            AffineMap colMap = AffineMap::get(3, 0, col, rewriter.getContext());
            Value colIndex = rewriter.create<AffineApplyOp>(
                loc, colMap,
                SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
            if (layout.rank == 1) indices.back() = rowIndex;
            if (layout.rank == 2) {
              assert(indices.size() >= 2);
              indices[indices.size() - 2] = rowIndex;
              indices[indices.size() - 1] = colIndex;
            }
            Value el = rewriter.create<memref::LoadOp>(loc, source, indices);
            auto vectorType = VectorType::get({1}, elementType);
            Value v = rewriter.create<vector::BroadcastOp>(loc, vectorType, el);
            SmallVector<int64_t> offsets{
                b0, b1, j * layout.shape[DimType::VecIdZ] + i, k};
            SmallVector<int64_t> strides{1};
            vector = rewriter.create<vector::InsertStridedSliceOp>(
                loc, v, vector, offsets, strides);
          }
        }
      }
    }
  }
  simdToSimtMap.try_emplace(result, vector);
  ops.push_back(readOp);
}

void distributeContracts(vector::ContractionOp contractOp,
                         DenseMap<Value, Layout> &layoutMap,
                         DenseMap<Value, Value> &simdToSimtMap,
                         OpBuilder &rewriter,
                         SmallVectorImpl<Operation *> &ops) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(contractOp);
  Value lhs = contractOp.getLhs();
  if (!layoutMap.count(lhs)) return;
  if (!simdToSimtMap.count(lhs)) return;
  Type elementType = lhs.getType().cast<ShapedType>().getElementType();
  Value rhs = contractOp.getRhs();
  if (!layoutMap.count(rhs)) return;
  if (!simdToSimtMap.count(rhs)) return;
  Location loc = contractOp.getLoc();
  Value contractResult = contractOp.getResult();
  Layout lhsLayout = layoutMap.at(lhs);
  Layout resultLayout = layoutMap.at(contractResult);
  SmallVector<int64_t> vecShape{
      resultLayout.shape[DimType::Batch0], resultLayout.shape[DimType::Batch1],
      resultLayout.shape[DimType::VecIdZ] * resultLayout.shape[DimType::VecIdY],
      resultLayout.shape[DimType::VecIdX]};
  auto vecType = VectorType::get(vecShape, elementType);
  Value result = rewriter.create<arith::ConstantOp>(
      loc, vecType, rewriter.getZeroAttr(vecType));
  int M = resultLayout.shape[DimType::Batch0];
  int N = resultLayout.shape[DimType::Batch1];
  int canonicalM = resultLayout.canonicalShape[0];
  int canonicalN = resultLayout.canonicalShape[1];
  int K = lhsLayout.shape[DimType::Batch1];
  int canonicalK = lhsLayout.canonicalShape[1];
  auto cType = VectorType::get({vecShape[2], vecShape[3]}, elementType);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      Value cMatrix = rewriter.create<arith::ConstantOp>(
          loc, cType, rewriter.getZeroAttr(cType));
      for (int k = 0; k < K; k++) {
        Value aMatrix = rewriter.create<vector::ExtractOp>(
            loc, simdToSimtMap.at(lhs), SmallVector<int64_t>{i, k});
        Value bMatrix = rewriter.create<vector::ExtractOp>(
            loc, simdToSimtMap.at(rhs), SmallVector<int64_t>{k, j});
        cMatrix = rewriter.create<nvgpu::MmaSyncOp>(
            loc, aMatrix, bMatrix, cMatrix,
            rewriter.getI64ArrayAttr({canonicalM, canonicalN, canonicalK}));
      }
      result = rewriter.create<vector::InsertOp>(loc, cMatrix, result,
                                                 SmallVector<int64_t>{i, j});
    }
  }
  simdToSimtMap.try_emplace(contractResult, result);
  ops.push_back(contractOp);
}

void distributeTransferWrites(vector::TransferWriteOp writeOp,
                              DenseMap<Value, Layout> &layoutMap,
                              DenseMap<Value, Value> &simdToSimtMap,
                              OpBuilder &rewriter,
                              SmallVectorImpl<Operation *> &ops) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(writeOp);
  Value vector = writeOp.getVector();
  Value source = writeOp.getSource();
  Location loc = writeOp.getLoc();
  SmallVector<Value> indices = writeOp.getIndices();
  Value threadIdX = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value threadIdY = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
  Value threadIdZ = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z);
  if (!layoutMap.count(vector)) return;
  if (!simdToSimtMap.count(vector)) return;
  Layout layout = layoutMap.at(vector);
  std::array<int, DimType::NumDims> state;
  for (int b0 = 0; b0 < layout.shape[DimType::Batch0]; b0++) {
    state[DimType::Batch0] = b0;
    for (int b1 = 0; b1 < layout.shape[DimType::Batch1]; b1++) {
      state[DimType::Batch1] = b1;
      for (int i = 0; i < layout.shape[DimType::VecIdZ]; i++) {
        state[DimType::VecIdZ] = i;
        for (int j = 0; j < layout.shape[DimType::VecIdY]; j++) {
          state[DimType::VecIdY] = j;
          for (int k = 0; k < layout.shape[DimType::VecIdX]; k++) {
            state[DimType::VecIdX] = k;
            Value v = rewriter.create<vector::ExtractOp>(
                loc, simdToSimtMap.at(vector),
                SmallVector<int64_t>{b0, b1,
                                     j * layout.shape[DimType::VecIdZ] + i, k});
            AffineExpr row = layout.computeDim(0, state, rewriter);
            AffineMap rowMap = AffineMap::get(3, 0, row, rewriter.getContext());
            Value rowIndex = rewriter.create<AffineApplyOp>(
                loc, rowMap,
                SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
            AffineExpr col = layout.computeDim(1, state, rewriter);
            AffineMap colMap = AffineMap::get(3, 0, col, rewriter.getContext());
            Value colIndex = rewriter.create<AffineApplyOp>(
                loc, colMap,
                SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
            if (layout.rank == 1) indices.back() = rowIndex;
            if (layout.rank == 2) {
              assert(indices.size() >= 2);
              indices[indices.size() - 2] = rowIndex;
              indices[indices.size() - 1] = colIndex;
            }
            rewriter.create<memref::StoreOp>(loc, v, source, indices);
          }
        }
      }
    }
  }
  ops.push_back(writeOp);
}

static bool isLaneId(int dimType) {
  return ((dimType == DimType::LaneIdX) || (dimType == DimType::LaneIdY) ||
          (dimType == DimType::LaneIdZ));
}

static bool isVectorId(int dimType) {
  return ((dimType == DimType::VecIdX) || (dimType == DimType::VecIdY) ||
          (dimType == DimType::VecIdZ));
}

static int getLaneIdIndex(std::array<int, 4> &order) {
  for (int i = 0; i < 4; i++) {
    if (isLaneId(order[i])) return i;
  }
  return -1;
}

static int isSingleLaneIdReduced(std::array<int, 4> &order) {
  int count{0};
  for (int i = 0; i < 4; i++) {
    if (isLaneId(order[i])) count++;
  }
  return count == 1;
}

static int getVecSizes(std::array<int, 4> &order, const Layout &layout) {
  int size = 1;
  for (int i = 0; i < 4; i++) {
    if (isVectorId(i)) size *= layout.shape[i];
  }
  return size;
}

using bodyType = std::function<void(std::array<int, DimType::NumDims> &)>;

/// This function iterates over the dimensions of a given column/row order
/// that are not LaneIdX, LaneIdY or LaneIdZ and executes the function body
/// inside the innermost loop. It keeps track of the induction variables
/// in the state array and passes them to the body function.
void iterate(int dimType, ArrayRef<int> order,
             std::array<int, DimType::NumDims> &state, const Layout &layout,
             bodyType body) {
  if (dimType == DimType::NumDims) {
    body(state);
    return;
  }
  if ((std::find(order.begin(), order.end(), dimType) != order.end()) &&
      (!isLaneId(dimType))) {
    for (int i = 0; i < layout.shape[dimType]; i++) {
      state[dimType] = i;
      iterate(dimType + 1, order, state, layout, body);
    }
  } else {
    iterate(dimType + 1, order, state, layout, body);
  }
}

void distributeReductionBroadcastTranspose(
    vector::MultiDimReductionOp reductionOp, vector::BroadcastOp broadcastOp,
    vector::TransposeOp transposeOp, DenseMap<Value, Layout> &layoutMap,
    DenseMap<Value, Value> &simdToSimtMap, OpBuilder &rewriter,
    SmallVectorImpl<Operation *> &ops) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(reductionOp);
  Value source = reductionOp.getSource();
  Type elementType = source.getType().cast<ShapedType>().getElementType();
  if (!layoutMap.count(source)) return;
  if (!simdToSimtMap.count(source)) return;
  if (!broadcastOp) return;
  Location loc = reductionOp.getLoc();
  Layout layout = layoutMap.at(source);
  auto reductionDims = llvm::to_vector<4>(
      reductionOp.getReductionDims().getAsRange<IntegerAttr>());
  vector::CombiningKind combiningKind = reductionOp.getKind();
  // Only support reduction on one dimension
  if (reductionDims.size() > 1) return;
  int reductionDim = reductionDims[0].getInt();
  std::array<int, 4> reductionOrder = layout.order[reductionDim];
  std::array<int, 4> parallelOrder = layout.order[!reductionDim];
  Value acc = reductionOp.getAcc();
  // TODO: Should be able to handle any accumulator type here without
  // too many problems
  APFloat floatValue(0.0);
  if (!matchPattern(acc, m_ConstantFloat(&floatValue))) return;
  SmallVector<int64_t> vecShape{
      layout.shape[DimType::Batch0], layout.shape[DimType::Batch1],
      layout.shape[DimType::VecIdZ] * layout.shape[DimType::VecIdY],
      layout.shape[DimType::VecIdX]};
  auto vecType = VectorType::get(vecShape, elementType);
  Value output = rewriter.create<arith::ConstantOp>(
      loc, vecType, rewriter.getZeroAttr(vecType));

  if (!isSingleLaneIdReduced(reductionOrder)) return;
  int dimIndex = getLaneIdIndex(reductionOrder);
  int dimType = reductionOrder[dimIndex];
  int offset{0};
  switch (dimType) {
    case DimType::LaneIdX:
      offset = 1;
      break;
    case DimType::LaneIdY:
      offset = layout.shape[DimType::LaneIdX];
      break;
    case DimType::LaneIdZ:
      offset = layout.shape[DimType::LaneIdX] * layout.shape[DimType::LaneIdY];
      break;
  }

  bodyType loopBody = [&](std::array<int, DimType::NumDims> &state) {
    Value result = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, floatValue));

    auto reduce = [&](std::array<int, DimType::NumDims> &state) {
      Value vector = simdToSimtMap.at(source);
      int vectorOffset =
          state[DimType::VecIdY] * layout.shape[DimType::VecIdZ] +
          state[DimType::VecIdZ];
      if (std::find(reductionOrder.begin(), reductionOrder.end(),
                    DimType::VecIdX) == reductionOrder.end()) {
        vector = rewriter.create<vector::TransposeOp>(
            loc, vector, ArrayRef<int64_t>{0, 1, 3, 2});
        vectorOffset = state[DimType::VecIdX];
      }

      vector = rewriter.create<vector::ExtractOp>(
          loc, vector,
          SmallVector<int64_t>{state[DimType::Batch0], state[DimType::Batch1],
                               vectorOffset});
      ArrayRef<int64_t> vShape = vector.getType().cast<VectorType>().getShape();
      assert(vShape.size() == 1);

      uint32_t size{32};
      Value mask;
      for (uint64_t i = offset; i < offset * layout.shape[dimType]; i <<= 1) {
        Value packed = packVectorToSupportedWidth(loc, rewriter, vector);
        auto shuffleOp = rewriter.create<gpu::ShuffleOp>(loc, packed, i, size,
                                                         gpu::ShuffleMode::XOR);
        Value unpacked =
            unpackToVector(loc, rewriter, shuffleOp.getShuffleResult(),
                           vector.getType().cast<VectorType>());
        vector = makeArithReduction(rewriter, loc, combiningKind, unpacked,
                                    vector, mask);
      }

      for (int i = 0; i < vShape[0]; i++) {
        Value v = rewriter.create<vector::ExtractOp>(loc, vector,
                                                     SmallVector<int64_t>{i});
        result =
            makeArithReduction(rewriter, loc, combiningKind, result, v, mask);
      }
    };

    // Iterate only over batch dimension
    std::array<int, 1> batchDim = {{reductionOrder.back()}};
    iterate(0, batchDim, state, layout, reduce);

    auto broadcastResult = [&](std::array<int, DimType::NumDims> &state) {
      output = rewriter.create<vector::InsertOp>(
          loc, result, output,
          SmallVector<int64_t>{
              state[DimType::Batch0], state[DimType::Batch1],
              state[DimType::VecIdY] * layout.shape[DimType::VecIdZ] +
                  state[DimType::VecIdZ],
              state[DimType::VecIdX]});
    };

    // Broadcast result to same shape as original
    iterate(0, reductionOrder, state, layout, broadcastResult);
  };

  std::array<int, DimType::NumDims> state;
  state.fill(0);
  iterate(0, parallelOrder, state, layout, loopBody);

  if (transposeOp)
    simdToSimtMap.try_emplace(transposeOp.getResult(), output);
  else
    simdToSimtMap.try_emplace(broadcastOp.getResult(), output);

  ops.push_back(reductionOp);
  ops.push_back(broadcastOp);
  if (transposeOp) ops.push_back(transposeOp);
}

static void eraseOps(SmallVectorImpl<Operation *> &opsToErase,
                     IRRewriter &rewriter) {
  for (int i = opsToErase.size() - 1; i >= 0; i--) {
    assert(opsToErase[i]->getUses().empty());
    rewriter.eraseOp(opsToErase[i]);
  }
}

}  // namespace

void doLayoutAnalysisAndDistribution(IRRewriter &rewriter,
                                     func::FuncOp funcOp) {
  // First compute the layouts (set MMA layouts and propagate to rest)
  DenseMap<Value, Layout> layoutMap;
  funcOp.walk([&](Operation *op) {
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      Value lhs = contractOp.getLhs();
      Value rhs = contractOp.getRhs();
      Value result = contractOp.getResult();
      setMMALayout(lhs, rhs, result, layoutMap);
    } else {
      propagateLayout(op, layoutMap);
    }
    return WalkResult::advance();
  });

  // Apply SIMD to SIMT conversion
  DenseMap<Value, Value> simdToSimtMap;
  SmallVector<Operation *> opsToErase;
  funcOp.walk([&](Operation *op) {
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      distributeTransferReads(readOp, layoutMap, simdToSimtMap, rewriter,
                              opsToErase);
    }
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      distributeContracts(contractOp, layoutMap, simdToSimtMap, rewriter,
                          opsToErase);
    }
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
      distributeTransferWrites(writeOp, layoutMap, simdToSimtMap, rewriter,
                               opsToErase);
    }
    if (auto reductionOp = dyn_cast<vector::MultiDimReductionOp>(op)) {
      auto [broadcastOp, transposeOp] =
          checkForReduceBroadcastTranspose(reductionOp);
      distributeReductionBroadcastTranspose(
          reductionOp, broadcastOp, transposeOp, layoutMap, simdToSimtMap,
          rewriter, opsToErase);
    }
    return WalkResult::advance();
  });

  // Erase old ops
  eraseOps(opsToErase, rewriter);
}

}  // namespace mlir::iree_compiler
