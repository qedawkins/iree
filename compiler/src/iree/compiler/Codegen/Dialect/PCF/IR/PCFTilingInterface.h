// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFTILINGINTERFACE_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFTILINGINTERFACE_H_

#include "mlir/IR/Value.h"

namespace mlir::iree_compiler::IREE::PCF {

/// Info about how an operand is provided to the distributed op.
struct DistributedOperandInfo {
  /// The replacement value (tensor tile, sref, or full operand).
  Value value;
  /// Whether |value| is a tile or the full operand.
  bool isTile;
};

/// Info about how a result should be produced.
struct DistributedResultInfo {
  /// If non-null, write the result to this sref instead of returning a tensor
  /// tile. Null means "return a tile".
  Value destSref;
};

/// Parameters for a single tiling level (subgroup or lane).
struct TilingLevelParams {
  /// The scope for this tiling level (must implement ScopeAttrInterface).
  Attribute scope;
  /// Tile sizes for each iteration domain dimension. Zero means don't tile.
  SmallVector<OpFoldResult> tileSizes;
};

/// Parameters for multi-level distributed tiling (subgroup + lane +
/// reduction).
struct MultiLevelTilingParams {
  /// Subgroup-level tiling parameters.
  TilingLevelParams subgroup;
  /// Lane-level tiling parameters.
  TilingLevelParams lane;
  /// Tile sizes for reduction dimensions.
  SmallVector<OpFoldResult> reductionTileSizes;
  /// Indices of operands to promote.
  SmallVector<unsigned> operandsToPromote;
  /// If set, use InnerTileDescAttrInterface-based conversion at the lane
  /// level instead of getDistributedImplementation. This is for MMA/intrinsic
  /// ops where lane distribution is handled by the inner tile descriptor.
  Attribute mmaKind;
};

} // namespace mlir::iree_compiler::IREE::PCF

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFTILINGINTERFACE_H_
