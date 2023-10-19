// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_UTILS_EXTERN_BUILDING_UTILS_H_
#define IREE_COMPILER_DIALECT_HAL_UTILS_EXTERN_BUILDING_UTILS_H_

#include "iree/compiler/Dialect/HAL/Utils/ExternBuildingUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

/// Helper to build a hal.dispatch.extern op with the given arguments. This
/// returns the block arguments of the workgroup count region and the extern
/// op. Note that the workgroup count region will not include the terminator
/// and that is left up to the user to properly populate.
static FailureOr<Operation *>
createDispatchExtern(PatternRewriter &rewriter, ValueRange workload,
                     TypeRange resultTypes, ValueRange resultDims,
                     ValueRange arguments, ValueRange argumentDims,
                     DenseI64ArrayAttr tiedOperands, DictionaryAttr attrDict) {
  Location rootLoc = (*arguments.begin()).getLoc();
  SmallVector<int64_t> tiedOperandsIntList(tiedOperands.asArrayRef());
  SmallVector<NamedAttribute> namedAttributes(attrDict.begin(), attrDict.end());
  Operation *externOp = rewriter.create<DispatchExternOp>(
      rootLoc, workload, resultTypes, resultDims, arguments, argumentDims,
      tiedOperandsIntList, namedAttributes);
  return externOp;
}

/// Helper to emplace a block on the given hal.dispatch.extern op. This returns
/// the block arguments of the updated workgroup count region. Note that the
/// workgroup count region will not include the terminator and that is left up
/// to the user to properly populate.
static FailureOr<ValueRange>
emplaceExternWorkgroupCountRegion(PatternRewriter &rewriter, Operation *op) {
  Location rootLoc = op->getLoc();
  auto externOp = dyn_cast<DispatchExternOp>(op);
  if (!externOp) {
    return failure();
  }

  SmallVector<Type> countTypes({rewriter.getType<IREE::HAL::DeviceType>()});
  SmallVector<Location> countLocs({rootLoc});
  for (auto workloadIdx : externOp.getWorkload()) {
    countTypes.push_back(workloadIdx.getType());
    countLocs.push_back(workloadIdx.getLoc());
  }

  auto &entryBlock = externOp.getWorkgroupCount().emplaceBlock();
  auto countArgs = entryBlock.addArguments(countTypes, countLocs);

  ArrayRef<BlockArgument> countArgsArray(countArgs.begin(), countArgs.end());

  /// Update the insertion point to the beginning of the block to enable
  /// contructing the workgroup count region.
  rewriter.setInsertionPointToStart(&entryBlock);
  return ValueRange(countArgsArray);
}
} // namespace

void registerExternDispatchRewriteFunction(PDLPatternModule &pdlPatterns) {
  pdlPatterns.registerRewriteFunction("create_dispatch_extern",
                                      createDispatchExtern);
  pdlPatterns.registerRewriteFunction("emplace_extern_workgroup_count",
                                      emplaceExternWorkgroupCountRegion);
}

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_HAL_UTILS_EXTERN_BUILDING_UTILS_H_
