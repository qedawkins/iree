// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/UtilCostExternalModels.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Value resolveAndHoist(RewriterBase &rewriter, Value v,
                             Operation *parent) {
  if (failed(moveValueDefinitions(rewriter, {v}, parent))) {
    return Value();
  }
  return v;
}

//===----------------------------------------------------------------------===//
// CostEstimateOpInterface
//===----------------------------------------------------------------------===//

template <typename OpTy, int64_t cost>
struct StaticCostOpInterface
    : public IREE::Util::CostEstimateOpInterface::ExternalModel<
          StaticCostOpInterface<OpTy, cost>, OpTy> {
  OpFoldResult getEstimatedCost(Operation *op, RewriterBase &rewriter) const {
    return rewriter.getIndexAttr(cost);
  }
};

template <typename OpTy>
struct LinalgCostOpInterface
    : public IREE::Util::CostEstimateOpInterface::ExternalModel<
          LinalgCostOpInterface<OpTy>, OpTy> {
  OpFoldResult getEstimatedCost(Operation *op, RewriterBase &rewriter) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);

    // Get the body to count the total cost of one iteration.
    Block *body = linalgOp.getBlock();
    if (!body) {
      return Value();
    }

    OpBuilder::InsertionGuard g(rewriter);

    OpFoldResult singleIterCost = rewriter.getIndexAttr(0);
    for (Operation &containedOp : body->getOperations()) {
      auto costEstimateOp =
          dyn_cast<IREE::Util::CostEstimateOpInterface>(&containedOp);
      if (!costEstimateOp) {
        return Value();
      }

      rewriter.setInsertionPoint(&containedOp);
      OpFoldResult estimatedCost = costEstimateOp.getEstimatedCost(rewriter);
      if (auto v = dyn_cast<Value>(estimatedCost)) {
        Value movedValue = resolveAndHoist(rewriter, v, op);
        if (!movedValue) {
          return Value();
        }
        estimatedCost = movedValue;
      }

      rewriter.setInsertionPoint(op);
      singleIterCost = IREE::LinalgExt::addOfrs(rewriter, op->getLoc(),
                                                singleIterCost, estimatedCost);
    }

    rewriter.setInsertionPoint(op);
    SmallVector<Range> iterationDomain =
        cast<TilingInterface>(op).getIterationDomain(rewriter);
    OpFoldResult numIters = rewriter.getIndexAttr(1);
    for (Range range : iterationDomain) {
      numIters = IREE::LinalgExt::mulOfrs(rewriter, op->getLoc(), numIters,
                                          range.size);
    }

    return IREE::LinalgExt::mulOfrs(rewriter, op->getLoc(), singleIterCost,
                                    numIters);
  }
};

/// Helper structures that iterates over all Op types in `OpTys` and registers
/// them with the given static cost.
template <int64_t cost, typename... Ops>
struct StaticCostOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<StaticCostOpInterface<Ops, cost>>(*context),
     ...);
  }
};

template <typename... Ops>
struct LinalgCostOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<LinalgCostOpInterface<Ops>>(*context), ...);
  }
};

} // namespace

void registerUtilCostExternalModels(DialectRegistry &registry) {
  // Must ensure that any dependent dialects are registered.
  registry.insert<arith::ArithDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<tensor::TensorDialect>();

  registry.addExtension(
      +[](MLIRContext *context, arith::ArithDialect *dialect) {
        // (inaccurately) mark all arith ops as cost 1.
        StaticCostOpInterfaceHelper<
            1, arith::BitcastOp, arith::ExtFOp, arith::ExtUIOp, arith::ExtSIOp,
            arith::FPToSIOp, arith::FPToUIOp, arith::IndexCastOp,
            arith::TruncFOp, arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp,
            arith::MulIOp, arith::DivUIOp>::registerOpInterface(context);
        // Bitcasts should be free.
        StaticCostOpInterfaceHelper<0, arith::BitcastOp>::registerOpInterface(
            context);
      });

  // Hoistable Op Interface registration.

  // Register hoistable op interfaces for linalg ops.
  // We have a specific allow-list for Linalg ops because we want to consider
  // new additions carefully.
  registry.addExtension(+[](MLIRContext *context,
                            linalg::LinalgDialect *dialect) {
    // Structured op implementations and other auxiliary ops.

    // Register all LinalgOps ops. `LinalgOp` is an interface and it is
    // not possible to attach an external interface to an existing
    // interface. Therefore, attach the `LinalgCostOpInterface` to all
    // ops one-by-one.
    LinalgCostOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >::registerOpInterface(context);
    // Index ops are ostensibly fake, and yields are for control flow.
    StaticCostOpInterfaceHelper<0, linalg::IndexOp,
                                linalg::YieldOp>::registerOpInterface(context);
  });
}

} // namespace mlir::iree_compiler
