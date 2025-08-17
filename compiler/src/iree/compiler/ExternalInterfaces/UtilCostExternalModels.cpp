// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/UtilCostExternalModels.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// InferIntDivisibilityOpInterface
//===----------------------------------------------------------------------===//

struct ArithDivUIInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          ArithDivUIInferIntDivisibilityOpInterface, arith::DivUIOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto divOp = cast<arith::DivUIOp>(op);

    APInt intVal;
    if (!matchPattern(divOp.getRhs(), m_ConstantInt(&intVal))) {
      return;
    }

    auto lhsDivisibility = getDivisibilityOfOperand(divOp.getLhs(), argDivs[0]);

    uint64_t divUDiv = lhsDivisibility.udiv() / intVal.getZExtValue();
    uint64_t divSDiv = lhsDivisibility.sdiv() / std::abs(intVal.getSExtValue());

    setResultDivs(divOp, IREE::Util::ConstantIntDivisibility(divUDiv, divSDiv));
  }
};

//===----------------------------------------------------------------------===//
// HoistableOpInterface
//===----------------------------------------------------------------------===//

template <typename OpTy, int64_t cost>
struct StaticCostOpInterface
    : public IREE::Util::CostEstimateOpInterface::ExternalModel<
          StaticCostOpInterface<OpTy, cost>, OpTy> {
  OpFoldResult getEstimatedCost(Operation *op, OpBuilder &builder) const {
    return builder.getIndexAttr(cost);
  }
};

template <typename OpTy>
struct LinalgCostOpInterface
    : public IREE::Util::CostEstimateOpInterface::ExternalModel<
          LinalgCostOpInterface<OpTy>, OpTy> {
  Value getEstimatedCost(Operation *op, OpBuilder &builder) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);

    // Get the body to count the total cost of one iteration.
    Block *body = linalgOp.getBlock();
    if (!body) {
      return Value();
    }

    OpBuilder::InsertionGuard g(builder);

    OpFoldResult singleIterCost = builder.getIndexAttr(0);
    for (Operation &containedOp : body->getOperations()) {
      auto costEstimateOp =
          dyn_cast<IREE::Util::CostEstimateOpInterface>(&containedOp);
      if (!costEstimateOp) {
        return Value();
      }

      builder.setInsertionPoint(&containedOp);
    }
  }
};

// The default interface is always hoistable. This acts as an override
// for other default hoistability checks as the interface is checked
// first.
template <typename OpTy>
struct AlwaysHoistableOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          AlwaysHoistableOpInterface<OpTy>, OpTy> {};

template <typename OpTy>
struct HoistableLinalgOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          HoistableLinalgOpInterface<OpTy>, OpTy> {
  bool isHoistableOp(Operation *) const { return true; }

  // Determines if a linalg op is a hoistable leaf, based on heuristics.
  bool isHoistableLeafOp(Operation *op) const {
    // Don't hoist bit extend ops because fusing them with their
    // consumers prevents materializing the high bit-width tensor and they
    // preform very little real computation.
    if (IREE::LinalgExt::isBitExtendOp(op)) {
      return false;
    }

    // Hoist all non-generic linalg ops except for fill ops which should be
    // fused with their consumers.
    auto genericOp = llvm::dyn_cast<linalg::GenericOp>(op);
    if (!genericOp) {
      return !isa<linalg::FillOp>(op);
    }

    // Don't hoist ops with no tensor inputs. They are likely to be fill-like
    // or sequences (from `linalg.index`) which can be fused with their
    // consumers.
    if (IREE::LinalgExt::hasOnlyScalarInputs(genericOp)) {
      return false;
    }

    // Don't hoist broadcast-like ops because fusing them makes the new
    // op cheaper.
    if (linalg::isaBroadcastOpInterface(genericOp).has_value()) {
      return false;
    }

    // Hoist all other ops.
    return true;
  }
  bool isAtomicallyHoistableOp(Operation *) const { return true; }
  bool isOperandHoistable(Operation *, OpOperand *) const { return true; }
};

/// Helper structures that iterates over all Op types in `OpTys` and registers
/// the associated Hoistable___OpInterface.
template <typename... Ops>
struct UnhoistableOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<UnhoistableOpInterface<Ops>>(*context), ...);
  }
};

template <typename... Ops>
struct HoistableNonLeafOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<HoistableNonLeafOpInterface<Ops>>(*context),
     ...);
  }
};

template <typename... Ops>
struct AlwaysHoistableOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<AlwaysHoistableOpInterface<Ops>>(*context),
     ...);
  }
};

template <typename... Ops>
struct HoistableLinalgOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<HoistableLinalgOpInterface<Ops>>(*context),
     ...);
  }
};

} // namespace

void registerUtilExternalModels(DialectRegistry &registry) {
  // Must ensure that any dependent dialects are registered.
  registry.insert<arith::ArithDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<tensor::TensorDialect>();

  registry.addExtension(
      +[](MLIRContext *context, ml_program::MLProgramDialect *dialect) {
        ml_program::GlobalOp::attachInterface<GlobalOpInterfaceExternalModel>(
            *context);
      });

  registry.addExtension(+[](MLIRContext *context,
                            arith::ArithDialect *dialect) {
    GenericNumericCastExternalModel::add<
        arith::BitcastOp, arith::ExtFOp, arith::ExtUIOp, arith::ExtSIOp,
        arith::FPToSIOp, arith::FPToUIOp, arith::IndexCastOp, arith::TruncFOp,
        arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp>(context);
    arith::ConstantOp::attachInterface<
        ArithConstantInferIntDivisibilityOpInterface>(*context);
    arith::MulIOp::attachInterface<ArithMulIInferIntDivisibilityOpInterface>(
        *context);
    arith::DivUIOp::attachInterface<ArithDivUIInferIntDivisibilityOpInterface>(
        *context);
  });

  registry.addExtension(
      +[](MLIRContext *context, tensor::TensorDialect *dialect) {
        tensor::InsertSliceOp::attachInterface<InsertSliceOpTiedOpInterface>(
            *context);
      });

  registry.addExtension(
      +[](MLIRContext *context, linalg::LinalgDialect *dialect) {
        // Register all Linalg structured ops. `LinalgOp` is an interface and it
        // is not possible to attach an external interface to an existing
        // interface. Therefore, attach the `TiedOpInterface` to all ops
        // one-by-one.
        LinalgOpTiedOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
            >::registerOpInterface(context);
      });

  // Hoistable Op Interface registration.

  // Register hoistable op interfaces for linalg ops.
  // We have a specific allow-list for Linalg ops because we want to consider
  // new additions carefully.
  registry.addExtension(
      +[](MLIRContext *context, linalg::LinalgDialect *dialect) {
        // Structured op implementations and a handful of pure ops are included.
        // Notably: IndexOp is not included because it establishes a hidden
        // dependency to the iterator and is non-const.

        // Register all LinalgOps ops. `LinalgOp` is an interface and it is
        // not possible to attach an external interface to an existing
        // interface. Therefore, attach the `HoistableLinalgOpInterface` to all
        // ops one-by-one.
        HoistableLinalgOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
            >::registerOpInterface(context);
        UnhoistableOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"
            >::registerOpInterface(context);

        AlwaysHoistableOpInterfaceHelper<
            linalg::PackOp, linalg::UnPackOp>::registerOpInterface(context);
      });
  // Register hoistable op interfaces for tensor ops.
  registry.addExtension(
      +[](MLIRContext *context, tensor::TensorDialect *dialect) {
        // Never hoist empty and other pure metadata ops as a leaf. It's fine to
        // hoist them as a part of a larger constant tree that does actual work.
        HoistableNonLeafOpInterfaceHelper<
            tensor::EmptyOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
            tensor::ExtractSliceOp>::registerOpInterface(context);
        // Cases of trivial pack/unpack should be handled as canonicalizations
        // before we get here, thus we're safe to always hoist.
        AlwaysHoistableOpInterfaceHelper<tensor::PadOp>::registerOpInterface(
            context);
      });
  registry.addExtension(
      +[](MLIRContext *context, IREE::Util::UtilDialect *dialect) {
        IREE::Util::AssumeIntOp::attachInterface<
            UtilAssumeIntValueBoundsOpInterface>(*context);
      });
}

} // namespace mlir::iree_compiler
