// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- RedirectCapturedReads.cpp ------------------------------------------===//
//
// Replaces direct uses of pcf.generic init operands inside the body region
// with pcf.read_slice from the corresponding sref block argument.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_REDIRECTCAPTUREDREADSPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

/// For a given pcf.generic, redirect any captured uses of tied init operands
/// inside the body to pcf.read_slice from the corresponding sref block arg.
static void redirectCapturedReads(PCF::GenericOp genericOp) {
  Region &body = genericOp.getRegion();
  if (body.empty()) {
    return;
  }

  int64_t numResults = genericOp.getNumResults();
  int64_t numLeadingArgs = genericOp.getNumLeadingArgs();

  for (int64_t i = 0; i < numResults; ++i) {
    if (!genericOp.getIsTied()[i]) {
      continue;
    }

    OpOperand *initOperand = genericOp.getTiedInit(i);
    if (!initOperand) {
      continue;
    }

    Value initValue = initOperand->get();
    auto initType = dyn_cast<RankedTensorType>(initValue.getType());
    if (!initType) {
      continue;
    }

    // The sref block arg for result i is at index numLeadingArgs + i.
    BlockArgument srefArg = body.getArgument(numLeadingArgs + i);

    // Collect uses of initValue that are inside the body region.
    SmallVector<OpOperand *> capturedUses;
    for (OpOperand &use : initValue.getUses()) {
      Operation *user = use.getOwner();
      if (body.isAncestor(user->getParentRegion())) {
        capturedUses.push_back(&use);
      }
    }

    if (capturedUses.empty()) {
      continue;
    }

    // Create a pcf.read_slice at the start of the body that reads the full
    // tensor from the sref. This bufferizes to a no-op memref view.
    OpBuilder builder(genericOp.getContext());
    builder.setInsertionPointToStart(&body.front());
    Location loc = genericOp.getLoc();

    int64_t rank = initType.getRank();
    SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (int64_t dim : initType.getShape()) {
      sizes.push_back(builder.getIndexAttr(dim));
    }
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

    Value readSlice = PCF::ReadSliceOp::create(builder, loc, initType,
                                               srefArg, offsets, sizes,
                                               strides);

    // Replace all captured uses.
    for (OpOperand *use : capturedUses) {
      use->set(readSlice);
    }
  }
}

struct RedirectCapturedReadsPass final
    : impl::RedirectCapturedReadsPassBase<RedirectCapturedReadsPass> {
  void runOnOperation() override {
    getOperation()->walk([](PCF::GenericOp genericOp) {
      redirectCapturedReads(genericOp);
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::PCF
