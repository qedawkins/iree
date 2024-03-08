// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-gpu-tile-to-serial-loops"

namespace mlir::iree_compiler {

namespace {
struct GPUTensorTensorTileToSerialLoopsPass final
    : public GPUTensorTileToSerialLoopsBase<
          GPUTensorTensorTileToSerialLoopsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    funcOp->walk([](tensor::PadOp padOp) {
      if (padOp->hasAttr("lowering_config")) {
        padOp->removeAttr("lowering_config");
      }
    });

    // // Fuse the pad with the workgroup level slice.
    // fusePadIntoConsumer(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After fusing pad once ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    concretizePadShape(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After concretizing pad shape ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Tile reductions based on the annotated tiling configuration.
    if (failed(tileReductionToSerialLoops(funcOp,
                                          /*fuseInputProducer=*/true))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to loops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Fuse pad with the slices in the tiled loops.
    fusePadIntoConsumer(funcOp);
    concretizePadShape(funcOp);
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUTensorTileToSerialLoops() {
  return std::make_unique<GPUTensorTensorTileToSerialLoopsPass>();
}

} // namespace mlir::iree_compiler
