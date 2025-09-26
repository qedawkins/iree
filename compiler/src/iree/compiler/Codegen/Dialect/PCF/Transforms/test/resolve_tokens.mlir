// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-gpu-combine-barrier-regions))" --split-input-file | FileCheck %s
