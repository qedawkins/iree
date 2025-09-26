// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-convert-sref-to-memref)" --split-input-file | FileCheck %s
