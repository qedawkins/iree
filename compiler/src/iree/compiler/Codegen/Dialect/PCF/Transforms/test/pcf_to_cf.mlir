// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-to-cf)" --split-input-file | FileCheck %s
