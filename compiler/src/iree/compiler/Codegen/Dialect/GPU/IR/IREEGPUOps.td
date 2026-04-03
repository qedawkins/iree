// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_IREEGPUOPS
#define IREE_CODEGEN_DIALECT_IREEGPUOPS

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td"
include "iree/compiler/Codegen/Interfaces/HoistableRegionOpInterface.td"
include "iree/compiler/Utils/CommonTypeConstraints.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// BarrierRegionOp
//===----------------------------------------------------------------------===//

def IREEGPU_BarrierRegionOp : Op<IREEGPU_Dialect, "barrier_region", [
    DeclareOpInterfaceMethods<HoistableRegionOpInterface>,
    Pure,
    SingleBlockImplicitTerminator<"mlir::iree_compiler::IREE::GPU::YieldOp">
    ]> {
  let summary = [{Synchronizes workers on a region of shared code.}];
  let description = [{
    This op is designed to represent synchronization of workers on the operands
    and results of the given region. This operation naturally arises when combining
    the regions of producer-consumer `scf.forall` operations that share a
    mapping type.

    For example, consider the following pair of parallel loops.
    ```mlir
      %0 = scf.forall (%idy, %idx) in (2, 32) shared_outs(%init = %empty) -> (tensor<4x128xf32>) {
        %in = ...
        %2 = affine.apply #affine_map<(d0) -> (d0 * 2)> (%idy)
        %3 = affine.apply #affine_map<(d0) -> (d0 * 4)> (%idx)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %in into %init[%2, %3] [2, 4] [1, 1]
            : tensor<2x4xf32> into tensor<4x128xf32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %1 = scf.forall (%idy, %idx) in (8, 8) -> (tensor<128x128xf32>) {
        %4 = affine.apply #affine_map<(d0) -> (d0 * 16)> (%idx)
        %extracted_slice = tensor.extract_slice %0[0, %4] [4, 16] [1, 1]
          : tensor<4x128xf32> to tensor<4x16xf32>
        ...
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    ```

    Because these loops share the same worker type and total count, the bodies
    of these two loops can be merged with a barrier an insert_slice and a
    shuffle where the boundary of the loops currently is.

    ```mlir
      %0 = scf.forall (%idy, %idx) in (8, 8) -> (tensor<4x128xf32>) {
        %alloc = bufferization.alloc_tensor {memory_space = #gpu.address_space<workgroup>}
          : tensor<4x128xf32>
        %barrier = iree_gpu.barrier_region %alloc {
        ^bb0(%shared: tensor<4x128xf32>):
          %ids = affine.delinearize_index %idy * 8 + %idx to (2, 32) : index
          %in = ...
          %2 = affine.apply #affine_map<(d0) -> (d0 * 2)> (%ids#0)
          %3 = affine.apply #affine_map<(d0) -> (d0 * 4)> (%ids#1)
          %inserted_slice = tensor.insert_slice %in into %shared[%2, %3] [2, 4] [1, 1]
            : tensor<2x4xf32> to tensor<4x128xf32>
          iree_gpu.yield %slice : tensor<4x16xf32>
        } : tensor<4x128xf32> -> tensor<4x16xf32>
        %4 = affine.apply #affine_map<(d0) -> (d0 * 16)> (%idx)
        %slice = tensor.extract_slice %barrier[0, %4] [4, 16] [1, 1] : tensor<4x128xf32> to tensor<4x16xf32>
        ...
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    ```

    A barrier_region can be lowered to two barriers, one on the input operands
    and a second one on the results.

    Movtivation and Intended Use Cases:

    The primary way this op is generated is when fusing parallel loops with
    tensor results. This operation helps to make lowerings more progressive
    and flexible.
      - Lowering directly to an alloc + reads and writes breaks the dependency
        chain making transformations like barrier placement and pipelining
        potentially more difficult.
      - Allows the option of non-vector based lowering paths.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs
  );
  let regions = (region SizedRegion<1>:$region);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    (`ins` `(` $inputs^ `:` type($inputs) `)` )?
    $region attr-dict `:` type($results)
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$result_types, "ValueRange":$inputs)>
  ];

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ValueBarrierOp
//===----------------------------------------------------------------------===//

def IREEGPU_ValueBarrierOp : Op<IREEGPU_Dialect, "value_barrier", [
  Pure,
  AllTypesMatch<["inputs", "results"]>]> {
  let summary = [{Synchronizes workers on a value semantic tensor or vector.}];
  let description = [{
    This operation acts as a barrier on a value semantic SSA values (tensor or
    vector). It takes multiple operands and produces a value equivalent to each
    input. This does not have copy and/or data movement semantics and simply
    represents a barrier on all writes in the tensor case, and a barrier until
    all threads acquire the input vector in the vector case.

    The inputs must be either all tensors, or all vectors.

    This operation is a no-op when not present in a parallel context. This
    operation is pure as it only requires synchronization for the value it
    produces.
  }];

  let arguments = (ins  Variadic<AnyRankedTensorOrVector>:$inputs);
  let results   = (outs Variadic<AnyRankedTensorOrVector>:$results);

  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs)>
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool hasTensorSemantics() {
      return isa<::mlir::RankedTensorType>(getOperand(0).getType());
    }
    ::mlir::ShapedType getInputType(int operandNum) {
      return ::llvm::cast<::mlir::ShapedType>(
          getInputs()[operandNum].getType());
    }
    SmallVector<::mlir::ShapedType> getInputTypes() {
      return llvm::map_to_vector(
          getInputs(),
          [](Value v) {
            return ::llvm::cast<::mlir::ShapedType>(v.getType());
          });
    }
  }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def IREEGPU_YieldOp : Op<IREEGPU_Dialect, "yield", [
    Pure, ReturnLike, Terminator,
    HasParent<"::mlir::iree_compiler::IREE::GPU::BarrierRegionOp">]> {
  let summary = [{Yield values from a iree_gpu region.}];
  let description = [{
     This operation is used to yield values from a within a region.
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($values^ `:` type($values))? }];
}

//===----------------------------------------------------------------------===//
//
// AMD Specific Operations
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BufferResourceCastOp
//===----------------------------------------------------------------------===//

def IREEGPU_BufferResourceCastOp : Op<IREEGPU_Dialect, "buffer_resource_cast", [
  Pure,
  AllTypesMatch<["input", "result"]>]> {
  let summary = [{Represents a cast to addr_space<7> (buffer resource) before bufferization.}];
  let description = [{
    Nominal cast of a tensor to AMDGPU buffer resource memory space before
    bufferization. This op takes the parameters with which to perform the cast
    if |input| bufferizes to `storage_buffer` memory space. If |input| resolves
    to any other memory space this op is silently dropped and has no effect.

    If |cache_swizzle_stride| is present, there is verification before
    bufferization that all producers of |input| are view-like and single source
    and user (i.e. trivially no alias). In all other cases this op is best
    effort and has no verification or failure modes.

    // TODO: Add other parameters for casting as needed.
  }];

  let arguments = (ins  AnyNonVectorShaped:$input,
                        Optional<Index>:$cache_swizzle_stride);
  let results   = (outs AnyNonVectorShaped:$result);

  let assemblyFormat = [{
    $input oilist (`cacheSwizzleStride` `(` $cache_swizzle_stride `)` )
    attr-dict `:` type($result)
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// CoalescedGatherDMAOp
//===----------------------------------------------------------------------===//

def IREEGPU_CoalescedGatherDMAOp : Op<IREEGPU_Dialect, "coalesced_gather_dma", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<ParallelCombiningOpInterface,
       ["getUpdatedDestinations", "getIteratingParent"]>]> {
  let summary = "Coalesced gather DMA operation for efficient GPU memory access";
  let description = [{
    Performs a coalesced gather operation.
    This operation can exist in two forms: a tensor-based
    (value-semantic) form and a buffer-based (memref-semantic) form.

    In both forms, it reads elements from a source operand based on the
    optional `indices` operand and writes the gathered data into the
    destination `init` operand using destination-passing style.

    The `indices` operand is optional. The `source` represents the data
    loaded by this thread, while `init` is the collective output for all
    threads in the subgroup. Therefore, `source` and `init` may have
    different shapes (typically source is smaller, representing one
    thread's portion).

    The operation is specifically designed for subgroup-level
    parallelism, where threads within a subgroup cooperatively gather
    data with coalesced memory accesses. It implements
    ParallelCombiningOpInterface and must live inside an op implementing
    `InParallelOpInterface`, such as `scf.forall.in_parallel`.

    ## Lowering Paths

    Two lowering strategies are supported:
    1. Lowers to `amdgpu.gather_to_lds` operations when lowering
       requirements are met.
    2. Default lowering using `vector.gather` operations.

    ## Operands and Results

    * `$indices`: The variadic `indices` operand is an optional tensor
      or vector of indices to gather from `source`. If the indices are
      present, their shape must be a prefix of the `init`/`result` type.
      Each element of `indices` must be a 1D tensor or vector whose
      length matches the length of the corresponding dimension of `source`.

      The values in `indices` form indices into the memref starting at
      `source` from which a given thread will gather data, and each
      tensor/vector component in `indices` corresponds to one index
      dimension in `source`.
      Any component that is not specified is implicitly assumed to be
      `[0, 1, ..., len - 1]` where `len` is the length of the
      corresponding dimension of source. That is, gather all the
      elements along that dimension.

      This operation will gather data into its result as by setting:
      ```
      forall (i0, i1, ... iN) in (dim(source, 0), dim(source, 1), ...
          dim(source, N)):
        result[i0, i1, ... iN-1, iN + lane_id * dim(init, N)] =
            source[indices[0][i0], indices[1][i1], ..., indices[N][iN]]
      ```
      where `lane_id` is the ID of the thread within its subgroup.

      Note that, in order to enable efficient gathers, the trailing
      dimension of `source` must have unspecified indices and the dim's
      size must be a supported DMA width for your target.

      `$indices` supports both index and i32 element types. The reason
      is that one lowering path (from linalg_ext.gather) already have
      indices in i32 type.

    * `$source`: Source tensor/memref containing the data to be gathered.
    * `$init`: Destination tensor/memref receiving the gathered data
      (destination-passing style).
    * `lane`: The lane that specifies the coalescing store's offset within the
      workgroup/shared memory.

    ## In-Bounds Attribute

    The optional `in_bounds` attribute is a boolean array with one entry per
    dimension of `init`. When not present, all dimensions are treated as
    in-bounds (source and init must have matching sizes for non-indexed dims).

    When present, `in_bounds[i] = false` indicates that the source may be
    smaller than init along dimension `i`. Reads beyond the source extent
    return zero (padding semantics). This enables fusion of `tensor.pad`
    with zero padding into the DMA operation.

    `in_bounds[i] = true` means the source and init sizes match along that
    dimension, and no padding is needed.

    ## Example of a single subgroup using coalesced_gather_dma in copy mode
       for transferring tensor<4x128xf32>, with an intended DMA width of 128 bits
       (4 x f32), with subgroup size 32:
    ```mlir
    scf.forall (%arg6) in (32) ... {
        %2 = %arg6 * 4
        %thread_slice = ... : tensor<4x128xf32>
        %dest_slice = ... : tensor<4x128xf32>
        scf.forall.in_parallel {
          iree_gpu.coalesced_gather_dma %thread_slice into %dest_slice lane(%arg6) :  ...
        }
      } {mapping = [#gpu.lane_id<linear_dim_0>]}
    ```
  }];

  let arguments = (ins
    AnyRankedTensorOrMemRef:$source,
    Variadic<RankedTensorOrVectorOrMemRefOf<[I32, Index]>>:$indices,
    AnyNonVectorShaped:$init,
    Index:$lane,
    OptionalAttr<BoolArrayAttr>:$in_bounds
  );

  let results = (outs Optional<AnyRankedTensorOrMemRef>:$result);

  let assemblyFormat = [{
    $source (`[` $indices^ `]`)? `into` $init `lane` `(` $lane `)`
    (`in_bounds` $in_bounds^)?
    attr-dict `:` type(operands) ( `->` type($result)^ )?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Get the rank of the init tensor/memref.
    unsigned getResultRank() {
      return llvm::cast<ShapedType>(getInit().getType()).getRank();
    }

    bool hasTensorSemantics() {
      return ::llvm::isa<RankedTensorType>(getInit().getType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// GlobalSubgroupBarrierOp
//===----------------------------------------------------------------------===//

def IREEGPU_GlobalSubgroupBarrierOp : Op<IREEGPU_Dialect,
    "global_subgroup_barrier", []> {
  let summary = "Synchronization-only barrier across all subgroups.";
  let description = [{
    All subgroups in the workgroup must reach any instance of this op before
    any can proceed past it. Unlike `gpu.barrier`, this op has no memory
    fence semantics - memory operations can be freely reordered with respect
    to it. Fences are handled separately.

    This operation operates on a single, global workgroup-wide barrier object
    that can implement such dynamic join and must not be used if such a barrier
    isn't available.
  }];

  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// PromoteOperandOp
//===----------------------------------------------------------------------===//

def IREEGPU_PromoteOperandOp : Op<IREEGPU_Dialect, "promote_operand", [
    Pure,
  ]> {
  let summary = [{Promotes an operand tile to a different scope.}];
  let description = [{
    Takes a pcf.sref at one scope and produces a pcf.sref at a different
    (typically inner) scope. The promotion attribute controls how the
    promotion is lowered (e.g., shared memory allocation + copy).

    The `symbols` attribute lists namespace symbol names whose values
    define the tile dimensions. The number of symbols must equal the
    rank of the source sref.

    Example:
    ```mlir
    %promoted = iree_gpu.promote_operand
        #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>
        %input_sref ["m_tile", "k_tile"]
        : !pcf.sref<64x128xf16, #wg_scope>
        -> !pcf.sref<64x128xf16, #sg_scope>
    ```
  }];

  let arguments = (ins
    IREEGPU_PromotionAttr:$promotion,
    AnyNonVectorShaped:$source,
    ArrayAttr:$symbols
  );

  let results = (outs AnyNonVectorShaped:$result);

  let assemblyFormat = [{
    $promotion $source $symbols attr-dict
    `:` type($source) `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DmaCopyOp
//===----------------------------------------------------------------------===//

def IREEGPU_DmaCopyOp : Op<IREEGPU_Dialect, "dma_copy", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = [{DMA copy between srefs at different scopes.}];
  let description = [{
    Copies data from a source sref to a destination sref using DMA.
    Each worker (lane) contributes one element per invocation, and the
    subgroup collectively fills the destination tile. The `transfer_type`
    determines the granularity of each lane's transfer.

    This is the PCF-level equivalent of `amdgpu.gather_to_lds`. It
    operates on srefs rather than memrefs and is designed to lower to
    hardware DMA instructions.

    The source and destination must have matching element types. The
    offsets/sizes/strides describe the tile region to copy from the source
    and the region to write in the destination.

    Example:
    ```mlir
    iree_gpu.dma_copy %source[%off0, %off1] [%sz0, %sz1] [1, 1]
                      to %dest[0, 0] [%sz0, %sz1] [1, 1]
                      : !pcf.sref<128x256xf16, #outer_scope>
                     -> !pcf.sref<32x16xf16, #inner_scope>
    ```
  }];

  let arguments = (ins
    AnyNonVectorShaped:$source,
    Variadic<Index>:$source_offsets,
    Variadic<Index>:$source_sizes,
    Variadic<Index>:$source_strides,
    DenseI64ArrayAttr:$static_source_offsets,
    DenseI64ArrayAttr:$static_source_sizes,
    DenseI64ArrayAttr:$static_source_strides,
    AnyNonVectorShaped:$dest,
    Variadic<Index>:$dest_offsets,
    Variadic<Index>:$dest_sizes,
    Variadic<Index>:$dest_strides,
    DenseI64ArrayAttr:$static_dest_offsets,
    DenseI64ArrayAttr:$static_dest_sizes,
    DenseI64ArrayAttr:$static_dest_strides
  );

  let results = (outs);

  let assemblyFormat = [{
    $source
    custom<DynamicIndexList>($source_offsets, $static_source_offsets)
    custom<DynamicIndexList>($source_sizes, $static_source_sizes)
    custom<DynamicIndexList>($source_strides, $static_source_strides)
    `to`
    $dest
    custom<DynamicIndexList>($dest_offsets, $static_dest_offsets)
    custom<DynamicIndexList>($dest_sizes, $static_dest_sizes)
    custom<DynamicIndexList>($dest_strides, $static_dest_strides)
    attr-dict `:` type($source) `->` type($dest)
  }];

  let hasVerifier = 1;
}

#endif // IREE_CODEGEN_DIALECT_IREEGPUOPS
