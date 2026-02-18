#!/usr/bin/env python3
"""Generate MLIR matmul test cases with numpy reference inputs/outputs.

Generates matmul_transpose_b test cases (f16 inputs, f32 accumulator/output)
organized by ukernel pattern (1-cluster, 2-cluster, 4-cluster).

Usage:
    python generate_tests.py [--subset SUBSET] [--seed SEED]

    --subset: "all", "small" (2-3 shapes for validation), or "1cluster"/"2cluster"/"4cluster"
    --seed: random seed for reproducibility (default: 42)
"""

import argparse
import os
import sys

import numpy as np


# ============================================================================
# Shape definitions organized by ukernel pattern
# ============================================================================

def get_shapes():
    """Return shape definitions organized by ukernel cluster pattern."""
    shapes = {}

    # 1-cluster: 128x128 tile, benefit 1
    # Constraints: M%128==0, N%128==0, K%64==0
    k_values_1c = [64, 128, 256, 512, 1024, 2048, 4096]
    mn_values_1c = [128, 256, 384, 512]
    shapes["1cluster"] = []
    for mn in mn_values_1c:
        for k in k_values_1c:
            shapes["1cluster"].append((mn, mn, k))

    # 2-cluster: 256x128 tile, benefit 2
    # Constraints: M%256==0, N%128==0, K%64==0
    k_values_2c = [64, 256, 1024, 4096]
    mn_pairs_2c = [(256, 128), (512, 256), (768, 384), (1024, 512)]
    shapes["2cluster"] = []
    for m, n in mn_pairs_2c:
        for k in k_values_2c:
            shapes["2cluster"].append((m, n, k))

    # 4-cluster: 256x256 tile, benefit 3
    # Constraints: M%256==0, N%256==0, K%64==0
    k_values_4c = [64, 256, 1024, 4096]
    mn_values_4c = [256, 512, 1024, 2048]
    shapes["4cluster"] = []
    for mn in mn_values_4c:
        for k in k_values_4c:
            shapes["4cluster"].append((mn, mn, k))

    return shapes


def get_small_shapes():
    """Return a small subset for local validation."""
    return {
        "small": [
            (128, 128, 64),
            (256, 256, 128),
            (256, 128, 64),
        ]
    }


# ============================================================================
# MLIR generation
# ============================================================================

MLIR_TEMPLATE = """\
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_{M}x{N}x{K}(%lhs: tensor<{M}x{K}xf16>, %rhs: tensor<{N}x{K}xf16>) -> tensor<{M}x{N}xf32> {{
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<{M}x{N}xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %result = linalg.generic {{
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  }} ins(%lhs, %rhs : tensor<{M}x{K}xf16>, tensor<{N}x{K}xf16>) outs(%fill : tensor<{M}x{N}xf32>) {{
  ^bb0(%in0: f16, %in1: f16, %out: f32):
    %0 = arith.extf %in0 : f16 to f32
    %1 = arith.extf %in1 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %2, %out : f32
    linalg.yield %3 : f32
  }} -> tensor<{M}x{N}xf32>
  return %result : tensor<{M}x{N}xf32>
}}
"""


def generate_mlir(m, n, k):
    """Generate MLIR source for a matmul_transpose_b operation."""
    return MLIR_TEMPLATE.format(M=m, N=n, K=k)


# ============================================================================
# Numpy input/reference generation
# ============================================================================

def generate_inputs_and_ref(m, n, k, rng):
    """Generate random f16 inputs and f32 reference output.

    Uses small-magnitude random values to avoid f16 overflow issues.
    Reference is computed in f32: cast f16 inputs to f32, matmul, keep f32.
    """
    # Use small values to stay within f16 range and avoid large accumulations
    scale = 1.0 / np.sqrt(k)
    lhs = (rng.standard_normal((m, k)) * scale).astype(np.float16)
    rhs = (rng.standard_normal((n, k)) * scale).astype(np.float16)

    # Reference: f16->f32, transpose_b matmul, f32 output
    ref = np.matmul(lhs.astype(np.float32), rhs.astype(np.float32).T)

    return lhs, rhs, ref


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate matmul test cases")
    parser.add_argument(
        "--subset", default="all",
        choices=["all", "small", "1cluster", "2cluster", "4cluster"],
        help="Which shape subset to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory (default: same directory as this script)"
    )
    args = parser.parse_args()

    if args.outdir:
        base_dir = args.outdir
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Create subdirectories.
    mlir_dir = os.path.join(base_dir, "mlir")
    inputs_dir = os.path.join(base_dir, "inputs")
    refs_dir = os.path.join(base_dir, "refs")
    for d in [mlir_dir, inputs_dir, refs_dir]:
        os.makedirs(d, exist_ok=True)

    # Select shapes.
    if args.subset == "small":
        shape_groups = get_small_shapes()
    elif args.subset == "all":
        shape_groups = get_shapes()
    else:
        all_shapes = get_shapes()
        shape_groups = {args.subset: all_shapes[args.subset]}

    rng = np.random.default_rng(args.seed)
    total = 0
    manifest_lines = []

    for group_name, shapes in shape_groups.items():
        print(f"Generating {group_name} ({len(shapes)} shapes)...")
        for m, n, k in shapes:
            name = f"matmul_{m}x{n}x{k}"

            # Generate MLIR.
            mlir_path = os.path.join(mlir_dir, f"{name}.mlir")
            with open(mlir_path, "w") as f:
                f.write(generate_mlir(m, n, k))

            # Generate inputs and reference.
            lhs, rhs, ref = generate_inputs_and_ref(m, n, k, rng)
            lhs_path = os.path.join(inputs_dir, f"{name}_lhs.npy")
            rhs_path = os.path.join(inputs_dir, f"{name}_rhs.npy")
            ref_path = os.path.join(refs_dir, f"{name}_ref.npy")
            np.save(lhs_path, lhs)
            np.save(rhs_path, rhs)
            np.save(ref_path, ref)

            manifest_lines.append(f"{group_name},{name},{m},{n},{k}")
            total += 1
            print(f"  {name}: lhs={lhs.shape} rhs={rhs.shape} ref={ref.shape}")

    # Write manifest for scripts to consume.
    manifest_path = os.path.join(base_dir, "manifest.csv")
    with open(manifest_path, "w") as f:
        f.write("group,name,M,N,K\n")
        for line in manifest_lines:
            f.write(line + "\n")

    print(f"\nGenerated {total} test cases.")
    print(f"  MLIR:     {mlir_dir}/")
    print(f"  Inputs:   {inputs_dir}/")
    print(f"  Refs:     {refs_dir}/")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
