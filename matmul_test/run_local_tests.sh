#!/usr/bin/env bash
# Run matmul tests locally on gfx1201 (RDNA4) with numerics validation.
#
# Usage:
#   ./run_local_tests.sh                    # Run all shapes in manifest
#   ./run_local_tests.sh --subset small     # Generate and run small subset
#   ./run_local_tests.sh --threshold 1.0    # Custom f32 threshold
#   ./run_local_tests.sh --jobs 4           # Parallel compilation (serial run)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../../iree-build" && pwd)"
IREE_COMPILE="$BUILD_DIR/tools/iree-compile"
IREE_RUN="$BUILD_DIR/tools/iree-run-module"
VENV="$(cd "$SCRIPT_DIR/../../../iree_venv" && pwd)/bin/activate"

# Defaults.
SUBSET="all"
THRESHOLD="0.5"
JOBS=1
TARGET="gfx1201"
EXTRA_COMPILE_FLAGS="--iree-codegen-llvmgpu-use-tile-and-fuse-matmul=false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subset)   SUBSET="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --jobs)     JOBS="$2"; shift 2 ;;
        --target)   TARGET="$2"; shift 2 ;;
        --extra-flags) EXTRA_COMPILE_FLAGS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Ensure tools exist.
if [[ ! -x "$IREE_COMPILE" ]]; then
    echo "ERROR: iree-compile not found at $IREE_COMPILE"
    exit 1
fi
if [[ ! -x "$IREE_RUN" ]]; then
    echo "ERROR: iree-run-module not found at $IREE_RUN"
    exit 1
fi

# Generate test data if needed.
MANIFEST="$SCRIPT_DIR/manifest.csv"
if [[ ! -f "$MANIFEST" ]] || [[ "$SUBSET" != "all" ]]; then
    echo "=== Generating test data (subset=$SUBSET) ==="
    source "$VENV"
    python3 "$SCRIPT_DIR/generate_tests.py" --subset "$SUBSET" --outdir "$SCRIPT_DIR"
fi

# Create vmfb output directory.
VMFB_DIR="$SCRIPT_DIR/vmfbs/local_${TARGET}"
mkdir -p "$VMFB_DIR"

# Read manifest (skip header).
PASS=0
FAIL=0
SKIP=0
TOTAL=0
ERRORS=""

while IFS=, read -r group name m n k; do
    [[ "$group" == "group" ]] && continue  # Skip header.
    TOTAL=$((TOTAL + 1))

    MLIR="$SCRIPT_DIR/mlir/${name}.mlir"
    VMFB="$VMFB_DIR/${name}.vmfb"
    LHS="$SCRIPT_DIR/inputs/${name}_lhs.npy"
    RHS="$SCRIPT_DIR/inputs/${name}_rhs.npy"
    REF="$SCRIPT_DIR/refs/${name}_ref.npy"

    if [[ ! -f "$MLIR" ]]; then
        echo "SKIP $name: MLIR file not found"
        SKIP=$((SKIP + 1))
        continue
    fi

    # Compile.
    echo -n "[$TOTAL] $name ($group): compile... "
    COMPILE_CMD="$IREE_COMPILE $MLIR \
        --iree-hal-target-device=hip \
        --iree-hip-target=$TARGET \
        $EXTRA_COMPILE_FLAGS \
        -o $VMFB"

    if ! COMPILE_OUTPUT=$(eval "$COMPILE_CMD" 2>&1); then
        echo "COMPILE FAIL"
        ERRORS="${ERRORS}\n  COMPILE FAIL: $name\n    $COMPILE_OUTPUT"
        FAIL=$((FAIL + 1))
        continue
    fi

    # Run and validate.
    echo -n "run... "
    RUN_CMD="$IREE_RUN \
        --device=hip \
        --module=$VMFB \
        --function=${name} \
        --input=@$LHS \
        --input=@$RHS \
        --expected_output=@$REF \
        --expected_f32_threshold=$THRESHOLD"

    RUN_OUTPUT=$(eval "$RUN_CMD" 2>&1) || true

    if echo "$RUN_OUTPUT" | grep -q '\[SUCCESS\]'; then
        echo "PASS"
        PASS=$((PASS + 1))
    elif echo "$RUN_OUTPUT" | grep -q '\[FAILED\]'; then
        echo "FAIL"
        FAIL_LINE=$(echo "$RUN_OUTPUT" | grep '\[FAILED\]' | head -1)
        ERRORS="${ERRORS}\n  NUMERICS FAIL: $name\n    $FAIL_LINE"
        FAIL=$((FAIL + 1))
    else
        echo "ERROR"
        ERRORS="${ERRORS}\n  RUN ERROR: $name\n    $(echo "$RUN_OUTPUT" | head -3)"
        FAIL=$((FAIL + 1))
    fi

done < "$MANIFEST"

# Summary.
echo ""
echo "========================================"
echo "  Results: $PASS pass, $FAIL fail, $SKIP skip (total $TOTAL)"
echo "  Target:  $TARGET"
echo "  Threshold: $THRESHOLD"
echo "========================================"
if [[ -n "$ERRORS" ]]; then
    echo ""
    echo "Failures:"
    echo -e "$ERRORS"
fi

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
