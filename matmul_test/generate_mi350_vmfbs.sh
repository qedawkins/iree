#!/usr/bin/env bash
# Compile all matmul shapes for gfx950 (MI350) with and without ukernels.
#
# Usage:
#   ./generate_mi350_vmfbs.sh                 # Compile all shapes
#   ./generate_mi350_vmfbs.sh --subset small  # Compile small subset only
#   ./generate_mi350_vmfbs.sh --jobs 4        # Parallel compilation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../../iree-build" && pwd)"
IREE_COMPILE="$BUILD_DIR/tools/iree-compile"
VENV="$(cd "$SCRIPT_DIR/../../../iree_venv" && pwd)/bin/activate"

TARGET="gfx950"
SUBSET="all"
JOBS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subset)  SUBSET="$2"; shift 2 ;;
        --jobs)    JOBS="$2"; shift 2 ;;
        --target)  TARGET="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -x "$IREE_COMPILE" ]]; then
    echo "ERROR: iree-compile not found at $IREE_COMPILE"
    exit 1
fi

# Generate test data if needed.
MANIFEST="$SCRIPT_DIR/manifest.csv"
if [[ ! -f "$MANIFEST" ]] || [[ "$SUBSET" != "all" ]]; then
    echo "=== Generating test data (subset=$SUBSET) ==="
    source "$VENV"
    python3 "$SCRIPT_DIR/generate_tests.py" --subset "$SUBSET" --outdir "$SCRIPT_DIR"
fi

# Create output directories.
VMFB_UKERNEL="$SCRIPT_DIR/vmfbs/${TARGET}_ukernel"
VMFB_BASELINE="$SCRIPT_DIR/vmfbs/${TARGET}_baseline"
mkdir -p "$VMFB_UKERNEL" "$VMFB_BASELINE"

UK_PASS=0; UK_FAIL=0
BL_PASS=0; BL_FAIL=0
TOTAL=0
ERRORS=""

while IFS=, read -r group name m n k; do
    [[ "$group" == "group" ]] && continue
    TOTAL=$((TOTAL + 1))

    MLIR="$SCRIPT_DIR/mlir/${name}.mlir"
    if [[ ! -f "$MLIR" ]]; then
        echo "SKIP $name: MLIR not found"
        continue
    fi

    # Compile WITH ukernels.
    VMFB_UK="$VMFB_UKERNEL/${name}.vmfb"
    echo -n "[$TOTAL] $name: ukernel... "
    UK_CMD="$IREE_COMPILE $MLIR \
        --iree-hal-target-device=hip \
        --iree-rocm-target=$TARGET \
        --iree-rocm-enable-tensor-ukernels \
        -o $VMFB_UK"

    if UK_OUT=$(eval "$UK_CMD" 2>&1); then
        echo -n "OK  "
        UK_PASS=$((UK_PASS + 1))
    else
        echo -n "FAIL  "
        ERRORS="${ERRORS}\n  UKERNEL COMPILE FAIL: $name\n    $(echo "$UK_OUT" | head -3)"
        UK_FAIL=$((UK_FAIL + 1))
    fi

    # Compile WITHOUT ukernels (baseline).
    VMFB_BL="$VMFB_BASELINE/${name}.vmfb"
    echo -n "baseline... "
    BL_CMD="$IREE_COMPILE $MLIR \
        --iree-hal-target-device=hip \
        --iree-rocm-target=$TARGET \
        -o $VMFB_BL"

    if BL_OUT=$(eval "$BL_CMD" 2>&1); then
        echo "OK"
        BL_PASS=$((BL_PASS + 1))
    else
        echo "FAIL"
        ERRORS="${ERRORS}\n  BASELINE COMPILE FAIL: $name\n    $(echo "$BL_OUT" | head -3)"
        BL_FAIL=$((BL_FAIL + 1))
    fi

done < "$MANIFEST"

TOTAL_FAIL=$((UK_FAIL + BL_FAIL))

echo ""
echo "========================================"
echo "  Ukernel:  $UK_PASS pass, $UK_FAIL fail"
echo "  Baseline: $BL_PASS pass, $BL_FAIL fail"
echo "  Total shapes: $TOTAL"
echo "  Target:   $TARGET"
echo "  Ukernel:  $VMFB_UKERNEL/"
echo "  Baseline: $VMFB_BASELINE/"
echo "========================================"
if [[ -n "$ERRORS" ]]; then
    echo ""
    echo "Failures:"
    echo -e "$ERRORS"
fi

if [[ $TOTAL_FAIL -gt 0 ]]; then
    exit 1
fi
