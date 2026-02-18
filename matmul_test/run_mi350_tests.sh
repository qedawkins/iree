#!/usr/bin/env bash
# Run matmul numerics validation on MI350 (gfx950).
# Compares both ukernel and baseline vmfbs against numpy reference.
#
# This script is intended to be run on a machine with MI350 hardware.
#
# Usage:
#   ./run_mi350_tests.sh                         # Run all
#   ./run_mi350_tests.sh --mode ukernel          # Only ukernel vmfbs
#   ./run_mi350_tests.sh --mode baseline         # Only baseline vmfbs
#   ./run_mi350_tests.sh --mode both             # Both (default)
#   ./run_mi350_tests.sh --threshold 1.0         # Custom threshold
#   ./run_mi350_tests.sh --iree-run /path/to/bin # Custom iree-run-module path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../../iree-build" && pwd)"
IREE_RUN="${IREE_RUN:-$BUILD_DIR/tools/iree-run-module}"

TARGET="gfx950"
MODE="both"
THRESHOLD="0.5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)      MODE="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --target)    TARGET="$2"; shift 2 ;;
        --iree-run)  IREE_RUN="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -x "$IREE_RUN" ]]; then
    echo "ERROR: iree-run-module not found at $IREE_RUN"
    exit 1
fi

MANIFEST="$SCRIPT_DIR/manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest.csv not found. Run generate_tests.py first."
    exit 1
fi

VMFB_UKERNEL="$SCRIPT_DIR/vmfbs/${TARGET}_ukernel"
VMFB_BASELINE="$SCRIPT_DIR/vmfbs/${TARGET}_baseline"

run_test() {
    local vmfb="$1"
    local name="$2"
    local label="$3"

    if [[ ! -f "$vmfb" ]]; then
        echo "    $label: SKIP (vmfb not found)"
        return 2
    fi

    local lhs="$SCRIPT_DIR/inputs/${name}_lhs.npy"
    local rhs="$SCRIPT_DIR/inputs/${name}_rhs.npy"
    local ref="$SCRIPT_DIR/refs/${name}_ref.npy"

    local output
    output=$("$IREE_RUN" \
        --device=hip \
        --module="$vmfb" \
        --function="$name" \
        --input="@$lhs" \
        --input="@$rhs" \
        --expected_output="@$ref" \
        --expected_f32_threshold="$THRESHOLD" 2>&1) || true

    if echo "$output" | grep -q '\[SUCCESS\]'; then
        echo "    $label: PASS"
        return 0
    elif echo "$output" | grep -q '\[FAILED\]'; then
        local fail_line
        fail_line=$(echo "$output" | grep '\[FAILED\]' | head -1)
        echo "    $label: FAIL - $fail_line"
        return 1
    else
        local err_line
        err_line=$(echo "$output" | grep -i "error\|INTERNAL" | head -1)
        echo "    $label: ERROR - $err_line"
        return 1
    fi
}

UK_PASS=0; UK_FAIL=0; UK_SKIP=0
BL_PASS=0; BL_FAIL=0; BL_SKIP=0
TOTAL=0

while IFS=, read -r group name m n k; do
    [[ "$group" == "group" ]] && continue
    TOTAL=$((TOTAL + 1))

    echo "[$TOTAL] $name ($group, ${m}x${n}x${k}):"

    if [[ "$MODE" == "ukernel" ]] || [[ "$MODE" == "both" ]]; then
        if run_test "$VMFB_UKERNEL/${name}.vmfb" "$name" "ukernel"; then
            UK_PASS=$((UK_PASS + 1))
        else
            rc=$?
            if [[ $rc -eq 2 ]]; then
                UK_SKIP=$((UK_SKIP + 1))
            else
                UK_FAIL=$((UK_FAIL + 1))
            fi
        fi
    fi

    if [[ "$MODE" == "baseline" ]] || [[ "$MODE" == "both" ]]; then
        if run_test "$VMFB_BASELINE/${name}.vmfb" "$name" "baseline"; then
            BL_PASS=$((BL_PASS + 1))
        else
            rc=$?
            if [[ $rc -eq 2 ]]; then
                BL_SKIP=$((BL_SKIP + 1))
            else
                BL_FAIL=$((BL_FAIL + 1))
            fi
        fi
    fi

done < "$MANIFEST"

echo ""
echo "========================================"
echo "  Results (target=$TARGET, threshold=$THRESHOLD)"
echo "========================================"
if [[ "$MODE" == "ukernel" ]] || [[ "$MODE" == "both" ]]; then
    echo "  Ukernel:  $UK_PASS pass, $UK_FAIL fail, $UK_SKIP skip"
fi
if [[ "$MODE" == "baseline" ]] || [[ "$MODE" == "both" ]]; then
    echo "  Baseline: $BL_PASS pass, $BL_FAIL fail, $BL_SKIP skip"
fi
echo "  Total shapes: $TOTAL"
echo "========================================"

TOTAL_FAIL=$((UK_FAIL + BL_FAIL))
if [[ $TOTAL_FAIL -gt 0 ]]; then
    exit 1
fi
