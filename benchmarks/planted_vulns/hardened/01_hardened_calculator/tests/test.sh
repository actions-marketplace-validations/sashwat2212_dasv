#!/bin/sh
set -eu
DIR=$(dirname "$0")
out=$(python3 "$DIR/test_calc.py")
printf '%s\n' "$out"
printf '%s\n' "$out" | grep -qx "DASV_REWARD=1.0"

