#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="data/raw"
OUT_DIR="data/processed"
NORM_MIN=150.0
NORM_MAX=320.0
FILL_VALUE=0.0

python3 src/preprocess.py \
    --raw-dir $RAW_DIR \
    --out-dir $OUT_DIR \
    --norm-min $NORM_MIN \
    --norm-max $NORM_MAX \
    --fill-value $FILL_VALUE