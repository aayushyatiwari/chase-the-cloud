#!/usr/bin/env bash
# Turn raw INSAT HDF5 files into .npy frames, then rebuild the manifest.
#
# Both steps run together on purpose: the manifest lists file paths, so
# preprocessing without rebuilding it leaves training pointed at the old data.
#
# Normalisation ranges are per channel and live in src/preprocess.py
# (NORM_RANGES), not here -- water vapour needs a different range from TIR1.
#
#   ./run_preprocess.sh                      # full sector, TIR1 (what training uses)
#   CHANNELS="TIR1 TIR2 WV" ./run_preprocess.sh
#   MODE=crop OUT_DIR=data/processed ./run_preprocess.sh   # old fixed 256x256 window
#   OVERWRITE=1 ./run_preprocess.sh          # redo files that already exist

set -euo pipefail

RAW_DIR="${RAW_DIR:-data/data}"
OUT_DIR="${OUT_DIR:-data/processed_full}"
MANIFEST="${MANIFEST:-data/manifest_full.json}"
CHANNELS="${CHANNELS:-TIR1}"
MODE="${MODE:-full}"          # full = keep whole sector, crop = old fixed window
T="${T:-6}"
STRIDE="${STRIDE:-7}"   # T+1 = non-overlapping sequences; 1 slides one frame

# Files already present are skipped unless OVERWRITE=1.
EXTRA=()
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA+=(--overwrite)
[[ "$MODE" == "crop" ]] && EXTRA+=(--crop)

echo "Preprocessing $RAW_DIR -> $OUT_DIR (channels: $CHANNELS, mode: $MODE)"
python -m src.preprocess \
    --raw-dir "$RAW_DIR" \
    --out-dir "$OUT_DIR" \
    --channels $CHANNELS \
    "${EXTRA[@]}"

echo "Building manifest -> $MANIFEST"
python -c "
from src.manifest import build
build('$OUT_DIR', T=$T, output_path='$MANIFEST', stride=$STRIDE)
"

echo "Done. Point config.yaml data.manifest_path at $MANIFEST"
