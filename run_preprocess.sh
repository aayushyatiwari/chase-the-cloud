#!/usr/bin/env bash
# Turn raw INSAT HDF5 granules into uint16 count frames, then rebuild the manifest.
#
# Frames are stored as raw 10-bit sensor counts, not decoded temperatures. The
# calibration tables from every granule go into one sidecar .npz, and the
# dataloader decodes a crop by indexing that frame's own table. See the module
# docstring in src/preprocess.py for why.
#
# Both steps run together on purpose: the manifest lists file paths, so
# preprocessing without rebuilding it leaves training pointed at the old data.
#
# Normalisation ranges are NOT applied here -- they live in data/norm_ranges.json
# and are applied at load time, so changing them needs no reprocessing.
#
#   ./run_preprocess.sh                      # 4 channels, full sector
#   CHANNELS="TIR1" ./run_preprocess.sh      # single channel
#   MODE=crop OUT_DIR=data/processed ./run_preprocess.sh   # old fixed 256x256 window
#   OVERWRITE=1 ./run_preprocess.sh          # redo files that already exist

set -euo pipefail

RAW_DIR="${RAW_DIR:-data/data}"
OUT_DIR="${OUT_DIR:-data/processed_counts}"
LUTS="${LUTS:-data/luts.npz}"
MANIFEST="${MANIFEST:-data/manifest_counts.json}"
# The predicted channel comes first: the model's head is narrower than its
# input, and dataset.target_channels slices the target from the front.
CHANNELS="${CHANNELS:-TIR1 TIR2 WV MIR}"
MODE="${MODE:-full}"          # full = keep whole sector, crop = old fixed window
T="${T:-6}"
STRIDE="${STRIDE:-7}"   # T+1 = non-overlapping sequences; 1 slides one frame
MAX_FILL="${MAX_FILL:-0.01}"  # drop granules that are mostly fill

# Files already present are skipped unless OVERWRITE=1.
EXTRA=()
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA+=(--overwrite)
[[ "$MODE" == "crop" ]] && EXTRA+=(--crop)

echo "Preprocessing $RAW_DIR -> $OUT_DIR (channels: $CHANNELS, mode: $MODE)"
python -m src.preprocess \
    --raw-dir "$RAW_DIR" \
    --out-dir "$OUT_DIR" \
    --lut-path "$LUTS" \
    --channels $CHANNELS \
    --max-fill-frac "$MAX_FILL" \
    "${EXTRA[@]}"

echo "Building manifest -> $MANIFEST"
python -c "
from src.manifest import build
build('$OUT_DIR', T=$T, output_path='$MANIFEST', stride=$STRIDE)
"

echo "Done. Point config.yaml data.manifest_path at $MANIFEST and data.lut_path at $LUTS"
