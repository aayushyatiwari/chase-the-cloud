"""
Turn raw INSAT HDF5 granules into uint16 count frames plus one LUT sidecar.

The imager stores 10-bit sensor counts (0..1023) and, in every file, the
calibration tables that turn a count into a physical value. Brightness
temperature is therefore a pure function of the count: bt = lut[count].

This script does NOT apply that function. It copies the counts out as uint16
and collects every file's tables into a single .npz, and the decode happens in
the dataloader instead (src/dataset.py). Two reasons:

  * Size. The counts carry 10 bits of information; storing decoded float32
    spends 32 bits on it. Counts are half the size of float32 and bit-exact.
  * Reversibility. Normalisation ranges, the choice of temperature vs
    radiance, and fill handling all become load-time decisions, changeable by
    editing a 48MB sidecar rather than reprocessing 72GB of granules.

The tables differ from file to file -- the instrument is recalibrated as it
drifts, by up to 13K on MIR -- so each frame must be decoded with the tables
from its own file. That is why the sidecar is keyed by filename.

    python -m src.preprocess --raw-dir data/data --out-dir data/processed_counts
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import h5py


@dataclass(frozen=True)
class Crop:
    row_start: int
    row_end: int
    col_start: int
    col_end: int


# Old fixed 256x256 crop over central-western India. Kept for reference and for
# reproducing earlier single-crop results; the multi-crop pipeline stores the
# whole sector instead and picks crops at training time.
DEFAULT_CROP = Crop(row_start=680, row_end=936, col_start=740, col_end=996)

DEFAULT_CHANNELS = ('TIR1', 'TIR2', 'WV', 'MIR')

# The imager is 10-bit, so counts index a 1024-entry table. Asserted per file:
# everything downstream indexes the tables with these counts directly.
N_COUNTS = 1024

# _FillValue on every IMG_* dataset. It collides with a real count -- lut[1023]
# is each channel's coldest entry, ~180K -- so it cannot be detected after
# decoding. A frame that is mostly fill is a dead granule, dropped below.
FILL_COUNT = 1023

# Drop a frame when this fraction of its pixels are fill. Three granules in the
# 2023-2024 set are blank or near-blank (31JUL2023_0420 is 100% fill,
# 11JUL2024_2015 is 98%, 22JUL2024_2057 is 11%); they normalise to a uniform
# frame and would otherwise be trained and validated against.
DEFAULT_MAX_FILL_FRAC = 0.01


def h5_to_counts(h5_path, channels):
    """Raw sensor counts and this file's own calibration tables."""
    with h5py.File(h5_path, 'r') as f:
        # Images are stored as (1, H, W), so take index 0.
        counts = np.stack([f[f'IMG_{c}'][0] for c in channels])
        luts = np.stack([f[f'IMG_{c}_TEMP'][:] for c in channels])

    if counts.dtype != np.uint16:
        raise ValueError(f"{h5_path.name}: counts are {counts.dtype}, expected uint16")
    if counts.max() >= N_COUNTS:
        raise ValueError(f"{h5_path.name}: count {counts.max()} exceeds the "
                         f"{N_COUNTS}-entry table")
    if luts.shape != (len(channels), N_COUNTS):
        raise ValueError(f"{h5_path.name}: tables are {luts.shape}, "
                         f"expected {(len(channels), N_COUNTS)}")
    return counts, luts.astype(np.float32)


def h5_to_luts(h5_path, channels):
    """Just the tables, for a frame whose .npy is already written."""
    with h5py.File(h5_path, 'r') as f:
        luts = np.stack([f[f'IMG_{c}_TEMP'][:] for c in channels])
    return luts.astype(np.float32)


def fill_fraction(counts):
    return float((counts == FILL_COUNT).mean())


def process_file(h5_path, out_path, channels, crop, max_fill_frac=DEFAULT_MAX_FILL_FRAC):
    """
    Write one .npy of shape (C, H, W) uint16 and return this file's tables.

    Returns (luts, fill_frac), or (None, fill_frac) when the frame is mostly
    fill and was dropped. The channel axis is always present, even for a single
    channel, so the dataset slices crops the same way regardless of C.
    """
    counts, luts = h5_to_counts(h5_path, channels)
    frac = fill_fraction(counts)
    if frac > max_fill_frac:
        return None, frac

    if crop is not None:
        counts = counts[:, crop.row_start:crop.row_end, crop.col_start:crop.col_end]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, counts)
    return luts, frac


def process_all(raw_dir, out_dir, lut_path, channels=DEFAULT_CHANNELS, crop=None,
                overwrite=False, max_fill_frac=DEFAULT_MAX_FILL_FRAC):
    files = sorted(Path(raw_dir).glob("*.h5"))
    if not files:
        print(f"No .h5 files found in {raw_dir}")
        return

    luts_by_stem = {}
    written = skipped = 0
    dropped = []

    for i, h5_path in enumerate(files, 1):
        out_path = Path(out_dir) / f"{h5_path.stem}.npy"

        # An already-written frame still needs its tables in the sidecar, or it
        # becomes undecodable. Read only the tables for it -- 16KB, against
        # 22MB for the counts -- so resuming a partial run stays cheap.
        if out_path.exists() and not overwrite:
            probe = np.load(out_path, mmap_mode='r')
            if probe.dtype != np.uint16:
                raise ValueError(
                    f"{out_path} is {probe.dtype}, not uint16. This directory holds "
                    f"output from the old decode-at-preprocess pipeline -- point "
                    f"--out-dir somewhere fresh, or pass --overwrite."
                )
            luts_by_stem[h5_path.stem] = h5_to_luts(h5_path, channels)
            skipped += 1
            continue

        try:
            luts, frac = process_file(h5_path, out_path, channels, crop, max_fill_frac)
        except Exception as e:
            print(f"Error processing {h5_path.name}: {e}")
            continue

        if luts is None:
            dropped.append((h5_path.name, frac))
            continue

        luts_by_stem[h5_path.stem] = luts
        written += 1
        if written % 100 == 0:
            print(f"  {i}/{len(files)} ... {written} written", flush=True)

    if not luts_by_stem:
        print("Nothing to write.")
        return

    stems = sorted(luts_by_stem)
    lut_path = Path(lut_path)
    lut_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        lut_path,
        luts=np.stack([luts_by_stem[s] for s in stems]),
        files=np.array(stems),
        channels=np.array(list(channels)),
    )

    print(f"\nWrote {written} frames, skipped {skipped} already present, "
          f"dropped {len(dropped)} mostly-fill.")
    for name, frac in dropped:
        print(f"  dropped {name}  ({100 * frac:.1f}% fill)")
    print(f"Tables for {len(stems)} frames -> {lut_path} "
          f"({lut_path.stat().st_size / 1e6:.1f}MB), channels={list(channels)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess INSAT HDF5 granules into uint16 count frames.")
    parser.add_argument("--raw-dir", default="data/data")
    parser.add_argument("--out-dir", default="data/processed_counts")
    parser.add_argument("--lut-path", default="data/luts.npz")
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS),
                        help="Channels to stack, in order. The predicted channel comes first.")
    parser.add_argument("--full-sector", action="store_true", default=True,
                        help="Keep the whole sector (default) so crops can be chosen while training.")
    parser.add_argument("--crop", action="store_false", dest="full_sector",
                        help="Instead cut the old fixed 256x256 window.")
    parser.add_argument("--max-fill-frac", type=float, default=DEFAULT_MAX_FILL_FRAC,
                        help="Drop a frame when more than this fraction of it is fill.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    process_all(args.raw_dir, args.out_dir, args.lut_path,
                channels=tuple(args.channels),
                crop=None if args.full_sector else DEFAULT_CROP,
                overwrite=args.overwrite,
                max_fill_frac=args.max_fill_frac)
