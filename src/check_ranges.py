"""
Measure how much real data NORM_RANGES actually clips.

preprocess.py maps brightness temperature to [0,1] with a fixed per-channel
range and clips whatever falls outside. This reports what that costs.

Exact, not sampled: every pixel of every file is accounted for. It never
materialises a decoded frame -- a 1024-bin histogram of the raw counts carries
the same information, since brightness temperature is a pure function of the
count through that file's own LUT.

    python -m src.check_ranges
    python -m src.check_ranges --limit 200
    python -m src.check_ranges --channels TIR1 WV
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

DEFAULT_CHANNELS = ('TIR1', 'TIR2', 'WV', 'MIR')


def load_ranges(path):
    with open(path) as f:
        return {k: tuple(v) for k, v in json.load(f).items()}


def check(raw_dir, channels, ranges, limit=None):
    files = sorted(Path(raw_dir).glob('*.h5'))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .h5 files in {raw_dir}")

    acc = {c: dict(n=0, below=0, above=0, fill=0,
                   lo=np.inf, hi=-np.inf,
                   worst_below=0.0, worst_above=0.0,
                   lut_below=0, lut_above=0) for c in channels}

    for i, path in enumerate(files, 1):
        with h5py.File(path, 'r') as f:
            for c in channels:
                raw = f[f'IMG_{c}'][0]
                lut = f[f'IMG_{c}_TEMP'][:].astype(np.float64)

                # How many pixels landed on each of the 1024 possible counts.
                hist = np.bincount(raw.ravel(), minlength=1024)
                seen = hist > 0

                lo, hi = ranges[c]
                a = acc[c]
                a['n'] += int(hist.sum())
                a['below'] += int(hist[seen & (lut < lo)].sum())
                a['above'] += int(hist[seen & (lut > hi)].sum())
                a['fill'] += int(hist[1023])

                bt_seen = lut[seen]
                a['lo'] = min(a['lo'], float(bt_seen.min()))
                a['hi'] = max(a['hi'], float(bt_seen.max()))
                a['worst_below'] = max(a['worst_below'], float(lo - bt_seen.min()))
                a['worst_above'] = max(a['worst_above'], float(bt_seen.max() - hi))

                # Table entries outside the range, whether or not any pixel
                # used them. These cost dynamic range, not data.
                a['lut_below'] += int((lut < lo).sum())
                a['lut_above'] += int((lut > hi).sum())

        if i % 200 == 0:
            print(f"  {i}/{len(files)} ...", flush=True)

    return acc, len(files)


def report(acc, n_files, ranges):
    print(f"\n{len(acc)} channels over {n_files} files\n")
    for c, a in acc.items():
        lo, hi = ranges[c]
        clipped = a['below'] + a['above']
        print(f"{c}")
        print(f"  range in use        {lo} .. {hi} K")
        print(f"  observed BT         {a['lo']:.3f} .. {a['hi']:.3f} K")
        print(f"  pixels              {a['n']:,}")
        print(f"  clipped low         {a['below']:,} ({100 * a['below'] / a['n']:.6f}%)")
        print(f"  clipped high        {a['above']:,} ({100 * a['above'] / a['n']:.6f}%)")
        print(f"  clipped total       {clipped:,} ({100 * clipped / a['n']:.6f}%)")
        print(f"  worst overshoot     {a['worst_below']:.3f} K below, "
              f"{a['worst_above']:.3f} K above")
        print(f"  fill count (1023)   {a['fill']:,} ({100 * a['fill'] / a['n']:.4f}%)")
        print(f"  LUT entries outside {a['lut_below'] / n_files:.1f} below, "
              f"{a['lut_above'] / n_files:.1f} above (of 1024, mean per file)")
        print(f"  tight range would be [{a['lo']:.2f}, {a['hi']:.2f}]")
        print()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Check what NORM_RANGES clips.")
    p.add_argument('--raw-dir', default='data/data')
    p.add_argument('--ranges', default='data/norm_ranges.json')
    p.add_argument('--channels', nargs='+', default=list(DEFAULT_CHANNELS))
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    ranges = load_ranges(args.ranges)
    acc, n = check(args.raw_dir, args.channels, ranges, args.limit)
    report(acc, n, ranges)
