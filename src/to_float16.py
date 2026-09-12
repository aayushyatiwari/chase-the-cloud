"""
Halve the size of processed .npy frames by storing them as float16.

Frames are normalised to [0,1] by preprocess.py, where float16 resolves to
about 2.4e-4 of full scale. Against TIR1's 180-300K range that is 0.03K --
an order of magnitude below the instrument's own noise, so nothing physical
is lost. It just makes the set cheap enough to move to a remote machine.

dataset.py needs no change: it calls .float() on every tensor it loads, so
float16 on disk still trains in float32.

    python -m src.to_float16 --in-dir data/processed_full --out-dir data/processed_f16
"""

import argparse
from pathlib import Path

import numpy as np


def convert_all(in_dir, out_dir, overwrite=False):
    files = sorted(Path(in_dir).glob('*.npy'))
    if not files:
        raise FileNotFoundError(f"No .npy files in {in_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    converted = skipped = 0
    max_err = 0.0
    for i, src in enumerate(files, 1):
        dst = Path(out_dir) / src.name
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        arr = np.load(src)
        half = arr.astype(np.float16)
        # Check the claim rather than trusting it: a frame that left the [0,1]
        # range upstream would lose real precision here, and should be noticed.
        max_err = max(max_err, float(np.abs(arr - half.astype(np.float32)).max()))
        np.save(dst, half)
        converted += 1
        if i % 200 == 0:
            print(f"  {i}/{len(files)} ...")

    print(f"Converted {converted}, skipped {skipped} already present.")
    print(f"Largest round-trip error: {max_err:.3e} of full scale "
          f"({max_err * 120:.4f}K on TIR1's 180-300K range)")
    print(f"Now rebuild the manifest against {out_dir} -- it stores absolute paths.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert processed .npy frames to float16.")
    parser.add_argument("--in-dir", default="data/processed_full")
    parser.add_argument("--out-dir", default="data/processed_f16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_all(args.in_dir, args.out_dir, overwrite=args.overwrite)
