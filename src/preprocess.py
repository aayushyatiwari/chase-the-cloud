import argparse
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

DEFAULT_WINDOW = 256
DEFAULT_NORM_MIN = 180.0
DEFAULT_NORM_MAX = 300.0
DEFAULT_FILL_VALUE = 0.0

# Per-channel true min/max, measured over every pixel of all 2964 raw files.
# Water vapour sits far colder than the window channels because it sees the
# upper troposphere, so reusing TIR1's range would squash its detail into the
# bottom of [0,1].
NORM_RANGES = {
    'TIR1': (179.86, 335.84),
    'TIR2': (179.93, 340.07),
    'WV':   (179.69, 308.57),
    'MIR':  (179.69, 339.79),
}

DEFAULT_CHANNELS = ('TIR1','TIR2', 'WV', 'MIR')

def h5_to_bt(h5_path, channel='TIR1'):
    """Read one channel and turn raw sensor counts into brightness temperature."""
    with h5py.File(h5_path, 'r') as f:
        # Images are stored as (1, H, W), so take index 0.
        raw = f[f'IMG_{channel}'][0]
        lut = f[f'IMG_{channel}_TEMP'][:]
        bt = lut[raw]
    return bt.astype(np.float32)

def normalize(bt, norm_min=DEFAULT_NORM_MIN, norm_max=DEFAULT_NORM_MAX):
    bt = np.asarray(bt, dtype=np.float32)
    normalized = (bt - norm_min) / (norm_max - norm_min)
    return np.clip(normalized, 0.0, 1.0)

def process_file(h5_path, out_path, channels, crop, fill_value, norm_ranges=NORM_RANGES):
    """
    Turn one HDF5 file into one .npy of shape (C, H, W).

    The channel axis is always present, even for a single channel, so the
    dataset can slice crops the same way no matter how many channels there are.
    Pass crop=None to keep the full sector, which is what multi-crop training
    wants -- crops are then chosen per sample while training.
    """
    try:
        planes = []
        for name in channels:
            bt = h5_to_bt(h5_path, name)
            if crop is not None:
                bt = bt[crop.row_start:crop.row_end, crop.col_start:crop.col_end]
            norm_min, norm_max = norm_ranges[name]
            bt = normalize(bt, norm_min=norm_min, norm_max=norm_max)
            bt = np.nan_to_num(bt, nan=fill_value, posinf=fill_value, neginf=fill_value)
            planes.append(bt)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.stack(planes).astype(np.float32))
        return True
    except Exception as e:
        print(f"Error processing {h5_path.name}: {e}")
        return False

def process_all(raw_dir, out_dir, channels=DEFAULT_CHANNELS, crop=None, fill_value=DEFAULT_FILL_VALUE,
                overwrite=False):
    files = sorted(Path(raw_dir).glob("*.h5"))
    if not files:
        print(f"No .h5 files found in {raw_dir}")
        return

    processed = 0
    for h5_path in files:
        out_path = Path(out_dir) / f"{h5_path.stem}.npy"
        if out_path.exists() and not overwrite:
            continue
        if process_file(h5_path, out_path, channels, crop, fill_value):
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} files...")
    print(f"Finished: {processed} files processed, channels={list(channels)}, crop={crop}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess INSAT HDF5 files into .npy frames.")
    parser.add_argument("--raw-dir", default="data/data")
    parser.add_argument("--out-dir", default="data/processed_full")
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS),
                        choices=list(NORM_RANGES), help="Channels to stack, in order.")
    parser.add_argument("--full-sector", action="store_true", default=True,
                        help="Keep the whole sector (default) so crops can be chosen while training.")
    parser.add_argument("--crop", action="store_false", dest="full_sector",
                        help="Instead cut the old fixed 256x256 window.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    process_all(args.raw_dir, args.out_dir,
                channels=tuple(args.channels),
                crop=None if args.full_sector else DEFAULT_CROP,
                overwrite=args.overwrite)
