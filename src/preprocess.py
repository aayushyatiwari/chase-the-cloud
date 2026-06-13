import argparse
import os
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

# Default crop for INSAT-3DR (roughly center region)
DEFAULT_CROP = Crop(row_start=680, row_end=936, col_start=740, col_end=996)

DEFAULT_WINDOW = 256
DEFAULT_NORM_MIN = 180.0
DEFAULT_NORM_MAX = 300.0
DEFAULT_FILL_VALUE = 0.0

def h5_to_bt(h5_path):
    with h5py.File(h5_path, 'r') as f:
        # TIR1 is typically at index 0 in the 3D array (1, H, W)
        raw = f['IMG_TIR1'][0]
        lut = f['IMG_TIR1_TEMP'][:]
        # Convert raw counts to Brightness Temperature using LUT
        bt = lut[raw]
    return bt.astype(np.float32)

def normalize(bt, norm_min=DEFAULT_NORM_MIN, norm_max=DEFAULT_NORM_MAX):
    bt = np.asarray(bt, dtype=np.float32)
    normalized = (bt - norm_min) / (norm_max - norm_min)
    return np.clip(normalized, 0.0, 1.0)

def process_file(h5_path, out_path, crop, norm_min, norm_max, fill_value):
    try:
        bt = h5_to_bt(h5_path)
        bt = bt[crop.row_start : crop.row_end, crop.col_start : crop.col_end]
        bt = normalize(bt, norm_min=norm_min, norm_max=norm_max)
        bt = np.nan_to_num(bt, nan=fill_value, posinf=fill_value, neginf=fill_value)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, bt.astype(np.float32))
        return True
    except Exception as e:
        print(f"Error processing {h5_path.name}: {e}")
        return False

def process_all(raw_dir, out_dir, crop=DEFAULT_CROP, norm_min=DEFAULT_NORM_MIN, norm_max=DEFAULT_NORM_MAX, fill_value=DEFAULT_FILL_VALUE, overwrite=False):
    files = sorted(Path(raw_dir).glob("*.h5"))
    if not files:
        print(f"No .h5 files found in {raw_dir}")
        return

    processed = 0
    for h5_path in files:
        out_path = Path(out_dir) / f"{h5_path.stem}.npy"
        if out_path.exists() and not overwrite:
            continue
        if process_file(h5_path, out_path, crop, norm_min, norm_max, fill_value):
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed} files...")
    print(f"Finished: {processed} files processed, crop={crop}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess INSAT HDF5 files into .npy frames.")
    parser.add_argument("--raw-dir", default="data/data")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--norm-min", type=float, default=DEFAULT_NORM_MIN)
    parser.add_argument("--norm-max", type=float, default=DEFAULT_NORM_MAX)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    process_all(args.raw_dir, args.out_dir, norm_min=args.norm_min, norm_max=args.norm_max, overwrite=args.overwrite)
