import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


@dataclass(frozen=True)
class Crop:
    row_start: int
    row_end: int
    col_start: int
    col_end: int


DEFAULT_CROP = Crop(row_start=392, row_end=648, col_start=1160, col_end=1416)

DEFAULT_WINDOW = 256
DEFAULT_STRIDE = 64
DEFAULT_BORDER = 200
DEFAULT_VALID_MIN = 150.0
DEFAULT_VALID_MAX = 320.0
DEFAULT_NORM_MIN = 150.0
DEFAULT_NORM_MAX = 320.0
DEFAULT_FILL_VALUE = 0.0


def _read_scalar(ds, name):
    return float(np.asarray(np.ma.filled(ds.variables[name][:], np.nan)))


def nc_to_bt(nc_path, use_dqf=True):
    with Dataset(nc_path) as ds:
        rad = np.ma.filled(ds.variables["Rad"][:], np.nan).astype(np.float32)
        fk1 = _read_scalar(ds, "planck_fk1")
        fk2 = _read_scalar(ds, "planck_fk2")
        bc1 = _read_scalar(ds, "planck_bc1")
        bc2 = _read_scalar(ds, "planck_bc2")
        dqf = None
        if use_dqf and "DQF" in ds.variables:
            dqf = np.ma.filled(ds.variables["DQF"][:], 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        bt = (fk2 / np.log(fk1 / rad + 1.0) - bc1) / bc2

    bt = np.asarray(bt, dtype=np.float32)
    bt[(rad <= 0) | ~np.isfinite(rad)] = np.nan
    if dqf is not None:
        bt[dqf != 0] = np.nan
    return bt


def _candidate_starts(size, window, border, stride):
    stop = size - window - border
    starts = list(range(border, stop, stride))
    if not starts:
        raise ValueError(
            f"image dimension {size} is too small for window={window} and border={border}"
        )
    return starts


def variance_map(
    bt,
    window=DEFAULT_WINDOW,
    stride=DEFAULT_STRIDE,
    border=DEFAULT_BORDER,
    valid_min=DEFAULT_VALID_MIN,
    valid_max=DEFAULT_VALID_MAX,
    min_valid_fraction=0.8,
):
    height, width = bt.shape
    rows = _candidate_starts(height, window, border, stride)
    cols = _candidate_starts(width, window, border, stride)

    bt_clean = np.array(bt, dtype=np.float32, copy=True)
    bt_clean[(bt_clean < valid_min) | (bt_clean > valid_max)] = np.nan

    min_valid = min_valid_fraction * window * window
    vmap = np.zeros((len(rows), len(cols)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(cols):
            patch = bt_clean[row : row + window, col : col + window]
            valid = np.count_nonzero(~np.isnan(patch))
            if valid > min_valid:
                vmap[row_idx, col_idx] = np.nanvar(patch)
    return vmap, rows, cols


def find_best_crop(
    nc_paths,
    window=DEFAULT_WINDOW,
    stride=DEFAULT_STRIDE,
    border=DEFAULT_BORDER,
    valid_min=DEFAULT_VALID_MIN,
    valid_max=DEFAULT_VALID_MAX,
    min_valid_fraction=0.8,
):
    avg_vmap = None
    rows = cols = None
    used = 0

    for nc_path in nc_paths:
        try:
            bt = nc_to_bt(nc_path)
            vmap, rows, cols = variance_map(
                bt,
                window=window,
                stride=stride,
                border=border,
                valid_min=valid_min,
                valid_max=valid_max,
                min_valid_fraction=min_valid_fraction,
            )
        except Exception as exc:
            print(f"skipping crop scan for {Path(nc_path).name}: {exc}")
            continue

        avg_vmap = vmap if avg_vmap is None else avg_vmap + vmap
        used += 1

    if used == 0 or avg_vmap is None or rows is None or cols is None:
        raise RuntimeError("could not compute a crop; no readable NetCDF files were found")

    avg_vmap /= used
    best_row_idx, best_col_idx = np.unravel_index(np.argmax(avg_vmap), avg_vmap.shape)
    row_start = rows[best_row_idx]
    col_start = cols[best_col_idx]
    crop = Crop(row_start, row_start + window, col_start, col_start + window)
    print(
        "Best crop: "
        f"ROW_START={crop.row_start}, ROW_END={crop.row_end}, "
        f"COL_START={crop.col_start}, COL_END={crop.col_end} "
        f"(scanned {used} files)"
    )
    return crop


def crop_bt(bt, crop):
    return bt[crop.row_start : crop.row_end, crop.col_start : crop.col_end]


def normalize(bt, norm_min=DEFAULT_NORM_MIN, norm_max=DEFAULT_NORM_MAX):
    if norm_max <= norm_min:
        raise ValueError("norm_max must be greater than norm_min")
    bt = np.asarray(bt, dtype=np.float32)
    normalized = (bt - norm_min) / (norm_max - norm_min)
    return np.clip(normalized, 0.0, 1.0)


def save_npy_atomic(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, array)
    os.replace(tmp_path, path)


def list_nc_files(raw_dir):
    return sorted(Path(raw_dir).glob("*.nc"))


def process_file(nc_path, out_path, crop, norm_min, norm_max, fill_value):
    bt = nc_to_bt(nc_path)
    bt = crop_bt(bt, crop)
    bt = normalize(bt, norm_min=norm_min, norm_max=norm_max)
    bt = np.nan_to_num(bt, nan=fill_value, posinf=fill_value, neginf=fill_value)
    bt = bt.astype(np.float32, copy=False)
    save_npy_atomic(out_path, bt)


def process_all(
    raw_dir="data/raw",
    out_dir="data/processed",
    find_crop=False,
    window=DEFAULT_WINDOW,
    stride=DEFAULT_STRIDE,
    border=DEFAULT_BORDER,
    valid_min=DEFAULT_VALID_MIN,
    valid_max=DEFAULT_VALID_MAX,
    norm_min=DEFAULT_NORM_MIN,
    norm_max=DEFAULT_NORM_MAX,
    fill_value=DEFAULT_FILL_VALUE,
    overwrite=False,
    max_crop_files=None,
):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    files = list_nc_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"no .nc files found in {raw_dir}")

    if find_crop:
        crop_files = files[:max_crop_files] if max_crop_files else files
        crop = find_best_crop(
            crop_files,
            window=window,
            stride=stride,
            border=border,
            valid_min=valid_min,
            valid_max=valid_max,
        )
    else:
        crop = DEFAULT_CROP
        print(f"Using default crop: {crop}")

    out_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    for nc_path in files:
        out_path = out_dir / f"{nc_path.stem}.npy"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        process_file(nc_path, out_path, crop, norm_min, norm_max, fill_value)
        processed += 1
        print(f"processed {out_path.name}")

    print(f"done: processed={processed}, skipped={skipped}, crop={crop}")
    return crop


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess GOES NetCDF files into .npy frames.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument(
        "--find-crop",
        action="store_true",
        help="Run variance search to discover a new crop. If omitted, uses the hardcoded default crop.",
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--border", type=int, default=DEFAULT_BORDER)
    parser.add_argument("--valid-min", type=float, default=DEFAULT_VALID_MIN)
    parser.add_argument("--valid-max", type=float, default=DEFAULT_VALID_MAX)
    parser.add_argument("--norm-min", type=float, default=DEFAULT_NORM_MIN)
    parser.add_argument("--norm-max", type=float, default=DEFAULT_NORM_MAX)
    parser.add_argument("--fill-value", type=float, default=DEFAULT_FILL_VALUE)
    parser.add_argument("--max-crop-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    process_all(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        find_crop=args.find_crop,
        window=args.window,
        stride=args.stride,
        border=args.border,
        valid_min=args.valid_min,
        valid_max=args.valid_max,
        norm_min=args.norm_min,
        norm_max=args.norm_max,
        fill_value=args.fill_value,
        overwrite=args.overwrite,
        max_crop_files=args.max_crop_files,
    )


if __name__ == "__main__":
    main()