import os
import json
import re
from datetime import datetime
from pathlib import Path
import numpy as np

_MONTHS = {m: i + 1 for i, m in enumerate(
    ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])}
_STAMP = re.compile(r'_(\d{2})([A-Z]{3})(\d{4})_(\d{2})(\d{2})_')

def _timestamp(path):
    """Acquisition time from an INSAT filename, e.g. 3RIMG_01JUL2023_0015_L1C_... ."""
    m = _STAMP.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse a timestamp from {path.name}")
    day, month, year, hour, minute = m.groups()
    return datetime(int(year), _MONTHS[month], int(day), int(hour), int(minute))

def _is_valid(path):
    # Frames are (C, H, W) now -- the channel axis is always there, even when
    # there is only one channel. mmap_mode avoids reading the whole frame just
    # to check its shape, which matters for full-sector files.
    try:
        arr = np.load(path, mmap_mode='r')
        return arr.ndim == 3
    except Exception:
        return False

def _is_continuous(window, step_minutes, tolerance_minutes):
    """True if every consecutive pair of frames is exactly one time step apart."""
    times = [_timestamp(f) for f in window]
    return all(
        abs((b - a).total_seconds() / 60.0 - step_minutes) <= tolerance_minutes
        for a, b in zip(times, times[1:])
    )

def build(data_dir, T=6, step_minutes=30, tolerance_minutes=5, output_path='data/manifest.json'):
    # Sort by parsed time, not filename: lexicographic order scrambles the series
    # once more than one month is present (01AUG < 01JUL < 01JUN).
    files = sorted(Path(data_dir).glob('*.npy'), key=_timestamp)
    valid = {f for f in files if _is_valid(f)}

    samples = []
    skipped_invalid = 0
    skipped_gap = 0
    for i in range(len(files) - T):
        window = files[i : i + T + 1]
        if not all(f in valid for f in window):
            skipped_invalid += 1
        elif not _is_continuous(window, step_minutes, tolerance_minutes):
            # A missing frame makes the window span more time than it claims,
            # so the target is further ahead than the model is told to predict.
            skipped_gap += 1
        else:
            samples.append({
                "input_frames": [str(f.resolve()) for f in window[:T]],
                "target_frame": str(window[T].resolve())
            })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(output_path), 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Built {len(samples)} samples, "
          f"skipped {skipped_gap} with time gaps, {skipped_invalid} with invalid frames")

if __name__ == '__main__':
    build('data/processed/')