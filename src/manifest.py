import os
import json
import re
from datetime import date, datetime
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

def build(data_dir, T=6, step_minutes=30, tolerance_minutes=5, output_path='data/manifest.json',
          stride=1):
    """
    Build a manifest of `T` input frames -> 1 target frame windows.

    stride is how far to advance after accepting a window. stride=1 slides one
    frame at a time, so consecutive windows share T of their T+1 frames -- 6.8x
    redundancy on this data, and a sample count that overstates how much
    independent weather is present. stride=T+1 partitions the series into
    non-overlapping sequences instead, which is what the FY-2G benchmark does.

    A rejected window advances by 1 regardless, so a gap costs only the windows
    that actually span it rather than knocking the whole partition out of phase.
    """
    # Sort by parsed time, not filename: lexicographic order scrambles the series
    # once more than one month is present (01AUG < 01JUL < 01JUN).
    files = sorted(Path(data_dir).glob('*.npy'), key=_timestamp)
    valid = {f for f in files if _is_valid(f)}

    samples = []
    skipped_invalid = 0
    skipped_gap = 0
    i = 0
    while i + T < len(files):
        window = files[i : i + T + 1]
        if not all(f in valid for f in window):
            skipped_invalid += 1
            i += 1
        elif not _is_continuous(window, step_minutes, tolerance_minutes):
            # A missing frame makes the window span more time than it claims,
            # so the target is further ahead than the model is told to predict.
            skipped_gap += 1
            i += 1
        else:
            samples.append({
                "input_frames": [str(f.resolve()) for f in window[:T]],
                "target_frame": str(window[T].resolve())
            })
            i += stride
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(output_path), 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Built {len(samples)} samples at stride {stride}, "
          f"skipped {skipped_gap} with time gaps, {skipped_invalid} with invalid frames")

def _as_date(value):
    """Accept a date from YAML either as a real date or as an ISO string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def split_indices(manifest_path, splits, verbose=True):
    """
    Manifest indices for each named split, selected by calendar date.

    `splits` maps a name to an inclusive [start, end] pair of dates, e.g.
    {'train': ['2023-07-01', '2024-07-15'], 'val': ['2024-07-17', ...]}.

    Dates, not fractions: a fraction means a different period every time data
    is added. When July 2024 arrived, a 0.8 train fraction moved the boundary
    from 25 Jul 2023 to 19 Jul 2024.

    A window joins a split only when every frame in it falls inside the range,
    so no split can reach across a boundary for its inputs. Days named by no
    split are buffers, keeping consecutive splits more than a day apart.
    """
    with open(manifest_path) as f:
        samples = json.load(f)

    days = [
        [_timestamp(Path(p)).date() for p in s['input_frames'] + [s['target_frame']]]
        for s in samples
    ]

    indices, owner = {}, {}
    for name, bounds in splits.items():
        try:
            lo, hi = (_as_date(b) for b in bounds)
        except (TypeError, ValueError) as e:
            raise ValueError(f"splits.{name} must be an inclusive [start, end] date pair") from e
        if lo > hi:
            raise ValueError(f"splits.{name}: start {lo} is after end {hi}")

        picked = [i for i, window in enumerate(days) if all(lo <= d <= hi for d in window)]
        if not picked:
            raise ValueError(
                f"splits.{name} ({lo}..{hi}) selects no complete window. "
                "Check the dates against the data actually present."
            )
        for i in picked:
            if i in owner:
                raise ValueError(
                    f"splits.{name} overlaps splits.{owner[i]}: both claim window {i} "
                    f"({days[i][-1]}). Splits must not share windows."
                )
            owner[i] = name
        indices[name] = picked

    if verbose:
        total = len(samples)
        for name, picked in indices.items():
            lo, hi = (_as_date(b) for b in splits[name])
            covered = sorted({days[i][-1] for i in picked})
            print(f"  {name:5s} {lo} .. {hi}  {len(picked):5d} windows "
                  f"({100 * len(picked) / total:4.1f}%, {len(covered)} days)")
        held = total - len(owner)
        print(f"  unused (buffer days and partial windows): {held} of {total}")

    return indices


if __name__ == '__main__':
    build('data/processed/')