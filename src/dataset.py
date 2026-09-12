import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Frames are stored as 10-bit sensor counts; see src/preprocess.py.
N_COUNTS = 1024

# Built tables are shared between the train, val and test datasets. Each copy
# is ~49MB for four channels, and DataLoader forks a worker per copy, so three
# independent loads would carry a few hundred MB of duplicate tables.
_LUT_CACHE = {}


def tile_grid(H, W, size=256, stride=None):
    """
    Top-left corners of a grid of crops covering an (H, W) frame.

    stride=size (the default) steps one tile at a time. A smaller stride gives
    more, overlapping crops -- more samples, but they share pixels so they are
    less independent.

    A frame is rarely an exact number of strides across, so a last row and
    column flush with the bottom and right edges are added. Without them the
    leftover strip is never covered: at stride 512 on a 1616x1737 sector that
    silently left the bottom 336 rows and right 457 columns out of validation.
    """
    stride = stride or size

    def starts(total):
        pos = list(range(0, total - size + 1, stride))
        if pos and pos[-1] != total - size:
            pos.append(total - size)
        return pos

    return [(r, c) for r in starts(H) for c in starts(W)]


def load_lut_table(lut_path, norm_ranges_path):
    """
    Per-file calibration tables, pre-normalised to [0,1].

    Decoding a frame is `lut[count]`, and normalising it is
    `(bt - lo) / (hi - lo)` clipped -- but the second step only ever sees the
    1024 values the first step can produce. Folding it into the table once
    turns per-crop work into a single gather with no arithmetic: 1024 divisions
    per channel per file, instead of 65,536 per crop.

    The tables are per file because the instrument is recalibrated as it
    drifts. The ranges are NOT: one global [lo, hi] per channel, from
    norm_ranges_path, so the same temperature maps to the same value in every
    frame. Per-file tables are what makes frames comparable; a per-file range
    would be what breaks it.

    Returns (lut_norm, row_of_stem, channels) where lut_norm is
    (n_files, C, 1024) float32.
    """
    key = (str(lut_path), str(norm_ranges_path))
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]

    with np.load(lut_path, allow_pickle=False) as z:
        luts = z['luts'].astype(np.float64)          # (N, C, 1024), kelvin
        stems = [str(s) for s in z['files']]
        channels = [str(c) for c in z['channels']]

    with open(norm_ranges_path) as f:
        ranges = json.load(f)
    missing = [c for c in channels if c not in ranges]
    if missing:
        raise KeyError(f"{norm_ranges_path} has no range for {missing}")

    lo = np.array([ranges[c][0] for c in channels], np.float64)[None, :, None]
    hi = np.array([ranges[c][1] for c in channels], np.float64)[None, :, None]

    # float64 for the division, then down to float32. Only 1024 entries per
    # channel per file, so the wider arithmetic is free here, and it keeps the
    # result independent of how the ranges happen to round in float32.
    lut_norm = np.clip((luts - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    # Fill decodes to each channel's coldest entry, which normalises to 0.0
    # anyway; pinning it makes that a decision rather than a coincidence.
    lut_norm[:, :, -1] = 0.0

    out = (lut_norm, {s: i for i, s in enumerate(stems)}, channels)
    _LUT_CACHE[key] = out
    return out


class Clouds(Dataset):
    """
    Sequences of satellite frames, cropped to a fixed window size.

    Frames on disk are uint16 sensor counts. A crop is decoded to normalised
    brightness temperature here, by indexing that frame's own calibration
    table -- see load_lut_table. The counts stay on disk, so the normalisation
    ranges can change without reprocessing the granules.

    A sample is one time window plus one crop position. The same position is
    used for all frames of that window -- moving the crop between frames would
    look like camera motion and the model would try to learn it as cloud motion.

    The frame is tiled by crop_stride and every window is paired with every
    tile, in a fixed order. All three splits tile the same way, so each pixel is
    trained on and scored equally. A random crop position instead would sample
    the interior 65,536x more often than the corners -- over half the sector
    sits within one crop-width of an edge -- while val and test covered it
    uniformly.

    window_range selects which manifest entries this dataset covers. Slicing by
    window (not after crops are expanded) keeps the train/validation split
    purely by time, so every crop of a timestamp lands on the same side of it.

    target_channels is how many leading channels the model predicts. The inputs
    can be wider than the target -- water vapour helps predict TIR1 without
    being predicted itself -- and the predicted channels come first, matching
    ConvLSTM's single-channel head and ResidualWrapper.
    """

    def __init__(self, manifest_path='data/manifest_counts.json',
                 lut_path='data/luts.npz', norm_ranges_path='data/norm_ranges.json',
                 T=6, window_range=None, crop_size=256, crop_stride=None,
                 target_channels=1):
        with open(manifest_path) as f:
            samples = json.load(f)
        self.samples = [samples[i] for i in (window_range if window_range is not None
                                             else range(len(samples)))]
        if not self.samples:
            raise ValueError(
                f"No windows selected from {manifest_path} (it has {len(samples)}). "
                "Check data.splits and that the manifest is not tiny."
            )
        self.T = T
        self.crop_size = crop_size

        self.lut_norm, self.lut_row, self.channels = load_lut_table(
            lut_path, norm_ranges_path)

        # Frame shape, read from a header only (mmap), not the whole array.
        probe = np.load(self.samples[0]['target_frame'], mmap_mode='r')
        self.C, self.H, self.W = probe.shape
        if probe.dtype != np.uint16:
            raise ValueError(
                f"{self.samples[0]['target_frame']} is {probe.dtype}, expected uint16 "
                "counts. This manifest points at output from the old "
                "decode-at-preprocess pipeline."
            )
        if self.C != len(self.channels):
            raise ValueError(
                f"Frames have {self.C} channels but {lut_path} has tables for "
                f"{len(self.channels)} ({self.channels}). They came from different runs."
            )
        if not 1 <= target_channels <= self.C:
            raise ValueError(
                f"target_channels={target_channels} outside 1..{self.C}")
        self.target_channels = target_channels

        self.crops = tile_grid(self.H, self.W, crop_size, crop_stride)

    def __len__(self):
        return len(self.samples) * len(self.crops)

    def __getitem__(self, idx):
        window_idx, crop_idx = divmod(idx, len(self.crops))
        top, left = self.crops[crop_idx]

        sample = self.samples[window_idx]
        # Load failures are raised, not substituted: silently swapping in a
        # random sample would draw from the whole dataset, leaking training
        # frames into validation and hiding corrupt data.
        inputs = np.stack([self._crop(p, top, left) for p in sample['input_frames']])
        target = self._crop(sample['target_frame'], top, left)[:self.target_channels]
        return torch.from_numpy(inputs).float(), torch.from_numpy(target).float()

    def _crop(self, path, top, left):
        """Read one crop out of a stored frame and decode it, leaving the rest on disk."""
        size = self.crop_size
        arr = np.load(path, mmap_mode='r')
        counts = np.ascontiguousarray(arr[:, top:top + size, left:left + size])

        stem = Path(path).stem
        try:
            lut = self.lut_norm[self.lut_row[stem]]
        except KeyError:
            raise KeyError(
                f"No calibration table for {stem}. The .npz and the frames must "
                "come from the same preprocessing run."
            ) from None

        # One gather per channel: each has its own table, so they cannot share
        # a single fancy-index without flattening.
        out = np.empty(counts.shape, np.float32)
        for c in range(counts.shape[0]):
            out[c] = lut[c][counts[c]]
        return out
