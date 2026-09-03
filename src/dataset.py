import json
import numpy as np
import torch
from torch.utils.data import Dataset


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


class Clouds(Dataset):
    """
    Sequences of satellite frames, cropped to a fixed window size.

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
    """

    def __init__(self, manifest_path='data/manifest.json', T=6, window_range=None,
                 crop_size=256, crop_stride=None):
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

        # Frame shape, read from a header only (mmap), not the whole array.
        probe = np.load(self.samples[0]['target_frame'], mmap_mode='r')
        self.C, self.H, self.W = probe.shape
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
        target = self._crop(sample['target_frame'], top, left)
        return torch.from_numpy(inputs).float(), torch.from_numpy(target).float()

    def _crop(self, path, top, left):
        """Read just one crop out of a stored frame, leaving the rest on disk."""
        size = self.crop_size
        arr = np.load(path, mmap_mode='r')
        return np.ascontiguousarray(arr[:, top:top + size, left:left + size])
