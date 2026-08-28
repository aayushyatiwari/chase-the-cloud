import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    Most recently written .pt under checkpoint_dir, searched recursively so it
    picks up the newest per-run subdirectory (e.g. 20260828_201053_L3_h64/).

    train.py only saves on a validation-loss improvement, so the newest
    checkpoint of a run is also its best so far.
    """
    paths = sorted(Path(checkpoint_dir).rglob('*.pt'), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No .pt checkpoints found under {checkpoint_dir}")
    return paths[-1]

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    Expects inputs in shape [B, C, H, W] and values in range [0, 1].
    """
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def csi_counts(preds, targets, threshold=0.5):
    """
    Contingency counts (hits, misses, false_alarms) for the cold-cloud class.

    preprocess.py normalizes so that cold cloud tops -> 0 and the warm surface
    -> 1, so a cloud pixel is one BELOW the threshold. The default 0.5
    corresponds to 240K, a standard cold cloud-top cutoff.

    Returns raw counts so they can be pooled across a whole epoch before
    forming the ratio: CSI is a ratio of sums, not a mean of per-batch ratios.
    """
    preds_bin = (preds < threshold).float()
    targets_bin = (targets < threshold).float()

    hits = (preds_bin * targets_bin).sum().item()
    misses = ((1 - preds_bin) * targets_bin).sum().item()
    false_alarms = (preds_bin * (1 - targets_bin)).sum().item()
    return hits, misses, false_alarms


def csi_from_counts(hits, misses, false_alarms):
    """
    Critical Success Index (CSI) / Threat Score.
    CSI = Hits / (Hits + Misses + FalseAlarms). Higher is better.
    True negatives are excluded, so clear sky cannot inflate the score.
    Undefined (NaN) when no cloud is present in either prediction or target.
    """
    denominator = hits + misses + false_alarms
    return hits / denominator if denominator > 0 else float('nan')


def calculate_csi(preds, targets, threshold=0.5):
    """CSI for a single batch. Prefer pooling csi_counts over a full epoch."""
    return torch.tensor(
        csi_from_counts(*csi_counts(preds, targets, threshold))
    ).to(preds.device)
