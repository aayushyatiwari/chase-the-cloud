import math

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

def squared_error_counts(pred, target):
    """
    Sum of squared errors, and the number of elements behind it.

    Returned raw because PSNR has to be formed once
    from a whole epoch's pooled MSE, not averaged over batches. See psnr_from_mse.
    """
    return torch.sum((pred - target) ** 2).item(), pred.numel()


def psnr_from_mse(mse, data_range=1.0):
    """
    Peak Signal-to-Noise Ratio in decibels, from an already-pooled MSE.

    This is MSE on a log scale: psnr = 10 * log10(range^2 / mse). It carries no
    information MSE does not already have, but video prediction papers report
    it, so it makes results comparable with them.

    Take the log ONCE, at the end, over the pooled MSE. Averaging per-batch PSNR
    averages logarithms, which is the log of the *geometric* mean of the batch
    MSEs -- a different number, always lower than the true PSNR, and inconsistent
    with the MSE reported next to it. A single near-perfect batch also sends its
    PSNR to infinity and poisons the whole epoch average.
    """
    if mse <= 0:
        return float('inf')
    return 10.0 * math.log10((data_range ** 2) / mse)


def psnr(pred, target, data_range=1.0):
    """
    PSNR for a single batch. Prefer pooling squared_error_counts over an epoch.
    """
    se, n = squared_error_counts(pred, target)
    return torch.tensor(psnr_from_mse(se / n, data_range), device=pred.device)
