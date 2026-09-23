import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def latest_checkpoint(checkpoint_dir='checkpoints', model=None):
    """
    Most recently written .pt under checkpoint_dir, searched recursively so it
    picks up the newest per-run subdirectory (e.g. 20260828_201053_L3_h64/).

    train.py only saves on a validation-loss improvement, so the newest
    checkpoint of a run is also its best so far.

    Pass `model` to only accept a checkpoint whose weights fit it: same keys
    and same shapes. checkpoint_dir holds every architecture's runs, so the
    newest file overall may belong to a different model (e.g. a PredRNN run
    when resuming a ConvLSTM). Matching on the weights themselves rather than
    the run name also covers old run directories that predate the model type
    being in the name, and catches hidden_dim / num_layers / residual changes.
    """
    paths = sorted(Path(checkpoint_dir).rglob('*.pt'), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No .pt checkpoints found under {checkpoint_dir}")
    if model is None:
        return paths[-1]

    expected = {k: v.shape for k, v in model.state_dict().items()}
    for path in reversed(paths):
        # mmap: only the header is read here, not every tensor in the file
        state = torch.load(path, map_location='cpu', mmap=True)['model_state_dict']
        if {k: v.shape for k, v in state.items()} == expected:
            if path != paths[-1]:
                print("latest_checkpoint: skipped newer checkpoints that don't fit this model")
            return path
    raise FileNotFoundError(
        f"No checkpoint under {checkpoint_dir} matches the current model "
        f"({len(paths)} .pt files checked). Set train.resume_from to null or a path.")

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


def gradient_difference(pred, target, p=1):
    """
    Gradient Difference Loss (Mathieu et al., 2016), on [B, C, H, W] tensors.

    MSE is minimised by a blur: when the model is unsure where an edge will be,
    the lowest-error answer is to smear it across both possibilities. Half a
    cloud edge in two places costs less than a sharp edge in the wrong one, so
    the model learns to hedge and the forecasts come out soft.

    This scores the *edges* instead of the pixels. Take the neighbour-to-
    neighbour difference along each axis -- large where the image changes
    quickly, near zero over flat sky -- and compare the prediction's to the
    target's:

        GDL = mean(| |dx(pred)| - |dx(target)| |^p)
            + mean(| |dy(pred)| - |dy(target)| |^p)

    A blurred prediction has small gradients everywhere, so it cannot score well
    however close its pixel values are. p=1 is the sharper and more forgiving of
    the two; p=2 behaves more like MSE.

    Note what this does NOT see: add a constant to every pixel and every
    gradient is unchanged, so the loss is identical. It is blind to overall
    brightness and must be paired with a pixel term -- see CombinedLoss.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"gradient_difference got {tuple(pred.shape)} and {tuple(target.shape)}; "
            f"these would broadcast into nonsense instead of erroring."
        )

    # Differences along width and height. One column/row shorter than the
    # input, which is why the two terms are averaged separately.
    dx_pred = torch.abs(pred[..., :, 1:] - pred[..., :, :-1])
    dx_true = torch.abs(target[..., :, 1:] - target[..., :, :-1])
    dy_pred = torch.abs(pred[..., 1:, :] - pred[..., :-1, :])
    dy_true = torch.abs(target[..., 1:, :] - target[..., :-1, :])

    return ((dx_pred - dx_true).abs().pow(p).mean()
            + (dy_pred - dy_true).abs().pow(p).mean())


class GradientDifferenceLoss(nn.Module):
    """gradient_difference as a module, so it can be swapped in as a criterion."""

    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def forward(self, pred, target):
        return gradient_difference(pred, target, p=self.p)

    def extra_repr(self):
        return f"p={self.p}"


class CombinedLoss(nn.Module):
    """
    A weighted sum of a pixel loss and the gradient loss:

        L = alpha * pixel(pred, target) + beta * GDL(pred, target)

    The pixel term (MSE or L1) pins down the brightness the gradient term
    cannot see; the gradient term buys back the sharpness the pixel term gives
    away. Either weight can be zero to run that term alone -- beta = 0 is
    exactly the old MSE training.

    The two terms are on quite different scales: on [0, 1] frames the pixel
    error is already small, and the neighbour-to-neighbour differences are
    smaller still, so beta = 1 may leave the gradient term contributing almost
    nothing. Read the split printed by --dry-run and set beta so the terms are
    comparable before committing to a run.

    `last_terms` holds the two weighted terms from the most recent call, for
    logging the split rather than only the total.
    """

    def __init__(self, pixel_loss, alpha=1.0, beta=1.0, p=1):
        super().__init__()
        self.pixel_loss = pixel_loss
        self.alpha = alpha
        self.beta = beta
        self.gdl = GradientDifferenceLoss(p=p)
        self.last_terms = {}

    def forward(self, pred, target):
        pixel = self.alpha * self.pixel_loss(pred, target) if self.alpha else None
        grad = self.beta * self.gdl(pred, target) if self.beta else None

        self.last_terms = {
            'pixel': float(pixel.detach()) if pixel is not None else 0.0,
            'gdl': float(grad.detach()) if grad is not None else 0.0,
        }

        if pixel is None and grad is None:
            raise ValueError("CombinedLoss has alpha = beta = 0; there is nothing to minimise.")
        if grad is None:
            return pixel
        if pixel is None:
            return grad
        return pixel + grad

    def extra_repr(self):
        return f"alpha={self.alpha}, beta={self.beta}"
