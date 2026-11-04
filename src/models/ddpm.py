"""
DDPM: a conditional denoising diffusion model for next-frame prediction.

Paper: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020), with the
conditioning-by-concatenation of SR3 ("Image Super-Resolution via Iterative
Refinement", Saharia et al., 2021) and the respaced sampler and cosine schedule
of "Improved DDPM" (Nichol & Dhariwal, 2021).

Every other model in this repo regresses: given the past, output one frame, and
MSE pulls that frame toward the *average* of all futures consistent with the
past. Average futures are blurry, which is why the forecasts soften with time.
A diffusion model instead learns to *sample* a future. It is trained on a much
easier task than forecasting -- take the true next frame, bury it in a known
amount of Gaussian noise, and predict that noise -- and at inference it starts
from pure noise and removes it a little at a time, steered by the past frames.

    Training (one step, cheap):
        x0      = target frame (or its change from the last input, see residual)
        t       ~ Uniform{0 .. timesteps-1}
        eps     ~ N(0, I)
        x_t     = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps
        loss    = || eps - UNet([x_t, past frames], t) ||^2

    Sampling (sample_steps network passes, expensive):
        x_T ~ N(0, I);  for t = T .. 1:  x_{t-1} ~ p(x_{t-1} | x_t, past frames)

Consequences worth knowing before comparing it to the regressors:

- A sample is one plausible future, not the mean one. It is sharper, but its
  MSE is typically *higher* than a blurry regressor's -- MSE rewards hedging.
  SSIM and visual sharpness are the fairer comparison; averaging several
  samples (ensemble_size) recovers a mean forecast if MSE is what matters.
- forward() runs the whole sampler, so validation costs sample_steps network
  passes per batch. --dry-run reports what that means for an epoch.
- An exponential moving average of the weights (ema_decay) is what samples are
  drawn from, as in the paper. Raw weights give noticeably worse samples.

Shape of the model:

    inputs (B, T, C, H, W) -> reshaped to (B, T*C, H, W): the condition
    x_t    (B, out_channels, H, W)
      -> concat -> U-Net with sinusoidal timestep embedding, GroupNorm resblocks,
         self-attention at the coarsest resolutions
      -> predicted noise (B, out_channels, H, W)
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def make_betas(schedule, timesteps):
    """
    Per-step noise variances beta_1..beta_T.

    linear: the original paper's 1e-4 -> 0.02. It destroys the signal quickly,
            so the last few hundred steps are close to pure noise already.
    cosine: Improved DDPM's schedule. abar falls along a cosine, so the signal
            fades more evenly and fewer steps are wasted near pure noise.
    """
    if schedule == 'linear':
        # Scaled so the schedule means the same thing at any timesteps count.
        scale = 1000 / timesteps
        return torch.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=torch.float64)
    if schedule == 'cosine':
        s = 0.008
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        abar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        abar = abar / abar[0]
        return (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    raise ValueError(f"Unknown beta schedule {schedule!r} (expected 'linear' or 'cosine')")


# ---------------------------------------------------------------------------
# U-Net denoiser
# ---------------------------------------------------------------------------

def timestep_embedding(t, dim):
    """Transformer-style sinusoidal embedding of integer timesteps, (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


def norm(channels):
    return nn.GroupNorm(min(32, channels), channels)


class ResBlock(nn.Module):
    """Two 3x3 convs with the timestep added in between, plus a skip connection."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.block1 = nn.Sequential(norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.block2 = nn.Sequential(norm(out_ch), nn.SiLU(), nn.Dropout(dropout),
                                    nn.Conv2d(out_ch, out_ch, 3, padding=1))
        # Zero-init the last conv so each block starts as the identity; deep
        # U-Nets train much more stably that way.
        nn.init.zeros_(self.block2[-1].weight)
        nn.init.zeros_(self.block2[-1].bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.block1(x)
        h = h + self.emb(emb)[:, :, None, None]
        h = self.block2(h)
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Multi-head self-attention over the H*W positions of one feature map."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.num_heads, C // self.num_heads, H * W).unbind(1)
        # (B, heads, HW, d) for scaled_dot_product_attention
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    """
    The DDPM U-Net (Ho et al.), sized down for 256x256 crops.

    channel_mult sets the width at each resolution; len(channel_mult) - 1
    downsamplings, so (1, 2, 2, 4) on 256 gives 256 -> 128 -> 64 -> 32.
    attn_levels lists which of those levels get self-attention -- only the
    coarse ones, since attention at 256x256 would be 65k tokens.
    """

    def __init__(self, in_channels, out_channels, base_channels=64,
                 channel_mult=(1, 2, 2, 4), num_res_blocks=2, attn_levels=(3,),
                 num_heads=4, dropout=0.0):
        super().__init__()
        emb_dim = base_channels * 4
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(nn.Linear(base_channels, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))

        self.inp = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder. skip_chs records the width of every tensor pushed onto the
        # skip stack, so the decoder knows what it will concatenate.
        self.down = nn.ModuleList()
        skip_chs = [base_channels]
        ch = base_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layer = nn.ModuleList([ResBlock(ch, base_channels * mult, emb_dim, dropout)])
                ch = base_channels * mult
                if level in attn_levels:
                    layer.append(AttnBlock(ch, num_heads))
                self.down.append(layer)
                skip_chs.append(ch)
            if level != len(channel_mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                skip_chs.append(ch)

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout), AttnBlock(ch, num_heads),
                                  ResBlock(ch, ch, emb_dim, dropout)])

        # Decoder: one more block per level than the encoder, consuming the skips.
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layer = nn.ModuleList([ResBlock(ch + skip_chs.pop(), base_channels * mult, emb_dim, dropout)])
                ch = base_channels * mult
                if level in attn_levels:
                    layer.append(AttnBlock(ch, num_heads))
                if level != 0 and i == num_res_blocks:
                    layer.append(nn.Upsample(scale_factor=2, mode='nearest'))
                    layer.append(nn.Conv2d(ch, ch, 3, padding=1))
                self.up.append(layer)

        self.out = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _run(layer, h, emb):
        for m in layer:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        return h

    def forward(self, x, t):
        emb = self.time_mlp(timestep_embedding(t, self.base_channels))
        h = self.inp(x)
        skips = [h]
        for layer in self.down:
            h = self._run(layer, h, emb)
            skips.append(h)
        h = self._run(self.mid, h, emb)
        for layer in self.up:
            h = self._run(layer, torch.cat([h, skips.pop()], dim=1), emb)
        return self.out(h)


# ---------------------------------------------------------------------------
# Diffusion wrapper
# ---------------------------------------------------------------------------

class DDPM(nn.Module):
    """
    Conditional DDPM over the predicted channels of the next frame.

    The interface matches the other models where it can:
        forward(inputs)               -> sampled forecast (B, out_channels, H, W),
                                         in the same [0, 1] space as the target
    and adds what diffusion needs that regression does not:
        training_loss(inputs, target) -> the noise-prediction loss to backprop
        after_step()                  -> EMA update, called after optimizer.step()

    residual: diffuse the change from the last input frame instead of the frame
    itself -- the same idea as ResidualWrapper, which cannot wrap this model
    because the residual has to be taken before noise is added, not after.
    The change is small (std of a few hundredths), so it is divided by
    residual_scale to bring it to roughly unit variance; a signal far smaller
    than the noise is drowned by the first few steps. `python train.py
    --dry-run` prints the std of x0, which is what residual_scale should make ~1.

    Without residual the target is mapped from [0, 1] to [-1, 1], the range the
    schedules were designed for.
    """

    def __init__(self, in_channels, T, out_channels=1, timesteps=1000,
                 beta_schedule='linear', sample_steps=50, residual=False,
                 residual_scale=0.1, ema_decay=0.9999, ensemble_size=1,
                 clip_x0=True, **unet_kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.timesteps = timesteps
        self.sample_steps = min(sample_steps, timesteps)
        self.residual = residual
        self.residual_scale = residual_scale
        self.ema_decay = ema_decay
        self.ensemble_size = ensemble_size
        self.clip_x0 = clip_x0

        # The condition is every channel of every past frame, stacked as channels.
        self.net = UNet(in_channels=out_channels + T * in_channels,
                        out_channels=out_channels, **unet_kwargs)
        # Samples come from an EMA copy of the weights. requires_grad=False keeps
        # it out of backprop; it still rides along in state_dict and checkpoints.
        self.ema = copy.deepcopy(self.net)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.register_buffer('ema_steps', torch.zeros((), dtype=torch.long))

        betas = make_betas(beta_schedule, timesteps)
        abar = torch.cumprod(1 - betas, dim=0)
        # Stored in float32 for use; computed in float64 so 1 - abar near t=0
        # does not round to nothing.
        self.register_buffer('alphas_cumprod', abar.float())
        self.register_buffer('sqrt_abar', abar.sqrt().float())
        self.register_buffer('sqrt_one_minus_abar', (1 - abar).sqrt().float())

    # -- target <-> diffusion space -------------------------------------------

    def _last(self, inputs):
        return inputs[:, -1, :self.out_channels]

    def encode_target(self, inputs, target):
        """Target in [0, 1] -> x0, the signal the diffusion runs on."""
        if self.residual:
            return (target - self._last(inputs)) / self.residual_scale
        return target * 2 - 1

    def decode_x0(self, inputs, x0):
        if self.residual:
            return self._last(inputs) + x0 * self.residual_scale
        return (x0 + 1) / 2

    def _x0_bounds(self):
        # Where x0 can legitimately be: target and last frame both lie in [0, 1].
        return (1.0 / self.residual_scale) if self.residual else 1.0

    @staticmethod
    def _cond(inputs):
        B, T, C, H, W = inputs.shape
        return inputs.reshape(B, T * C, H, W)

    # -- training --------------------------------------------------------------

    def training_loss(self, inputs, target):
        x0 = self.encode_target(inputs, target)
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = (self.sqrt_abar[t].view(B, 1, 1, 1) * x0
               + self.sqrt_one_minus_abar[t].view(B, 1, 1, 1) * noise)
        pred = self.net(torch.cat([x_t, self._cond(inputs)], dim=1), t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def after_step(self):
        """
        Move the EMA weights toward the live ones.

        The decay is warmed up (as in many DDPM codebases), so early in training
        the EMA tracks the net closely instead of averaging in its random init
        for the first ~10k steps.
        """
        self.ema_steps += 1
        n = self.ema_steps.item()
        decay = min(self.ema_decay, (1 + n) / (10 + n))
        for e, p in zip(self.ema.parameters(), self.net.parameters()):
            e.lerp_(p, 1 - decay)
        for e, b in zip(self.ema.buffers(), self.net.buffers()):
            e.copy_(b)

    # -- sampling --------------------------------------------------------------

    def _sample_timesteps(self):
        """
        The subsequence of the training timesteps the sampler visits.

        sample_steps == timesteps is the paper's sampler, one pass per step.
        Fewer is Improved DDPM's respacing: the same ancestral sampler over an
        evenly spaced subset, with each jump's beta recomputed from abar so the
        chain stays a valid DDPM -- just a coarser one.
        """
        return torch.linspace(0, self.timesteps - 1, self.sample_steps).round().long().tolist()

    @torch.no_grad()
    def sample(self, inputs, net=None, generator=None):
        net = net or self.ema
        cond = self._cond(inputs)
        B, _, H, W = cond.shape
        x = torch.randn(B, self.out_channels, H, W, device=cond.device, generator=generator)
        steps = self._sample_timesteps()
        bound = self._x0_bounds()

        for i in reversed(range(len(steps))):
            t = steps[i]
            abar_t = self.alphas_cumprod[t]
            abar_prev = self.alphas_cumprod[steps[i - 1]] if i > 0 else abar_t.new_tensor(1.0)
            beta = 1 - abar_t / abar_prev  # the effective beta of this (possibly long) jump

            eps = net(torch.cat([x, cond], dim=1), torch.full((B,), t, device=x.device))
            x0 = (x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()
            if self.clip_x0:
                x0 = x0.clamp(-bound, bound)

            # Posterior q(x_{t-1} | x_t, x0): Ho et al. eq. 7.
            mean = (abar_prev.sqrt() * beta / (1 - abar_t)) * x0 \
                 + ((1 - beta).sqrt() * (1 - abar_prev) / (1 - abar_t)) * x
            if i > 0:
                var = beta * (1 - abar_prev) / (1 - abar_t)
                noise = torch.randn(x.shape, device=x.device, generator=generator)
                x = mean + var.sqrt() * noise
            else:
                x = mean
        return self.decode_x0(inputs, x)

    def forward(self, inputs):
        """
        A forecast, comparable to any other model's output.

        ensemble_size > 1 averages that many independent samples: a mean
        forecast, lower MSE, but blurrier -- trading back toward a regressor.
        """
        out = self.sample(inputs)
        for _ in range(self.ensemble_size - 1):
            out = out + self.sample(inputs)
        return out / self.ensemble_size
