"""
SimVPv2: SimVP with a gSTA (gated spatiotemporal attention) translator.

Paper: "SimVP: Towards Simple yet Powerful Spatiotemporal Predictive Learning"
github: https://github.com/chengtan9907/OpenSTL

Same three-part shape as simvp.py -- spatial encoder, temporal translator,
spatial decoder -- with only the middle replaced. v1's translator is a stack of
Inception blocks: several kernel sizes in parallel, summed. v2 swaps each for a
MetaFormer block whose token mixer is gSTA: a large receptive field built from
cheap depthwise convolutions (5x5 depthwise, then 7x7 depthwise dilated by 3,
covering ~21x21) used as a *gate* -- it multiplies the features rather than
being added to them, so the block learns where to look, not just what to see.

The encoder and decoder are imported from simvp.py unchanged, so a v1 and a v2
run differ only in the translator and stay directly comparable.
"""
import torch
import torch.nn as nn

from .simvp import Encoder, Decoder


class AttentionModule(nn.Module):
    """
    The gate itself: a large effective receptive field on the cheap.

    A 5x5 depthwise conv followed by a 7x7 depthwise conv dilated by 3 sees
    about 21x21 pixels for a small fraction of the parameters a real 21x21
    kernel would cost, and depthwise means each channel keeps its own filter.
    The 1x1 then mixes channels. The result multiplies the input, so a value
    near zero suppresses a location and a large value amplifies it.
    """
    def __init__(self, dim, kernel_size=21, dilation=3):
        super(AttentionModule, self).__init__()
        # d_k / d_p: the small depthwise conv. d_k is chosen so the pair of
        # convs covers exactly `kernel_size` pixels.
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1)) // 2

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p,
                                      groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    """
    The gate wrapped in the usual project -> activate -> gate -> project, with a
    residual connection so an untrained block starts close to the identity.
    """
    def __init__(self, dim, kernel_size=21, dilation=3):
        super(SpatialAttention, self).__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim, kernel_size, dilation)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class MixMlp(nn.Module):
    """
    The MetaFormer channel-mixing half: 1x1 up, depthwise 3x3, GELU, 1x1 down.
    The depthwise conv in the middle is what makes it "Mix" rather than a plain
    per-pixel MLP -- it lets neighbouring pixels talk during the channel mix.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MixMlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1,
                                bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GASubBlock(nn.Module):
    """
    One MetaFormer block: norm -> gated attention -> add, norm -> MLP -> add.

    layer_scale_1/2 are learnable per-channel scales initialised at 1e-2, so
    each branch starts contributing almost nothing and the block begins life as
    the identity. Without them a deep stack of these is hard to train.
    """
    def __init__(self, dim, kernel_size=21, mlp_ratio=4., drop=0., init_value=1e-2):
        super(GASubBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MixMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        return x


class MetaBlock(nn.Module):
    """
    A GASubBlock that may change channel width.

    The block is residual, which only works when in and out widths match. When
    they differ (the first and last blocks of the translator) a 1x1 conv on the
    shortcut reconciles them.
    """
    def __init__(self, in_channels, out_channels, kernel_size=21, mlp_ratio=4., drop=0.):
        super(MetaBlock, self).__init__()
        self.block = GASubBlock(in_channels, kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio, drop=drop)
        self.reduction = (nn.Conv2d(in_channels, out_channels, 1)
                          if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        return self.reduction(self.block(x))


class MidMetaNet(nn.Module):
    """
    The translator: the T input frames' latents stacked along channels, run
    through N_T MetaBlocks, and reshaped into T_out output frames.

    Same contract as simvp.Mid_Xnet -- (B, T, C, H, W) in, (B, T_out, C, H, W)
    out -- so the surrounding encoder and decoder are untouched. Unlike v1 this
    is a plain stack with no U-net skips; the residual inside every block is
    what carries information forward instead.
    """
    def __init__(self, channel_in, channel_hid, N_T, T_out, hid_S,
                 kernel_size=21, mlp_ratio=4., drop=0.):
        super(MidMetaNet, self).__init__()
        assert N_T >= 2, "MidMetaNet needs at least 2 blocks (one in, one out)"
        self.N_T = N_T
        self.T_out = T_out

        layers = [MetaBlock(channel_in, channel_hid, kernel_size, mlp_ratio, drop)]
        for _ in range(1, N_T - 1):
            layers.append(MetaBlock(channel_hid, channel_hid, kernel_size, mlp_ratio, drop))
        # Final block widens to T_out frames' worth of latent channels, exactly
        # as v1's last Inception does.
        layers.append(MetaBlock(channel_hid, T_out * hid_S, kernel_size, mlp_ratio, drop))
        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        z = x.reshape(B, T * C, H, W)
        z = self.enc(z)
        # C (per-frame latent width) is unchanged; only the frame count differs.
        return z.reshape(B, self.T_out, C, H, W)


class SimVPv2(nn.Module):
    """
    SimVPv2, adapted -- like simvp.SimVP -- to predict T_out frames (default 1)
    from T_in frames rather than the paper's T_in == T_out setup.

    Constructor mirrors SimVP so the two swap in place, with three extra knobs
    for the translator: gSTA's receptive field, the MLP expansion and dropout.
    v1's `groups` and `incep_ker` have no counterpart here.
    """
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, T_out=1,
                 mlp_ratio=4., drop=0., spatio_kernel=21, out_channels=None):
        super(SimVPv2, self).__init__()
        T, C, H, W = shape_in
        self.T_out = T_out
        # As in v1: the decoder emits out_channels, not C. The input is wider
        # than the target by design (water vapour helps predict TIR1 without
        # being predicted), so the decoder never computes channels nothing scores.
        self.out_channels = C if out_channels is None else out_channels

        self.enc = Encoder(C, hid_S, N_S)
        self.hid = MidMetaNet(T * hid_S, hid_T, N_T, T_out, hid_S,
                              kernel_size=spatio_kernel, mlp_ratio=mlp_ratio, drop=drop)
        self.dec = Decoder(hid_S, self.out_channels, N_S)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        _, Cs, Hs, Ws = skip.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)                                  # (B, T_out, C_, H_, W_)
        hid = hid.reshape(B * self.T_out, C_, H_, W_)

        # skip has one entry per INPUT frame (batch B*T); the decoder needs one
        # per OUTPUT frame. Use each batch item's most recent T_out input frames.
        skip = skip.view(B, T, Cs, Hs, Ws)[:, -self.T_out:].reshape(B * self.T_out, Cs, Hs, Ws)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, self.T_out, self.out_channels, H, W)
        Y = Y.squeeze(1)  # T_out=1: (B, 1, C, H, W) -> (B, C, H, W), matching ConvLSTM
        return Y
