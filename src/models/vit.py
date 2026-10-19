"""
ViT: a plain Vision Transformer adapted to next-frame prediction.

Paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021), with the
tubelet embedding from "ViViT: A Video Vision Transformer" (Arnab et al., 2021).

The convolutional models in this repo see the sky through small kernels and
build a large view by stacking them. A transformer instead cuts the input into
patches and lets every patch attend to every other one in every layer, so a
cloud in one corner can inform a prediction in the opposite corner from the
first block. The price is that attention cost grows with the square of the
number of tokens, and that a patch is predicted as one block of pixels, so the
output is only as fine as the head can make it.

Shape of the model:

    (B, T, C, H, W)
      -> tubelet embed: a (tubelet x patch x patch) chunk of all C channels
         becomes one token, so T/tubelet time slots x (H/patch)(W/patch) tokens
      -> + learned spatial and temporal position embeddings
      -> depth x pre-norm transformer block, joint space-time attention
      -> the time slots of each spatial patch concatenated and fused to one token
      -> linear head to patch x patch x out_channels pixels, reassembled
    (B, out_channels, H, W)

At the defaults (256 crop, patch 16, tubelet 2, T=6) that is 3 x 256 = 768
tokens, which full attention handles comfortably.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TubeletEmbed(nn.Module):
    """
    Cut the clip into non-overlapping tubelets and project each to one token.

    A Conv3d whose kernel equals its stride is exactly "flatten each chunk and
    apply a shared linear layer" -- the standard ViT patch embedding, extended
    over time. tubelet=1 gives one token per frame per patch.
    """
    def __init__(self, in_channels, embed_dim, patch_size, tubelet):
        super(TubeletEmbed, self).__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=(tubelet, patch_size, patch_size),
                              stride=(tubelet, patch_size, patch_size))

    def forward(self, x):
        # (B, T, C, H, W) -> (B, C, T, H, W), the layout Conv3d expects
        x = self.proj(x.permute(0, 2, 1, 3, 4))   # (B, D, T', H', W')
        B, D, Tp, Hp, Wp = x.shape
        # (B, T', H'*W', D): one row of spatial tokens per time slot
        return x.permute(0, 2, 3, 4, 1).reshape(B, Tp, Hp * Wp, D)


class Attention(nn.Module):
    """
    Multi-head self-attention. scaled_dot_product_attention picks a fused
    (flash / memory-efficient) kernel when one is available, so the full
    tokens x tokens matrix is never held in memory at once.
    """
    def __init__(self, dim, num_heads, drop=0.):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"embed_dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = drop
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, D = x.shape
        # (B, N, 3D) -> 3 x (B, heads, N, D/heads)
        q, k, v = (self.qkv(x)
                   .reshape(B, N, 3, self.num_heads, D // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
        x = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.drop if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    """The per-token feed-forward half of a block: Linear up, GELU, Linear down."""
    def __init__(self, dim, hidden_dim, drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """
    Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x)).

    Normalising before each branch rather than after keeps the residual path an
    unobstructed identity, which is what lets a deep stack train without warmup
    tricks. LayerNorm, not BatchNorm: it normalises each token on its own, so it
    does not depend on batch size.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    ViT predicting one frame of out_channels from T input frames of C channels.

    Constructor takes shape_in like SimVP, since the position embeddings are
    sized from it: the model only accepts the crop size and T it was built for.
    """
    def __init__(self, shape_in, patch_size=16, tubelet=2, embed_dim=384, depth=8,
                 num_heads=6, mlp_ratio=4., drop=0., out_channels=None):
        super(ViT, self).__init__()
        T, C, H, W = shape_in
        if H % patch_size or W % patch_size:
            raise ValueError(f"crop {H}x{W} is not a multiple of patch_size {patch_size}")
        if T % tubelet:
            raise ValueError(f"T={T} is not a multiple of tubelet {tubelet}")
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.Tp = T // tubelet
        # As in SimVP: the input is wider than the target by design, so the head
        # only emits the channels that are scored.
        self.out_channels = C if out_channels is None else out_channels

        self.embed = TubeletEmbed(C, embed_dim, patch_size, tubelet)
        # Factorised position embeddings: one per spatial patch plus one per time
        # slot, added together. Fewer parameters than one per token, and the
        # spatial table is shared by every time slot.
        self.pos_space = nn.Parameter(torch.zeros(1, 1, self.Hp * self.Wp, embed_dim))
        self.pos_time = nn.Parameter(torch.zeros(1, self.Tp, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Each spatial patch has Tp tokens after attention, one per time slot.
        # Concatenating and projecting them lets the model weigh the recent slot
        # more than the old ones, rather than averaging them equally.
        self.fuse = nn.Linear(self.Tp * embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, patch_size * patch_size * self.out_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_space, std=0.02)
        nn.init.trunc_normal_(self.pos_time, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # A zero head outputs all zeros at step 0. Under ResidualWrapper that is
        # exactly the persistence forecast, so training starts from the baseline
        # instead of from noise added to it.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.shape[0]
        z = self.embed(x)                                  # (B, T', N, D)
        z = z + self.pos_space + self.pos_time
        z = self.pos_drop(z.flatten(1, 2))                 # (B, T'*N, D)

        for block in self.blocks:
            z = block(z)
        z = self.norm(z)

        N, D = self.Hp * self.Wp, z.shape[-1]
        # (B, T', N, D) -> (B, N, T'*D): every time slot of one patch side by side
        z = z.reshape(B, self.Tp, N, D).permute(0, 2, 1, 3).reshape(B, N, self.Tp * D)
        z = self.head(self.fuse(z))                        # (B, N, p*p*out)

        # Unpatchify: (B, H', W', p, p, out) -> (B, out, H'*p, W'*p)
        p = self.patch_size
        z = z.reshape(B, self.Hp, self.Wp, p, p, self.out_channels)
        z = z.permute(0, 5, 1, 3, 2, 4)
        return z.reshape(B, self.out_channels, self.Hp * p, self.Wp * p)
