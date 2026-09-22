"""
PhyDNet: Disentangling Physical Dynamics from Unknown Factors for Unsupervised
Video Prediction (Le Guen & Thome, CVPR 2020).
github: https://github.com/vincent-leguen/PhyDNet

Two recurrent branches share one latent space:

  * PhyCell  -- the physical branch. Its update is a discretised PDE: predict
                the next latent by applying a bank of spatial differential
                operators to the current one, then correct that prediction with
                the new observation, Kalman-filter style.
  * ConvLSTM -- the residual branch, for everything the PDE cannot express
                (appearance, illumination, whatever else the data does).

Their latents are summed and decoded, so neither has to explain the whole
signal alone. For cloud advection the split is the point: motion is close to an
advection-diffusion process, and the physical branch can represent that with far
fewer parameters than a ConvLSTM needs to learn it from scratch.

What makes PhyCell physical rather than just another conv is the moment
regulariser: each of its F_hidden_dim filters is constrained so its moment
matrix matches one differential operator -- filter k approximating d^(i+j)/dx^i dy^j.
That constraint is a LOSS TERM, not a layer. It is exposed here as
`model.moment_loss()`; nothing in src/engine.py adds it yet, so training this
model as-is gives a PhyCell with unconstrained filters (still a working model,
just no longer guaranteed to be a PDE). To turn it on, add

    loss = criterion(pred, target) + model.moment_loss()

in Trainer.train_one_epoch. See `moment_loss` below for the weighting.

Like the other models here it predicts a single next frame: the branches are
run over the T input frames and the decoder fires once, at the end.
"""
import torch
import torch.nn as nn

from .convlstm import ConvLSTMCell


# ---------------------------------------------------------------- moments ---

class K2M(nn.Module):
    """
    Kernel -> moment matrix, for square 2-D kernels.

    The moment of order (i, j) of a kernel K is

        m[i, j] = sum_{u,v} K[u, v] * (u - c)^i * (v - c)^j / (i! j!)

    which is exactly M @ K @ M.T for the matrix M[i, u] = (u - c)^i / i!. A
    kernel whose moment matrix is 1 at (i, j) and 0 elsewhere applies, to first
    order, the derivative d^(i+j)/dx^i dy^j -- that is the Taylor expansion read
    backwards. Constraining moments therefore constrains what operator the
    filter *is*, without fixing its weights.

    The original implements this for arbitrary rank via repeated tensordot;
    convolution kernels here are always 2-D, so the two matmuls below are the
    same thing, written plainly.
    """
    def __init__(self, shape):
        super(K2M, self).__init__()
        h, w = shape
        assert h == w, f"K2M expects a square kernel, got {shape}"
        self.register_buffer('M', self._moment_matrix(h))

    @staticmethod
    def _moment_matrix(size):
        import math
        c = (size - 1) // 2
        M = torch.zeros(size, size, dtype=torch.float64)
        for i in range(size):
            for u in range(size):
                M[i, u] = float(u - c) ** i / math.factorial(i)
        return M

    def forward(self, k):
        # k: (..., h, w) -> m: (..., h, w)
        M = self.M.to(k.dtype)
        return M @ k @ M.t()


# ---------------------------------------------------------------- PhyCell ---

class PhyCell_Cell(nn.Module):
    """
    One PhyCell step, in two halves.

    Prediction:   h~ = h + F(h)           -- Euler step of dh/dt = F(h), where F
                                             is the constrained operator bank
    Correction:   h' = h~ + K * (x - h~)  -- move towards the observation by a
                                             learned, input-dependent gate K

    Setting K to 0 gives a pure simulation; setting it to 1 throws the physics
    away and copies the observation. The gate learns the trade-off per pixel.
    """
    def __init__(self, input_dim, F_hidden_dim, kernel_size=(7, 7), bias=True):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # conv1's kernels are the ones the moment regulariser constrains: one
        # differential operator each, which is why F_hidden_dim is usually
        # kernel_size[0] * kernel_size[1] (49 for 7x7) -- all the operators up
        # to order 6 in each direction. conv2 is the free linear combination of
        # them, i.e. the PDE's coefficients.
        self.F = nn.Sequential(
            nn.Conv2d(input_dim, F_hidden_dim, kernel_size, stride=1, padding=padding),
            nn.GroupNorm(7, F_hidden_dim),
            nn.Conv2d(F_hidden_dim, input_dim, kernel_size=(1, 1), stride=1, padding=0),
        )

        self.convgate = nn.Conv2d(2 * input_dim, input_dim, kernel_size=(3, 3),
                                  padding=(1, 1), bias=bias)

    def forward(self, x, hidden):
        hidden_tilde = hidden + self.F(hidden)                  # prediction
        combined = torch.cat([x, hidden_tilde], dim=1)
        K = torch.sigmoid(self.convgate(combined))              # Kalman-ish gain
        return hidden_tilde + K * (x - hidden_tilde)            # correction


class PhyCell(nn.Module):
    """A stack of PhyCell_Cells, each with its own persistent hidden state."""
    def __init__(self, input_dim, F_hidden_dims, n_layers, kernel_size=(7, 7)):
        super(PhyCell, self).__init__()
        if isinstance(F_hidden_dims, int):
            F_hidden_dims = [F_hidden_dims] * n_layers
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.cell_list = nn.ModuleList([
            PhyCell_Cell(input_dim, F_hidden_dims[i], kernel_size)
            for i in range(n_layers)
        ])

    def forward(self, x, hidden):
        # Every layer works in the same latent width, so a layer's output is
        # directly the next layer's observation.
        cur = x
        for i, cell in enumerate(self.cell_list):
            hidden[i] = cell(cur, hidden[i])
            cur = hidden[i]
        return cur, hidden

    def init_hidden(self, batch, h, w, device):
        return [torch.zeros(batch, self.input_dim, h, w, device=device)
                for _ in range(self.n_layers)]


class ConvLSTMBranch(nn.Module):
    """
    The residual branch: a plain stacked ConvLSTM over the same latent, reusing
    the cell from convlstm.py so the two models share one implementation.
    """
    def __init__(self, input_dim, hidden_dims, n_layers, kernel_size=3):
        super(ConvLSTMBranch, self).__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * n_layers
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        cells = []
        for i in range(n_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cells.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size, bias=True))
        self.cell_list = nn.ModuleList(cells)

    def forward(self, x, hidden, cell):
        cur = x
        for i, c in enumerate(self.cell_list):
            hidden[i], cell[i] = c(cur, (hidden[i], cell[i]))
            cur = hidden[i]
        return cur, hidden, cell

    def init_hidden(self, batch, h, w, device):
        hidden = [torch.zeros(batch, d, h, w, device=device) for d in self.hidden_dims]
        cell = [torch.zeros(batch, d, h, w, device=device) for d in self.hidden_dims]
        return hidden, cell


# ------------------------------------------------------- encoder / decoder ---

def dcgan_conv(nin, nout, stride=2):
    return nn.Sequential(
        nn.Conv2d(nin, nout, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(4, nout),
        nn.LeakyReLU(0.2, inplace=True),
    )


def dcgan_upconv(nin, nout, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(nin, nout, kernel_size=3, stride=stride, padding=1,
                           output_padding=stride // 2),
        nn.GroupNorm(4, nout),
        nn.LeakyReLU(0.2, inplace=True),
    )


class EncoderE(nn.Module):
    """Frames -> latent, downsampled by 4. Both branches recur in this space."""
    def __init__(self, nc=1, nf=32, latent_dim=64):
        super(EncoderE, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=2)
        self.c2 = dcgan_conv(nf, nf, stride=1)
        self.c3 = dcgan_conv(nf, latent_dim, stride=2)

    def forward(self, x):
        return self.c3(self.c2(self.c1(x)))


class DecoderD(nn.Module):
    """Latent -> frame, the mirror of EncoderE. No output activation, so the
    prediction is unbounded and stays comparable with the other models here."""
    def __init__(self, nc=1, nf=32, latent_dim=64):
        super(DecoderD, self).__init__()
        self.upc1 = dcgan_upconv(latent_dim, nf, stride=2)
        self.upc2 = dcgan_conv(nf, nf, stride=1)
        self.upc3 = nn.ConvTranspose2d(nf, nc, kernel_size=3, stride=2,
                                       padding=1, output_padding=1)

    def forward(self, x):
        return self.upc3(self.upc2(self.upc1(x)))


# ----------------------------------------------------------------- PhyDNet ---

class PhyDNet(nn.Module):
    """
    Encoder -> (PhyCell + ConvLSTM, summed) -> decoder, predicting one frame.

    The per-branch 1x1 "specific" convs let each branch keep its own view of the
    shared latent, as in the paper, without duplicating the expensive encoder.
    """
    def __init__(self, input_dim, out_channels=1, nf=32, latent_dim=64,
                 phy_hidden_dims=49, phy_layers=1, phy_kernel_size=7,
                 conv_hidden_dims=(128, 128, 64), conv_layers=3, conv_kernel_size=3):
        super(PhyDNet, self).__init__()
        self.latent_dim = latent_dim
        self.phy_kernel_size = phy_kernel_size

        self.encoder = EncoderE(nc=input_dim, nf=nf, latent_dim=latent_dim)
        self.encoder_Ep = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
        self.encoder_Er = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)

        self.phycell = PhyCell(latent_dim, phy_hidden_dims, phy_layers,
                               kernel_size=(phy_kernel_size, phy_kernel_size))
        conv_hidden_dims = list(conv_hidden_dims)[:conv_layers]
        self.convcell = ConvLSTMBranch(latent_dim, conv_hidden_dims, conv_layers,
                                       kernel_size=conv_kernel_size)

        self.decoder_Dp = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
        self.decoder_Dr = nn.Conv2d(conv_hidden_dims[-1], latent_dim, kernel_size=1)
        self.decoder = DecoderD(nc=out_channels, nf=nf, latent_dim=latent_dim)

        self.k2m = K2M((phy_kernel_size, phy_kernel_size))
        self.register_buffer('moment_targets', self._moment_targets(phy_kernel_size,
                                                                    self.phycell.cell_list[0].F_hidden_dim))

    @staticmethod
    def _moment_targets(k, n_filters):
        """
        Filter n gets moment matrix e_(i,j): 1 at one position, 0 everywhere
        else, walking the kernel in raster order. With n_filters == k*k that
        assigns every operator from the identity (0,0) up to d^(k-1)/dx^(k-1)
        exactly once; with fewer filters, the first n_filters in that order.
        """
        targets = torch.zeros(n_filters, k, k)
        for n in range(n_filters):
            i, j = divmod(n, k)
            if i < k:
                targets[n, i, j] = 1.0
        return targets

    def moment_loss(self, weight=1.0):
        """
        The PDE constraint, as a scalar to add to the training loss.

        For each input channel of PhyCell's first conv, take that channel's
        F_hidden_dim filters, turn them into moment matrices, and score them
        against the targets. The paper weights this at 1 alongside an MSE
        image loss; it is a hard constraint in spirit but a soft one in practice.

        Returns a tensor on the model's device, so it can be added directly.
        """
        conv1 = self.phycell.cell_list[0].F[0]
        loss = conv1.weight.new_zeros(())
        for b in range(conv1.weight.shape[1]):          # over input channels
            filters = conv1.weight[:, b]                 # (F_hidden_dim, k, k)
            m = self.k2m(filters)
            loss = loss + nn.functional.mse_loss(m, self.moment_targets)
        return weight * loss

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.size()
        device = x.device
        # The encoder downsamples by 4, which is where the recurrence happens.
        hl, wl = h // 4, w // 4

        phy_h = self.phycell.init_hidden(b, hl, wl, device)
        conv_h, conv_c = self.convcell.init_hidden(b, hl, wl, device)

        out_phys = out_conv = None
        for seq_idx in range(t):
            latent = self.encoder(x[:, seq_idx])
            out_phys, phy_h = self.phycell(self.encoder_Ep(latent), phy_h)
            out_conv, conv_h, conv_c = self.convcell(self.encoder_Er(latent),
                                                     conv_h, conv_c)

        # One decode, at the end: the two branches' latents are summed, which is
        # what forces them to split the signal rather than each model all of it.
        fused = self.decoder_Dp(out_phys) + self.decoder_Dr(out_conv)
        return self.decoder(fused)
