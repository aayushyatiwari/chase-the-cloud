"""
SIMVP: A Simple Video Prediction Model
github: https://github.com/A4Bio/SimVP/blob/master/
"""
import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    """
    Basic convolutional block : with transpose facility, group normalization and leaky relu activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose = False, act_norm = False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            # output_padding=stride//2 makes this exactly invert a stride-2 encoder
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=stride // 2)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act_norm:
            x = self.norm(x)
            x = self.act(x)
        return x

class ConvSC(nn.Module):
    """
    Convolutional module with custom stride
    """
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm = True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride, padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        return self.conv(x)

class GroupConv2d(nn.Module):
    """
    divide channels into groups and apply convolution to each group separately
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm = False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act_norm:
            x = self.norm(x)
            x = self.activate(x)
        return x

class Inception(nn.Module):
    """
    Inception/translator module from paper 
    task: look at samples from different kernel sizes and combine them to get a better representation of the input
    """
    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups = 8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            # Output width is C_out (this module's actual output), not C_hid --
            # the next Inception in the stack is built expecting C_out channels.
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0 # add all the outputs from different kernel sizes
        for layer in self.layers:
            y += layer(x)
        return y


### ----- main.py from github repo ----- ###


def stride_generator(N, reverse=False):
    """
    Generate list of strides for encoder and decoder
    """
    stride = [1, 2] * 10
    if reverse: stride = stride[::-1]
    return stride[:N]

class Encoder(nn.Module):
    """
    Encoder module 
    task: encode the input sequence into a latent representation
    """
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator (N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1 
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Decoder (nn.Module):
    """
    Decoder module 
    task: decode the latent representation into output sequece
    """
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator (N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            # 2*C_hid in: this layer consumes cat([hid, enc1]) in forward(). Output
            # stays C_hid (not C_out) so GroupNorm(2, ...) inside it has a channel
            # count it can actually divide -- the final projection to C_out happens
            # in self.readout below, a plain conv with no normalization.
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True),
        )

        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(len(self.dec)-1):
            hid = self.dec[i](hid)
        y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        y = self.readout(y)
        return y

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, T_out, hid_S, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        self.T_out = T_out
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, T_out*hid_S, incep_ker= incep_ker, groups=groups)) # final layer outputs T_out*hid_S channels, not channel_hid

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        # C (per-frame channel width) is unchanged; only the frame count differs.
        y = z.reshape(B, self.T_out, C, H, W)
        return y

class SimVP(nn.Module):
    """
    SimVP model, adapted to predict T_out frames (default 1) from T_in frames
    instead of the paper's T_in == T_out video-prediction setup.
    """
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, T_out=1, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        self.T_out = T_out
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, T_out, hid_S, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        _, Cs, Hs, Ws = skip.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)                                  # (B, T_out, C_, H_, W_)
        hid = hid.reshape(B*self.T_out, C_, H_, W_)

        # skip has one entry per INPUT frame (batch B*T); the decoder needs one
        # per OUTPUT frame (batch B*T_out). Use each batch item's most recent
        # T_out input frames' skip features as the closest available match.
        skip = skip.view(B, T, Cs, Hs, Ws)[:, -self.T_out:].reshape(B*self.T_out, Cs, Hs, Ws)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, self.T_out, C, H, W)
        Y = Y.squeeze(1)  # T_out=1: (B, 1, C, H, W) -> (B, C, H, W), matching ConvLSTM
        return Y
