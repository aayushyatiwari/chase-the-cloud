import torch
import torch.nn as nn


class BoundedOutput(nn.Module):
    """
    Squash a model's output into [0,1], the range frames are scaled to.

    ResidualWrapper bounds its own output. Without the same bound on the plain
    path, a residual run and a bare run would be trained against different
    objectives -- one bounded, one not -- and could not be compared fairly. The
    bare models end in a 1x1 conv with no activation, so nothing else keeps
    them in range.

    Sigmoid, not clamp. A fresh ConvLSTM's output starts near zero and is
    often entirely negative -- 5 of 8 seeds tested -- so a clamp would flatten
    every pixel onto the boundary, where its gradient is zero, and the model
    would never train at all. Sigmoid is bounded everywhere and differentiable
    everywhere, and starts predictions near 0.5 instead of 0.

    ResidualWrapper can still use a clamp because it adds the last frame first,
    which lands the prediction inside the range to begin with.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


class ResidualWrapper(nn.Module):
    """
    Turn any model into one that predicts the *change* from the last input
    frame, instead of the whole next frame.

        output = last_input_frame + model(inputs)

    Persistence (just repeating the last frame) is already a strong forecast
    over 30 minutes, so a plain model spends most of its capacity relearning
    "the sky looks like it did 30 minutes ago" before it can improve on that.
    Here the model starts at persistence for free -- predicting all zeros gives
    exactly the persistence forecast -- and everything it learns goes into the
    correction. The change is also much smaller and more centred than the frame
    itself, which is usually easier to fit.

    Only the predicted channels are carried over. With extra input channels
    (e.g. water vapour) the input is wider than the target, and the predicted
    channel (TIR1) comes first.
    """

    def __init__(self, model, out_channels=1):
        super().__init__()
        self.model = model
        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, T, C, H, W)  ->  last frame: (B, out_channels, H, W)
        pred = x[:, -1, :self.out_channels] + self.model(x)
        # Frames are scaled to [0,1], so a value outside that is not a real
        # brightness temperature. Without this the model can lower its training
        # loss by drifting out of range, where evaluation just clips it away.
        return pred.clamp(0.0, 1.0)
