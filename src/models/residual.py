import torch.nn as nn


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
