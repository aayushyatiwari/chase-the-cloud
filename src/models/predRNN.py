"""
PredRNN: Recurrent Neural Networks for Predictive Learning using
Spatiotemporal LSTMs (Wang et al., NeurIPS 2017).
github: https://github.com/thuml/predrnn-pytorch

The idea in one line: a stacked ConvLSTM carries memory *forward in time*, one
independent stream per layer, so what the top layer notices never reaches the
bottom one. PredRNN adds a second memory M that zigzags -- up through every
layer within a timestep, then across to the next timestep from the top layer
back down to the bottom. The cell state C stays the layer's own long-term
memory; M is the shared short-term one that lets a detail spotted deep in the
stack inform the first layer's next step.

    C:  layer-wise, horizontal      M:  zigzag, bottom -> top -> next t bottom

Only the v1 ST-LSTM is implemented. PredRNN-V2 adds memory decoupling (a loss
term penalising C and M for learning the same thing) and reverse scheduled
sampling; both need hooks in the training loop -- an extra loss term and a
per-epoch schedule -- rather than only a model, so they are left out. Adding
them means touching src/engine.py, not just this file.

Like ConvLSTM here, this predicts a single next frame rather than rolling out a
sequence: the readout runs once, on the last timestep's top hidden state.
"""
import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):
    """
    One ST-LSTM cell: a ConvLSTM cell plus a second, parallel set of gates for
    the zigzag memory M, with the two memories fused at the output.

    Three convolutions feed the gates, each done as one wide conv and split:
      conv_x -> 7 chunks: i, f, g for C;  i', f', g' for M;  o's input term
      conv_h -> 4 chunks: i, f, g for C;  o's hidden term
      conv_m -> 3 chunks: i', f', g' for M

    C updates exactly as in a ConvLSTM. M updates the same way from its own
    gates. The output gate then sees both, and h comes from a 1x1 conv over
    [C, M] concatenated -- which is the only place the two memories mix.
    """
    def __init__(self, in_channel, num_hidden, kernel_size=3, stride=1, bias=True):
        super(SpatioTemporalLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        padding = kernel_size // 2

        self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        # The output gate's view of the fused memory, and the fusion itself.
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=bias)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        # Standard LSTM update on the layer's own memory C.
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t

        # The same update on the zigzag memory M, from its own gates.
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class PredRNN(nn.Module):
    """
    A stack of ST-LSTM cells over the input sequence, with a 1x1 readout.

    Constructor matches ConvLSTM's (input_dim, hidden_dim, kernel_size,
    num_layers) plus out_channels, so the two are swapped by config alone.
    hidden_dim may be an int (same width everywhere) or a list, one per layer,
    which is how the paper sizes it -- e.g. [128, 64, 64, 64].
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=4, out_channels=1):
        super(PredRNN, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        assert len(hidden_dim) == num_layers, \
            f"hidden_dim has {len(hidden_dim)} entries but num_layers is {num_layers}"

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        cells = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cells.append(SpatioTemporalLSTMCell(cur_input_dim, hidden_dim[i], kernel_size))
        self.cell_list = nn.ModuleList(cells)

        self.conv_last = nn.Conv2d(hidden_dim[-1], out_channels, kernel_size=1, bias=False)

        # M is handed from the top layer down to the bottom one between
        # timesteps, so every layer must carry it at the same width.
        self.m_dim = hidden_dim[0]
        if any(d != self.m_dim for d in hidden_dim):
            # A varying stack needs M projected between widths; rather than
            # inventing a projection the paper does not have, require it uniform.
            raise ValueError(
                "PredRNN's zigzag memory M is passed between all layers, so every "
                f"entry of hidden_dim must match; got {hidden_dim}.")

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, _, h, w = x.size()
        device = x.device

        hidden_states = [torch.zeros(b, d, h, w, device=device) for d in self.hidden_dim]
        cell_states = [torch.zeros(b, d, h, w, device=device) for d in self.hidden_dim]
        # One M for the whole stack -- this is what makes it PredRNN rather than
        # a stacked ConvLSTM. It is never reset between layers, only between runs.
        memory = torch.zeros(b, self.m_dim, h, w, device=device)

        for seq_idx in range(t):
            cur_input = x[:, seq_idx]
            # Within a timestep M travels upward, layer 0 -> layer L-1; it then
            # carries over to the next timestep's layer 0. That is the zigzag.
            for layer_idx in range(self.num_layers):
                h_next, c_next, memory = self.cell_list[layer_idx](
                    cur_input,
                    hidden_states[layer_idx],
                    cell_states[layer_idx],
                    memory,
                )
                hidden_states[layer_idx] = h_next
                cell_states[layer_idx] = c_next
                cur_input = h_next

        return self.conv_last(hidden_states[-1])
