import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell that processes one timestep at a time.
    Think of this as a standard LSTM cell, but with Convolutional layers instead of Linear layers.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # One big convolution to compute all 4 gates (input, forget, output, cell) at once
        # Input: [Batch, input_dim + hidden_dim, H, W]
        # Output: [Batch, 4 * hidden_dim, H, W]
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and previous hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Apply the convolution
        combined_conv = self.conv(combined)
        
        # Split the output into the 4 gate components
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Sigmoid activations for gates, Tanh for the candidate cell state
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Update cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):
    """
    The full ConvLSTM model that iterates through the sequence of T frames.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Create a list of ConvLSTM cells for each layer
        cells = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cells.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, bias=True))
        
        self.cell_list = nn.ModuleList(cells)
        
        # Final 1x1 convolution to convert the last hidden state back to 1 channel (the prediction)
        self.conv_last = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Time, Channels, Height, Width)
        b, t, _, h, w = x.size()
        
        # Initialize hidden and cell states with zeros
        hidden_states = [torch.zeros(b, self.hidden_dim, h, w).to(x.device) for _ in range(self.num_layers)]
        cell_states = [torch.zeros(b, self.hidden_dim, h, w).to(x.device) for _ in range(self.num_layers)]

        # Iterate through each frame in the sequence
        for seq_idx in range(t):
            cur_input = x[:, seq_idx, :, :, :]
            # Pass through each ConvLSTM layer
            for layer_idx in range(self.num_layers):
                h_next, c_next = self.cell_list[layer_idx](cur_input, (hidden_states[layer_idx], cell_states[layer_idx]))
                hidden_states[layer_idx] = h_next
                cell_states[layer_idx] = c_next
                cur_input = h_next # The output of this layer is the input for the next layer

        # The final prediction is generated from the last hidden state of the last layer
        return self.conv_last(hidden_states[-1])
