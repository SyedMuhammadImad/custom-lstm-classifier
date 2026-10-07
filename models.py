"""project implementation retained from author/assisted source; see ATTRIBUTION.md."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
NUM_CLASSES=4
class MyLSTMCell(nn.Module):
    """
    Single LSTM timestep.
    Uses one combined Linear: [h_{t-1}; x_t] -> 4*H raw activations.
    Split into f, i, g, o of size H each.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        if any(not isinstance(v, int) or v < 1 for v in (input_size, hidden_size)): raise ValueError("Invalid recurrent dimensions")
        self.hidden_size = hidden_size
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """x_t (N,I), h_prev (N,H), c_prev (N,H) -> h_t (N,H), c_t (N,H)."""
        combined = torch.cat([h_prev, x_t], dim=1)
        f_raw, i_raw, g_raw, o_raw = self.gates(combined).chunk(4, dim=1)
        f_t = torch.sigmoid(f_raw)
        i_t = torch.sigmoid(i_raw)
        g_t = torch.tanh(g_raw)
        o_t = torch.sigmoid(o_raw)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return (h_t, c_t)

class MyLSTM(nn.Module):
    """
    Stacked LSTM built from MyLSTMCell.
    Input (N, T, input_size) -> final h of top layer (N, hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        if any(not isinstance(v, int) or v < 1 for v in (input_size, hidden_size)): raise ValueError("Invalid recurrent dimensions")
        self.hidden_size = hidden_size
        if not isinstance(num_layers, int) or num_layers < 1: raise ValueError("Invalid layer count")
        self.num_layers = num_layers
        self.cells = nn.ModuleList([MyLSTMCell(input_size if layer == 0 else hidden_size, hidden_size) for layer in range(num_layers)])

    def forward(self, x):
        if x.ndim != 3 or min(x.shape) < 1 or x.shape[2] != self.cells[0].gates.in_features-self.hidden_size:
            raise ValueError("Expected a nonempty sequence with matching features")
        if not torch.isfinite(x).all(): raise ValueError("Non-finite sequence")
        N, T, _ = x.shape
        H = self.hidden_size
        h = [torch.zeros(N, H, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        c = [torch.zeros(N, H, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        for t in range(T):
            layer_input = x[:, t, :]
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(layer_input, h[layer], c[layer])
                layer_input = h[layer]
        return h[-1]

class LSTMClassifier(nn.Module):

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=NUM_CLASSES, dropout=0.35):
        super().__init__()
        self.lstm = MyLSTM(input_size, hidden_size, num_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 96), nn.ReLU(), nn.Dropout(0.2), nn.Linear(96, num_classes))

    def forward(self, x):
        return self.head(self.lstm(x))
