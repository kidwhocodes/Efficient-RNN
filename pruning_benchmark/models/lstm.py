"""LSTM-based recurrent model compatible with the CTRNN API."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """
    Wrapper around nn.LSTM exposing the same interface as CTRNN/GRUNet.

    Inputs are (T, B, I) and outputs follow CTRNN.forward().
    """

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, *, bias: bool = True):
        super().__init__()
        self.I = input_dim
        self.H = hidden_size
        self.O = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_size, bias=bias)
        self.readout_layer = nn.Linear(hidden_size, output_dim, bias=bias)
        # expose pseudo-CTRNN attributes for metric/pruning compatibility
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.hidden_layer.weight.zero_()
        self.alpha = 1.0

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Tensor of shape (T, B, input_dim)
        Returns:
            logits: (T, B, output_dim)
            hidden_seq: (T, B, hidden_size)
        """
        hidden_seq, _ = self.lstm(inputs)
        logits = self.readout_layer(hidden_seq)
        return logits, hidden_seq

    def forward_sequence(self, x: torch.Tensor):
        logits, _ = self.forward(x)
        return logits

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str = "cpu"):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)


__all__ = ["LSTMNet"]
