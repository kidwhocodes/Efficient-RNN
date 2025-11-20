"""Simple GRU-based recurrent model compatible with the CTRNN API."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUNet(nn.Module):
    """
    Lightweight wrapper around nn.GRU exposing the same methods as CTRNN.

    Inputs are assumed to be shaped (T, B, I) and outputs mirror CTRNN.forward().
    """

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, *, bias: bool = True):
        super().__init__()
        self.I = input_dim
        self.H = hidden_size
        self.O = output_dim

        self.gru = nn.GRU(input_dim, hidden_size, bias=bias)
        self.readout_layer = nn.Linear(hidden_size, output_dim, bias=bias)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Tensor of shape (T, B, input_dim)
        Returns:
            logits: (T, B, output_dim)
            hidden_seq: (T, B, hidden_size)
        """
        hidden_seq, _ = self.gru(inputs)
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


__all__ = ["GRUNet"]
