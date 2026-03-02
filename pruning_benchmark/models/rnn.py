"""Vanilla RNN model compatible with the CTRNN API."""

from __future__ import annotations

import torch
import torch.nn as nn


class _RNNWeightProxy(nn.Module):
    def __init__(self, rnn: nn.RNN, attr: str):
        super().__init__()
        # Avoid registering the full RNN as a submodule to prevent duplicate parameters.
        self.__dict__["_rnn"] = rnn
        self._attr = attr
        param = getattr(rnn, attr)
        # share the same Parameter object so pruning mutates the true RNN weights.
        self.weight = param


class RNNNet(nn.Module):
    """
    Wrapper around nn.RNN exposing the same interface as CTRNN/GRUNet/LSTMNet.

    Inputs are (T, B, I) and outputs follow CTRNN.forward().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        *,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ):
        super().__init__()
        self.I = input_dim
        self.H = hidden_size
        self.O = output_dim

        self.rnn = nn.RNN(input_dim, hidden_size, bias=bias, nonlinearity=nonlinearity)
        self.readout_layer = nn.Linear(hidden_size, output_dim, bias=bias)
        self.input_layer = _RNNWeightProxy(self.rnn, "weight_ih_l0")
        self.hidden_layer = _RNNWeightProxy(self.rnn, "weight_hh_l0")
        self.alpha = 1.0

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Tensor of shape (T, B, input_dim)
        Returns:
            logits: (T, B, output_dim)
            hidden_seq: (T, B, hidden_size)
        """
        hidden_seq, _ = self.rnn(inputs)
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

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        filtered = {}
        for key, value in state.items():
            if key.startswith("input_layer._rnn") or key.startswith("hidden_layer._rnn"):
                continue
            filtered[key] = value
        return filtered

    def load_state_dict(self, state_dict, strict: bool = True):
        filtered = {}
        for key, value in state_dict.items():
            if key.startswith("input_layer") or key.startswith("hidden_layer"):
                continue
            filtered[key] = value
        return super().load_state_dict(filtered, strict=False)


__all__ = ["RNNNet"]
