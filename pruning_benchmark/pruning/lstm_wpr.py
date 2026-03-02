"""Utilities for LSTM Web-PageRank structural pruning."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def lstm_wpr_prune(
    layer: nn.LSTM,
    amount: float,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Apply WPR structural pruning to an LSTM layer."""
    hidden = layer.hidden_size
    drop = int(round(hidden * float(amount)))
    if drop <= 0:
        return {"lstm_keep_fraction": 1.0}

    device = layer.weight_hh_l0.device
    dtype = layer.weight_hh_l0.dtype
    W_hh = layer.weight_hh_l0.detach().clone()
    W_hh_T = W_hh.transpose(0, 1)  # (hidden, 4*hidden)
    gates = [W_hh_T[:, k * hidden : (k + 1) * hidden] for k in range(4)]
    adjacency = sum(torch.abs(G) for G in gates)

    degree = torch.sum(adjacency, dim=1, keepdim=True) + 1e-10
    M_T = (adjacency / degree).transpose(0, 1)

    scores = torch.full((hidden, 1), 1.0 / hidden, device=device, dtype=dtype)
    personalization = scores.clone()
    for _ in range(max_iter):
        prev = scores.clone()
        scores = alpha * (M_T @ scores) + (1 - alpha) * personalization
        if torch.norm(scores - prev) < tol:
            break

    keep = torch.topk(scores.flatten(), hidden - drop, largest=True).indices
    keep_mask = torch.zeros(hidden, dtype=torch.bool, device=device)
    keep_mask[keep] = True

    def _zero_rows(mat: torch.Tensor):
        if not mat.numel():
            return
        for gate in range(4):
            rows = slice(gate * hidden, (gate + 1) * hidden)
            gate_rows = mat[rows]
            gate_rows[~keep_mask] = 0

    def _zero_rows_and_cols(mat: torch.Tensor):
        if not mat.numel():
            return
        for gate in range(4):
            rows = slice(gate * hidden, (gate + 1) * hidden)
            gate_rows = mat[rows]
            gate_rows[~keep_mask] = 0
            gate_rows[:, ~keep_mask] = 0

    _zero_rows(layer.weight_ih_l0.data)
    _zero_rows_and_cols(layer.weight_hh_l0.data)
    if layer.bias:
        _zero_rows(layer.bias_ih_l0.data.unsqueeze(1))
        _zero_rows(layer.bias_hh_l0.data.unsqueeze(1))

    return {"lstm_keep_fraction": float(keep_mask.float().mean().item())}


__all__ = ["lstm_wpr_prune"]
