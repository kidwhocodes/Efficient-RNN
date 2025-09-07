import torch
from .core import CTRNN

def count_nonzero_and_total(t: torch.Tensor):
    nz = int((t != 0).sum().item())
    tot = t.numel()
    return nz, tot

@torch.no_grad()
def recurrent_sparsity(model: CTRNN):
    W = model.hidden_layer.weight
    nz, tot = count_nonzero_and_total(W)
    return 1.0 - (nz / tot)

@torch.no_grad()
def spectral_radius(W: torch.Tensor):
    ev = torch.linalg.eigvals(W.float().cpu())
    return float(ev.abs().max().item())

@torch.no_grad()
def ctrnn_stability_proxy(model: CTRNN):
    W = model.hidden_layer.weight.detach()
    rho = spectral_radius(W)
    return model.alpha * rho

@torch.no_grad()
def neuron_pruning_stats(model: CTRNN):
    """
    Returns counts for neuron-level pruning on the recurrent matrix (H x H).

    rows_zero: neurons whose outgoing weights are all zero (row == 0)
    cols_zero: neurons whose incoming weights are all zero (col == 0)
    isolated:  neurons that are both rows_zero AND cols_zero (fully disconnected)
    """
    W = model.hidden_layer.weight.detach()
    row_zero = (W.abs().sum(dim=1) == 0)  # shape [H]
    col_zero = (W.abs().sum(dim=0) == 0)  # shape [H]
    rows_zero = int(row_zero.sum().item())
    cols_zero = int(col_zero.sum().item())
    isolated  = int((row_zero & col_zero).sum().item())
    return {"rows_zero": rows_zero, "cols_zero": cols_zero, "isolated": isolated}
