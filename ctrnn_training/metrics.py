import torch
from .core import CTRNN
import numpy as np 

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
def spectral_radius(W: torch.Tensor) -> float:
    W = W.detach().to("cpu", dtype=torch.float32)
    W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)  # NEW
    try:
        eigvals = np.linalg.eigvals(W.numpy())
        return float(np.max(np.abs(eigvals)))
    except Exception:
        return float("nan")

@torch.no_grad()
def ctrnn_stability_proxy(model: CTRNN):
    W = model.hidden_layer.weight.detach()
    rho = spectral_radius(W)
    return model.alpha * rho

@torch.no_grad()
def layer_sparsities(model: CTRNN):
    """Sparsity per layer (fraction of zeros)."""
    layers = {
        "input": model.input_layer.weight,
        "recurrent": model.hidden_layer.weight,
        "readout": model.readout_layer.weight,
    }
    out = {}
    for name, W in layers.items():
        nz = int((W != 0).sum().item())
        tot = W.numel()
        out[name] = 1.0 - (nz / tot)
    return out

@torch.no_grad()
def neuron_keep_fraction(model: CTRNN):
    """Fraction of hidden neurons effectively kept (not isolated)."""
    W = model.hidden_layer.weight.detach()
    row_zero = (W.abs().sum(dim=1) == 0)
    col_zero = (W.abs().sum(dim=0) == 0)
    removed = int((row_zero & col_zero).sum().item())
    H = W.size(0)
    kept = H - removed
    return kept / max(1, H)


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
