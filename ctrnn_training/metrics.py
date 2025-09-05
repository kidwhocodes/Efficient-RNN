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
