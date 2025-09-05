import torch
import torch.nn as nn
import torch.nn.utils.prune as P
from .core import CTRNN

# ---------- helpers ----------
def _enforce_no_self_connections(model: CTRNN):
    with torch.no_grad():
        if getattr(model, "no_self_connections", False):
            model.hidden_layer.weight.data.fill_diagonal_(0.0)

def _consolidate_if_pruned(layer: nn.Module):
    try:
        P.remove(layer, "weight")
    except Exception:
        pass

# ---------- strategies ----------
def prune_random_recurrent(model: CTRNN, amount: float):
    P.random_unstructured(model.hidden_layer, name="weight", amount=amount)

def prune_l1_recurrent(model: CTRNN, amount: float):
    P.l1_unstructured(model.hidden_layer, name="weight", amount=amount)

def prune_structured_recurrent(model: CTRNN, amount: float, dim: int = 0):
    """
    Structured pruning (entire rows or cols). Uses ln_structured for compatibility.
    dim=0 -> prune rows (outgoing fan) = delete neurons' outputs
    dim=1 -> prune cols (incoming fan) = delete neurons' inputs
    """
    P.ln_structured(model.hidden_layer, name="weight", amount=amount, n=1, dim=dim)

def prune_global_unstructured(model: CTRNN, amount: float,
                              include_readout: bool = False, include_input: bool = False):
    params = [(model.hidden_layer, "weight")]
    if include_readout:
        params.append((model.readout_layer, "weight"))
    if include_input:
        params.append((model.input_layer, "weight"))
    P.global_unstructured(params, pruning_method=P.L1Unstructured, amount=amount)

def iterative_magnitude_pruning(model: CTRNN, opt, data, device, criterion,
                                rounds=5, prune_each=0.2, ft_steps=100, last_only=True):
    from .train_eval import train_epoch
    for _ in range(rounds):
        P.l1_unstructured(model.hidden_layer, name="weight", amount=prune_each)
        _consolidate_if_pruned(model.hidden_layer)
        _enforce_no_self_connections(model)
        _ = train_epoch(model, data, device, opt, criterion, steps=ft_steps, last_only=last_only)
    _consolidate_if_pruned(model.hidden_layer)
    _enforce_no_self_connections(model)

def movement_scores(model: CTRNN, data, device, criterion, batches=10, last_only=True):
    model.train()  # must be in train mode so grads are tracked
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    scores = None
    with torch.enable_grad():  # force autograd to track
        for _ in range(batches):
            x, y = data.sample_batch()
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits[-1], y[-1]) if last_only else \
                   criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward()
        W = model.hidden_layer.weight
        scores = (W.grad * W).abs().detach().clone()
    return scores



def prune_movement_recurrent(model: CTRNN, score: torch.Tensor, amount: float):
    W = model.hidden_layer.weight.data
    k = int(W.numel() * amount)
    thresh = torch.topk(score.flatten(), k, largest=False).values.max()
    mask = (score > thresh).to(W.dtype)
    model.hidden_layer.weight.data = W * mask
    _enforce_no_self_connections(model)
