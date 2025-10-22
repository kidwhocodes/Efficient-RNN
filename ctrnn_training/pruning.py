import torch
import torch.nn as nn
import torch.nn.utils.prune as P
from .core import CTRNN
import math
from typing import Optional
import contextlib

# ---------- helpers ----------

@torch.no_grad()
def enforce_constraints(model: CTRNN):
    """Apply Dale and no-self constraints to the underlying weight_orig if present."""
    layer = model.hidden_layer
    W = getattr(layer, "weight_orig", layer.weight)

    # Dale's law (sign-constrained weights)
    if getattr(model, "use_dale", False):
        W.mul_(model.dale_sign).abs_().mul_(model.dale_sign)

    # No self-connections (zero diagonal)
    if getattr(model, "no_self_connections", False):
        W.fill_diagonal_(0.0)
        M = getattr(layer, "weight_mask", None)
        if M is not None:
            M.fill_diagonal_(0.0)


def _global_threshold_from_tensors(tensors, amount: float) -> float:
    """Return a value 'thresh' such that ~`amount` fraction will be pruned (score <= thresh)."""
    flat = torch.cat([t.reshape(-1) for t in tensors if t is not None])
    k_prune = int(round(amount * flat.numel()))
    if k_prune <= 0:
        return float(-1e30)
    if k_prune >= flat.numel():
        return float(+1e30)
    # smallest k_prune elements → pruned
    thresh = torch.topk(flat, k_prune, largest=False).values.max().item()
    return float(thresh)

@torch.no_grad()
def mask_from_scores(model: CTRNN, scores: dict, amount: float,
                     structured: bool = False, global_: bool = True):
    """
    Apply an unstructured mask to the recurrent weights using a dict of per-param scores.
    Only 'hidden_layer.weight' is used (clean + matches your metrics).
    """
    layer = model.hidden_layer
    s = None
    # scores keys come from model.named_parameters() in synflow_scores
    for name, t in scores.items():
        if name.endswith("hidden_layer.weight"):
            s = t
            break
    if s is None:
        # fallback: try attribute
        s = getattr(layer, "weight")
        s = s.new_ones(s.shape)

    thresh = _global_threshold_from_tensors([s], amount)
    keep = (s > thresh).to(dtype=layer.weight.dtype)
    P.custom_from_mask(layer, name="weight", mask=keep)
    enforce_constraints(model)  # keep your diagonal/Dale constraints

@torch.no_grad()
def neuron_mask_from_scores(model: CTRNN, neuron_scores: torch.Tensor, amount: float):
    """
    Expand per-neuron scores [H] into consistent masks for in/rec/out and apply.
    """
    keep = _neuron_keep_mask_from_scores(neuron_scores, amount)
    _apply_neuron_keep_mask(model, keep)
    enforce_constraints(model)

def _consolidate_if_pruned(layer: nn.Module):
    try:
        P.remove(layer, "weight")
    except Exception:
        pass

@torch.no_grad()
def _offdiag(M: torch.Tensor) -> torch.Tensor:
    return M - torch.diag(torch.diag(M))

@torch.no_grad()
def _simulate_covariance(
    A: torch.Tensor,
    sigma: float = 1.0,
    dt: float = 1e-2,
    steps: int = 20_000,
    burnin: int = 2_000,
    seed: Optional[int] = None,
    ) -> torch.Tensor:
    """Euler–Maruyama for dx = A x dt + sigma dW_t to estimate C = E[x x^T]."""
    if seed is not None:
        torch.manual_seed(seed)
    N = A.shape[0]
    x = torch.zeros(N, device=A.device, dtype=A.dtype)
    S = torch.zeros((N, N), device=A.device, dtype=A.dtype)
    cnt = 0
    sqrt_dt = math.sqrt(dt)
    for t in range(steps):
        x = x + dt * (A @ x) + sigma * sqrt_dt * torch.randn_like(x)
        if t >= burnin:
            S += torch.outer(x, x)
            cnt += 1
    return S / max(1, cnt)

def _get_rec_name_and_tensor(model, scores_dict):
    """Find the recurrent weight name/tensor in a score dict."""
    rec_name = None
    for n in scores_dict.keys():
        if n.endswith("hidden_layer.weight"):
            rec_name = n
            break
    if rec_name is None:
        # fall back to model attr; synthesize scores if needed
        rec_name = "hidden_layer.weight"
        scores_dict[rec_name] = model.hidden_layer.weight.detach().abs().clone()
    return rec_name, scores_dict[rec_name]

def _synapse_scores_to_neuron(scores_tensor: torch.Tensor, reduce="sumabs"):
    """
    Collapse synapse scores (H x H) to per-neuron scores [H].
    Combine in+out connections to reflect neuron importance.
    """
    # scores_tensor shape: [H, H] (row: out, col: in) for recurrent matrix
    if reduce == "sumabs":
        row = scores_tensor.abs().sum(dim=1)  # outgoing
        col = scores_tensor.abs().sum(dim=0)  # incoming
        return row + col
    elif reduce == "l2":
        row = torch.linalg.vector_norm(scores_tensor, ord=2, dim=1)
        col = torch.linalg.vector_norm(scores_tensor, ord=2, dim=0)
        return row + col
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")

def _zscore(x: torch.Tensor, eps=1e-8):
    m = x.mean()
    s = x.std(unbiased=False)
    return (x - m) / (s + eps)

def combined_noise_neuron_scores(
    model,
    data_loader_fn,
    batches=4,
    sigma=0.05,
    last_only=True,
    fisher_w=1.0,
    noiseprobe_w=1.0,
    activity_w=1.0,
    reduce="sumabs",
):
    """
    Build a neuron score by z-scored weighted combo of:
      - Fisher diagonal (synapse -> neuron)
      - Noise probe synapse scores (synapse -> neuron)
      - Activity variance (already neuron-level)
    Returns: [H] neuron scores (higher = keep).
    """
    device = next(model.parameters()).device
    H = getattr(model, "H", model.hidden_layer.weight.shape[0])

    # 1) Fisher (synapse -> neuron)
    fisher = None
    if fisher_w != 0.0:
        fs = fisher_diag_scores(model, data_loader_fn=data_loader_fn, batches=batches)
        _, fW = _get_rec_name_and_tensor(model, fs)
        fisher = _synapse_scores_to_neuron(fW, reduce=reduce)

    # 2) Noise probe (synapse -> neuron)
    nprobe = None
    if noiseprobe_w != 0.0:
        ns = noise_probe_scores(model, data_loader_fn=data_loader_fn, batches=batches, sigma=sigma, last_only=last_only)
        _, nW = _get_rec_name_and_tensor(model, ns)
        nprobe = _synapse_scores_to_neuron(nW, reduce=reduce)

    # 3) Activity variance (neuron)
    act = None
    if activity_w != 0.0:
        act = activity_neuron_scores(model, data_loader_fn=data_loader_fn, batches=batches)
        # ensure shape [H]
        if act.numel() != H:
            act = act.view(H)

    # Combine (z-score each non-None and weight)
    parts = []
    if fisher is not None:
        parts.append(fisher_w * _zscore(fisher.to(device)))
    if nprobe is not None:
        parts.append(noiseprobe_w * _zscore(nprobe.to(device)))
    if act is not None:
        parts.append(activity_w * _zscore(act.to(device)))

    if len(parts) == 0:
        # fallback: keep all
        return torch.ones(H, device=device)

    combo = torch.stack(parts).sum(dim=0)
    # higher score = more important; neuron_mask_from_scores will prune the lowest
    return combo


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

def iterative_magnitude_pruning(model, opt, data, device, criterion,
                                rounds=5, prune_each=0.2, ft_steps=100, last_only=True):
    from .train_eval import train_epoch

    for _ in range(rounds):
        # magnitude-based pruning on recurrent weights
        P.l1_unstructured(model.hidden_layer, name="weight", amount=prune_each)

        # NEW: enforce constraints on weight_orig + mask (no self-connections, Dale if enabled)
        enforce_constraints(model)

        # brief fine-tune
        _ = train_epoch(model, data, device, opt, criterion, steps=ft_steps, last_only=last_only)

def movement_scores(model: CTRNN, data, device, criterion, batches=10, last_only=True):
    model.train()
    # zero existing grads once
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    with torch.enable_grad():
        for _ in range(batches):
            x, y = data.sample_batch()
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits[-1], y[-1]) if last_only else \
                   criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()  # accumulate across batches

    W = model.hidden_layer.weight
    g = W.grad if W.grad is not None else torch.zeros_like(W)
    scores = (g * W).abs().detach().clone()
    return scores

@torch.no_grad()
def _keep_mask_from_scores(score: torch.Tensor, amount: float, *, seed: int = 0) -> torch.Tensor:
    n = score.numel()
    k_prune = int(round(amount * n))
    if k_prune <= 0:
        return torch.ones_like(score, dtype=torch.uint8)
    if k_prune >= n:
        return torch.zeros_like(score, dtype=torch.uint8)

    flat = score.flatten()
    # tiny noise to break ties deterministically
    g = torch.Generator(device=flat.device)
    g.manual_seed(seed)
    noise = 1e-12 * torch.rand_like(flat, generator=g)

    idx = torch.argsort(flat + noise, descending=True)
    keep_n = n - k_prune
    keep_idx = idx[:keep_n]

    mask = torch.zeros(n, dtype=torch.uint8, device=score.device)
    mask[keep_idx] = 1
    return mask.view_as(score)


def snip_scores(model: CTRNN, data, device, criterion, batches: int = 1, last_only: bool = True) -> torch.Tensor:
    """
    SNIP-style saliency for recurrent weights: |∂L/∂W * W|.
    Compute on a few mini-batches without optimizer steps.
    """
    model.train()
    layer = model.hidden_layer
    # zero just in case
    if layer.weight.grad is not None:
        layer.weight.grad.zero_()

    with torch.enable_grad():
        for _ in range(batches):
            x, y = data.sample_batch()
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits[-1], y[-1]) if last_only else \
                   criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)

    W = layer.weight
    g = W.grad if W.grad is not None else torch.zeros_like(W)
    score = (g * W).abs().detach().clone()
    # clean up
    layer.weight.grad = None
    return score

@torch.no_grad()
def prune_snip_recurrent(model: CTRNN, score: torch.Tensor, amount: float, *, seed: int = 0):
    import torch.nn.utils.prune as P
    keep = _keep_mask_from_scores(score, amount, seed=seed).to(dtype=model.hidden_layer.weight.dtype)
    P.custom_from_mask(model.hidden_layer, name="weight", mask=keep)
    enforce_constraints(model)


def prune_movement_recurrent(model: CTRNN, score: torch.Tensor, amount: float):
    layer = model.hidden_layer
    W = layer.weight  # do NOT use .data; pruning may wrap this as weight_orig * mask

    # pick a threshold so that 'amount' fraction of entries are pruned
    k = int(score.numel() * amount)
    if k <= 0:
        return
    thresh = torch.topk(score.flatten(), k, largest=False).values.max()
    mask = (score > thresh).to(dtype=W.dtype).reshape_as(W)

    # persistent mask so zeros stay zero during fine-tuning
    P.custom_from_mask(layer, name="weight", mask=mask)

    # NEW: clean diagonal / apply Dale on the underlying weight_orig & mask
    enforce_constraints(model)

def _infer_input_dim(model):
    # Try common attribute names first
    for name in ("I", "input_dim"):
        if hasattr(model, name):
            return int(getattr(model, name))
    # Fall back to the first Linear we can find
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            return int(m.in_features)
    raise AttributeError("Could not infer input dimension for SynFlow scoring.")

def synflow_scores(model):
    """
    Data-free SynFlow: score params by |grad * weight| after making all weights positive.
    Returns a dict mapping param names -> score tensors with same shapes.
    """
    was_training = model.training
    model.eval()

    # 1) make all params positive and stash originals
    backups = {}
    for n, p in model.named_parameters():
        backups[n] = p.data.clone()
        p.data = p.data.abs()

    # 2) forward ones input (T=1, B=1, I=in_features)
    device = next(model.parameters()).device
    I = _infer_input_dim(model)
    x = torch.ones(1, 1, I, device=device)

    # zero grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Your CTRNN forward returns (logits, hidden) when given [T,B,I]
    logits, _ = model(x)
    loss = logits.abs().sum()
    loss.backward()

    # 3) collect scores: |grad * weight|
    out = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        out[n] = (p.grad * p).detach().abs().clone()

    # 4) restore weights & mode
    for n, p in model.named_parameters():
        p.data.copy_(backups[n])
    if was_training:
        model.train()

    return out

def noise_probe_scores(model, data_loader_fn, batches=2, sigma=0.05, last_only=True):
    """
    Data-aware 'noise probing' scorer:
    - Add small Gaussian noise to inputs twice (independent draws)
    - Measure output disagreement (variance proxy)
    - Backprop disagreement to accumulate |grad| as importance
    Returns: dict name->score tensor (same shapes as params)
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.train()  # enable grads

    # init accumulators
    scores = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}

    for _ in range(batches):
        X, Y = data_loader_fn()  # X: [T,B,I], Y: [T,B] or [B]
        X = X.to(device)

        # two independent noise draws (same shape as X)
        eps1 = torch.randn_like(X) * sigma
        eps2 = torch.randn_like(X) * sigma

        # forward 1
        model.zero_grad(set_to_none=True)
        logits1, _ = model(X + eps1)
        # forward 2 (no grad clear yet, we’ll recompute)
        logits2, _ = model(X + eps2)

        # compare final-step logits (last_only) or full sequence
        if logits1.dim() == 3:  # [T,B,C]
            diff = logits1[-1] - logits2[-1] if last_only else (logits1 - logits2).view(-1, logits1.size(-1))
        else:  # [B,C]
            diff = logits1 - logits2

        # disagreement loss ~ variance proxy
        loss = (diff ** 2).mean()
        loss.backward()

        # accumulate |grad| (or |grad*weight|)
        for n, p in model.named_parameters():
            if p.grad is None: 
                continue
            scores[n] += p.grad.detach().abs()

    if not was_training:
        model.eval()
    # average across batches
    for n in scores:
        scores[n] /= float(batches)
    return scores



def fisher_diag_scores(model, data_loader_fn, batches=4):
    # Accumulate squared grads as Fisher diagonal proxy
    device = next(model.parameters()).device
    scores = {n: torch.zeros_like(p, device=device) for n,p in model.named_parameters() if p.requires_grad}
    was_training = model.training
    model.train()
    for _ in range(batches):
        X, Y = data_loader_fn()  # returns [T,B,I], [T,B] or [B] depending on last_only
        model.zero_grad(set_to_none=True)
        logits = model.forward_sequence(X)  # adapt if your API differs
        # use final step targets if last_only, else full sequence CE
        if logits.dim()==3:  # [T,B,C]
            loss = nn.CrossEntropyLoss()(logits[-1], Y[-1])
        else:                # [B,C]
            loss = nn.CrossEntropyLoss()(logits, Y[-1] if Y.dim()==2 else Y)
        loss.backward()
        for (n,p) in model.named_parameters():
            if p.grad is None: continue
            scores[n] += (p.grad.detach() ** 2)
    if was_training is False:
        model.eval()
    return {n: s / float(batches) for n,s in scores.items()}

def activity_neuron_scores(model, data_loader_fn, batches=4):
    # score each hidden unit by its activity variance across time/batch
    device = next(model.parameters()).device
    # infer H from model
    H = getattr(model, "H", None)
    if H is None:
        H = model.hidden_layer.weight.shape[0]

    act_sum = torch.zeros(H, device=device)
    act_sq_sum = torch.zeros(H, device=device)
    count = 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            X, _ = data_loader_fn()
            # get hidden sequence [T,B,H]
            try:
                h = model.hidden_sequence(X)
            except AttributeError:
                _, h = model(X)  # your forward returns (logits, hidden_seq)
            T, B, _ = h.shape
            h = h.reshape(T * B, H)
            act_sum += h.abs().sum(dim=0)
            act_sq_sum += (h ** 2).sum(dim=0)
            count += T * B
    if was_training:
        model.train()

    mean = act_sum / count
    var = (act_sq_sum / count) - (mean ** 2)
    # low-variance units are least useful → prune them first
    return var.clamp_min(0)

# -------- Neuron-level pruning helpers --------

def _neuron_keep_mask_from_scores(scores: torch.Tensor, amount: float) -> torch.Tensor:
    """
    scores: [H], larger = keep. Returns uint8 mask [H] with 1=keep, 0=prune.
    """
    H = scores.numel()
    k_prune = int(round(amount * H))
    if k_prune <= 0:
        return torch.ones(H, dtype=torch.uint8, device=scores.device)
    # deterministic tie-safe keep set
    idx = torch.argsort(scores, descending=True)
    keep = torch.zeros(H, dtype=torch.uint8, device=scores.device)
    keep[idx[: H - k_prune]] = 1
    return keep

def _apply_neuron_keep_mask(model: CTRNN, keep: torch.Tensor):
    """
    Apply the same neuron keep mask to input (rows), recurrent (rows & cols), and readout (cols).
    """
    H = model.H
    assert keep.numel() == H
    keep_r = keep.view(-1, 1)   # rows
    keep_c = keep.view(1, -1)   # cols

    # Consolidate any previous pruning to avoid stacking reparametrizations
    for layer in (model.input_layer, model.hidden_layer, model.readout_layer):
        _consolidate_if_pruned(layer)

    # Recurrent HxH
    mask_hh = (keep_r & keep_c).to(dtype=model.hidden_layer.weight.dtype)
    if getattr(model, "no_self_connections", False):
        mask_hh.fill_diagonal_(0.0)
    P.custom_from_mask(model.hidden_layer, name="weight", mask=mask_hh)

    # Input HxI (Linear(out=in_hidden, in=input)): rows correspond to hidden neurons
    Wi = model.input_layer.weight
    mask_in = keep_r.expand_as(Wi).to(dtype=Wi.dtype)
    P.custom_from_mask(model.input_layer, name="weight", mask=mask_in)

    # Readout OxH: columns correspond to hidden neurons
    Wo = model.readout_layer.weight
    mask_out = keep_c.expand_as(Wo).to(dtype=Wo.dtype)
    P.custom_from_mask(model.readout_layer, name="weight", mask=mask_out)

def _neuron_scores_from_weight(W: torch.Tensor, mode: str = "l1", combine: str = "max") -> torch.Tensor:
    """
    Per-neuron scores from recurrent weight W (H x H).
    mode: 'l1' or 'l2'; combine: 'max' (default) or 'sum' of row/col norms.
    """
    if mode == "l1":
        row = W.abs().sum(dim=1)
        col = W.abs().sum(dim=0)
    elif mode == "l2":
        row = (W**2).sum(dim=1).sqrt()
        col = (W**2).sum(dim=0).sqrt()
    else:
        raise ValueError("mode must be 'l1' or 'l2'")
    return torch.maximum(row, col) if combine == "max" else (row + col)

# -------- Neuron-level strategies --------

def prune_neurons_random(model: CTRNN, amount: float):
    H = model.H
    scores = torch.randn(H, device=model.hidden_layer.weight.device)
    keep = _neuron_keep_mask_from_scores(scores, amount)
    _apply_neuron_keep_mask(model, keep)

def prune_neurons_l1(model: CTRNN, amount: float, combine: str = "max"):
    W = model.hidden_layer.weight.detach()
    scores = _neuron_scores_from_weight(W, mode="l1", combine=combine)
    keep = _neuron_keep_mask_from_scores(scores, amount)
    _apply_neuron_keep_mask(model, keep)

def prune_neurons_movement(model: CTRNN, score_matrix: torch.Tensor, amount: float, combine: str = "max"):
    """
    score_matrix: |grad * weight|, shape HxH. Aggregate to per-neuron scores.
    """
    row = score_matrix.sum(dim=1)
    col = score_matrix.sum(dim=0)
    scores = torch.maximum(row, col) if combine == "max" else (row + col)
    keep = _neuron_keep_mask_from_scores(scores, amount)
    _apply_neuron_keep_mask(model, keep)

@torch.no_grad()
def prune_noise_synapse(
    model,
    amount: float,
    *,
    leak: float = 1.0,
    sigma: float = 1.0,
    sim_dt: float = 1e-2,
    sim_steps: int = 20_000,
    sim_burnin: int = 2_000,
    seed: Optional[int] = None,
    match_diagonal: bool = True,
):
    """
    Noise-driven strengthen-or-prune on the recurrent weights.
    Keeps ~ (1 - amount) of off-diagonal edges in expectation.
    """
    layer = model.hidden_layer
    W = layer.weight.detach()
    H = W.shape[0]

    # A = -leak*I + W
    A = -torch.eye(H, device=W.device, dtype=W.dtype) * leak + W
    A_off = _offdiag(A)
    W_off = A_off.clone()

    # Covariance estimate
    C = _simulate_covariance(A, sigma=sigma, dt=sim_dt, steps=sim_steps, burnin=sim_burnin, seed=seed)

    # p_ij ∝ |w_ij| * (C_ii + C_jj - sign(w_ij) * 2 C_ij), i ≠ j
    Cii = torch.diag(C).view(1, -1).expand(H, H)
    Cjj = Cii.t()
    base = W_off.abs() * (Cii + Cjj - torch.sign(W_off) * 2.0 * C)
    base = _offdiag(base).clamp(min=0.0)

    off_count = H * (H - 1)
    target_density = 1.0 - float(amount)
    target_edges = target_density * off_count
    denom = float(base.sum().item())
    K = (target_edges / max(denom, 1e-12)) if denom > 0 else 0.0
    p = (K * base).clamp(min=0.0, max=1.0)

    # Sample + rescale
    bern = torch.rand_like(p)
    keep = (bern < p).to(W.dtype)
    p_safe = torch.where((p <= 1e-12) & (W_off != 0), torch.full_like(p, 1e-8), p)
    A_sparse_off = keep * (A_off / p_safe)

    # Diagonal handling
    A_diag = torch.diag(A)
    if match_diagonal:
        in_new = torch.sum(A_sparse_off.abs(), dim=1)
        in_old = torch.sum(A_off.abs(), dim=1)
        Delta = in_new - in_old
        A_diag = A_diag - Delta
    A_sparse = A_sparse_off + torch.diag(A_diag)

    # Recover W_sparse from A_sparse = -leak*I + W_sparse
    W_sparse = _offdiag(A_sparse)

    # Persist zeros via prune mask
    import torch.nn.utils.prune as P
    mask = (keep > 0).to(dtype=W.dtype)              # 1=keep, 0=prune
    P.custom_from_mask(layer, name="weight", mask=mask)
    layer.weight_orig.data.copy_(W_sparse)

    # Re-apply your constraints (diagonal/Dale etc.)
    enforce_constraints(model)

@torch.no_grad()
def finalize_pruning(model: nn.Module) -> None:
    """
    Remove prune reparametrizations so final checkpoints are truly sparse.
    Call this once after fine-tuning.
    """
    for mod in model.modules():
        for pname, _ in list(mod.named_parameters(recurse=False)):
            if hasattr(mod, f"{pname}_mask") and hasattr(mod, f"{pname}_orig"):
                try:
                    P.remove(mod, pname)
                except Exception:
                    pass