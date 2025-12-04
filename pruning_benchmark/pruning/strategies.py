"""Core pruning utilities focused on the primary RNN baselines."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from ..models import CTRNN
from .pruners import (
    BasePruner,
    PruneContext,
    available_pruning_strategies,
    register_pruner,
)

PRUNE_AMOUNT_STEP = 0.10
_STEP_EPS = 1e-6


def validate_prune_fraction(amount: float, *, step: float = PRUNE_AMOUNT_STEP) -> float:
    """Ensure pruning fractions follow the global step granularity (defaults to 10%)."""
    if math.isnan(amount):
        raise ValueError("Pruning amount cannot be NaN")
    if amount < 0.0 or amount > 1.0:
        raise ValueError(f"Pruning amount must be in [0, 1], got {amount}")
    scaled = round(amount / step)
    normalized = round(scaled * step, 10)
    if abs(normalized - amount) > _STEP_EPS:
        raise ValueError(
            f"Pruning amount {amount} must be a multiple of {step:.2f}. "
            f"Suggested value: {normalized:.1f}."
        )
    return float(normalized)


@torch.no_grad()
def enforce_constraints(model: CTRNN) -> None:
    """Apply Dale's law and remove self-connections when requested by the model."""
    if not hasattr(model, "hidden_layer"):
        return
    layer = model.hidden_layer
    weight = getattr(layer, "weight_orig", layer.weight)

    if getattr(model, "use_dale", False):
        weight.mul_(model.dale_sign).abs_().mul_(model.dale_sign)

    if getattr(model, "no_self_connections", False):
        weight.fill_diagonal_(0.0)
        mask = getattr(layer, "weight_mask", None)
        if mask is not None:
            mask.fill_diagonal_(0.0)


def _consolidate_if_pruned(layer: nn.Module) -> None:
    try:
        prune.remove(layer, "weight")
    except (ValueError, AttributeError):
        pass


def _neuron_keep_mask_from_scores(scores: torch.Tensor, amount: float) -> torch.Tensor:
    amount = validate_prune_fraction(float(amount))
    H = scores.numel()
    k_prune = int(round(amount * H))
    if k_prune <= 0:
        return torch.ones(H, dtype=torch.uint8, device=scores.device)
    keep = torch.zeros(H, dtype=torch.uint8, device=scores.device)
    idx = torch.argsort(scores, descending=True)
    keep[idx[: H - k_prune]] = 1
    return keep


def _weight_scores_to_mask(scores: torch.Tensor, amount: float) -> torch.Tensor:
    amount = validate_prune_fraction(float(amount))
    flat = scores.flatten()
    k_prune = int(round(amount * flat.numel()))
    if k_prune <= 0:
        return torch.ones_like(scores, dtype=torch.uint8)
    if k_prune >= flat.numel():
        return torch.zeros_like(scores, dtype=torch.uint8)
    sorted_vals, sorted_idx = torch.sort(flat)
    thresh = sorted_vals[k_prune - 1]
    mask = (scores > thresh).to(torch.uint8)
    # bring mask exactly to target sparsity if many scores equal threshold
    current_kept = mask.sum().item()
    target_kept = scores.numel() - k_prune
    if current_kept < target_kept:
        equal_mask = (scores == thresh).flatten()
        equal_indices = torch.nonzero(equal_mask, as_tuple=False).view(-1)
        needed = target_kept - current_kept
        if needed > 0 and equal_indices.numel() > 0:
            keep_indices = equal_indices[:needed]
            flat_mask = mask.view(-1)
            flat_mask[keep_indices] = 1
            mask = flat_mask.view_as(scores)
    return mask


def _apply_neuron_keep_mask(model: CTRNN, keep: torch.Tensor) -> None:
    H = model.H
    if keep.numel() != H:
        raise ValueError(f"Neuron mask has shape {keep.shape}, expected {H}")

    keep = keep.to(dtype=torch.uint8, device=model.hidden_layer.weight.device)
    keep_r = keep.view(-1, 1)
    keep_c = keep.view(1, -1)

    for layer in (model.input_layer, model.hidden_layer, model.readout_layer):
        _consolidate_if_pruned(layer)

    mask_hh = (keep_r & keep_c).to(dtype=model.hidden_layer.weight.dtype)
    if getattr(model, "no_self_connections", False):
        mask_hh.fill_diagonal_(0.0)
    prune.custom_from_mask(model.hidden_layer, name="weight", mask=mask_hh)

    input_mask = keep_r.expand_as(model.input_layer.weight).to(dtype=model.input_layer.weight.dtype)
    prune.custom_from_mask(model.input_layer, name="weight", mask=input_mask)

    readout_mask = keep_c.expand_as(model.readout_layer.weight).to(dtype=model.readout_layer.weight.dtype)
    prune.custom_from_mask(model.readout_layer, name="weight", mask=readout_mask)
    enforce_constraints(model)


def prune_random_unstructured(model: CTRNN, amount: float) -> None:
    amount = validate_prune_fraction(float(amount))
    prune.random_unstructured(model.hidden_layer, name="weight", amount=amount)
    enforce_constraints(model)


def prune_l1_unstructured(model: CTRNN, amount: float) -> None:
    amount = validate_prune_fraction(float(amount))
    prune.l1_unstructured(model.hidden_layer, name="weight", amount=amount)
    enforce_constraints(model)


def prune_scores_unstructured(model: CTRNN, scores: torch.Tensor, amount: float) -> None:
    mask = _weight_scores_to_mask(scores, amount).to(dtype=model.hidden_layer.weight.dtype)
    _consolidate_if_pruned(model.hidden_layer)
    prune.custom_from_mask(model.hidden_layer, name="weight", mask=mask)
    enforce_constraints(model)


def noise_prune_recurrent(
    model: CTRNN,
    amount: float,
    *,
    sigma: float = 1.0,
    eps: float = 0.3,
    leak_shift: float = 0.0,
    matched_diagonal: bool = True,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 5,
) -> Dict[str, float]:
    amount = validate_prune_fraction(float(amount))
    stats: Dict[str, float] = {}
    if amount <= 0.0:
        return stats

    rng = rng or np.random.default_rng()
    weight = model.hidden_layer.weight.detach().cpu().numpy()
    desired_density = float(max(0.0, min(1.0, 1.0 - amount)))
    current_shift = float(leak_shift)
    used_shift = current_shift

    for attempt in range(max_attempts):
        base_shift = 1.0 + current_shift
        shifted = weight - base_shift * np.eye(weight.shape[0], dtype=weight.dtype)
        from .noise_prune import noise_prune as ct_noise_prune
        try:
            pruned, noise_stats = ct_noise_prune(
                shifted,
                sigma=float(sigma),
                eps=float(eps),
                matched_diagonal=bool(matched_diagonal),
                rng=rng,
                target_density=desired_density,
            )
            stats = {k: float(v) for k, v in noise_stats.items()}
            stats["leak_shift"] = float(current_shift)
            used_shift = current_shift
            break
        except ValueError as exc:
            if attempt >= max_attempts - 1:
                raise
            current_shift = max(0.5, current_shift * 2.0 if current_shift > 0 else 0.5)
    else:  # pragma: no cover - loop exhaustion defensive clause
        raise RuntimeError("noise_prune failed to converge")

    restored = pruned + (1.0 + used_shift) * np.eye(pruned.shape[0], dtype=pruned.dtype)
    tensor = torch.tensor(restored, dtype=model.hidden_layer.weight.dtype, device=model.hidden_layer.weight.device)
    _consolidate_if_pruned(model.hidden_layer)
    model.hidden_layer.weight.data.copy_(tensor)
    mask = (tensor != 0).to(dtype=model.hidden_layer.weight.dtype)
    prune.custom_from_mask(model.hidden_layer, name="weight", mask=mask)
    enforce_constraints(model)

    stats.update({
        "amount": float(amount),
        "target_density": desired_density,
    })
    return stats


def prune_movement_synapse(model: CTRNN, amount: float, *, scores: torch.Tensor) -> None:
    prune_scores_unstructured(model, scores, amount)


def prune_snip_synapse(model: CTRNN, amount: float, *, scores: torch.Tensor) -> None:
    prune_scores_unstructured(model, scores, amount)


def prune_synflow(model: CTRNN, amount: float, *, scores: torch.Tensor) -> None:
    prune_scores_unstructured(model, scores, amount)


def prune_fisher_synapse(model: CTRNN, amount: float, *, scores: torch.Tensor) -> None:
    prune_scores_unstructured(model, scores, amount)


def _collect_gradients(
    model: CTRNN,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    last_only: bool,
) -> torch.Tensor:
    grads = []
    was_training = model.training
    model.train()
    for x_batch, y_batch in batches:
        model.zero_grad(set_to_none=True)
        logits, _ = model(x_batch)
        if last_only:
            loss = criterion(logits[-1], y_batch[-1])
        else:
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        loss.backward()
        grad = model.hidden_layer.weight.grad
        if grad is None:
            grad = torch.zeros_like(model.hidden_layer.weight)
        grads.append(grad.detach().clone().reshape(-1))
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    if not grads:
        raise ValueError("No gradients collected for WoodFisher/OBS computation.")
    return torch.stack(grads, dim=0)


def _conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)
    if rs_old == 0:
        return x
    for _ in range(max_iter):
        Ap = matvec(p)
        denom = torch.dot(p, Ap)
        if torch.abs(denom) < 1e-12:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def _estimate_inverse_hessian_diag(
    model: CTRNN,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    last_only: bool,
    num_samples: int = 4,
    damping: float = 1e-3,
    cg_iters: int = 50,
) -> torch.Tensor:
    weight = model.hidden_layer.weight
    device = weight.device
    dtype = weight.dtype
    n = weight.numel()
    num_samples = max(1, int(num_samples))
    batches = list(batches)
    if not batches:
        raise ValueError("OBS requires sampled batches.")

    def hvp(vec_flat: torch.Tensor) -> torch.Tensor:
        vec = vec_flat.view_as(weight)
        hv_total = torch.zeros_like(weight)
        for x_batch, y_batch in batches:
            model.zero_grad(set_to_none=True)
            logits, _ = model(x_batch)
            if last_only:
                loss = criterion(logits[-1], y_batch[-1])
            else:
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            grad = torch.autograd.grad(loss, weight, create_graph=True)[0]
            grad_vec = torch.sum(grad * vec)
            hv = torch.autograd.grad(grad_vec, weight, retain_graph=False)[0]
            hv_total += hv
        hv_total /= len(batches)
        hv_total += damping * vec
        return hv_total.reshape(-1).detach()

    diag_est = torch.zeros(n, device=device, dtype=dtype)
    for _ in range(num_samples):
        v = torch.randint(0, 2, (n,), device=device, dtype=dtype)
        v = v * 2 - 1
        sol = _conjugate_gradient(hvp, v, max_iter=cg_iters, tol=1e-5)
        diag_est += sol * v
    diag_est /= num_samples
    return diag_est.view_as(weight).abs().clamp_min(1e-8)


def _woodfisher_inverse_diag(
    grads: torch.Tensor,
    *,
    damping: float = 1e-3,
) -> torch.Tensor:
    if grads.ndim != 2:
        raise ValueError("Grad matrix for WoodFisher must be 2-D.")
    device = grads.device
    dtype = grads.dtype
    m, _ = grads.shape
    if damping <= 0:
        raise ValueError("WoodFisher damping must be positive.")
    lambda_inv = 1.0 / damping
    GGt = torch.matmul(grads, grads.T)
    A = torch.eye(m, device=device, dtype=dtype) + lambda_inv * GGt
    A_inv = torch.linalg.inv(A)
    tmp = torch.matmul(A_inv, grads)
    diag_term = (grads * tmp).sum(dim=0)
    inv_diag = lambda_inv - (lambda_inv**2) * diag_term
    return inv_diag.clamp_min(1e-8)


def _causal_neuron_scores(
    model: CTRNN,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    last_only: bool,
) -> torch.Tensor:
    if not hasattr(model, "readout_layer"):
        raise ValueError("Model needs a readout layer for causal pruning.")
    weight = model.hidden_layer.weight
    device = weight.device
    H = weight.shape[0]
    readout_effect = model.readout_layer.weight.detach().abs().sum(dim=0)
    scores = torch.zeros(H, device=device, dtype=weight.dtype)
    count = 0
    model.eval()
    with torch.no_grad():
        for x_batch, _ in batches:
            logits, hidden_seq = model(x_batch)
            if last_only:
                hidden = hidden_seq[-1]
            else:
                hidden = hidden_seq.mean(dim=0)
            scores += hidden.abs().mean(dim=0) * readout_effect
            count += 1
    if count > 0:
        scores /= count
    return scores

def movement_scores(
    model: CTRNN,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    last_only: bool = True,
    simulated_lr: float = 1e-3,
) -> torch.Tensor:
    batches = list(batches)
    if not batches:
        return torch.zeros_like(model.hidden_layer.weight)
    was_training = model.training
    model.train()
    scores = torch.zeros_like(model.hidden_layer.weight)
    with torch.enable_grad():
        for x_batch, y_batch in batches:
            original = model.hidden_layer.weight.data.clone()
            logits, _ = model(x_batch)
            if last_only:
                loss = criterion(logits[-1], y_batch[-1])
            else:
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            grad = torch.autograd.grad(loss, model.hidden_layer.weight, retain_graph=False)[0]
            delta = -simulated_lr * grad
            scores += delta.abs()
            model.hidden_layer.weight.data.copy_(original)
    if not was_training:
        model.eval()
    return scores / max(1, len(batches))


def snip_scores(
    model: CTRNN,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    last_only: bool = True,
) -> torch.Tensor:
    batches = list(batches)
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        for x_batch, y_batch in batches:
            logits, _ = model(x_batch)
            if last_only:
                loss = criterion(logits[-1], y_batch[-1])
            else:
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            loss.backward(retain_graph=False)
    weight = model.hidden_layer.weight
    grad = weight.grad if weight.grad is not None else torch.zeros_like(weight)
    scores = (grad * weight).abs().detach()
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return scores


def fisher_diag_scores(
    model: CTRNN,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    last_only: bool = True,
) -> torch.Tensor:
    batches = list(batches)
    was_training = model.training
    model.train()
    fisher = torch.zeros_like(model.hidden_layer.weight)
    with torch.enable_grad():
        for x_batch, y_batch in batches:
            logits, _ = model(x_batch)
            if last_only:
                loss = criterion(logits[-1], y_batch[-1])
            else:
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            grad = torch.autograd.grad(loss, model.hidden_layer.weight, retain_graph=False)[0]
            fisher += grad.detach() ** 2
    if not was_training:
        model.eval()
    return fisher / max(1, len(batches))


def grasp_scores(
    model: CTRNN,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    last_only: bool = True,
) -> torch.Tensor:
    batches = list(batches)
    if not batches:
        raise ValueError("GraSP pruning requires sampled batches.")
    was_training = model.training
    model.train()
    scores = torch.zeros_like(model.hidden_layer.weight)
    with torch.enable_grad():
        for x_batch, y_batch in batches:
            model.zero_grad(set_to_none=True)
            logits, _ = model(x_batch)
            if last_only:
                loss = criterion(logits[-1], y_batch[-1])
            else:
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            grad_w = torch.autograd.grad(loss, model.hidden_layer.weight, create_graph=True)[0]
            grad_norm = grad_w / (grad_w.norm() + 1e-8)
            with torch.enable_grad():
                inner = (grad_norm * model.hidden_layer.weight).sum()
            hvp = torch.autograd.grad(inner, model.hidden_layer.weight, retain_graph=False)[0]
            scores += (-hvp).detach()
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return scores / max(1, len(batches))


def synflow_scores(model: CTRNN) -> torch.Tensor:
    was_training = model.training
    model.eval()
    backups = {}
    for name, param in model.named_parameters():
        backups[name] = param.data.clone()
        param.data = param.data.abs()

    model.zero_grad(set_to_none=True)
    input_dim = getattr(model, "I", None)
    if input_dim is None:
        raise AttributeError("CTRNN must expose input dimension (I) for SynFlow scoring")
    device = next(model.parameters()).device
    x = torch.ones(1, 1, input_dim, device=device)

    with torch.enable_grad():
        logits, _ = model(x)
        loss = logits.abs().sum()
        loss.backward()

    weight = model.hidden_layer.weight
    grad = weight.grad if weight.grad is not None else torch.zeros_like(weight)
    scores = (grad * weight).abs().detach()

    for name, param in model.named_parameters():
        param.data.copy_(backups[name])
        if param.grad is not None:
            param.grad.zero_()

    if was_training:
        model.train()

    return scores


def finalize_pruning(model: CTRNN) -> None:
    """Remove pruning reparameterisations so saved checkpoints are dense tensors."""
    for layer_name in ("input_layer", "hidden_layer", "readout_layer"):
        if not hasattr(model, layer_name):
            continue
        layer = getattr(model, layer_name)
        try:
            prune.remove(layer, "weight")
        except (ValueError, AttributeError):
            continue
    enforce_constraints(model)


class NoisePruneStrategy(BasePruner):
    name = "noise_prune"
    description = "Covariance-guided pruning on the continuous-time operator."

    def apply(
        self,
        context: PruneContext,
        state: Mapping[str, object],
        **kwargs,
    ) -> Mapping[str, float]:
        return noise_prune_recurrent(context.model, context.amount, **kwargs)


class RandomUnstructuredPruner(BasePruner):
    name = "random_unstructured"
    aliases = ("random",)

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        prune_random_unstructured(context.model, context.amount)
        return {}


class L1UnstructuredPruner(BasePruner):
    name = "l1_unstructured"
    aliases = ("l1",)

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        prune_l1_unstructured(context.model, context.amount)
        return {}


class MovementSynapsePruner(BasePruner):
    name = "movement"
    aliases = ("movement_synapse",)
    requires_batches = True
    default_batch_count = 20

    def prepare(self, context: PruneContext) -> Mapping[str, object]:
        if not context.batches:
            raise ValueError("Movement pruning requires sampled batches.")
        scores = movement_scores(
            context.model,
            context.batches,
            context.criterion,
            last_only=context.last_only,
        )
        return {"scores": scores}

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        prune_movement_synapse(context.model, context.amount, scores=state["scores"])
        return {}


class SnipPruner(BasePruner):
    name = "snip"
    requires_batches = True
    default_batch_count = 20
    supports_pretrain = True

    def prepare(self, context: PruneContext) -> Mapping[str, object]:
        if not context.batches:
            raise ValueError("SNIP pruning requires sampled batches.")
        scores = snip_scores(
            context.model,
            context.batches,
            context.criterion,
            last_only=context.last_only,
        )
        return {"scores": scores}

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        prune_snip_synapse(context.model, context.amount, scores=state["scores"])
        return {}


class FisherPruner(BasePruner):
    name = "fisher"
    requires_batches = True
    default_batch_count = 20

    def prepare(self, context: PruneContext) -> Mapping[str, object]:
        if not context.batches:
            raise ValueError("Fisher pruning requires sampled batches.")
        scores = fisher_diag_scores(
            context.model,
            context.batches,
            context.criterion,
            last_only=context.last_only,
        )
        return {"scores": scores}

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        prune_fisher_synapse(context.model, context.amount, scores=state["scores"])
        return {}


class SynflowPruner(BasePruner):
    name = "synflow"
    supports_pretrain = True

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        scores = synflow_scores(context.model)
        prune_synflow(context.model, context.amount, scores=scores)
        return {}


class GraspPruner(BasePruner):
    name = "grasp"
    requires_batches = True
    default_batch_count = 20
    supports_pretrain = True

    def prepare(self, context: PruneContext) -> Mapping[str, object]:
        if not context.batches:
            raise ValueError("GraSP pruning requires sampled batches.")
        scores = grasp_scores(
            context.model,
            context.batches,
            context.criterion,
            last_only=context.last_only,
        )
        return {"scores": scores}

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        prune_scores_unstructured(context.model, state["scores"], context.amount)
        return {}


class OBDPruner(BasePruner):
    name = "obd"
    description = "Optimal Brain Damage with diagonal Hessian approximation."
    requires_batches = True
    default_batch_count = 20

    def prepare(self, context: PruneContext) -> Mapping[str, object]:
        if not context.batches:
            raise ValueError("OBD pruning requires sampled batches for Hessian diagonal.")
        diag = fisher_diag_scores(
            context.model,
            context.batches,
            context.criterion,
            last_only=context.last_only,
        )
        return {"diag": diag}

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        diag = state["diag"].to(context.model.hidden_layer.weight.device)
        weight = context.model.hidden_layer.weight
        saliency = 0.5 * diag * (weight.detach() ** 2)
        prune_scores_unstructured(context.model, saliency, context.amount)
        return {}


class SETPruner(BasePruner):
    name = "set"
    description = "Sparse Evolutionary Training-inspired rewiring (drop poor weights, regrow random ones)."
    requires_batches = True
    default_batch_count = 10
    supports_pretrain = True

    def __init__(self, rewire_iterations: int = 2, simulated_lr: float = 1e-3):
        self.rewire_iterations = rewire_iterations
        self.simulated_lr = simulated_lr

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        if not context.batches:
            raise ValueError("SET pruning requires batches to perform supervised regrowth.")
        weight = context.model.hidden_layer.weight.data
        amount = validate_prune_fraction(context.amount)
        num_remove_total = 0
        was_training = context.model.training
        context.model.train()
        optimizer = torch.optim.SGD(context.model.parameters(), lr=self.simulated_lr)

        for _ in range(self.rewire_iterations):
            num_remove_total += self._drop_and_regrow(weight, amount, regrow=True)
            for x_batch, y_batch in context.batches:
                optimizer.zero_grad()
                logits, _ = context.model(x_batch)
                if context.last_only:
                    loss = context.criterion(logits[-1], y_batch[-1])
                else:
                    loss = context.criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                loss.backward()
                optimizer.step()
        # Final drop with no regrowth to enforce sparsity mask
        num_remove_total += self._drop_and_regrow(weight, amount, regrow=False)
        _consolidate_if_pruned(context.model.hidden_layer)
        mask = (context.model.hidden_layer.weight != 0).to(dtype=context.model.hidden_layer.weight.dtype)
        prune.custom_from_mask(context.model.hidden_layer, name="weight", mask=mask)
        enforce_constraints(context.model)
        if not was_training:
            context.model.eval()
        return {"set_regrown": float(num_remove_total)}

    def _drop_and_regrow(self, weight: torch.Tensor, amount: float, *, regrow: bool) -> int:
        num_weights = weight.numel()
        k_remove = int(round(amount * num_weights))
        if k_remove <= 0:
            return 0
        flat = weight.flatten()
        abs_w = flat.abs()
        thresh = torch.topk(abs_w, k_remove, largest=False).values[-1]
        remove_mask = abs_w <= thresh
        k_remove = int(remove_mask.sum().item())
        if k_remove <= 0:
            return 0
        flat[remove_mask] = 0.0
        if not regrow:
            return k_remove
        zero_indices = torch.nonzero(flat == 0, as_tuple=False).view(-1)
        if zero_indices.numel() == 0:
            return 0
        perm = torch.randperm(zero_indices.numel(), device=flat.device)
        selected = zero_indices[perm[:k_remove]]
        std = flat.std()
        if std == 0:
            std = 1e-3
        flat[selected] = torch.randn(selected.size(0), device=flat.device, dtype=flat.dtype) * std
        return k_remove


class OBSPruner(BasePruner):
    name = "obs"
    description = "Optimal Brain Surgeon with inverse-diagonal approximation."
    requires_batches = True
    default_batch_count = 10

    def __init__(self, damping: float = 1e-3, num_samples: int = 4, cg_iters: int = 50):
        self.damping = damping
        self.num_samples = num_samples
        self.cg_iters = cg_iters

    def prepare(self, context: PruneContext) -> Mapping[str, object]:
        if not context.batches:
            raise ValueError("OBS requires sampled batches.")
        inv_diag = _estimate_inverse_hessian_diag(
            context.model,
            context.batches,
            context.criterion,
            last_only=context.last_only,
            num_samples=self.num_samples,
            damping=self.damping,
            cg_iters=self.cg_iters,
        )
        return {"inv_diag": inv_diag}

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        inv_diag = state["inv_diag"].to(context.model.hidden_layer.weight.device)
        weight = context.model.hidden_layer.weight.detach()
        scores = 0.5 * weight.pow(2) / inv_diag.clamp_min(1e-8)
        prune_scores_unstructured(context.model, scores, context.amount)
        return {"obs_num_samples": float(self.num_samples)}


class WoodFisherPruner(BasePruner):
    name = "woodfisher"
    description = "WoodFisher low-rank Fisher inverse approximation."
    requires_batches = True
    default_batch_count = 20

    def __init__(self, damping: float = 1e-3):
        self.damping = damping

    def prepare(self, context: PruneContext) -> Mapping[str, object]:
        if not context.batches:
            raise ValueError("WoodFisher pruning requires sampled batches.")
        grads = _collect_gradients(
            context.model,
            context.batches,
            context.criterion,
            last_only=context.last_only,
        )
        inv_diag = _woodfisher_inverse_diag(grads, damping=self.damping)
        return {"inv_diag": inv_diag.view_as(context.model.hidden_layer.weight)}

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        inv_diag = state["inv_diag"].to(context.model.hidden_layer.weight.device)
        weight = context.model.hidden_layer.weight.detach()
        scores = 0.5 * weight.pow(2) / inv_diag.clamp_min(1e-8)
        prune_scores_unstructured(context.model, scores, context.amount)
        return {}


class CausalPruner(BasePruner):
    name = "causal"
    description = "Causal neuron pruning based on readout contributions."
    requires_batches = True
    default_batch_count = 10

    def prepare(self, context: PruneContext) -> Mapping[str, object]:
        if not context.batches:
            raise ValueError("Causal pruning requires sampled batches.")
        scores = _causal_neuron_scores(
            context.model,
            context.batches,
            last_only=context.last_only,
        )
        return {"scores": scores}

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        keep_mask = _neuron_keep_mask_from_scores(state["scores"], context.amount)
        _apply_neuron_keep_mask(context.model, keep_mask)
        return {}


# Register built-in strategies
register_pruner(NoisePruneStrategy())
register_pruner(RandomUnstructuredPruner())
register_pruner(L1UnstructuredPruner())
register_pruner(MovementSynapsePruner())
register_pruner(SnipPruner())
register_pruner(FisherPruner())
register_pruner(SynflowPruner())
register_pruner(GraspPruner())
register_pruner(OBDPruner())
register_pruner(SETPruner())
register_pruner(OBSPruner())
register_pruner(WoodFisherPruner())
register_pruner(CausalPruner())


__all__ = [
    "PRUNE_AMOUNT_STEP",
    "available_pruning_strategies",
    "enforce_constraints",
    "finalize_pruning",
    "fisher_diag_scores",
    "grasp_scores",
    "movement_scores",
    "noise_prune_recurrent",
    "prune_l1_unstructured",
    "prune_movement_synapse",
    "prune_random_unstructured",
    "prune_snip_synapse",
    "prune_synflow",
    "prune_fisher_synapse",
    "snip_scores",
    "synflow_scores",
    "validate_prune_fraction",
]
