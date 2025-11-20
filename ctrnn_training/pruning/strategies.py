"""Core pruning utilities focused on the primary RNN baselines."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from ..models import CTRNN

PRUNE_AMOUNT_STEP = 0.05
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


def prune_neurons_random(model: CTRNN, amount: float) -> None:
    scores = torch.rand(model.H, device=model.hidden_layer.weight.device)
    mask = _neuron_keep_mask_from_scores(scores, amount)
    _apply_neuron_keep_mask(model, mask)


def prune_neurons_l1(model: CTRNN, amount: float, combine: str = "max") -> None:
    weight = model.hidden_layer.weight.detach()
    if combine not in {"max", "sum"}:
        raise ValueError("combine must be 'max' or 'sum'")
    row = weight.abs().sum(dim=1)
    col = weight.abs().sum(dim=0)
    scores = torch.maximum(row, col) if combine == "max" else (row + col)
    mask = _neuron_keep_mask_from_scores(scores, amount)
    _apply_neuron_keep_mask(model, mask)


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
        try:
            from noise_prune import noise_prune as ct_noise_prune
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError("noise_prune module is required for the noise_prune strategy") from exc
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
    model.hidden_layer.weight.data.copy_(tensor)
    enforce_constraints(model)

    stats.update({
        "amount": float(amount),
        "target_density": desired_density,
    })
    return stats


def prune_movement_synapse(model: CTRNN, amount: float, *, scores: torch.Tensor) -> None:
    prune_scores_unstructured(model, scores, amount)


def prune_movement_neuron(model: CTRNN, amount: float, *, scores: torch.Tensor) -> None:
    row = scores.sum(dim=1)
    col = scores.sum(dim=0)
    combined = torch.maximum(row, col)
    keep = _neuron_keep_mask_from_scores(combined, amount)
    _apply_neuron_keep_mask(model, keep)


def prune_snip_synapse(model: CTRNN, amount: float, *, scores: torch.Tensor) -> None:
    prune_scores_unstructured(model, scores, amount)


def prune_synflow(model: CTRNN, amount: float, *, scores: torch.Tensor) -> None:
    prune_scores_unstructured(model, scores, amount)


def prune_fisher_synapse(model: CTRNN, amount: float, *, scores: torch.Tensor) -> None:
    prune_scores_unstructured(model, scores, amount)


def movement_scores(
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
            model.zero_grad(set_to_none=True)
            logits, _ = model(x_batch)
            if last_only:
                loss = criterion(logits[-1], y_batch[-1])
            else:
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            loss.backward(retain_graph=False)
            grad = model.hidden_layer.weight.grad
            if grad is not None:
                fisher += grad.detach() ** 2
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return fisher / max(1, len(batches))


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


PRUNING_REGISTRY = {
    "random_neuron": prune_neurons_random,
    "l1_neuron": prune_neurons_l1,
    "random_unstructured": prune_random_unstructured,
    "l1_unstructured": prune_l1_unstructured,
    "noise_prune": noise_prune_recurrent,
    "movement": prune_movement_synapse,
    "movement_neuron": prune_movement_neuron,
    "snip": prune_snip_synapse,
    "synflow": prune_synflow,
    "fisher": prune_fisher_synapse,
}

PRUNING_ALIASES = {
    "random": "random_unstructured",
    "random_prune": "random_neuron",
    "l1": "l1_unstructured",
    "movement_synapse": "movement",
}


def available_pruning_strategies() -> Dict[str, callable]:
    return dict(PRUNING_REGISTRY)


def apply_pruning(model: CTRNN, strategy: str, amount: float, **kwargs) -> Dict[str, float]:
    key = strategy.lower()
    if key in {"", "none"}:
        return {}
    key = PRUNING_ALIASES.get(key, key)
    if key not in PRUNING_REGISTRY:
        raise ValueError(f"Unknown pruning strategy '{strategy}'")
    normalized = validate_prune_fraction(float(amount))
    result = PRUNING_REGISTRY[key](model, normalized, **kwargs)
    return result or {}


__all__ = [
    "PRUNE_AMOUNT_STEP",
    "apply_pruning",
    "available_pruning_strategies",
    "enforce_constraints",
    "finalize_pruning",
    "fisher_diag_scores",
    "movement_scores",
    "noise_prune_recurrent",
    "prune_l1_unstructured",
    "prune_movement_neuron",
    "prune_movement_synapse",
    "prune_neurons_l1",
    "prune_neurons_random",
    "prune_random_unstructured",
    "prune_snip_synapse",
    "prune_synflow",
    "prune_fisher_synapse",
    "snip_scores",
    "synflow_scores",
    "validate_prune_fraction",
]
