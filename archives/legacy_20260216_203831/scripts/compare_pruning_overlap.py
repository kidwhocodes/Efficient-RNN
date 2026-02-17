#!/usr/bin/env python3
"""Compare overlap of pruning masks between strategies on a fixed checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.experiments.runner import fresh_model
from pruning_benchmark.tasks import (
    SynthCfg,
    SynthContextCfg,
    SynthHierContextCfg,
    SynthMultiRuleCfg,
    SynthNBackCfg,
    SyntheticContextDM,
    SyntheticDM,
    SyntheticHierContextDM,
    SyntheticMultiRuleDM,
    SyntheticNBackDM,
)
from pruning_benchmark.pruning.strategies import (
    _weight_scores_to_mask,
    fisher_diag_scores,
    movement_scores,
    prune_l1_unstructured,
    prune_random_unstructured,
    noise_prune_recurrent,
)
from pruning_benchmark.tasks.modcog import ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import NeuroGymDatasetDM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pruning overlaps.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/scratch_prune_smoke_synthetic_seed0.pt",
        help="Baseline checkpoint path.",
    )
    parser.add_argument("--amount", type=float, default=0.9)
    parser.add_argument("--model_type", default="ctrnn")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="synthetic")
    parser.add_argument("--decision_delay", type=int, default=20)
    parser.add_argument("--ng_T", type=int, default=400)
    parser.add_argument("--ng_B", type=int, default=64)
    parser.add_argument("--batch_count", type=int, default=20)
    return parser.parse_args()


def _infer_dims(state: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    input_w = state.get("input_layer.weight")
    hidden_w = state.get("hidden_layer.weight")
    readout_w = state.get("readout_layer.weight")
    if input_w is None or hidden_w is None or readout_w is None:
        raise ValueError("Checkpoint missing expected CTRNN layer weights.")
    hidden_size = int(hidden_w.shape[0])
    input_dim = int(input_w.shape[1])
    output_dim = int(readout_w.shape[0])
    return input_dim, hidden_size, output_dim


def _load_model(checkpoint: Path, model_type: str, device: str) -> torch.nn.Module:
    state = torch.load(checkpoint, map_location=device)
    input_dim, hidden_size, output_dim = _infer_dims(state)
    model = fresh_model(
        input_dim=input_dim,
        hidden_size=hidden_size,
        output_dim=output_dim,
        model_type=model_type,
        device=device,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def _mask_from_hidden(model: torch.nn.Module) -> torch.Tensor:
    layer = getattr(model, "hidden_layer", None)
    if layer is None or not hasattr(layer, "weight"):
        raise ValueError("Model has no hidden_layer weights.")
    mask = getattr(layer, "weight_mask", None)
    if mask is None:
        # Not pruned; keep all
        return torch.ones_like(layer.weight, dtype=torch.bool)
    return mask.to(dtype=torch.bool)


def _build_dataset(task: str, *, decision_delay: int, ng_T: int, ng_B: int, device: str):
    dataset_task = task
    if dataset_task.startswith("modcog:"):
        env_id = ensure_modcog_env_id(dataset_task)
        if env_id is None:
            raise ValueError(f"Unknown Mod_Cog task: {dataset_task}")
        dataset_task = env_id

    if dataset_task == "synthetic":
        cfg = SynthCfg(decision_delay=decision_delay)
        return SyntheticDM(cfg)
    if dataset_task == "synthetic_context":
        cfg = SynthContextCfg(decision_delay=decision_delay)
        return SyntheticContextDM(cfg)
    if dataset_task == "synthetic_multirule":
        cfg = SynthMultiRuleCfg(decision_delay=decision_delay)
        return SyntheticMultiRuleDM(cfg)
    if dataset_task == "synthetic_hiercontext":
        cfg = SynthHierContextCfg(decision_delay=decision_delay)
        return SyntheticHierContextDM(cfg)
    if dataset_task == "synthetic_nback":
        cfg = SynthNBackCfg(decision_delay=decision_delay)
        return SyntheticNBackDM(cfg)

    return NeuroGymDatasetDM(
        dataset_task,
        T=ng_T,
        B=ng_B,
        device=device,
        last_only=True,
        seed=0,
    )


def _neuron_mask_from_weight(mask: torch.Tensor) -> torch.Tensor:
    # Keep neurons with any surviving in/out connection.
    keep_row = mask.any(dim=1)
    keep_col = mask.any(dim=0)
    return keep_row | keep_col


def _overlap(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    jaccard = inter / union if union else 1.0
    frac_same = (a == b).float().mean().item()
    return jaccard, frac_same


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    torch.manual_seed(args.seed)

    masks: Dict[str, torch.Tensor] = {}
    data = _build_dataset(
        args.task,
        decision_delay=args.decision_delay,
        ng_T=args.ng_T,
        ng_B=args.ng_B,
        device=args.device,
    )
    criterion = torch.nn.CrossEntropyLoss()
    batches = [data.sample_batch() for _ in range(args.batch_count)]

    for name, pruner in (
        ("l1_unstructured", prune_l1_unstructured),
        ("random_unstructured", prune_random_unstructured),
    ):
        model = _load_model(checkpoint, args.model_type, args.device)
        pruner(model, args.amount, include_feedforward=False)
        masks[name] = _mask_from_hidden(model).cpu()

    model = _load_model(checkpoint, args.model_type, args.device)
    noise_prune_recurrent(model, args.amount, include_feedforward=False)
    masks["noise_prune"] = _mask_from_hidden(model).cpu()

    model = _load_model(checkpoint, args.model_type, args.device)
    fisher_scores = fisher_diag_scores(model, batches, criterion, last_only=True)
    fisher_mask = _weight_scores_to_mask(fisher_scores, args.amount).to(dtype=torch.bool)
    masks["fisher"] = fisher_mask.cpu()

    model = _load_model(checkpoint, args.model_type, args.device)
    movement_scores_mask = movement_scores(model, batches, criterion, last_only=True)
    movement_mask = _weight_scores_to_mask(movement_scores_mask, args.amount).to(dtype=torch.bool)
    masks["movement"] = movement_mask.cpu()

    # Synapse overlap
    print("Synapse mask overlap (Jaccard, exact match fraction)")
    pairs = [
        ("l1_unstructured", "random_unstructured"),
        ("l1_unstructured", "noise_prune"),
        ("l1_unstructured", "fisher"),
        ("l1_unstructured", "movement"),
    ]
    for a, b in pairs:
        jaccard, frac_same = _overlap(masks[a], masks[b])
        print(f"  {a} vs {b}: jaccard={jaccard:.4f} same={frac_same:.4f}")

    # Neuron overlap
    neuron_masks = {k: _neuron_mask_from_weight(v) for k, v in masks.items()}
    print("\nNeuron keep overlap (Jaccard, exact match fraction)")
    for a, b in pairs:
        jaccard, frac_same = _overlap(neuron_masks[a], neuron_masks[b])
        print(f"  {a} vs {b}: jaccard={jaccard:.4f} same={frac_same:.4f}")


if __name__ == "__main__":
    main()
