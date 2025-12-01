"""
Quick diagnostic script for verifying dataset label distributions.

This helps ensure NeuroGym tasks expose only the final decision label
so pruning/evaluation aren't dominated by fixation timesteps.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Tuple

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pruning_benchmark.tasks.synthetic import (
    SynthCfg,
    SynthContextCfg,
    SynthMultiRuleCfg,
    SynthHierContextCfg,
    SynthNBackCfg,
    SyntheticContextDM,
    SyntheticDM,
    SyntheticMultiRuleDM,
    SyntheticHierContextDM,
    SyntheticNBackDM,
)
from pruning_benchmark.tasks.neurogym import NeuroGymDM


def build_dataset(
    task: str,
    *,
    ng_kwargs: dict | None = None,
    ng_T: int | None = None,
    ng_B: int | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[torch.utils.data.Dataset, str]:
    """Mirror the runner task selection for diagnostics."""
    if task == "synthetic":
        cfg = SynthCfg(T=60, B=64)
        return SyntheticDM(cfg), "synthetic"
    if task == "synthetic_context":
        cfg = SynthCfg(T=40, B=64)
        return SyntheticContextDM(cfg), "synthetic_context"
    if task == "synthetic_multirule":
        cfg = SynthMultiRuleCfg(T=60, B=64)
        return SyntheticMultiRuleDM(cfg), "synthetic_multirule"
    if task == "synthetic_hiercontext":
        cfg = SynthHierContextCfg(T=70, B=64)
        return SyntheticHierContextDM(cfg), "synthetic_hiercontext"
    if task == "synthetic_nback":
        cfg = SynthNBackCfg(T=50, B=64)
        return SyntheticNBackDM(cfg), "synthetic_nback"
    if task.startswith("ng:"):
        import neurogym as ngym  # lazy import to avoid dependency for synthetic tasks

        task_name = task.split("ng:", 1)[1]
        if not any(task_name.endswith(sfx) for sfx in ("-v0", "-v1", "-v2", "-v3")):
            task_name = f"{task_name}-v0"

        env_kwargs = ng_kwargs or {}
        env = ngym.make(task_name, **env_kwargs)
        T = int(ng_T) if ng_T is not None else 400
        B = int(ng_B) if ng_B is not None else 64
        dataset = NeuroGymDM(env, T=T, B=B, device=device, last_only=True, seed=seed)
        return dataset, task_name
    raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose dataset label distributions.")
    parser.add_argument("--task", required=True, help="Task name (e.g., ng:DelayMatchSample-v0)")
    parser.add_argument("--samples", type=int, default=50, help="Number of batches to sample")
    parser.add_argument("--ng_kwargs", type=str, default=None, help="JSON dict for neurogym kwargs")
    parser.add_argument("--ng_T", type=int, default=None, help="NeuroGym trial length override")
    parser.add_argument("--ng_B", type=int, default=None, help="NeuroGym batch size override")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ng_kwargs = None
    if args.ng_kwargs:
        ng_kwargs = json.loads(args.ng_kwargs)

    dataset, task_name = build_dataset(
        args.task,
        ng_kwargs=ng_kwargs,
        ng_T=args.ng_T,
        ng_B=args.ng_B,
        seed=args.seed,
    )

    final_counts: Counter[int] = Counter()
    batch_size = getattr(dataset, "B", 0)
    for _ in range(args.samples):
        _, labels = dataset.sample_batch()
        final = labels[-1].flatten()
        final_counts.update(int(v) for v in final.tolist())
        if not batch_size:
            batch_size = final.numel()

    total = sum(final_counts.values())
    majority_label, majority_count = final_counts.most_common(1)[0]
    maj_freq = majority_count / max(total, 1)

    print(f"Task: {task_name}")
    print(f"Samples: {args.samples} batches (B≈{batch_size})")
    print("Final-label distribution:")
    for lbl, cnt in sorted(final_counts.items()):
        print(f"  label {lbl}: {cnt} ({cnt / max(total,1):.3f})")
    print(f"Majority label {majority_label} frequency: {maj_freq:.3f}")
    print(f"Chance accuracy (uniform): {1.0 / max(len(final_counts), 1):.3f}")
    print("If majority frequency ≈ observed plateau accuracy, pruning results are likely uninformative.")


if __name__ == "__main__":
    main()
