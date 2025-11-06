"""Utilities for training baseline checkpoints prior to pruning sweeps."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .runner import run_prune_experiment
from ..utils import make_run_id


def _slugify(task: str) -> str:
    return task.replace(":", "_").replace("-", "_")


def _merge_config(defaults: Dict, specific: Dict) -> Dict:
    merged = dict(defaults)
    for key, value in specific.items():
        if key in {"task", "seeds", "checkpoint_name"}:
            continue
        merged[key] = value
    return merged


def train_baselines(config_path: str, *, overwrite: bool = False) -> List[str]:
    """
    Train (or reuse) baseline models described in a JSON configuration file.

    Returns the list of checkpoint paths produced.
    """
    with open(config_path, "r") as fh:
        cfg = json.load(fh)

    tasks: List[Dict] = cfg.get("tasks", [])
    if not tasks:
        raise ValueError("Baseline config contains no tasks.")

    defaults: Dict = cfg.get("defaults", {})
    out_dir = Path(cfg.get("output_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[str] = []

    total_runs = sum(len(spec.get("seeds", [0])) for spec in tasks)
    completed = 0

    for spec in tasks:
        task_name = spec["task"]
        slug = _slugify(task_name)
        seeds = spec.get("seeds", [0])
        merged = _merge_config(defaults, spec)
        for seed in seeds:
            run_id = spec.get("run_id_prefix", f"baseline_{slug}") + f"_seed{seed}"
            checkpoint_template = spec.get("checkpoint_name", f"{slug}_seed{{seed}}.pt")
            checkpoint_path = out_dir / checkpoint_template.format(seed=seed, task=slug)
            if checkpoint_path.exists() and not overwrite:
                completed += 1
                print(
                    f"[baseline {completed}/{total_runs}] pre-existing checkpoint for {task_name} seed {seed} -> {checkpoint_path}"
                )
                produced.append(str(checkpoint_path))
                continue

            train_steps = int(merged.get("train_steps", 600))
            ft_steps = int(merged.get("ft_steps", 0))

            completed += 1
            print(
                f"[baseline {completed}/{total_runs}] training {task_name} seed {seed} (steps={train_steps})"
            )

            run_prune_experiment(
                strategy="none",
                amount=0.0,
                train_steps=train_steps,
                ft_steps=ft_steps,
                last_only=bool(merged.get("last_only", True)),
                seed=int(seed),
                device=merged.get("device", "cpu"),
                movement_batches=int(merged.get("movement_batches", 20)),
                task=task_name,
                no_prune=True,
                run_id=run_id,
                hidden_size=merged.get("hidden_size"),
                ng_kwargs=merged.get("ng_kwargs"),
                ng_T=merged.get("ng_T"),
                ng_B=merged.get("ng_B"),
                save_model_path=str(checkpoint_path),
            )
            produced.append(str(checkpoint_path))

    return produced


__all__ = ["train_baselines"]
