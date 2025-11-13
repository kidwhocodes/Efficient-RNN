"""Evaluate trained checkpoints to build the task portfolio baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .runner import append_results_csv, run_prune_experiment


def _slugify(task: str) -> str:
    return task.replace(":", "_").replace("-", "_")


def evaluate_portfolio(
    config_path: str,
    *,
    out_csv: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Evaluate saved checkpoints described in a config file."""

    with open(config_path, "r") as fh:
        cfg = json.load(fh)

    tasks: List[Dict] = cfg.get("tasks", [])
    if not tasks:
        raise ValueError("Portfolio config contains no tasks.")

    defaults: Dict = cfg.get("defaults", {})
    csv_path = Path(out_csv or cfg.get("output_csv", "results/portfolio.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and csv_path.exists():
        csv_path.unlink()

    total = sum(len(spec.get("seeds", [spec.get("seed", 0)])) for spec in tasks)
    completed = 0

    for spec in tasks:
        task = spec["task"]
        slug = _slugify(task)
        seeds = spec.get("seeds", [spec.get("seed", 0)])
        checkpoint_template = spec.get("checkpoint") or defaults.get("checkpoint")
        if checkpoint_template is None:
            raise ValueError(f"Missing checkpoint template for task {task}")

        merged = dict(defaults)
        merged.update(spec)

        for seed in seeds:
            completed += 1
            checkpoint_path = Path(checkpoint_template.format(seed=seed, task=slug))
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            run_id_prefix = merged.get("run_id_prefix", f"portfolio_{slug}")
            run_id = f"{run_id_prefix}_seed{seed}"

            print(f"[portfolio {completed}/{total}] evaluating {task} seed {seed}")

            row = run_prune_experiment(
                strategy="none",
                amount=0.0,
                train_steps=0,
                ft_steps=0,
                last_only=bool(merged.get("last_only", True)),
                seed=int(seed),
                device=merged.get("device", "cpu"),
                movement_batches=int(merged.get("movement_batches", 20)),
                task=task,
                no_prune=True,
                run_id=run_id,
                skip_training=True,
                load_model_path=str(checkpoint_path),
                hidden_size=merged.get("hidden_size"),
                eval_steps_pre0=int(merged.get("eval_steps_pre0", 100)),
                eval_steps_pre=int(merged.get("eval_steps_pre", 120)),
                eval_steps_post0=int(merged.get("eval_steps_post0", 0)),
                eval_steps_post=int(merged.get("eval_steps_post", 0)),
            )

            append_results_csv([row], str(csv_path))

    return str(csv_path)


__all__ = ["evaluate_portfolio"]
