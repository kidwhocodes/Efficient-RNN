"""
Train harder ContextDM baselines and evaluate multiple pruning strategies.

Updated harder-yet-manageable settings:
- Smaller CTRNN (hidden_size=24) to reduce redundancy.
- Trials length 600 with delay window [300, 800] to increase memory load without
  exploding runtime.
- Shorter training budget (train_steps=400) plus smaller batches to keep walltime sane.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctrnn_training.experiments.runner import append_results_csv, run_prune_experiment


def _load_completed_run_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    import csv

    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "run_id" not in reader.fieldnames:
            return set()
        return {row.get("run_id") for row in reader if row.get("run_id")}


def main():
    seeds = (0, 1, 2)
    prune_strategies = ("noise_prune", "movement", "synflow")
    prune_amounts = (0.1, 0.3, 0.5, 0.7, 0.9)

    checkpoint_dir = Path("checkpoints/hard_ctxdm24")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path("results/hard_ctxdm24_pruning.csv")
    completed_run_ids = _load_completed_run_ids(csv_path)

    results = []
    base_kwargs = {
        "task": "ng:ContextDecisionMaking-v0",
        "ng_T": 550,
        "ng_kwargs": {"timing": {"delay": [250, 700]}},
        "hidden_size": 24,
        "ng_B": 32,
        "device": "cpu",
        "last_only": True,
        "eval_last_only": True,
        "eval_sample_batches": 32,
        "eval_steps_pre0": 60,
        "eval_steps_pre": 80,
        "eval_steps_post0": 80,
        "eval_steps_post": 80,
        "movement_batches": 16,
    }

    total_runs = len(seeds) * (1 + len(prune_strategies) * len(prune_amounts))
    executed = 0

    # Stage 1: train three harder baselines (skip_training=False).
    for seed in seeds:
        checkpoint_path = checkpoint_dir / f"ctxdm_hard_seed{seed}.pt"
        baseline_run_id = f"hard_ctxdm24_baseline_seed{seed}"
        if baseline_run_id in completed_run_ids:
            print(f"[skip] {baseline_run_id} already logged.")
        elif checkpoint_path.exists():
            executed += 1
            print(f"[{executed}/{total_runs}] evaluating cached baseline seed={seed}...")
            row = run_prune_experiment(
                strategy="none",
                amount=0.0,
                train_steps=0,
                ft_steps=0,
                seed=seed,
                skip_training=True,
                no_prune=True,
                load_model_path=str(checkpoint_path),
                run_id=baseline_run_id,
                **base_kwargs,
            )
            results.append(row)
        else:
            executed += 1
            print(f"[{executed}/{total_runs}] training baseline seed={seed} (quick settings)...")
            row = run_prune_experiment(
                strategy="none",
                amount=0.0,
                train_steps=400,
                ft_steps=0,
                seed=seed,
                skip_training=False,
                no_prune=True,
                save_model_path=str(checkpoint_path),
                run_id=baseline_run_id,
                **base_kwargs,
            )
            results.append(row)

        # Stage 2: run pruning sweeps on the trained baseline.
        for strategy in prune_strategies:
            for amount in prune_amounts:
                run_id = f"hard_ctxdm24_{strategy}_a{int(amount * 100):02d}_seed{seed}"
                if run_id in completed_run_ids:
                    print(f"[skip] {run_id} already completed.")
                    continue
                executed += 1
                print(
                    f"[{executed}/{total_runs}] pruning seed={seed} strategy={strategy} amount={amount:.1f}..."
                )
                row = run_prune_experiment(
                    strategy=strategy,
                    amount=amount,
                    train_steps=0,
                    ft_steps=0,
                    seed=seed,
                    skip_training=True,
                    load_model_path=str(checkpoint_path),
                    run_id=run_id,
                    **base_kwargs,
                )
                results.append(row)

    append_results_csv(results, str(csv_path))
    print("All requested runs finished (new rows appended).")


if __name__ == "__main__":
    main()
