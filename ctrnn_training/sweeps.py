"""Batch sweep runner for pruning experiments."""

# ctrnn_training/sweeps.py
from __future__ import annotations
import os, time
from typing import Iterable, Optional
from .experiments import run_prune_experiment, append_results_csv
from .utils import make_run_id

def run_sweep(
    out_csv: str,
    strategies: Iterable[str] = ("l1_neuron", "movement_neuron", "random_neuron", "noise_synapse"),
    amounts: Iterable[float] = (0.1, 0.3, 0.5),
    seeds: Iterable[int] = (0, 1),
    *,
    train_steps: int = 120,
    ft_steps: int = 40,
    last_only: bool = True,
    device: str = "cpu",
    movement_batches: int = 10,
    task: str = "synthetic",
    run_id: Optional[str] = None,
) -> str:
    """
    Run (strategy × amount × seed) and append results to `out_csv`.
    Returns the path to the CSV.
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    results = []
    started = time.time()

    strategies = tuple(strategies)
    amounts    = tuple(amounts)
    seeds      = tuple(seeds)

    sweep_id = run_id or make_run_id("sweep")

    print(f"[sweep] id={sweep_id} task={task} device={device} "
          f"strategies={strategies} amounts={amounts} seeds={seeds}")
    print(f"[sweep] steps: train={train_steps}, ft={ft_steps}, last_only={last_only}, "
          f"movement_batches={movement_batches}")

    total = len(strategies) * len(amounts) * len(seeds)
    idx = 0

    for strat in strategies:
        for amt in amounts:
            for seed in seeds:
                idx += 1
                tag = f"[{idx}/{total}] {strat} amt={amt} seed={seed}"
                run_tag = f"{sweep_id}_{idx}"
                try:
                    row = run_prune_experiment(
                        strategy=strat,
                        amount=float(amt),
                        train_steps=train_steps,
                        ft_steps=ft_steps,
                        last_only=last_only,
                        seed=int(seed),
                        device=device,
                        movement_batches=movement_batches,
                        task=task,
                        run_id=run_tag,
                    )
                    # Ensure key fields
                    row.setdefault("task", task)
                    row.setdefault("strategy", strat)
                    row.setdefault("amount", float(amt))
                    row.setdefault("seed", int(seed))
                    row.setdefault("run_id", run_tag)
                    results.append(row)

                    # Write periodically so kills don't lose progress
                    if len(results) >= 5:
                        append_results_csv(results, out_csv)
                        results.clear()

                    print(f"{tag} ✓")
                except Exception as e:
                    err_row = {
                        "task": task, "strategy": strat, "amount": float(amt), "seed": int(seed),
                        "error": repr(e),
                    }
                    results.append(err_row)
                    append_results_csv(results, out_csv)
                    results.clear()
                    print(f"{tag} ✗  {e}")

    if results:
        append_results_csv(results, out_csv)

    elapsed = time.time() - started
    print(f"[sweep] wrote: {out_csv}  (elapsed {elapsed:.1f}s)")
    return out_csv
