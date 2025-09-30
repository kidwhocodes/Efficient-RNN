# ctrnn_training/sweeps.py
import os, time, csv
from typing import Sequence
from .experiments import run_prune_experiment

def run_sweep(
    out_csv: str,
    strategies: Sequence[str] = ("l1_neuron", "movement_neuron", "random_neuron", "noise_synapse"),
    amounts: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    seeds: Sequence[int] = (0,1,2,3,4),
    train_steps: int = 600,
    ft_steps: int = 400,
    last_only: bool = True,
    device: str = "cpu",
    movement_batches: int = 50,
    task: str = "synthetic",     
):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = None
    with open(out_csv, "w", newline="") as f:
        writer = None
        for strat in strategies:
            for amt in amounts:
                for seed in seeds:
                    t0 = time.time()
                    res = run_prune_experiment(
                        strategy=strat,
                        amount=amt,
                        train_steps=train_steps,
                        ft_steps=ft_steps,
                        last_only=last_only,
                        seed=seed,
                        device=device,
                        movement_batches=movement_batches,  # optional passthrough
                        task = task,
                    )
                    res["runtime_s"] = round(time.time() - t0, 2)
                    if fieldnames is None:
                        fieldnames = list(res.keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                    writer.writerow(res)
                    f.flush()
    return out_csv
