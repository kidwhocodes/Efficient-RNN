from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

from .core import CTRNN
from .data import SynthCfg, SyntheticDM
from .train_eval import train_epoch, evaluate
from .metrics import recurrent_sparsity, ctrnn_stability_proxy, neuron_pruning_stats
from . import pruning as PR
from typing import Optional

def fresh_model(device="cpu"):
    return CTRNN(
        input_dim=3, hidden_size=64, output_dim=3,
        dt=100, tau=100, activation="relu",
        preact_noise=0.05, postact_noise=0.0,
        use_dale=False, no_self_connections=True, scaling=1.0
    ).to(device)

def append_results_csv(results_list, csv_path="results.csv"):
    import csv, os
    if not results_list:
        return
    # union of keys across rows (so adding new fields later is fine)
    keys = sorted({k for r in results_list for k in r.keys()})
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new_file:
            w.writeheader()
        for r in results_list:
            w.writerow(r)


# make sure you have: from typing import Optional  (already in your file)
def run_prune_experiment(
    strategy: str,
    amount: float,
    *,
    seed: int = 0,
    device: Optional[str] = None,
    last_only: bool = True,
    global_include_input: bool = False,
    global_include_readout: bool = False,
    base_model: Optional[CTRNN] = None,
    train_steps: int = 300,
    ft_steps: int = 50,
):
    # --- setup ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed); np.random.seed(seed)

    # build model (or deep-copy a provided base model)
    model = fresh_model(device) if base_model is None else deepcopy(base_model).to(device)

    # data/task
    cfg = SynthCfg(T=60, B=64, coh_levels=(0.0, 0.05, 0.1, 0.2), stim_std=0.6)
    data = SyntheticDM(cfg)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    # --- baseline train & pre metrics ---
    _ = train_epoch(model, data, device, opt, criterion, steps=train_steps, last_only=last_only)
    pre_loss, pre_acc = evaluate(model, data, device, criterion, steps=100, last_only=last_only)
    pre_spars = recurrent_sparsity(model)
    pre_alpha_rho = ctrnn_stability_proxy(model)
    pre_np = neuron_pruning_stats(model)

    # --- apply pruning ---
    if strategy == "none":
        pass
    elif strategy == "random":
        PR.prune_random_recurrent(model, amount)
    elif strategy == "l1_unstructured":
        PR.prune_l1_recurrent(model, amount)
    elif strategy == "global_unstructured":
        PR.prune_global_unstructured(
            model, amount,
            include_readout=global_include_readout,
            include_input=global_include_input
        )
    elif strategy == "structured_out":
        PR.prune_structured_recurrent(model, amount, dim=0)
    elif strategy == "structured_in":
        PR.prune_structured_recurrent(model, amount, dim=1)
    elif strategy == "movement":
        score = PR.movement_scores(model, data, device, criterion, batches=20, last_only=last_only)
        PR.prune_movement_recurrent(model, score, amount)
    elif strategy == "imp":
        R = 5
        prune_each = 1 - (1 - amount) ** (1 / R)  # compound to ~final 'amount'
        PR.iterative_magnitude_pruning(
            model, opt, data, device, criterion,
            rounds=R, prune_each=prune_each, ft_steps=ft_steps, last_only=last_only
        )
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    # keep masks during fine-tune so zeros can't regrow
    PR._enforce_no_self_connections(model)
    _ = train_epoch(model, data, device, opt, criterion, steps=ft_steps, last_only=last_only)

    # --- post metrics ---
    post_loss, post_acc = evaluate(model, data, device, criterion, steps=100, last_only=last_only)
    post_spars = recurrent_sparsity(model)
    post_alpha_rho = ctrnn_stability_proxy(model)
    post_np = neuron_pruning_stats(model)

    return {
        "seed": seed,
        "strategy": strategy,
        "amount": amount,
        "pre_loss": pre_loss, "pre_acc": pre_acc,
        "post_loss": post_loss, "post_acc": post_acc,
        "pre_sparsity": pre_spars, "post_sparsity": post_spars,
        "pre_alpha_rho": pre_alpha_rho, "post_alpha_rho": post_alpha_rho,
        "pre_rows_zero": pre_np["rows_zero"], "pre_cols_zero": pre_np["cols_zero"], "pre_isolated": pre_np["isolated"],
        "post_rows_zero": post_np["rows_zero"], "post_cols_zero": post_np["cols_zero"], "post_isolated": post_np["isolated"],
    }



if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run pruning sweeps and log to CSV")
    p.add_argument("--strategies", type=str,
                   default="none,random,l1_unstructured,global_unstructured,structured_out,structured_in,movement,imp")
    p.add_argument("--amounts", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.8])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--csv", type=str, default="results.csv")
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--ft-steps", type=int, default=50)
    p.add_argument("--global-include-input", action="store_true")
    p.add_argument("--global-include-readout", action="store_true")
    args = p.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    results = []

    for seed in args.seeds:
        for strat in strategies:
            for amount in args.amounts:
                kwargs = {}
                if strat == "global_unstructured":
                    kwargs["global_include_input"] = args.global_include_input
                    kwargs["global_include_readout"] = args.global_include_readout

                res = run_prune_experiment(
                    strat,
                    amount,
                    seed=seed,
                    train_steps=args.train_steps,
                    ft_steps=args.ft_steps,
                    **kwargs,
                )
                print(res)
                results.append(res)

    append_results_csv(results, csv_path=args.csv)
    print(f"→ Appended {len(results)} rows to {args.csv}")
