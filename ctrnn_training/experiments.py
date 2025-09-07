from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

from .core import CTRNN
from .data import SynthCfg, SyntheticDM
from .train_eval import train_epoch, evaluate
from .metrics import recurrent_sparsity, ctrnn_stability_proxy, neuron_pruning_stats
from . import pruning as PR

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


def run_prune_experiment(strategy: str, amount: float,
                         base_model: CTRNN = None,
                         train_steps=600, ft_steps=200,
                         last_only=True, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data / loss
    cfg = SynthCfg(T=60, B=64, coh_levels=(0.0, 0.05, 0.1, 0.2), stim_std=0.6)
    data = SyntheticDM(cfg)
    criterion = nn.CrossEntropyLoss()

    # model & opt
    model = fresh_model(device) if base_model is None else deepcopy(base_model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # pretrain
    _ = train_epoch(model, data, device, opt, criterion, steps=train_steps, last_only=last_only)
    pre_loss, pre_acc = evaluate(model, data, device, criterion, steps=200, last_only=last_only)
    pre_spars = recurrent_sparsity(model)
    pre_alpha_rho = ctrnn_stability_proxy(model)

    pre_np = neuron_pruning_stats(model)

    # prune
    if strategy == "none":
        pass
    elif strategy == "random":
        PR.prune_random_recurrent(model, amount)
    elif strategy == "l1_unstructured":
        PR.prune_l1_recurrent(model, amount)
    elif strategy == "structured_out":
        PR.prune_structured_recurrent(model, amount, dim=0)
    elif strategy == "structured_in":
        PR.prune_structured_recurrent(model, amount, dim=1)
    elif strategy == "global_unstructured":
        PR.prune_global_unstructured(model, amount, include_readout=False, include_input=False)
    elif strategy == "movement":
        score = PR.movement_scores(model, data, device, criterion, batches=20, last_only=last_only)
        PR.prune_movement_recurrent(model, score, amount)
    elif strategy == "imp":
        PR.iterative_magnitude_pruning(model, opt, data, device, criterion,
                                       rounds=5, prune_each=amount/5, ft_steps=50, last_only=last_only)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # consolidate masks if any and re-zero diagonal
    PR._enforce_no_self_connections(model)

    # Only consolidate if you really want plain tensors afterwards
    # PR._consolidate_if_pruned(model.hidden_layer)

    # brief fine-tune
    _ = train_epoch(model, data, device, opt, criterion, steps=ft_steps, last_only=last_only)

    # post metrics
    post_loss, post_acc = evaluate(model, data, device, criterion, steps=200, last_only=last_only)
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
    sweep = [
        ("none", 0.0),
        ("random", 0.5),
        ("l1_unstructured", 0.5),
        ("global_unstructured", 0.5),
        ("structured_out", 0.3),
        ("structured_in", 0.3),
        ("movement", 0.5),
        ("imp", 0.8),
    ]
    results = [run_prune_experiment(m, a) for (m, a) in sweep]
    for r in results:
        print(r)
    append_results_csv(results, csv_path="results.csv")
    print("→ Appended to results.csv")