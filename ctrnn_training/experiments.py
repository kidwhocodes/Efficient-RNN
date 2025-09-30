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


def fresh_model(input_dim=3, hidden_size=128, output_dim=2, device="cpu"):
    return CTRNN(
        input_dim=input_dim, hidden_size=hidden_size, output_dim=output_dim,
        dt=100, tau=100, activation="relu",
        preact_noise=0.0, postact_noise=0.0,
        use_dale=False, ei_ratio=0.8,
        no_self_connections=True, scaling=1.0, bias=True
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
    strategy,
    amount,
    train_steps=600,
    ft_steps=200,
    last_only=True,
    seed=0,
    device="cpu",
    movement_batches=20,   # NEW: minibatches used to compute movement scores
    base_model=None,       # NEW: optional pre-trained model to clone
    task: str = "synthetic",
    **kwargs,              # future-proof: safely ignore extras from callers
):

    # --- pick data/task ---
    if task == "synthetic":
        cfg = SynthCfg(T=60, B=64, coh_levels=(0.0, 0.05, 0.1, 0.2), stim_std=0.6)
        data = SyntheticDM(cfg)
        # Prefer attributes if your SyntheticDM exposes them; otherwise fall back to constants
        input_dim = getattr(cfg, "input_dim", 3)
        output_dim = getattr(cfg, "output_dim", 2)

    elif task.startswith("ng:"):
        import neurogym as ngym
        from .neurogym_data import NeuroGymDM
        task_name = task.split("ng:", 1)[1]
        env = ngym.make(task_name)
        T, B = 60, 64     # dev-friendly; raise later for research runs
        data = NeuroGymDM(env, T=T, B=B, device=device)
        # NeuroGymDM should expose these:
        input_dim = getattr(data, "input_dim", None)
        output_dim = getattr(data, "n_classes", None)

    else:
        raise ValueError(f"unknown task: {task}")

    # Final safety: if any dim is still None, infer from a tiny sample
    if (input_dim is None) or (output_dim is None):
        X_tmp, Y_tmp = data.sample_batch()
        input_dim = X_tmp.size(-1)
        # If labels are last-only, this still works; otherwise you can also do Y_tmp.max()
        output_dim = int(max(2, int(Y_tmp.max().item()) + 1))

    # --- build model AFTER we know dims ---
    if base_model is None:
        model = fresh_model(input_dim=input_dim, hidden_size=64, output_dim=output_dim, device=device)
    else:
        model = deepcopy(base_model).to(device)

    PR.enforce_constraints(model)

    # --- Learning setup ---
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("DEBUG lr:", opt.param_groups[0]["lr"])
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
    elif strategy == "random_neuron":
        PR.prune_neurons_random(model, amount)
    elif strategy == "l1_neuron":
        PR.prune_neurons_l1(model, amount, combine="max")
    elif strategy == "movement_neuron":
        score = PR.movement_scores(
            model, data, device, criterion,
            batches=movement_batches,     # use the new argument
            last_only=last_only
        )
        PR.prune_neurons_movement(model, score, amount, combine="max")
        PR.enforce_constraints(model)
    elif strategy == "noise_synapse":
        PR.prune_noise_synapse(
            model, amount,
            leak=1.0,
            sigma=1.0,
            sim_dt=1e-2, sim_steps=20_000, sim_burnin=2_000,
            seed=seed,
            match_diagonal=True,
        )
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    # keep masks during fine-tune so zeros can't regrow
    PR.enforce_constraints(model)
    _ = train_epoch(model, data, device, opt, criterion, steps=ft_steps, last_only=last_only)

    # --- post metrics ---
    post_loss, post_acc = evaluate(model, data, device, criterion, steps=100, last_only=last_only)
    post_spars = recurrent_sparsity(model)
    post_alpha_rho = ctrnn_stability_proxy(model)
    post_np = neuron_pruning_stats(model)

    PR.finalize_pruning(model)

    # after you compute post_* metrics and PR.finalize_pruning(model)
    def _tensor_sparsity(x):
        n = x.numel()
        return 0.0 if n == 0 else float((x == 0).sum().item() / n)

    rec_sp = _tensor_sparsity(model.hidden_layer.weight)
    in_sp  = _tensor_sparsity(model.input_layer.weight)   if hasattr(model, "input_layer")  else None
    out_sp = _tensor_sparsity(model.readout_layer.weight) if hasattr(model, "readout_layer") else None

    row = {
        "seed": seed,
        "strategy": strategy,
        "amount": amount,
        "pre_loss": pre_loss, "pre_acc": pre_acc,
        "post_loss": post_loss, "post_acc": post_acc,
        "pre_sparsity": pre_spars, "post_sparsity": post_spars,
        "pre_alpha_rho": pre_alpha_rho, "post_alpha_rho": post_alpha_rho,
        "pre_rows_zero": pre_np["rows_zero"], "pre_cols_zero": pre_np["cols_zero"], "pre_isolated": pre_np["isolated"],
        "post_rows_zero": post_np["rows_zero"], "post_cols_zero": post_np["cols_zero"], "post_isolated": post_np["isolated"],
        "rec_sparsity": rec_sp,
    }
    if in_sp  is not None: row["in_sparsity"]  = in_sp
    if out_sp is not None: row["out_sparsity"] = out_sp
    return row
