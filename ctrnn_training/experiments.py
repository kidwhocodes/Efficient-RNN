"""Experiment runners for evaluating pruning strategies on CTRNNs."""

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import json

from .core import CTRNN
from .data import SynthCfg, SyntheticDM
from .train_eval import train_epoch, evaluate
from .metrics import recurrent_sparsity, ctrnn_stability_proxy, neuron_pruning_stats
from . import pruning as PR
from typing import Optional

from .config import ExperimentConfig
from .utils import make_run_id, set_global_seed


def fresh_model(input_dim=3, hidden_size=128, output_dim=2, device="cpu"):
    return CTRNN(
        input_dim=input_dim, hidden_size=hidden_size, output_dim=output_dim,
        dt=10, tau=100, activation="tanh",
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
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new_file:
            w.writeheader()
        for r in results_list:
            w.writerow(r)


def run_prune_experiment(
    strategy,
    amount,
    train_steps=600,
    ft_steps=200,
    last_only=True,
    seed=0,
    device="cpu",
    movement_batches=20,
    base_model=None,
    task: str = "synthetic",
    no_prune: bool = False,
    run_id: Optional[str] = None,
    **kwargs,
):
    extra_kwargs = dict(kwargs)
    extra_kwargs.pop("run_id", None)
    kw_run_id = kwargs.pop("run_id", None)
    config = ExperimentConfig(
        strategy=strategy,
        amount=amount,
        train_steps=train_steps,
        ft_steps=ft_steps,
        last_only=last_only,
        seed=seed,
        device=device,
        movement_batches=movement_batches,
        task=task,
        no_prune=no_prune,
        run_id=run_id or kw_run_id or make_run_id(),
        extra=extra_kwargs,
    )
    set_global_seed(config.seed)
    # align local variables with the normalized config
    strategy = config.strategy
    amount = config.amount
    train_steps = config.train_steps
    ft_steps = config.ft_steps
    last_only = config.last_only
    seed = config.seed
    device = config.device
    movement_batches = config.movement_batches
    task = config.task
    no_prune = config.no_prune
    run_id = config.run_id

    hidden_size_override = kwargs.pop("hidden_size", None)
    # ---------- pick data/task ----------
    if task == "synthetic":
        cfg = SynthCfg(T=60, B=64, coh_levels=(0.0, 0.05, 0.1, 0.2), stim_std=0.6)
        data = SyntheticDM(cfg)
        input_dim = getattr(cfg, "input_dim", 3)
        output_dim = getattr(cfg, "output_dim", 2)
    elif task.startswith("ng:"):
        import neurogym as ngym
        from .neurogym_data import NeuroGymDM
        task_name = task.split("ng:", 1)[1]
        if not any(task_name.endswith(sfx) for sfx in ("-v0", "-v1", "-v2", "-v3")):
            task_name = f"{task_name}-v0"

        # ----- parse ng_kwargs from CLI / code -----
        env_kwargs = kwargs.pop("ng_kwargs", None)
        if env_kwargs is None:
            env_kwargs = {}
        elif isinstance(env_kwargs, str):
            try:
                env_kwargs = json.loads(env_kwargs)
            except json.JSONDecodeError as e:
                raise ValueError(f"ng_kwargs must be JSON if passed as a string. Got: {env_kwargs}") from e
        elif not isinstance(env_kwargs, dict):
            raise TypeError(f"ng_kwargs must be a dict or JSON string, got {type(env_kwargs)}")

        env = ngym.make(task_name, **env_kwargs)

        # ----- allow T/B overrides; choose harder defaults -----
        T = kwargs.pop("ng_T", None)
        B = kwargs.pop("ng_B", None)

        if T is None: T = 400
        if B is None: B = 64

        # supervise the whole sequence to make it harder
        data = NeuroGymDM(env, T=T, B=B, device=device, last_only=False, seed=seed)

        input_dim = getattr(data, "input_dim", None)
        output_dim = getattr(data, "n_classes", None)
    else:
        raise ValueError(f"unknown task: {task}")

    # Fallback: infer dims from a small sample if missing
    if (input_dim is None) or (output_dim is None):
        X_tmp, Y_tmp = data.sample_batch()
        input_dim = X_tmp.size(-1)
        output_dim = int(max(2, int(Y_tmp.max().item()) + 1))

    # ---------- build model AFTER we know dims ----------
    if base_model is None:
        hs = 128 if hidden_size_override is None else int(hidden_size_override)
        model = fresh_model(
            input_dim=input_dim,
            hidden_size=hs,
            output_dim=output_dim,
            device=device,
        )
    else:
        model = deepcopy(base_model).to(device)
    PR.enforce_constraints(model)

    # ---------- opt/loss ----------
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ---------- PHASE A: before any training ----------
    pre0_loss, pre0_acc = evaluate(model, data, device, criterion, steps=50, last_only=last_only)

    # ---------- PHASE B: baseline train -> PRE metrics ----------
    _ = train_epoch(model, data, device, opt, criterion, steps=train_steps, last_only=last_only)
    pre_loss, pre_acc = evaluate(model, data, device, criterion, steps=100, last_only=last_only)
    pre_spars = recurrent_sparsity(model)
    pre_alpha_rho = ctrnn_stability_proxy(model)
    pre_np = neuron_pruning_stats(model)

    # ---------- PHASE C: prune (or skip) and measure POST0 ----------
    if not no_prune and strategy != "none":
        prune_last_only = kwargs.get("last_only", last_only)
        if strategy == "l1_neuron":
            PR.prune_neurons_l1(model, amount, combine="max")
        elif strategy == "movement_neuron":
            score = PR.movement_scores(
                model, data, device, criterion,
                batches=movement_batches, last_only=last_only
            )
            PR.prune_neurons_movement(model, score, amount, combine="max")
        elif strategy == "random_neuron":
            PR.prune_neurons_random(model, amount)
        elif strategy == "l1_unstructured":
            PR.prune_l1_recurrent(model, amount)
        elif strategy == "global_unstructured":
            PR.prune_global_unstructured(model, amount, include_readout=False, include_input=False)
        elif strategy == "noise_synapse":
            PR.prune_noise_synapse(
                model, amount,
                leak=1.0, sigma=1.0,
                sim_dt=1e-2, sim_steps=20_000, sim_burnin=2_000,
                seed=seed, match_diagonal=True,
            )
        elif strategy == "synflow":
            from .pruning import synflow_scores, mask_from_scores
            scores = synflow_scores(model)
            P = mask_from_scores(model, scores, amount, structured=False, global_=True)
        elif strategy == "fisher":
            from .pruning import fisher_diag_scores, mask_from_scores
            scores = fisher_diag_scores(model, data_loader_fn=lambda: data.sample_batch(), batches=4)
            P = mask_from_scores(model, scores, amount, structured=False, global_=True)
        elif strategy == "activity_neuron":
            from .pruning import activity_neuron_scores, neuron_mask_from_scores
            ns = activity_neuron_scores(model, data_loader_fn=lambda: data.sample_batch(), batches=4)
            P = neuron_mask_from_scores(model, ns, amount)   # your helper should expand unit scores to in/rec/out masks
        elif strategy == "snip":
            score = PR.snip_scores(model, data, device, criterion, batches=1, last_only=last_only)
            PR.prune_snip_recurrent(model, score, amount)
        elif strategy == "global_unstructured_all":
            PR.prune_global_unstructured(model, amount, include_readout=True, include_input=True)
        elif strategy == "noise_probe":
            from .pruning import noise_probe_scores, mask_from_scores
            batches = kwargs.get("score_batches", 4)
            scores = noise_probe_scores(model, data_loader_fn=lambda: data.sample_batch(), batches=batches, sigma=0.05, last_only=prune_last_only)
            mask_from_scores(model, scores, amount)
        elif strategy == "oja_synapse":
            PR.prune_oja_recurrent(
                model,
                data_loader_fn=lambda: data.sample_batch(),
                amount=amount,
                batches=kwargs.get("score_batches", 4),
                last_only=prune_last_only,
            )
        elif strategy == "variational_dropout":
            PR.prune_variational_dropout(
                model,
                data_loader_fn=lambda: data.sample_batch(),
                amount=amount,
                batches=kwargs.get("score_batches", 4),
                last_only=prune_last_only,
                eps=kwargs.get("vd_eps", 1e-8),
            )
        elif strategy == "stdp_synapse":
            PR.prune_stdp_synapse(
                model,
                data_loader_fn=lambda: data.sample_batch(),
                amount=amount,
                batches=kwargs.get("score_batches", 4),
                lag=kwargs.get("stdp_lag", 1),
                combine=kwargs.get("stdp_mode", "causal"),
                center=bool(kwargs.get("stdp_center", True)),
            )
        elif strategy == "turnover_synapse":
            PR.prune_turnover_synapse(
                model,
                amount=amount,
                regrow_frac=kwargs.get("turnover_regrow", 0.1),
                regrow_scale=kwargs.get("turnover_scale", 0.1),
                seed=seed,
            )
        elif strategy == "energy_neuron":
            PR.prune_energy_neuron(
                model,
                data_loader_fn=lambda: data.sample_batch(),
                amount=amount,
                batches=kwargs.get("score_batches", 4),
                beta=kwargs.get("energy_beta", 0.5),
                eps=kwargs.get("energy_eps", 1e-6),
            )
        elif strategy == "noise_combo":
            from .pruning import combined_noise_neuron_scores, neuron_mask_from_scores
            batches = kwargs.get("score_batches", 4)
            last_only = prune_last_only
            fisher_w = kwargs.get("fisher_w", 1.0)
            noiseprobe_w = kwargs.get("noiseprobe_w", 1.0)
            activity_w = kwargs.get("activity_w", 1.0)
            reduce = kwargs.get("reduce", "sumabs")
            debug_scores = kwargs.get("debug_scores", False)

            if debug_scores:
                combo, parts = combined_noise_neuron_scores(
                    model, data_loader_fn=lambda: data.sample_batch(),
                    batches=batches, sigma=0.05, last_only=last_only,
                    fisher_w=fisher_w, noiseprobe_w=noiseprobe_w, activity_w=activity_w,
                    reduce=reduce, debug=True
                )
                # simple stats
                for k, v in parts.items():
                    print(f"[noise_combo] {k}: mean={v.mean().item():.4f}, std={v.std(unbiased=False).item():.4f}")
                # pairwise correlations (Pearson)
                pk = list(parts.keys())
                for i in range(len(pk)):
                    for j in range(i+1, len(pk)):
                        vi, vj = parts[pk[i]], parts[pk[j]]
                        # corrcoef on centered
                        ci = (vi - vi.mean())
                        cj = (vj - vj.mean())
                        corr = (ci * cj).mean() / (ci.std(unbiased=False) * cj.std(unbiased=False) + 1e-8)
                        print(f"[noise_combo] corr({pk[i]}, {pk[j]}) = {corr.item():.4f}")
                ns = combo
            else:
                ns = combined_noise_neuron_scores(
                    model, data_loader_fn=lambda: data.sample_batch(),
                    batches=batches, sigma=0.05, last_only=last_only,
                    fisher_w=fisher_w, noiseprobe_w=noiseprobe_w, activity_w=activity_w,
                    reduce=reduce, debug=False
                )
            neuron_mask_from_scores(model, ns, amount)
        elif strategy == "homeostatic_neuron":
            PR.prune_neurons_homeostatic(
                model,
                data_loader_fn=lambda: data.sample_batch(),
                amount=amount,
                batches=kwargs.get("score_batches", 4),
                target=kwargs.get("homeo_target", 0.05),
                activity_mode=kwargs.get("homeo_mode", "relu"),
                var_weight=kwargs.get("homeo_var_weight", 0.0),
            )
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        PR.enforce_constraints(model)
        post0_loss, post0_acc = evaluate(model, data, device, criterion, steps=100, last_only=last_only)
    else:
        # control path: no pruning
        post0_loss, post0_acc = pre_loss, pre_acc

    # ---------- PHASE D: fine-tune -> POST metrics ----------
    if ft_steps > 0:
        _ = train_epoch(model, data, device, opt, criterion, steps=ft_steps, last_only=last_only)
    post_loss, post_acc = evaluate(model, data, device, criterion, steps=100, last_only=last_only)

    # finalize so weights are truly sparse for logging/checkpointing
    PR.finalize_pruning(model)

    # ---------- layerwise sparsities ----------
    def _tensor_sparsity(x: torch.Tensor) -> float:
        n = x.numel()
        return 0.0 if n == 0 else float((x == 0).sum().item() / n)

    rec_sp = _tensor_sparsity(model.hidden_layer.weight)
    in_sp  = _tensor_sparsity(model.input_layer.weight)   if hasattr(model, "input_layer")  else None
    out_sp = _tensor_sparsity(model.readout_layer.weight) if hasattr(model, "readout_layer") else None

    # ---------- post structural stats ----------
    post_spars = recurrent_sparsity(model)
    post_alpha_rho = ctrnn_stability_proxy(model)
    post_np = neuron_pruning_stats(model)

    # ---------- row ----------
    row = {
        "seed": seed,
        "task": task,
        "strategy": strategy if not no_prune else f"{strategy}+no_prune",
        "amount": amount,
        "run_id": run_id,
        "pre0_loss": pre0_loss, "pre0_acc": pre0_acc,
        "pre_loss": pre_loss,   "pre_acc": pre_acc,
        "post0_loss": post0_loss, "post0_acc": post0_acc,
        "post_loss": post_loss,   "post_acc": post_acc,
        "pre_sparsity": pre_spars, "post_sparsity": post_spars,
        "pre_alpha_rho": pre_alpha_rho, "post_alpha_rho": post_alpha_rho,
        "pre_rows_zero": pre_np["rows_zero"], "pre_cols_zero": pre_np["cols_zero"], "pre_isolated": pre_np["isolated"],
        "post_rows_zero": post_np["rows_zero"], "post_cols_zero": post_np["cols_zero"], "post_isolated": post_np["isolated"],
        "rec_sparsity": rec_sp,
    }
    if in_sp  is not None: row["in_sparsity"]  = in_sp
    if out_sp is not None: row["out_sparsity"] = out_sp
    if config.extra:
        row["config_extra"] = json.dumps(config.extra, default=str, sort_keys=True)
    return row
