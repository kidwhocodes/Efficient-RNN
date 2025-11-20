"""Experiment runners for evaluating pruning strategies on CTRNNs."""

from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..analysis import compile_run_metrics, save_metrics, snapshot_model
from ..config import ExperimentConfig
from ..data import (
    NeuroGymDM,
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
from ..models import CTRNN, GRUNet
from ..pruning import (
    PRUNE_AMOUNT_STEP,
    apply_pruning,
    enforce_constraints,
    fisher_diag_scores,
    finalize_pruning,
    movement_scores,
    validate_prune_fraction,
    snip_scores,
    synflow_scores,
)
from ..training import evaluate, train_epoch
from ..utils import make_run_id, set_global_seed


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ensure_run_directory(run_id: str) -> Path:
    path = Path("results") / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    return {}


def _format_yaml_scalar(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps(value, sort_keys=True)


def _yaml_lines(data: Any, indent: int = 0) -> List[str]:
    prefix = "  " * indent
    if isinstance(data, dict):
        lines: List[str] = []
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {_format_yaml_scalar(value)}")
        return lines
    if isinstance(data, list):
        lines: List[str] = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 1))
            else:
                lines.append(f"{prefix}- {_format_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_format_yaml_scalar(data)}"]


def _write_config_snapshot(run_dir: Path, snapshot: Dict[str, Any]) -> Dict[str, Path]:
    json_path = run_dir / "config.json"
    json_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    yaml_path = run_dir / "config.yaml"
    yaml_content = "\n".join(["---"] + _yaml_lines(snapshot)) + "\n"
    yaml_path.write_text(yaml_content)
    return {"json": json_path, "yaml": yaml_path}


def _extract_prune_kwargs(strategy: str, options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prune_kwargs: Dict[str, Any] = {}
    prune_meta: Dict[str, Any] = {}
    if strategy == "noise_prune":
        sigma = float(options.pop("noise_sigma", 1.0))
        eps = float(options.pop("noise_eps", 0.3))
        leak_shift = float(options.pop("noise_leak_shift", 0.0))
        matched_diagonal = bool(options.pop("noise_matched_diagonal", True))
        rng_seed = options.pop("noise_rng_seed", None)
        prune_kwargs.update({
            "sigma": sigma,
            "eps": eps,
            "leak_shift": leak_shift,
            "matched_diagonal": matched_diagonal,
        })
        prune_meta.update(prune_kwargs)
        if rng_seed is not None:
            rng_seed = int(rng_seed)
            prune_meta["rng_seed"] = rng_seed
            prune_kwargs["rng"] = np.random.default_rng(rng_seed)
    else:
        for key in (
            "noise_sigma",
            "noise_eps",
            "noise_leak_shift",
            "noise_matched_diagonal",
            "noise_rng_seed",
        ):
            options.pop(key, None)
    return prune_kwargs, prune_meta


# ---------------------------------------------------------------------------
# Seeding helpers and lightweight dataset/model factories
# ---------------------------------------------------------------------------


@contextmanager
def temporary_seed(seed: Optional[int]):
    if seed is None:
        yield
        return
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    else:
        cuda_states = None
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def fresh_model(
    input_dim=3,
    hidden_size=128,
    output_dim=2,
    device="cpu",
    *,
    model_type: str = "ctrnn",
    **model_kwargs,
):
    model_type = model_type.lower()
    if model_type == "ctrnn":
        activation = model_kwargs.get("activation", "tanh")
        return CTRNN(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            dt=10,
            tau=100,
            activation=activation,
            preact_noise=0.0,
            postact_noise=0.0,
            use_dale=model_kwargs.get("use_dale", False),
            ei_ratio=model_kwargs.get("ei_ratio", 0.8),
            no_self_connections=model_kwargs.get("no_self_connections", True),
            scaling=1.0,
            bias=True,
        ).to(device)
    if model_type == "gru":
        return GRUNet(input_dim, hidden_size, output_dim).to(device)
    raise ValueError(f"Unknown model_type '{model_type}'")


def append_results_csv(results_list: Iterable[Dict[str, Any]], csv_path: str = "results.csv"):
    import csv

    rows = list(results_list)
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    existing_rows: List[Dict[str, Any]] = []
    existing_fields: List[str] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []

    all_rows = existing_rows + rows
    keys = sorted({k for row in all_rows for k in row.keys() if k is not None})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        def _sanitize(row: Dict[str, Any]) -> Dict[str, Any]:
            sanitized = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list)):
                    sanitized[key] = json.dumps(value, sort_keys=True)
                else:
                    sanitized[key] = value
            return sanitized

        for r in existing_rows:
            writer.writerow(_sanitize(r))
        for r in rows:
            writer.writerow(_sanitize(r))


def evaluate_on_fixed_batches(
    model: torch.nn.Module,
    batches,
    criterion,
    *,
    dataset_last_only: bool,
    eval_last_only: bool | None,
) -> Dict[str, float]:
    prev_mode = model.training
    model.eval()
    if eval_last_only is None:
        eval_last_only = dataset_last_only
    total_loss = 0.0
    total_loss_weight = 0
    total_decision_correct = 0
    total_decision_count = 0
    total_seq_correct = 0
    total_seq_count = 0
    with torch.no_grad():
        for x_batch, y_batch in batches:
            logits, _ = model(x_batch)
            decision_logits = logits[-1]
            decision_targets = y_batch[-1]
            decision_loss = criterion(decision_logits, decision_targets)
            decision_N = decision_targets.numel()

            if eval_last_only:
                loss_val = decision_loss
                loss_weight = decision_N
            else:
                seq_logits = logits.view(-1, logits.size(-1))
                seq_targets = y_batch.view(-1)
                loss_val = criterion(seq_logits, seq_targets)
                loss_weight = seq_targets.numel()

            total_loss += float(loss_val) * loss_weight
            total_loss_weight += loss_weight

            decision_correct = (decision_logits.argmax(dim=-1) == decision_targets).sum().item()
            total_decision_correct += int(decision_correct)
            total_decision_count += int(decision_N)

            seq_correct = (logits.argmax(dim=-1) == y_batch).sum().item()
            seq_total = y_batch.numel()
            total_seq_correct += int(seq_correct)
            total_seq_count += int(seq_total)
    if prev_mode:
        model.train()
    mean_loss = total_loss / max(1, total_loss_weight)
    decision_acc = total_decision_correct / max(1, total_decision_count)
    sequence_acc = total_seq_correct / max(1, total_seq_count)
    return {"loss": mean_loss, "acc": decision_acc, "acc_sequence": sequence_acc}


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def run_prune_experiment(
    strategy,
    amount,
    train_steps=600,
    ft_steps=200,
    last_only=True,
    model_type="ctrnn",
    eval_last_only=None,
    seed=0,
    device="cpu",
    movement_batches=20,
    base_model=None,
    task: str = "synthetic",
    no_prune: bool = False,
    run_id: Optional[str] = None,
    return_model: bool = False,
    **kwargs,
):
    save_model_path = kwargs.pop("save_model_path", None)
    load_model_path = kwargs.pop("load_model_path", None)
    skip_training = bool(kwargs.pop("skip_training", False))
    eval_seed_base = kwargs.pop("eval_seed", None)
    eval_sample_batches = int(kwargs.pop("eval_sample_batches", 0))
    eval_steps_pre0 = kwargs.pop("eval_steps_pre0", 50)
    eval_steps_pre = kwargs.pop("eval_steps_pre", 100)
    eval_steps_post0 = kwargs.pop("eval_steps_post0", 100)
    eval_steps_post = kwargs.pop("eval_steps_post", 100)
    hidden_size_override = kwargs.pop("hidden_size", None)
    model_kwargs = {}
    for key in ("use_dale", "ei_ratio", "no_self_connections", "activation"):
        if key in kwargs:
            model_kwargs[key] = kwargs.pop(key)

    ng_kwargs_raw = kwargs.pop("ng_kwargs", None)
    ng_T = kwargs.pop("ng_T", None)
    ng_B = kwargs.pop("ng_B", None)

    prune_kwargs, prune_meta = _extract_prune_kwargs(strategy, kwargs)
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unsupported keyword arguments for run_prune_experiment: {unknown}")

    resolved_run_id = run_id or make_run_id()
    config = ExperimentConfig(
        strategy=strategy,
        amount=float(amount),
        train_steps=int(train_steps),
        ft_steps=int(ft_steps),
        last_only=bool(last_only),
        model_type=str(model_type),
        seed=int(seed),
        device=device,
        movement_batches=int(movement_batches),
        task=task,
        no_prune=bool(no_prune),
        run_id=resolved_run_id,
    )

    pruned = not config.no_prune and config.strategy != "none"
    normalized_amount = validate_prune_fraction(config.amount) if pruned else 0.0
    config.amount = normalized_amount

    if eval_sample_batches > 0 and eval_seed_base is None:
        eval_seed_base = config.seed

    if eval_last_only is None:
        eval_last_only = config.last_only

    set_global_seed(config.seed)
    run_dir = _ensure_run_directory(config.run_id)

    # ------------------------------------------------------------------
    # Build dataset/task
    # ------------------------------------------------------------------
    task_meta: Dict[str, Any] = {"task": task, "last_only_eval": bool(last_only)}
    env_kwargs: Dict[str, Any] = {}
    if task == "synthetic":
        cfg = SynthCfg(T=60, B=64, coh_levels=(0.0, 0.05, 0.1, 0.2), stim_std=0.6)
        data = SyntheticDM(cfg)
        task_meta.update(_dataclass_to_dict(cfg))
        input_dim = cfg.input_dim
        output_dim = cfg.output_dim
    elif task == "synthetic_context":
        cfg = SynthContextCfg(T=40, B=64)
        data = SyntheticContextDM(cfg)
        task_meta.update(_dataclass_to_dict(cfg))
        input_dim = cfg.input_dim
        output_dim = cfg.output_dim
    elif task == "synthetic_multirule":
        cfg = SynthMultiRuleCfg(T=60, B=64)
        data = SyntheticMultiRuleDM(cfg)
        task_meta.update(_dataclass_to_dict(cfg))
        input_dim = cfg.input_dim
        output_dim = cfg.output_classes
    elif task == "synthetic_hiercontext":
        cfg = SynthHierContextCfg(T=70, B=64)
        data = SyntheticHierContextDM(cfg)
        task_meta.update(_dataclass_to_dict(cfg))
        input_dim = cfg.input_dim
        output_dim = cfg.output_dim
    elif task == "synthetic_nback":
        cfg = SynthNBackCfg(T=50, B=64)
        data = SyntheticNBackDM(cfg)
        task_meta.update(_dataclass_to_dict(cfg))
        input_dim = cfg.input_dim
        output_dim = cfg.output_dim
    elif task.startswith("ng:"):
        import warnings

        warnings.filterwarnings("ignore", message=".*render_modes.*")
        warnings.filterwarnings("ignore", message=".*env.seed.*")
        warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

        task_name = task.split("ng:", 1)[1]
        if not any(task_name.endswith(sfx) for sfx in ("-v0", "-v1", "-v2", "-v3")):
            task_name = f"{task_name}-v0"

        if ng_kwargs_raw is None:
            env_kwargs = {}
        elif isinstance(ng_kwargs_raw, str):
            try:
                env_kwargs = json.loads(ng_kwargs_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"ng_kwargs must be JSON-serialisable: {ng_kwargs_raw}") from exc
        elif isinstance(ng_kwargs_raw, dict):
            env_kwargs = dict(ng_kwargs_raw)
        else:
            raise TypeError(f"ng_kwargs must be dict or JSON string, got {type(ng_kwargs_raw)}")

        import neurogym as ngym

        env = ngym.make(task_name, **env_kwargs)
        T = int(ng_T) if ng_T is not None else 400
        B = int(ng_B) if ng_B is not None else 64
        data = NeuroGymDM(env, T=T, B=B, device=device, last_only=last_only, seed=config.seed)
        task_meta.update({
            "env": task_name,
            "T": T,
            "B": B,
            "dataset_last_only": bool(last_only),
            "env_kwargs": env_kwargs,
        })
        input_dim = data.input_dim
        output_dim = data.n_classes
    else:
        raise ValueError(f"Unknown task: {task}")

    if input_dim is None or output_dim is None:
        X_tmp, Y_tmp = data.sample_batch()
        input_dim = X_tmp.size(-1)
        output_dim = int(max(2, int(Y_tmp.max().item()) + 1))

    fixed_batches = None
    dataset_last_only_flag = bool(getattr(data, "last_only", last_only))

    if eval_sample_batches > 0:
        fixed_batches = []
        seed_offset = 9
        base_seed = None if eval_seed_base is None else int(eval_seed_base) * 10 + seed_offset
        with temporary_seed(base_seed):
            for _ in range(eval_sample_batches):
                x_batch, y_batch = data.sample_batch()
                fixed_batches.append((x_batch.to(device), y_batch.to(device)))

    state_dict_cached = None
    if load_model_path is not None:
        try:
            state_dict_cached = torch.load(load_model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict_cached = torch.load(load_model_path, map_location=device)
        if hidden_size_override is None and state_dict_cached is not None:
            hh_weight = state_dict_cached.get("hidden_layer.weight")
            if hh_weight is None:
                hh_weight = state_dict_cached.get("gru.weight_hh_l0")
            if hh_weight is not None:
                hidden_size_override = hh_weight.shape[0]

    # ------------------------------------------------------------------
    # Build model and optimiser
    # ------------------------------------------------------------------
    if base_model is None:
        hidden_size = hidden_size_override or 128
        model = fresh_model(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            output_dim=output_dim,
            device=device,
            model_type=model_type,
            **model_kwargs,
        )
    else:
        model = deepcopy(base_model).to(device)

    if load_model_path is not None:
        state = state_dict_cached
        if state is None:
            try:
                state = torch.load(load_model_path, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(load_model_path, map_location=device)
        model.load_state_dict(state)

    enforce_constraints(model)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    def run_eval(steps: int, offset: int) -> Tuple[float, float]:
        if fixed_batches is not None:
            return evaluate_on_fixed_batches(
                model,
                fixed_batches,
                criterion,
                dataset_last_only=dataset_last_only_flag,
                eval_last_only=eval_last_only,
            )
        seed_val = None if eval_seed_base is None else int(eval_seed_base) * 10 + offset
        with temporary_seed(seed_val):
            return evaluate(
                model,
                data,
                device,
                criterion,
                steps=steps,
                dataset_last_only=dataset_last_only_flag,
                eval_last_only=eval_last_only,
            )

    # ------------------------------------------------------------------
    # Phase A: baseline evaluation
    # ------------------------------------------------------------------
    pre0_metrics = run_eval(eval_steps_pre0, 0)
    pre0_loss = pre0_metrics["loss"]
    pre0_acc = pre0_metrics["acc"]
    pre0_snapshot = snapshot_model(model)

    # ------------------------------------------------------------------
    # Phase B: baseline training
    # ------------------------------------------------------------------
    if not skip_training and train_steps > 0:
        train_epoch(model, data, device, opt, criterion, steps=train_steps, last_only=last_only)
        pre_metrics = run_eval(eval_steps_pre, 1)
        pre_loss = pre_metrics["loss"]
        pre_acc = pre_metrics["acc"]
        pre_snapshot = snapshot_model(model)
    else:
        pre_metrics = dict(pre0_metrics)
        pre_loss, pre_acc = pre0_loss, pre0_acc
        pre_snapshot = dict(pre0_snapshot)

    if save_model_path is not None:
        os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_model_path, _use_new_zipfile_serialization=True)

    prune_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Phase C: pruning
    # ------------------------------------------------------------------
    if pruned:
        score_batches: List[tuple[torch.Tensor, torch.Tensor]] = []
        needs_batches = strategy in {"movement", "movement_neuron", "snip", "fisher"}
        if needs_batches:
            num_batches = max(1, movement_batches)
            for _ in range(num_batches):
                xb, yb = data.sample_batch()
                score_batches.append((xb.to(device), yb.to(device)))

        if strategy in {"movement", "movement_neuron"}:
            scores = movement_scores(model, score_batches, criterion, last_only=last_only)
            prune_kwargs["scores"] = scores
        elif strategy == "snip":
            scores = snip_scores(model, score_batches, criterion, last_only=last_only)
            prune_kwargs["scores"] = scores
        elif strategy == "fisher":
            scores = fisher_diag_scores(model, score_batches, criterion, last_only=last_only)
            prune_kwargs["scores"] = scores
        elif strategy == "synflow":
            prune_kwargs["scores"] = synflow_scores(model)

        prune_stats = apply_pruning(model, strategy, normalized_amount, **prune_kwargs)
        post0_metrics = run_eval(eval_steps_post0, 2)
        post0_loss = post0_metrics["loss"]
        post0_acc = post0_metrics["acc"]
        post0_snapshot = snapshot_model(model)
    else:
        post0_metrics = dict(pre_metrics)
        post0_loss, post0_acc = pre_loss, pre_acc
        post0_snapshot = dict(pre_snapshot)

    # ------------------------------------------------------------------
    # Phase D: optional fine-tuning
    # ------------------------------------------------------------------
    if ft_steps > 0:
        train_epoch(model, data, device, opt, criterion, steps=ft_steps, last_only=last_only)
    post_metrics = run_eval(eval_steps_post, 3)
    post_loss = post_metrics["loss"]
    post_acc = post_metrics["acc"]
    post_snapshot = snapshot_model(model)

    finalize_pruning(model)

    # ------------------------------------------------------------------
    # Assemble metrics and config snapshots
    # ------------------------------------------------------------------
    phase_metrics = {
        "pre0": {**pre0_snapshot, "loss": pre0_loss, "acc": pre0_acc},
        "pre": {**pre_snapshot, "loss": pre_loss, "acc": pre_acc},
        "post0": {**post0_snapshot, "loss": post0_loss, "acc": post0_acc},
        "post": {**post_snapshot, "loss": post_loss, "acc": post_acc},
    }
    # merge additional evaluation metrics (e.g., sequence accuracies)
    for name, metrics in (
        ("pre0", pre0_metrics),
        ("pre", pre_metrics),
        ("post0", post0_metrics),
        ("post", post_metrics),
    ):
        for key, value in metrics.items():
            if key in {"loss", "acc"}:
                continue
            phase_metrics[name][key] = value

    extras = {
        "delta_post0_acc": pre_acc - post0_acc,
        "delta_post_acc": pre_acc - post_acc,
        "pruned": bool(pruned),
        "amount": normalized_amount,
        "amount_step": PRUNE_AMOUNT_STEP,
        "ft_steps": ft_steps,
        "train_steps": train_steps,
    }
    metrics_report = compile_run_metrics(phase_metrics, extras=extras)
    metrics_path = save_metrics(run_dir, metrics_report)

    model_meta = {
        "class": type(model).__name__,
        "model_type": model_type,
        "input_dim": getattr(model, "I", None),
        "hidden_size": getattr(model, "H", None),
        "output_dim": getattr(model, "O", None),
        "alpha": getattr(model, "alpha", None),
        "activation": getattr(model, "_activation_name", None),
        "preact_noise": getattr(model, "preact_noise", None),
        "postact_noise": getattr(model, "postact_noise", None),
        "use_dale": bool(getattr(model, "use_dale", False)),
        "no_self_connections": bool(getattr(model, "no_self_connections", False)),
    }

    config_metadata = config.to_metadata()
    config_metadata.update({
        "skip_training": bool(skip_training),
        "eval_seed": eval_seed_base,
        "eval_sample_batches": eval_sample_batches,
        "eval_steps_pre0": eval_steps_pre0,
        "eval_steps_pre": eval_steps_pre,
        "eval_steps_post0": eval_steps_post0,
        "eval_steps_post": eval_steps_post,
        "eval_last_only": bool(eval_last_only),
        "hidden_size_override": hidden_size_override,
        "movement_batches": movement_batches,
        "model_kwargs": model_kwargs,
    })
    if save_model_path is not None:
        config_metadata["save_model_path"] = save_model_path
    if load_model_path is not None:
        config_metadata["load_model_path"] = load_model_path

    config_snapshot = {
        "experiment": config_metadata,
        "task": task_meta,
        "model": model_meta,
        "pruning": {
            "strategy": strategy,
            "amount": normalized_amount,
            "step": PRUNE_AMOUNT_STEP,
            "applied": bool(pruned),
            "options": prune_meta,
        },
    }
    config_paths = _write_config_snapshot(run_dir, config_snapshot)

    row: Dict[str, Any] = {
        **config_metadata,
        **metrics_report,
        "run_dir": str(run_dir),
        "config_json": str(config_paths["json"]),
        "config_yaml": str(config_paths["yaml"]),
        "metrics_json": str(metrics_path),
    }
    if prune_stats:
        for key, value in prune_stats.items():
            row[f"prune_{key}"] = value

    if return_model:
        return row, model.cpu()
    return row


__all__ = ["append_results_csv", "fresh_model", "run_prune_experiment", "temporary_seed"]
