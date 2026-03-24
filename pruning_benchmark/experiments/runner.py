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
from ..tasks import (
    ModCogTrialDM,
    NeuroGymDM,
    NeuroGymDatasetDM,
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
from ..tasks.modcog import resolve_modcog_callable
from ..models import CTRNN, GRUNet
from ..pruning import (
    PRUNE_AMOUNT_STEP,
    PruneContext,
    enforce_constraints,
    finalize_pruning,
    get_pruner,
    validate_prune_fraction,
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
    if strategy == "obs":
        damping = float(options.pop("obs_damping", 1e-3))
        num_samples = int(options.pop("obs_num_samples", 4))
        cg_iters = int(options.pop("obs_cg_iters", 50))
        prune_kwargs.update({
            "damping": damping,
            "num_samples": num_samples,
            "cg_iters": cg_iters,
        })
        prune_meta.update({
            "obs_damping": damping,
            "obs_num_samples": num_samples,
            "obs_cg_iters": cg_iters,
        })
    else:
        for key in ("obs_damping", "obs_num_samples", "obs_cg_iters"):
            options.pop(key, None)
    if strategy == "woodfisher":
        damping = float(options.pop("woodfisher_damping", 1e-3))
        prune_kwargs["damping"] = damping
        prune_meta["woodfisher_damping"] = damping
    else:
        options.pop("woodfisher_damping", None)
    return prune_kwargs, prune_meta


def _coerce_kwargs(payload: Any, label: str) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"{label} must be JSON-serialisable: {payload}") from exc
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"{label} must be a dict or JSON string, got {type(payload)}")


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
    if model_type == "lstm":
        from ..models import LSTMNet

        return LSTMNet(input_dim, hidden_size, output_dim).to(device)
    if model_type == "rnn":
        from ..models import RNNNet

        return RNNNet(input_dim, hidden_size, output_dim).to(device)
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
            decision_valid = decision_targets >= 0
            decision_N = int(decision_valid.sum().item())
            decision_loss = criterion(decision_logits, decision_targets)

            if eval_last_only:
                loss_val = decision_loss
                loss_weight = decision_N
            else:
                seq_logits = logits.view(-1, logits.size(-1))
                seq_targets = y_batch.view(-1)
                loss_val = criterion(seq_logits, seq_targets)
                loss_weight = int((seq_targets >= 0).sum().item())

            if loss_weight > 0:
                total_loss += float(loss_val) * loss_weight
                total_loss_weight += loss_weight

            decision_pred = decision_logits.argmax(dim=-1)
            if decision_N > 0:
                decision_correct = ((decision_pred == decision_targets) & decision_valid).sum().item()
                total_decision_correct += int(decision_correct)
                total_decision_count += int(decision_N)

            seq_pred = logits.argmax(dim=-1)
            seq_valid = y_batch >= 0
            seq_total = int(seq_valid.sum().item())
            if seq_total > 0:
                seq_correct = ((seq_pred == y_batch) & seq_valid).sum().item()
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
    lr = float(kwargs.pop("lr", 1e-3))
    clip = float(kwargs.pop("clip", 1.0))
    hidden_size_override = kwargs.pop("hidden_size", None)
    model_kwargs = {}
    for key in ("use_dale", "ei_ratio", "no_self_connections", "activation"):
        if key in kwargs:
            model_kwargs[key] = kwargs.pop(key)

    ng_kwargs_raw = kwargs.pop("ng_kwargs", None)
    ng_T = kwargs.pop("ng_T", None)
    ng_B = kwargs.pop("ng_B", None)
    ng_dataset_kwargs_raw = kwargs.pop("ng_dataset_kwargs", None)
    score_batch_max_resamples = int(kwargs.pop("score_batch_max_resamples", 10) or 10)
    score_batch_min_valid = int(kwargs.pop("score_batch_min_valid", 1) or 1)

    prune_phase = kwargs.pop("prune_phase", "post")
    if prune_phase not in {"pre", "post"}:
        raise ValueError("prune_phase must be 'pre' or 'post'")
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
        prune_phase=prune_phase,
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
    uses_ng_env = task.startswith("ng:")
    uses_modcog = task.startswith("modcog:")
    env_kwargs = _coerce_kwargs(ng_kwargs_raw, "ng_kwargs") if (uses_ng_env or uses_modcog) else None
    dataset_kwargs = _coerce_kwargs(ng_dataset_kwargs_raw, "ng_dataset_kwargs") if uses_modcog else None
    if uses_modcog and last_only:
        raise ValueError("Mod-Cog tasks do not support last_only=True; use full-sequence training/eval.")

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
    elif uses_ng_env:
        import warnings

        warnings.filterwarnings("ignore", message=".*render_modes.*")
        warnings.filterwarnings("ignore", message=".*env.seed.*")
        warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

        task_name = task.split("ng:", 1)[1]
        if not any(task_name.endswith(sfx) for sfx in ("-v0", "-v1", "-v2", "-v3")):
            task_name = f"{task_name}-v0"

        import neurogym as ngym

        env = ngym.make(task_name, **(env_kwargs or {}))
        T = int(ng_T) if ng_T is not None else 400
        B = int(ng_B) if ng_B is not None else 64
        data = NeuroGymDM(env, T=T, B=B, device=device, last_only=last_only, seed=config.seed)
        task_meta.update({
            "env": task_name,
            "T": T,
            "B": B,
            "dataset_last_only": bool(last_only),
            "env_kwargs": env_kwargs or {},
            "backend": "gym_env",
        })
        input_dim = data.input_dim
        output_dim = data.n_classes
    elif uses_modcog:
        env_suffix = task.split("modcog:", 1)[1].strip()
        if not env_suffix:
            raise ValueError("Mod_Cog task identifier missing after 'modcog:' prefix.")
        env_kwargs_copy = dict(env_kwargs or {})
        dataset_kwargs = dict(dataset_kwargs or {})
        try:
            builder_info = resolve_modcog_callable(env_suffix)
        except ImportError as exc:
            raise ImportError(
                "Mod_Cog tasks requested but the package is not installed. "
                "Install it from https://github.com/mikailkhona/Mod_Cog (pip install -e .)."
            ) from exc
        T = int(ng_T) if ng_T is not None else 400
        B = int(ng_B) if ng_B is not None else 64
        if builder_info is not None:
            canonical_name, builder_fn = builder_info
            env_id = f"Mod_Cog-{canonical_name}-v0"
            env_label = canonical_name
            dataset_backend = "mod_cog_builder"
            dataset_env_source = builder_fn(**env_kwargs_copy)
            dataset_env_kwargs = None
        else:
            env_id = env_suffix if env_suffix.lower().startswith("mod_cog") else f"Mod_Cog-{env_suffix}"
            env_label = env_id
            dataset_backend = "mod_cog_dataset"
            dataset_env_source = env_id
            dataset_env_kwargs = env_kwargs_copy
        data = ModCogTrialDM(
            dataset_env_source,
            T=T,
            B=B,
            device=device,
            last_only=last_only,
            seed=config.seed,
            env_kwargs=dataset_env_kwargs,
            mask_fixation=True,
        )
        task_meta.update({
            "env": env_label,
            "env_id": env_id,
            "T": T,
            "B": B,
            "dataset_last_only": bool(last_only),
            "env_kwargs": env_kwargs_copy,
            "backend": dataset_backend,
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
    def _infer_hidden_from_state(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
        hh_weight = state_dict.get("hidden_layer.weight")
        if hh_weight is not None:
            return int(hh_weight.shape[0])
        gru_hh = state_dict.get("gru.weight_hh_l0")
        if gru_hh is not None:
            return int(gru_hh.shape[1])
        lstm_hh = state_dict.get("lstm.weight_hh_l0")
        if lstm_hh is not None:
            return int(lstm_hh.shape[1])
        rnn_hh = state_dict.get("rnn.weight_hh_l0")
        if rnn_hh is not None:
            return int(rnn_hh.shape[1])
        return None

    if load_model_path is not None:
        try:
            state_dict_cached = torch.load(load_model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict_cached = torch.load(load_model_path, map_location=device)
        if hidden_size_override is None and state_dict_cached is not None:
            inferred = _infer_hidden_from_state(state_dict_cached)
            if inferred is not None:
                hidden_size_override = inferred

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

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    pruner = get_pruner(strategy) if pruned else None

    def _batch_valid_count(yb: torch.Tensor, use_last_only: bool) -> int:
        if use_last_only:
            return int((yb[-1] != -1).sum().item())
        return int((yb != -1).sum().item())

    def sample_batches(num: int | None):
        if not num or num <= 0:
            return None
        batches: List[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num):
            chosen_x = None
            chosen_y = None
            attempts = max(1, score_batch_max_resamples)
            for _attempt in range(attempts):
                xb, yb = data.sample_batch()
                if _batch_valid_count(yb, last_only) >= score_batch_min_valid:
                    chosen_x, chosen_y = xb, yb
                    break
                if chosen_x is None:
                    # Keep first draw as a fallback to avoid infinite loops.
                    chosen_x, chosen_y = xb, yb
            batches.append((chosen_x.to(device), chosen_y.to(device)))
        return batches

    pre_prune_stats: Dict[str, Any] = {}
    pruned_pretraining = False
    if pruned and prune_phase == "pre":
        if skip_training or load_model_path is not None:
            raise ValueError("Cannot use prune_phase='pre' with pre-trained checkpoints or skip_training=True.")
        if pruner is None or not getattr(pruner, "supports_pretrain", False):
            raise ValueError(f"Strategy '{strategy}' does not support prune_phase='pre'.")
        batch_count = pruner.resolved_batch_count(movement_batches)
        score_batches = sample_batches(batch_count)
        context = PruneContext(
            model=model,
            amount=normalized_amount,
            criterion=criterion,
            last_only=last_only,
            device=device,
            batches=score_batches,
            metadata={"phase": "pre", "run_id": config.run_id},
        )
        stats = pruner.pretrain(context, **prune_kwargs)
        pre_prune_stats = dict(stats) if stats else {}
        pruned_pretraining = True

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
        train_epoch(
            model,
            data,
            device,
            opt,
            criterion,
            steps=train_steps,
            last_only=last_only,
            clip=clip,
        )
        pre_metrics = run_eval(eval_steps_pre, 1)
        pre_loss = pre_metrics["loss"]
        pre_acc = pre_metrics["acc"]
        pre_snapshot = snapshot_model(model)
    else:
        pre_metrics = dict(pre0_metrics)
        pre_loss, pre_acc = pre0_loss, pre0_acc
        pre_snapshot = dict(pre0_snapshot)

    prune_stats_post: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Phase C: pruning
    # ------------------------------------------------------------------
    if pruned and prune_phase == "post":
        batch_count = pruner.resolved_batch_count(movement_batches)
        score_batches: Optional[List[tuple[torch.Tensor, torch.Tensor]]] = None
        if batch_count > 0:
            score_batches = sample_batches(batch_count)

        context = PruneContext(
            model=model,
            amount=normalized_amount,
            criterion=criterion,
            last_only=last_only,
            device=device,
            batches=score_batches,
            metadata={"phase": "post", "run_id": config.run_id},
        )
        prune_stats_post = pruner.run(context, **prune_kwargs)
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
        train_epoch(
            model,
            data,
            device,
            opt,
            criterion,
            steps=ft_steps,
            last_only=last_only,
            clip=clip,
        )
        post_metrics = run_eval(eval_steps_post, 3)
        post_loss = post_metrics["loss"]
        post_acc = post_metrics["acc"]
        post_snapshot = snapshot_model(model)
    else:
        # When no fine-tuning occurs, the post-prune model is identical to post0.
        post_metrics = dict(post0_metrics)
        post_loss = post0_loss
        post_acc = post0_acc
        post_snapshot = dict(post0_snapshot)

    finalize_pruning(model)

    if save_model_path is not None:
        os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_model_path, _use_new_zipfile_serialization=True)

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
        "prune_phase": prune_phase,
        "pruned_pretraining": pruned_pretraining,
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
        "lr": lr,
        "clip": clip,
        "hidden_size_override": hidden_size_override,
        "movement_batches": movement_batches,
        "model_kwargs": model_kwargs,
        "ng_T": task_meta.get("T", ng_T),
        "ng_B": task_meta.get("B", ng_B),
        "ng_kwargs": ng_kwargs_raw,
        "ng_dataset_kwargs": ng_dataset_kwargs_raw,
        "score_batch_max_resamples": score_batch_max_resamples,
        "score_batch_min_valid": score_batch_min_valid,
    })
    if strategy == "noise_prune":
        config_metadata.update({
            "noise_sigma": prune_meta.get("sigma"),
            "noise_eps": prune_meta.get("eps"),
            "noise_leak_shift": prune_meta.get("leak_shift"),
            "noise_matched_diagonal": prune_meta.get("matched_diagonal"),
            "noise_rng_seed": prune_meta.get("rng_seed"),
            # Keep compatibility with existing analysis scripts that read prune_* columns.
            "prune_sigma": prune_meta.get("sigma"),
            "prune_eps": prune_meta.get("eps"),
            "prune_leak_shift": prune_meta.get("leak_shift"),
            "prune_matched_diagonal": prune_meta.get("matched_diagonal"),
            "prune_rng_seed": prune_meta.get("rng_seed"),
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
    combined_prune_stats = dict(pre_prune_stats)
    combined_prune_stats.update(prune_stats_post)
    if combined_prune_stats:
        for key, value in combined_prune_stats.items():
            row[f"prune_{key}"] = value

    if return_model:
        return row, model.cpu()
    return row


__all__ = ["append_results_csv", "fresh_model", "run_prune_experiment", "temporary_seed"]
