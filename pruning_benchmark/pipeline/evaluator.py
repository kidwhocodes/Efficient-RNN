"""Minimal training/pruning loop for the benchmark."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from pruning_benchmark.models.ctrnn import CTRNN
from pruning_benchmark.pruning import (
    enforce_constraints,
    finalize_pruning,
    noise_prune_recurrent,
    prune_l1_unstructured,
    prune_random_unstructured,
    validate_prune_fraction,
)
from pruning_benchmark.tasks.modcog import ensure_modcog_env_id, estimate_modcog_T
from pruning_benchmark.tasks.neurogym import ModCogTrialDM, NeuroGymDatasetDM
from pruning_benchmark.tasks.synthetic import (
    SynthContextCfg,
    SynthHierContextCfg,
    SynthMultiRuleCfg,
    SynthNBackCfg,
    SyntheticContextDM,
    SyntheticHierContextDM,
    SyntheticMultiRuleDM,
    SyntheticNBackDM,
)

SYNTHETIC_TASKS: Mapping[str, Tuple[Any, Any]] = {
    "synthetic_multirule": (SynthMultiRuleCfg, SyntheticMultiRuleDM),
    "synthetic_hiercontext": (SynthHierContextCfg, SyntheticHierContextDM),
    "synthetic_context": (SynthContextCfg, SyntheticContextDM),
    "synthetic_nback": (SynthNBackCfg, SyntheticNBackDM),
}

REAL_TASK_DEFAULTS = {"T": 600, "B": 32}


def _dataclass_kwargs(cfg_cls, overrides: Mapping[str, Any]) -> Dict[str, Any]:
    if not is_dataclass(cfg_cls):
        return {}
    fields = getattr(cfg_cls, "__dataclass_fields__", {})
    return {name: overrides[name] for name in fields if name in overrides}


def build_task(
    name: str,
    *,
    options: Mapping[str, Any],
    device: torch.device,
    last_only: bool,
    seed: int,
):
    if name in SYNTHETIC_TASKS:
        cfg_cls, task_cls = SYNTHETIC_TASKS[name]
        cfg_kwargs = _dataclass_kwargs(cfg_cls, options)
        cfg = cfg_cls(**cfg_kwargs)
        return task_cls(cfg)

    opts = dict(REAL_TASK_DEFAULTS)
    opts.update(options or {})
    T = int(opts.get("T", REAL_TASK_DEFAULTS["T"])) if "T" in opts else None
    if T is not None and T <= 0:
        T = None
    B = int(opts.get("B", REAL_TASK_DEFAULTS["B"]))
    mask_last_k = int(opts.get("mask_last_k", 0))
    env_kwargs = dict(opts.get("env_kwargs", {}))
    dataset_kwargs = dict(opts.get("dataset_kwargs", {}))
    task_seed = int(opts.get("seed", seed))

    if name.startswith("modcog:"):
        if last_only:
            raise ValueError("Mod-Cog tasks do not support last_only=True; use full-sequence training/eval.")
        env_id = ensure_modcog_env_id(name)
        if env_id is None:
            raise ValueError(f"Unknown Mod_Cog task '{name}'.")
        env = env_id
        if T is None:
            try:
                import gymnasium as gym  # type: ignore
            except Exception:  # pragma: no cover
                try:
                    import gym  # type: ignore
                except Exception:
                    gym = None
            if gym is not None:
                try:
                    env_probe = gym.make(env_id, **env_kwargs)
                    T = estimate_modcog_T(env_probe)
                except Exception:
                    T = REAL_TASK_DEFAULTS["T"]
            else:
                T = REAL_TASK_DEFAULTS["T"]
    elif name.startswith("ng:"):
        env = name.split(":", 1)[1]
    else:
        raise ValueError(f"Unknown task specifier '{name}'.")

    if T is None:
        T = REAL_TASK_DEFAULTS["T"]
    if name.startswith("modcog:"):
        return ModCogTrialDM(
            env,
            T=T,
            B=B,
            device=str(device),
            last_only=last_only,
            seed=task_seed,
            env_kwargs=env_kwargs,
            mask_fixation=True,
        )
    return NeuroGymDatasetDM(
        env,
        T=T,
        B=B,
        device=str(device),
        last_only=last_only,
        seed=task_seed,
        env_kwargs=env_kwargs,
        dataset_kwargs=dataset_kwargs,
        mask_fixation=name.startswith("modcog:"),
        mask_last_k=mask_last_k,
    )


def constrain_feedforward(model: CTRNN, limit: float | None) -> None:
    if limit is None or limit <= 0:
        return
    with torch.no_grad():
        for layer_name in ("input_layer", "readout_layer"):
            layer = getattr(model, layer_name, None)
            if layer is None or not hasattr(layer, "weight"):
                continue
            layer.weight.data.clamp_(-limit, limit)
            if layer.bias is not None:
                layer.bias.data.clamp_(-limit, limit)


def train_baseline(
    task,
    *,
    input_dim: int,
    hidden_size: int,
    output_dim: int,
    activation: str,
    steps: int,
    device: torch.device,
    feedforward_limit: float | None,
) -> CTRNN:
    model = CTRNN(
        input_dim=input_dim,
        hidden_size=hidden_size,
        output_dim=output_dim,
        activation=activation,
    ).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    for _ in range(max(1, steps)):
        x, y = task.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits[-1], y[-1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        enforce_constraints(model)
        constrain_feedforward(model, feedforward_limit)
    return model


def evaluate(model: CTRNN, task, *, batches: int = 40, device: torch.device | None = None) -> Tuple[float, float]:
    model.eval()
    device = device or next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = task.sample_batch()
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits[-1], y[-1])
            total_loss += loss.item() * y[-1].numel()
            pred = logits[-1].argmax(dim=-1)
            total_correct += (pred == y[-1]).sum().item()
            total += y[-1].numel()
    return total_loss / max(1, total), total_correct / max(1, total)


def _apply_pruning(model: CTRNN, strategy: str, amount: float) -> Dict[str, float]:
    normalized = validate_prune_fraction(amount)
    strat = strategy.lower()
    if strat in {"random", "random_unstructured"}:
        prune_random_unstructured(model, normalized, include_feedforward=True)
        return {}
    if strat in {"l1", "l1_unstructured", "magnitude"}:
        prune_l1_unstructured(model, normalized, include_feedforward=True)
        return {}
    if strat == "noise_prune":
        return noise_prune_recurrent(model, normalized, include_feedforward=True)
    raise ValueError(f"Unsupported pruning strategy '{strategy}'.")


def _sample_dims(task) -> Tuple[int, int]:
    x, y = task.sample_batch()
    input_dim = x.size(-1)
    y_max = y.max().item()
    output_dim = int(y_max) + 1 if y_max >= 0 else int(y[-1].numel())
    return input_dim, output_dim


def _load_model(model_path: Path, model: CTRNN):
    state = torch.load(model_path, map_location=model.hidden_layer.weight.device)
    model.load_state_dict(state)
    return model


def run_suite(config_path: str) -> Dict:
    cfg = json.loads(Path(config_path).read_text())
    defaults = cfg.get("defaults", {})
    device = torch.device(defaults.get("device", "cpu"))
    results = []
    for run_cfg in cfg.get("runs", []):
        run = dict(defaults)
        run.update(run_cfg)
        task = build_task(
            run["task"],
            options=run.get("task_options", {}),
            device=device,
            last_only=bool(run.get("last_only", True)),
            seed=int(run.get("seed", 0)),
        )
        input_dim, output_dim = _sample_dims(task)
        hidden_size = int(run.get("hidden_size", 128))
        activation = run.get("activation", defaults.get("activation", "tanh"))
        train_steps = int(run.get("train_steps", defaults.get("train_steps", 300)))
        clip_limit = run.get("feedforward_clip", defaults.get("feedforward_clip"))
        strategy = run.get("strategy", "none")
        amount = float(run.get("amount", 0.0))

        if strategy == "none":
            model = train_baseline(
                task,
                input_dim=input_dim,
                hidden_size=hidden_size,
                output_dim=output_dim,
                activation=activation,
                steps=train_steps,
                device=device,
                feedforward_limit=clip_limit,
            )
        else:
            load_path = run.get("load_model_path")
            if load_path:
                model = CTRNN(
                    input_dim=input_dim,
                    hidden_size=hidden_size,
                    output_dim=output_dim,
                    activation=activation,
                ).to(device)
                _load_model(Path(load_path), model)
            else:
                model = train_baseline(
                    task,
                    input_dim=input_dim,
                    hidden_size=hidden_size,
                    output_dim=output_dim,
                    activation=activation,
                    steps=train_steps,
                    device=device,
                    feedforward_limit=clip_limit,
                )
            _apply_pruning(model, strategy, amount)
        finalize_pruning(model)

        save_path = run.get("save_model_path")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path)

        loss, acc = evaluate(model, task, device=device)
        results.append({"run_id": run["run_id"], "post_loss": loss, "post_acc": acc})
    return {"runs": results}
