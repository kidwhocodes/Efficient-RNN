#!/usr/bin/env python3
"""Diagnostics for recurrence usage in Mod_Cog RNN baselines."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn

from pruning_benchmark.experiments.runner import fresh_model
from pruning_benchmark.tasks.modcog import resolve_modcog_callable
from pruning_benchmark.tasks.neurogym import NeuroGymDatasetDM
from pruning_benchmark.training.loops import evaluate


def _build_modcog_dataset(
    task: str,
    *,
    T: int,
    B: int,
    device: str,
    last_only: bool,
    seed: int,
    env_kwargs: Dict | None = None,
    dataset_kwargs: Dict | None = None,
    mask_last_k: int = 0,
):
    env_suffix = task.split("modcog:", 1)[1].strip()
    env_kwargs = dict(env_kwargs or {})
    dataset_kwargs = dict(dataset_kwargs or {})
    builder_info = resolve_modcog_callable(env_suffix)
    if builder_info is not None:
        _, builder_fn = builder_info
        dataset_env_source = builder_fn(**env_kwargs)
        dataset_env_kwargs = None
    else:
        env_id = env_suffix if env_suffix.lower().startswith("mod_cog") else f"Mod_Cog-{env_suffix}"
        dataset_env_source = env_id
        dataset_env_kwargs = env_kwargs

    return NeuroGymDatasetDM(
        dataset_env_source,
        T=T,
        B=B,
        device=device,
        last_only=last_only,
        seed=seed,
        env_kwargs=dataset_env_kwargs,
        dataset_kwargs=dataset_kwargs,
        mask_last_k=mask_last_k,
    )


def _get_recurrent_params(model: nn.Module) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if hasattr(model, "rnn"):
        return model.rnn.weight_hh_l0, model.rnn.weight_ih_l0
    if hasattr(model, "hidden_layer") and hasattr(model, "input_layer"):
        return model.hidden_layer.weight, model.input_layer.weight
    return None, None


def _grad_norm(param: torch.Tensor | None) -> float:
    if param is None or param.grad is None:
        return 0.0
    return float(param.grad.detach().norm().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose recurrence usage for Mod_Cog RNNs.")
    parser.add_argument("--task", default="modcog:ctxdlydm2intl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", default="rnn", choices=["ctrnn", "gru", "lstm", "rnn"])
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ng_T", type=int, default=400)
    parser.add_argument("--ng_B", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_last_k", type=int, default=0)
    parser.add_argument("--last_only", action="store_true", help="Use last-only targets from dataset.")
    parser.add_argument("--window_k", type=int, default=150)
    parser.add_argument("--eval_steps", type=int, default=20)
    args = parser.parse_args()

    if not args.task.startswith("modcog:"):
        raise ValueError("This diagnostic currently supports modcog:* tasks only.")

    data = _build_modcog_dataset(
        args.task,
        T=args.ng_T,
        B=args.ng_B,
        device=args.device,
        last_only=args.last_only,
        seed=args.seed,
        env_kwargs=None,
        dataset_kwargs=None,
        mask_last_k=args.mask_last_k,
    )
    input_dim = data.input_dim
    output_dim = data.n_classes

    model = fresh_model(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        output_dim=output_dim,
        device=args.device,
        model_type=args.model_type,
    )
    model.load(args.checkpoint, map_location=args.device)

    print("== Model ==")
    readout_in = getattr(model.readout_layer, "in_features", None)
    print(f"model_type: {args.model_type}")
    print(f"input_dim: {input_dim}, hidden_size: {args.hidden_size}, output_dim: {output_dim}")
    print(f"readout_in_features: {readout_in}")
    print(f"direct input->readout: {readout_in == input_dim}")

    X, Y = data.sample_batch()
    T = X.size(0)
    k = min(args.window_k, T)
    last_abs = X[-1].abs().mean().item()
    lastk_abs = X[-k:].abs().mean().item()
    lastk_nz = (X[-k:].abs() > 1e-6).float().mean().item()
    print("\n== Input activity (decision window) ==")
    print(f"last_step_abs_mean: {last_abs:.6f}")
    print(f"last_{k}_abs_mean: {lastk_abs:.6f}")
    print(f"last_{k}_nonzero_frac: {lastk_nz:.6f}")

    if Y.size(0) > 1:
        pre = Y[:-1]
        pre_nonzero = (pre != 0).float().mean().item()
        pre_match_last = (pre == Y[-1].unsqueeze(0)).float().mean().item()
    else:
        pre_nonzero = 0.0
        pre_match_last = 0.0
    print("\n== Target leakage checks ==")
    print(f"pre_step_nonzero_frac: {pre_nonzero:.6f}")
    print(f"pre_step_matches_final_frac: {pre_match_last:.6f}")

    print("\n== Hidden dynamics ==")
    with torch.no_grad():
        logits, hidden_seq = model(X)
        hidden_delta = (hidden_seq[1:] - hidden_seq[:-1]).norm(dim=-1).mean().item()
        hidden_std = hidden_seq.std(dim=0).mean().item()
    print(f"mean_hidden_delta: {hidden_delta:.6f}")
    print(f"mean_hidden_std: {hidden_std:.6f}")

    print("\n== Evaluation ==")
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(
        model,
        data,
        args.device,
        criterion,
        steps=args.eval_steps,
        dataset_last_only=args.last_only,
        eval_last_only=args.last_only,
        response_window_k=args.window_k,
    )
    for key in ("acc", "acc_sequence", "acc_response_window"):
        if key in metrics:
            print(f"{key}: {metrics[key]:.6f}")

    print("\n== Gradient check ==")
    model.train()
    X, Y = data.sample_batch()
    logits, _ = model(X)
    if args.last_only:
        loss = criterion(logits[-1], Y[-1])
    else:
        loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
    for p in model.parameters():
        p.grad = None
    loss.backward()
    rec_param, in_param = _get_recurrent_params(model)
    print(f"grad_norm_input: {_grad_norm(in_param):.6f}")
    print(f"grad_norm_recurrent: {_grad_norm(rec_param):.6f}")

    print("\n== Recurrent ablation (one batch) ==")
    if rec_param is not None:
        with torch.no_grad():
            saved = rec_param.clone()
            rec_param.zero_()
        with torch.no_grad():
            logits, _ = model(X)
            preds = logits[-1].argmax(dim=-1)
            acc = (preds == Y[-1]).float().mean().item()
        with torch.no_grad():
            rec_param.copy_(saved)
        print(f"acc_with_recurrent_zeroed: {acc:.6f}")
    else:
        print("recurrent_param: not found")


if __name__ == "__main__":
    main()
