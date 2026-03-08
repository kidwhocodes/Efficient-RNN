#!/usr/bin/env python3
"""Analyze pre/post-pruning prediction confidence from a pruning suite CSV."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pruning_benchmark.experiments.runner import fresh_model, temporary_seed
from pruning_benchmark.pruning import PruneContext, get_pruner
from pruning_benchmark.tasks.modcog import ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import ModCogTrialDM


def _to_int(v, default: int) -> int:
    try:
        if v is None or v == "":
            return default
        return int(v)
    except Exception:
        return default


def _to_float(v, default: float) -> float:
    try:
        if v is None or v == "":
            return default
        out = float(v)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _infer_hidden_size(state: Dict[str, torch.Tensor]) -> int:
    if "hidden_layer.weight" in state:
        return int(state["hidden_layer.weight"].shape[0])
    for key in ("gru.weight_hh_l0", "lstm.weight_hh_l0", "rnn.weight_hh_l0"):
        if key in state:
            return int(state[key].shape[1])
    raise ValueError("Could not infer hidden size from checkpoint state_dict.")


def _extract_prune_kwargs(row: Dict[str, str], strategy: str, prune_seed: Optional[int]) -> Dict[str, object]:
    if strategy != "noise_prune":
        return {}
    kwargs: Dict[str, object] = {
        "sigma": _to_float(row.get("prune_sigma"), 1.0),
        "eps": _to_float(row.get("prune_eps"), 0.3),
        "leak_shift": _to_float(row.get("prune_leak_shift"), 0.0),
        "matched_diagonal": True,
    }
    if prune_seed is not None:
        kwargs["rng"] = np.random.default_rng(int(prune_seed))
    return kwargs


def _sample_score_batches(
    data: ModCogTrialDM,
    num: int,
    device: str,
    max_resamples: int,
    min_valid: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(max(0, num)):
        chosen = None
        for _ in range(max(1, max_resamples)):
            xb, yb = data.sample_batch()
            valid = int((yb >= 0).sum().item())
            if valid >= min_valid:
                chosen = (xb, yb)
                break
            if chosen is None:
                chosen = (xb, yb)
        xb, yb = chosen
        out.append((xb.to(device), yb.to(device)))
    return out


def _confidence_stats(
    model: torch.nn.Module,
    eval_batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, float]:
    total_conf = 0.0
    total_entropy = 0.0
    total_n = 0
    conf_correct = 0.0
    n_correct = 0
    conf_wrong = 0.0
    n_wrong = 0

    model.eval()
    with torch.no_grad():
        for xb, yb in eval_batches:
            logits, _ = model(xb)  # [T, B, C]
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            valid = yb >= 0
            if not valid.any():
                continue
            conf_v = conf[valid]
            pred_v = pred[valid]
            y_v = yb[valid]
            p_v = probs[valid]
            entropy = -(p_v * torch.log(p_v.clamp_min(1e-12))).sum(dim=-1)

            n = int(conf_v.numel())
            total_conf += float(conf_v.sum().item())
            total_entropy += float(entropy.sum().item())
            total_n += n

            correct_mask = pred_v == y_v
            if correct_mask.any():
                conf_correct += float(conf_v[correct_mask].sum().item())
                n_correct += int(correct_mask.sum().item())
            wrong_mask = ~correct_mask
            if wrong_mask.any():
                conf_wrong += float(conf_v[wrong_mask].sum().item())
                n_wrong += int(wrong_mask.sum().item())

    return {
        "mean_confidence": total_conf / max(1, total_n),
        "mean_entropy": total_entropy / max(1, total_n),
        "mean_confidence_correct": conf_correct / max(1, n_correct),
        "mean_confidence_wrong": conf_wrong / max(1, n_wrong),
        "acc_sequence": n_correct / max(1, total_n),
        "n_valid": float(total_n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre/post confidence analysis from suite CSV.")
    parser.add_argument("--suite_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--plot_out", default=None, help="Optional mean confidence-vs-amount plot path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--score_batch_max_resamples", type=int, default=10)
    parser.add_argument("--score_batch_min_valid", type=int, default=1)
    parser.add_argument("--prune_rng_seed", type=int, default=None)
    parser.add_argument("--strategy", default=None, help="Optional strategy filter.")
    parser.add_argument("--task_substr", default=None, help="Optional task substring filter (e.g., 'seq').")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*env\\.gt.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*env\\.new_trial.*", category=UserWarning)

    rows = list(csv.DictReader(Path(args.suite_csv).open()))
    out_rows: List[Dict[str, str]] = []
    grouped_delta: Dict[Tuple[str, float], List[float]] = defaultdict(list)

    for row in rows:
        strategy = (row.get("strategy") or "").strip()
        if args.strategy and strategy != args.strategy:
            continue
        task = (row.get("task") or "").strip()
        if args.task_substr and args.task_substr not in task:
            continue
        ckpt = (row.get("load_model_path") or "").strip()
        if not task.startswith("modcog:") or not ckpt or not Path(ckpt).exists():
            continue

        env_id = ensure_modcog_env_id(task)
        if env_id is None:
            continue

        T = _to_int(row.get("ng_T"), 40)
        B = _to_int(row.get("ng_B"), 256)
        seed = _to_int(row.get("seed"), 0)
        amount = _to_float(row.get("amount"), 0.0)
        data = ModCogTrialDM(
            env_id,
            T=T,
            B=B,
            device=args.device,
            last_only=False,
            seed=seed,
            mask_fixation=True,
        )

        state = torch.load(ckpt, map_location=args.device)
        hidden = _infer_hidden_size(state)
        model_type = (row.get("model_type") or "ctrnn").strip()
        model = fresh_model(
            input_dim=data.input_dim,
            hidden_size=hidden,
            output_dim=data.n_classes,
            device=args.device,
            model_type=model_type,
        )
        model.load_state_dict(state)

        with temporary_seed(seed * 10 + 9):
            eval_batches = [(xb.to(args.device), yb.to(args.device)) for xb, yb in (data.sample_batch() for _ in range(max(1, args.eval_batches)))]

        pre = _confidence_stats(model, eval_batches)

        pruner = get_pruner(strategy)
        score_count = pruner.resolved_batch_count(_to_int(row.get("movement_batches"), 20))
        with temporary_seed(seed * 10 + 11):
            score_batches = _sample_score_batches(
                data,
                score_count,
                args.device,
                max_resamples=args.score_batch_max_resamples,
                min_valid=args.score_batch_min_valid,
            ) if score_count > 0 else None

        context = PruneContext(
            model=model,
            amount=amount,
            criterion=torch.nn.CrossEntropyLoss(ignore_index=-1),
            last_only=False,
            device=args.device,
            batches=score_batches,
            metadata={"phase": "post", "run_id": row.get("run_id", "")},
        )
        prune_kwargs = _extract_prune_kwargs(row, strategy, args.prune_rng_seed)
        pruner.run(context, **prune_kwargs)
        post = _confidence_stats(model, eval_batches)

        delta_conf = post["mean_confidence"] - pre["mean_confidence"]
        grouped_delta[(strategy, amount)].append(delta_conf)
        out_rows.append({
            "task": task,
            "strategy": strategy,
            "amount": f"{amount:.1f}",
            "pre_confidence": f"{pre['mean_confidence']:.8f}",
            "post_confidence": f"{post['mean_confidence']:.8f}",
            "delta_confidence": f"{delta_conf:.8f}",
            "pre_entropy": f"{pre['mean_entropy']:.8f}",
            "post_entropy": f"{post['mean_entropy']:.8f}",
            "pre_acc_sequence": f"{pre['acc_sequence']:.8f}",
            "post_acc_sequence": f"{post['acc_sequence']:.8f}",
            "pre_confidence_wrong": f"{pre['mean_confidence_wrong']:.8f}",
            "post_confidence_wrong": f"{post['mean_confidence_wrong']:.8f}",
            "n_valid": str(int(pre["n_valid"])),
        })
        if not args.quiet:
            print(f"{task}\t{strategy}\t{amount:.1f}\tdelta_conf={delta_conf:.6f}")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        fieldnames = [
            "task",
            "strategy",
            "amount",
            "pre_confidence",
            "post_confidence",
            "delta_confidence",
            "pre_entropy",
            "post_entropy",
            "pre_acc_sequence",
            "post_acc_sequence",
            "pre_confidence_wrong",
            "post_confidence_wrong",
            "n_valid",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out_path}")

    if args.plot_out and grouped_delta:
        by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for (strategy, amount), vals in grouped_delta.items():
            by_method[strategy].append((amount, float(np.mean(vals))))
        p = Path(args.plot_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.6, 4.8))
        for method in sorted(by_method):
            pts = sorted(by_method[method], key=lambda x: x[0])
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            plt.plot(xs, ys, marker="o", label=method)
        plt.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)
        plt.xlabel("Amount pruned")
        plt.ylabel("Delta confidence (post - pre)")
        plt.title("Prediction confidence shift vs pruning amount")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()

