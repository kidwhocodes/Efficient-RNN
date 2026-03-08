#!/usr/bin/env python3
"""Analyze calibration metrics (ECE, Brier) before/after pruning from a suite CSV.

This script replays pruning runs from a suite CSV using the stored baseline
checkpoints, computes calibration on fixed eval batches, and writes:
1) raw per-run metrics CSV
2) grouped summary CSV (strategy, amount)
3) line plots for post-ECE and post-Brier vs amount by strategy
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    raise ValueError("Could not infer hidden size from checkpoint.")


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
        attempts = max(1, max_resamples)
        for _ in range(attempts):
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


def _task_matches(task: str, include_substrs: Sequence[str], exclude_substrs: Sequence[str]) -> bool:
    low = task.lower()
    if include_substrs and not all(s in low for s in include_substrs):
        return False
    if exclude_substrs and any(s in low for s in exclude_substrs):
        return False
    return True


def _strategy_matches(strategy: str, keep: Optional[Sequence[str]]) -> bool:
    if not keep:
        return True
    return strategy in keep


def _ece_and_brier(
    model: torch.nn.Module,
    eval_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    n_classes: int,
    n_bins: int,
) -> Dict[str, float]:
    # Bin accumulators for ECE
    bin_counts = torch.zeros(n_bins, dtype=torch.float64)
    bin_conf = torch.zeros(n_bins, dtype=torch.float64)
    bin_acc = torch.zeros(n_bins, dtype=torch.float64)

    brier_sum = 0.0
    n_valid = 0

    model.eval()
    with torch.no_grad():
        for xb, yb in eval_batches:
            logits, _ = model(xb)  # [T,B,C]
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            valid = yb >= 0
            if not valid.any():
                continue

            p = probs[valid]  # [N,C]
            c = conf[valid]
            y = yb[valid].long()
            pr = pred[valid]
            correct = (pr == y).to(torch.float64)

            # ECE bins
            # conf in [0,1], right edge handled by clamp.
            bidx = torch.clamp((c * n_bins).long(), min=0, max=n_bins - 1)
            for i in range(n_bins):
                mask = bidx == i
                if mask.any():
                    cnt = int(mask.sum().item())
                    bin_counts[i] += cnt
                    bin_conf[i] += float(c[mask].sum().item())
                    bin_acc[i] += float(correct[mask].sum().item())

            # Multiclass Brier: sum_k (p_k - 1[y=k])^2
            onehot = torch.nn.functional.one_hot(y, num_classes=n_classes).to(p.dtype)
            sq = (p - onehot).pow(2).sum(dim=-1)
            brier_sum += float(sq.sum().item())
            n_valid += int(y.numel())

    ece = 0.0
    if n_valid > 0:
        for i in range(n_bins):
            cnt = int(bin_counts[i].item())
            if cnt == 0:
                continue
            mean_conf = float(bin_conf[i].item()) / cnt
            mean_acc = float(bin_acc[i].item()) / cnt
            ece += (cnt / n_valid) * abs(mean_acc - mean_conf)
    brier = brier_sum / max(1, n_valid)
    return {"ece": ece, "brier": brier, "n_valid": float(n_valid)}


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _plot_grouped(
    grouped: Dict[Tuple[str, float], List[float]],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for (strategy, amount), vals in grouped.items():
        if not vals:
            continue
        by_method[strategy].append((amount, float(np.mean(vals))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.8, 4.8))
    for method in sorted(by_method):
        pts = sorted(by_method[method], key=lambda x: x[0])
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("Amount pruned")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration analysis (ECE + Brier) for pruning suite rows.")
    parser.add_argument("--suite_csv", required=True)
    parser.add_argument("--output_raw_csv", required=True)
    parser.add_argument("--output_summary_csv", required=True)
    parser.add_argument("--plot_ece_out", required=True)
    parser.add_argument("--plot_brier_out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--ece_bins", type=int, default=15)
    parser.add_argument("--score_batch_max_resamples", type=int, default=10)
    parser.add_argument("--score_batch_min_valid", type=int, default=1)
    parser.add_argument("--prune_rng_seed", type=int, default=None)
    parser.add_argument(
        "--task_include",
        default="",
        help="Comma-separated substrings all required in task (e.g. ctx,dly,dm).",
    )
    parser.add_argument(
        "--task_exclude",
        default="",
        help="Comma-separated substrings excluded from task (e.g. seq).",
    )
    parser.add_argument(
        "--strategies",
        default="",
        help="Optional comma-separated strategy whitelist.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*env\\.gt.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*env\\.new_trial.*", category=UserWarning)

    include_substrs = tuple(s.strip().lower() for s in args.task_include.split(",") if s.strip())
    exclude_substrs = tuple(s.strip().lower() for s in args.task_exclude.split(",") if s.strip())
    keep_strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip()) or None

    rows = list(csv.DictReader(Path(args.suite_csv).open()))
    out_rows: List[Dict[str, str]] = []
    grouped_post_ece: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    grouped_post_brier: Dict[Tuple[str, float], List[float]] = defaultdict(list)

    for row in rows:
        task = (row.get("task") or "").strip()
        strategy = (row.get("strategy") or "").strip()
        if not task.startswith("modcog:"):
            continue
        if not _task_matches(task, include_substrs, exclude_substrs):
            continue
        if not _strategy_matches(strategy, keep_strategies):
            continue

        ckpt = (row.get("load_model_path") or "").strip()
        if not ckpt or not Path(ckpt).exists():
            continue
        env_id = ensure_modcog_env_id(task)
        if env_id is None:
            continue

        amount = _to_float(row.get("amount"), 0.0)
        seed = _to_int(row.get("seed"), 0)
        T = _to_int(row.get("ng_T"), 40)
        B = _to_int(row.get("ng_B"), 256)
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

        pre = _ece_and_brier(model, eval_batches, data.n_classes, args.ece_bins)

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
        post = _ece_and_brier(model, eval_batches, data.n_classes, args.ece_bins)

        out_rows.append(
            {
                "task": task,
                "strategy": strategy,
                "amount": f"{amount:.1f}",
                "pre_ece": f"{pre['ece']:.10f}",
                "post_ece": f"{post['ece']:.10f}",
                "delta_ece": f"{(post['ece'] - pre['ece']):.10f}",
                "pre_brier": f"{pre['brier']:.10f}",
                "post_brier": f"{post['brier']:.10f}",
                "delta_brier": f"{(post['brier'] - pre['brier']):.10f}",
                "n_valid": str(int(post["n_valid"])),
            }
        )
        grouped_post_ece[(strategy, amount)].append(post["ece"])
        grouped_post_brier[(strategy, amount)].append(post["brier"])

        if not args.quiet:
            print(
                f"{task}\t{strategy}\t{amount:.1f}\t"
                f"post_ece={post['ece']:.6f}\tpost_brier={post['brier']:.6f}"
            )

    if not out_rows:
        raise ValueError("No matching rows were processed. Check filters and paths.")

    raw_path = Path(args.output_raw_csv)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", newline="") as fh:
        fields = ["task", "strategy", "amount", "pre_ece", "post_ece", "delta_ece", "pre_brier", "post_brier", "delta_brier", "n_valid"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {raw_path}")

    summary_rows: List[Dict[str, str]] = []
    for key in sorted(set(list(grouped_post_ece.keys()) + list(grouped_post_brier.keys())), key=lambda x: (x[0], x[1])):
        strategy, amount = key
        ece_vals = grouped_post_ece.get(key, [])
        brier_vals = grouped_post_brier.get(key, [])
        ece_m, ece_s = _mean_std(ece_vals)
        brier_m, brier_s = _mean_std(brier_vals)
        summary_rows.append(
            {
                "strategy": strategy,
                "amount": f"{amount:.1f}",
                "n": str(max(len(ece_vals), len(brier_vals))),
                "post_ece_mean": f"{ece_m:.10f}",
                "post_ece_std": f"{ece_s:.10f}",
                "post_brier_mean": f"{brier_m:.10f}",
                "post_brier_std": f"{brier_s:.10f}",
            }
        )

    summary_path = Path(args.output_summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as fh:
        fields = ["strategy", "amount", "n", "post_ece_mean", "post_ece_std", "post_brier_mean", "post_brier_std"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {summary_path}")

    _plot_grouped(grouped_post_ece, "post ECE", "ECE vs pruning amount", Path(args.plot_ece_out))
    print(f"Wrote {args.plot_ece_out}")
    _plot_grouped(grouped_post_brier, "post Brier score", "Brier score vs pruning amount", Path(args.plot_brier_out))
    print(f"Wrote {args.plot_brier_out}")


if __name__ == "__main__":
    main()

