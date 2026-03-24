#!/usr/bin/env python3
"""Estimate finite-time maximum Lyapunov exponent (MLE) from pruning CSV rows.

This script reconstructs each model from its baseline checkpoint, reapplies the
recorded pruning method/amount, and measures pre/post finite-time MLE using a
shadow-trajectory estimator under identical task inputs.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.analysis.replay import (
    resolve_modcog_B,
    resolve_modcog_T,
    resolve_noise_prune_kwargs,
    resolve_row_seed,
    resolve_score_batch_setting,
)
from pruning_benchmark.experiments.runner import fresh_model
from pruning_benchmark.pruning.pruners import PruneContext, get_pruner
from pruning_benchmark.tasks.modcog import ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import ModCogTrialDM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument(
        "--methods",
        default="noise_prune,l1_unstructured,random_unstructured,obd",
        help="Comma-separated methods to analyze.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--score_batches", type=int, default=20, help="Fallback batches for score-based pruners.")
    p.add_argument("--eval_batches", type=int, default=4, help="Batches for MLE estimation.")
    p.add_argument("--delta0", type=float, default=1e-5, help="Initial perturbation norm.")
    p.add_argument("--seed", type=int, default=None, help="Optional override for dataset/eval seed.")
    p.add_argument("--output_csv", default="", help="Optional output CSV path.")
    p.add_argument("--plot_out", default="", help="Optional output PNG path (post_mle vs amount).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-row output; print summary only.")
    p.add_argument("--score_batch_max_resamples", type=int, default=10)
    p.add_argument("--score_batch_min_valid", type=int, default=1)
    return p.parse_args()


def _to_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v: str, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_bool(v: str, default: bool = True) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return default


def _infer_hidden_size(state: Dict[str, torch.Tensor]) -> int:
    if "hidden_layer.weight" in state:
        return int(state["hidden_layer.weight"].shape[0])
    if "gru.weight_hh_l0" in state:
        return int(state["gru.weight_hh_l0"].shape[1])
    if "lstm.weight_hh_l0" in state:
        return int(state["lstm.weight_hh_l0"].shape[1])
    raise ValueError("Could not infer hidden size from checkpoint.")


def _sample_score_batches(
    data: ModCogTrialDM,
    n: int,
    device: str,
    *,
    max_resamples: int,
    min_valid: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(max(0, n)):
        chosen = None
        attempts = max(1, int(max_resamples))
        for _a in range(attempts):
            xb, yb = data.sample_batch()
            valid = int((yb != -1).sum().item())
            if chosen is None:
                chosen = (xb, yb)
            if valid >= int(min_valid):
                chosen = (xb, yb)
                break
        xb, yb = chosen
        out.append((xb.to(device), yb.to(device)))
    return out


@torch.no_grad()
def finite_time_mle(model, x: torch.Tensor, delta0: float = 1e-5, seed: int = 0) -> float:
    """Shadow-trajectory MLE estimate under fixed input sequence x (T,B,I)."""
    model.eval()
    T, B, _ = x.shape
    fr1, v1 = model.init_state(B, x.device)
    fr2, v2 = model.init_state(B, x.device)

    # Random perturbation in state (v-space), normalized per sample.
    gen = torch.Generator(device=v1.device)
    gen.manual_seed(int(seed))
    d = torch.randn(v1.shape, generator=gen, device=v1.device, dtype=v1.dtype)
    d = d / (torch.linalg.norm(d, dim=1, keepdim=True) + 1e-12)
    d = d * float(delta0)
    v2 = v1 + d
    fr2 = model.act(v2)

    log_sum = torch.zeros(B, device=x.device)
    d0 = float(delta0)
    eps = 1e-12

    for t in range(T):
        fr1, v1 = model.step(fr1, v1, x[t])
        fr2, v2 = model.step(fr2, v2, x[t])
        dv = v2 - v1
        dist = torch.linalg.norm(dv, dim=1) + eps
        log_sum += torch.log(dist / d0)

        # Renormalize perturbation to keep within linear neighborhood.
        dv = dv / dist.unsqueeze(1) * d0
        v2 = v1 + dv
        fr2 = model.act(v2)

    mle_per_sample = log_sum / max(1, T)
    return float(mle_per_sample.mean().item())


def _extract_prune_kwargs(row: Dict[str, str], strategy: str) -> Dict[str, object]:
    if strategy != "noise_prune":
        return {}
    return resolve_noise_prune_kwargs(row)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise SystemExit(f"Missing CSV: {in_path}")

    warnings.filterwarnings(
        "ignore",
        message=r".*env\.new_trial to get variables from other wrappers is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*env\.ob to get variables from other wrappers is deprecated.*",
        category=UserWarning,
    )

    aliases = {"fischer": "fisher"}
    methods = set()
    for m in args.methods.split(","):
        mm = m.strip()
        if not mm:
            continue
        methods.add(aliases.get(mm, mm))
    rows = list(csv.DictReader(in_path.open()))
    selected = []
    for r in rows:
        s = (r.get("strategy") or "").strip()
        s = aliases.get(s, s)
        if s in methods:
            r["strategy"] = s
            selected.append(r)
    if not selected:
        raise SystemExit("No matching rows for requested methods.")

    out_rows: List[Dict[str, str]] = []
    grouped: Dict[Tuple[str, float], List[float]] = {}
    if not args.quiet:
        print("task\tstrategy\tamount\tpre_mle\tpost_mle\tdelta_mle")

    for row in selected:
        task = (row.get("task") or "").strip()
        strategy = (row.get("strategy") or "").strip()
        amount = _to_float(row.get("amount", ""), 0.0)
        ckpt = (row.get("load_model_path") or "").strip()
        if not task or not ckpt or not Path(ckpt).exists():
            continue

        env_id = ensure_modcog_env_id(task)
        T = resolve_modcog_T(row, task)
        B = resolve_modcog_B(row)
        data_seed = resolve_row_seed(row, args.seed)
        score_batch_max_resamples = resolve_score_batch_setting(
            row, "score_batch_max_resamples", args.score_batch_max_resamples
        )
        score_batch_min_valid = resolve_score_batch_setting(
            row, "score_batch_min_valid", args.score_batch_min_valid
        )
        data = ModCogTrialDM(
            env_id,
            T=T,
            B=B,
            device=args.device,
            last_only=False,
            seed=data_seed,
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
        model.eval()

        # Fixed eval inputs for fair pre/post comparison.
        eval_batches = [data.sample_batch()[0].to(args.device) for _ in range(max(1, args.eval_batches))]
        pre_vals = [
            finite_time_mle(model, xb, delta0=args.delta0, seed=data_seed + i) for i, xb in enumerate(eval_batches)
        ]
        pre_mle = sum(pre_vals) / len(pre_vals)

        pruner = get_pruner(strategy)
        score_count = pruner.resolved_batch_count(_to_int(row.get("movement_batches", ""), args.score_batches))
        score_batches = (
            _sample_score_batches(
                data,
                score_count,
                args.device,
                max_resamples=score_batch_max_resamples,
                min_valid=score_batch_min_valid,
            )
            if score_count > 0
            else None
        )
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        context = PruneContext(
            model=model,
            amount=float(amount),
            criterion=criterion,
            last_only=False,
            device=args.device,
            batches=score_batches,
            metadata={"phase": "post", "run_id": row.get("run_id", "")},
        )
        pruner.run(context, **_extract_prune_kwargs(row, strategy))

        post_vals = [
            finite_time_mle(model, xb, delta0=args.delta0, seed=data_seed + i) for i, xb in enumerate(eval_batches)
        ]
        post_mle = sum(post_vals) / len(post_vals)
        delta = post_mle - pre_mle
        if not args.quiet:
            print(f"{task}\t{strategy}\t{amount:.1f}\t{pre_mle:.6f}\t{post_mle:.6f}\t{delta:.6f}")

        out_rows.append(
            {
                "task": task,
                "strategy": strategy,
                "amount": f"{amount:.1f}",
                "pre_mle": f"{pre_mle:.8f}",
                "post_mle": f"{post_mle:.8f}",
                "delta_mle": f"{delta:.8f}",
            }
        )
        grouped.setdefault((strategy, float(amount)), []).append(post_mle)

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["task", "strategy", "amount", "pre_mle", "post_mle", "delta_mle"])
            w.writeheader()
            w.writerows(out_rows)
        print(f"Wrote {out}")

    if args.plot_out and grouped:
        by_method: Dict[str, List[Tuple[float, float]]] = {}
        for (strategy, amount), vals in grouped.items():
            m = sum(vals) / len(vals)
            by_method.setdefault(strategy, []).append((amount, m))

        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.5, 4.8))
        for method in sorted(by_method):
            pts = sorted(by_method[method], key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=method)
        plt.axhline(0.0, linestyle="--", linewidth=1.0, color="gray")
        plt.xlabel("Amount pruned")
        plt.ylabel("Post-prune finite-time MLE")
        plt.title("Hidden-layer stability (MLE) vs pruning amount")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
