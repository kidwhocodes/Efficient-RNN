#!/usr/bin/env python3
"""Analyze recurrent contribution vs pruning amount and plot by method.

For each CSV row:
1) Load baseline checkpoint (load_model_path).
2) Measure recurrent contribution on fixed task batches.
3) Reapply the row's pruning method/amount.
4) Measure recurrent contribution again.

Contribution metric (CTRNN):
    mean_t,b ||alpha * W_rec h_t|| / (|| (1-alpha) v_t + alpha * W_in x_t || + eps)
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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


ALIASES = {
    "fischer": "fisher",
    "l1": "l1_unstructured",
    "random": "random_unstructured",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--methods", default="noise_prune,l1_unstructured,random_unstructured,obd,fisher")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None, help="Optional override for dataset/eval seed.")
    p.add_argument("--eval_batches", type=int, default=2, help="Batches used for contribution estimate.")
    p.add_argument("--score_batches", type=int, default=20, help="Fallback score batches for score-based methods.")
    p.add_argument("--output_csv", default="")
    p.add_argument("--plot_out", default="")
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


def _extract_prune_kwargs(row: Dict[str, str], strategy: str) -> Dict[str, object]:
    if strategy != "noise_prune":
        return {}
    return resolve_noise_prune_kwargs(row)


def _sample_batches(
    data: ModCogTrialDM,
    n: int,
    device: str,
    *,
    max_resamples: int,
    min_valid: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    out = []
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
def recurrent_contribution_ratio(model, x: torch.Tensor, eps: float = 1e-8) -> float:
    """Average ratio of recurrent update norm vs non-recurrent update norm."""
    model.eval()
    T, B, _ = x.shape
    fr, v = model.init_state(B, x.device)
    ratios = []
    for t in range(T):
        in_affine = F.linear(x[t], model.input_layer.weight, model.input_layer.bias)
        rec_linear = F.linear(fr, model.hidden_layer.weight, None)
        rec_bias = model.hidden_layer.bias if model.hidden_layer.bias is not None else 0.0
        rec_term = model.alpha * rec_linear
        nonrec_term = model.oneminusalpha * v + model.alpha * (in_affine + rec_bias)
        rec_n = torch.linalg.norm(rec_term, dim=1)
        nonrec_n = torch.linalg.norm(nonrec_term, dim=1)
        ratios.append(rec_n / (nonrec_n + eps))
        v = nonrec_term + rec_term
        fr = model.act(v)
    return float(torch.stack(ratios, dim=0).mean().item())


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, var**0.5


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

    methods = set()
    for m in args.methods.split(","):
        mm = m.strip().lower()
        if not mm:
            continue
        methods.add(ALIASES.get(mm, mm))

    rows = list(csv.DictReader(in_path.open()))
    selected = [r for r in rows if (r.get("strategy") or "").strip().lower() in methods]
    if not selected:
        raise SystemExit("No matching rows for requested methods.")

    out_rows: List[Dict[str, str]] = []
    grouped_post: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    grouped_ratio: Dict[Tuple[str, float], List[float]] = defaultdict(list)

    if not args.quiet:
        print("task\tstrategy\tamount\tpre_contrib\tpost_contrib\tpost_over_pre")
    for row in selected:
        strategy = (row.get("strategy") or "").strip().lower()
        strategy = ALIASES.get(strategy, strategy)
        task = (row.get("task") or "").strip()
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

        eval_x = [data.sample_batch()[0].to(args.device) for _ in range(max(1, args.eval_batches))]
        pre_vals = [recurrent_contribution_ratio(model, xb) for xb in eval_x]
        pre_contrib = sum(pre_vals) / len(pre_vals)

        pruner = get_pruner(strategy)
        score_count = pruner.resolved_batch_count(_to_int(row.get("movement_batches", ""), args.score_batches))
        score_batches = (
            _sample_batches(
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

        post_vals = [recurrent_contribution_ratio(model, xb) for xb in eval_x]
        post_contrib = sum(post_vals) / len(post_vals)
        post_over_pre = post_contrib / max(pre_contrib, 1e-12)
        if not args.quiet:
            print(
                f"{task}\t{strategy}\t{amount:.1f}\t{pre_contrib:.6f}\t{post_contrib:.6f}\t{post_over_pre:.6f}"
            )

        out_rows.append(
            {
                "task": task,
                "strategy": strategy,
                "amount": f"{amount:.1f}",
                "pre_contrib": f"{pre_contrib:.8f}",
                "post_contrib": f"{post_contrib:.8f}",
                "post_over_pre": f"{post_over_pre:.8f}",
            }
        )
        grouped_post[(strategy, amount)].append(post_contrib)
        grouped_ratio[(strategy, amount)].append(post_over_pre)

    print("\nsummary (grouped):")
    print("strategy\tamount\tn\tpost_contrib_mean\tpost_contrib_std\tpost_over_pre_mean")
    summary_rows: List[Dict[str, str]] = []
    for key in sorted(grouped_post.keys(), key=lambda x: (x[0], x[1])):
        strat, amount = key
        post_vals = grouped_post[key]
        ratio_vals = grouped_ratio[key]
        pm, ps = _mean_std(post_vals)
        rm, _ = _mean_std(ratio_vals)
        print(f"{strat}\t{amount:.1f}\t{len(post_vals)}\t{pm:.6f}\t{ps:.6f}\t{rm:.6f}")
        summary_rows.append(
            {
                "strategy": strat,
                "amount": f"{amount:.1f}",
                "n": str(len(post_vals)),
                "post_contrib_mean": f"{pm:.8f}",
                "post_contrib_std": f"{ps:.8f}",
                "post_over_pre_mean": f"{rm:.8f}",
            }
        )

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "strategy",
                    "amount",
                    "n",
                    "post_contrib_mean",
                    "post_contrib_std",
                    "post_over_pre_mean",
                ],
            )
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote {out}")

    if args.plot_out:
        by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for r in summary_rows:
            by_method[r["strategy"]].append((float(r["amount"]), float(r["post_contrib_mean"])))
        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.5, 4.8))
        for method in sorted(by_method):
            pts = sorted(by_method[method], key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=method)
        plt.xlabel("Amount pruned")
        plt.ylabel("Recurrent contribution ratio")
        plt.title("Recurrent contribution vs pruning amount")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
