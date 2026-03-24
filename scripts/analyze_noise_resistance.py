#!/usr/bin/env python3
"""Analyze and plot noise resistance vs pruning amount by pruning method.

For each CSV row:
1) Load baseline checkpoint.
2) Reapply the row's pruning method/amount.
3) Evaluate clean and noisy sequence accuracy on fixed sampled batches.

Noise resistance metrics:
  - acc_drop = acc_clean - acc_noisy
  - acc_ratio = acc_noisy / max(acc_clean, eps)
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
    p.add_argument(
        "--methods",
        default="noise_prune,l1_unstructured,random_unstructured,obd,fisher",
        help="Comma-separated methods to include.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None, help="Optional override for dataset/eval seed.")
    p.add_argument("--eval_batches", type=int, default=4, help="Number of fixed eval batches per row.")
    p.add_argument("--score_batches", type=int, default=20, help="Fallback score batches for score-based methods.")
    p.add_argument("--noise_std", type=float, default=0.1, help="Std of additive Gaussian input noise.")
    p.add_argument("--output_csv", default="")
    p.add_argument("--plot_out", default="")
    p.add_argument("--quiet", action="store_true")
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
    max_resamples: int = 1,
    min_valid: int = 0,
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
def _acc_sequence(model, batches: List[Tuple[torch.Tensor, torch.Tensor]], noise_std: float, seed: int) -> float:
    model.eval()
    correct = 0
    total = 0
    for i, (xb, yb) in enumerate(batches):
        x = xb
        if noise_std > 0:
            gen = torch.Generator(device=xb.device)
            gen.manual_seed(int(seed + i))
            noise = torch.randn(xb.shape, generator=gen, device=xb.device, dtype=xb.dtype) * noise_std
            x = xb + noise
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)
        valid = yb != -1
        if valid.any():
            correct += int((pred[valid] == yb[valid]).sum().item())
            total += int(valid.sum().item())
    if total == 0:
        return float("nan")
    return float(correct / total)


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, var ** 0.5


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
    selected = []
    for r in rows:
        s = (r.get("strategy") or "").strip().lower()
        s = ALIASES.get(s, s)
        if s in methods:
            r["strategy"] = s
            selected.append(r)
    if not selected:
        raise SystemExit("No matching rows for requested methods.")

    out_rows: List[Dict[str, str]] = []
    grouped_drop: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    grouped_ratio: Dict[Tuple[str, float], List[float]] = defaultdict(list)

    if not args.quiet:
        print("task\tstrategy\tamount\tacc_clean\tacc_noisy\tacc_drop\tacc_ratio")

    for row in selected:
        strategy = (row.get("strategy") or "").strip().lower()
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

        eval_batches = _sample_batches(data, max(1, args.eval_batches), args.device)
        acc_clean = _acc_sequence(model, eval_batches, noise_std=0.0, seed=data_seed)

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

        acc_noisy = _acc_sequence(model, eval_batches, noise_std=float(args.noise_std), seed=data_seed + 1000)
        acc_drop = acc_clean - acc_noisy
        acc_ratio = acc_noisy / max(acc_clean, 1e-12)

        if not args.quiet:
            print(
                f"{task}\t{strategy}\t{amount:.1f}\t{acc_clean:.6f}\t{acc_noisy:.6f}\t{acc_drop:.6f}\t{acc_ratio:.6f}"
            )

        out_rows.append(
            {
                "task": task,
                "strategy": strategy,
                "amount": f"{amount:.1f}",
                "acc_clean": f"{acc_clean:.8f}",
                "acc_noisy": f"{acc_noisy:.8f}",
                "acc_drop": f"{acc_drop:.8f}",
                "acc_ratio": f"{acc_ratio:.8f}",
                "noise_std": f"{float(args.noise_std):.6f}",
            }
        )
        grouped_drop[(strategy, float(amount))].append(acc_drop)
        grouped_ratio[(strategy, float(amount))].append(acc_ratio)

    print("\nsummary (grouped):")
    print("strategy\tamount\tn\tdrop_mean\tdrop_std\tratio_mean")
    summary_rows: List[Dict[str, str]] = []
    for key in sorted(grouped_drop.keys(), key=lambda x: (x[0], x[1])):
        strategy, amount = key
        drops = grouped_drop[key]
        ratios = grouped_ratio[key]
        dm, ds = _mean_std(drops)
        rm, _ = _mean_std(ratios)
        print(f"{strategy}\t{amount:.1f}\t{len(drops)}\t{dm:.6f}\t{ds:.6f}\t{rm:.6f}")
        summary_rows.append(
            {
                "strategy": strategy,
                "amount": f"{amount:.1f}",
                "n": str(len(drops)),
                "drop_mean": f"{dm:.8f}",
                "drop_std": f"{ds:.8f}",
                "ratio_mean": f"{rm:.8f}",
                "noise_std": f"{float(args.noise_std):.6f}",
            }
        )

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["strategy", "amount", "n", "drop_mean", "drop_std", "ratio_mean", "noise_std"],
            )
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Wrote {out}")

    if args.plot_out:
        by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for r in summary_rows:
            by_method[r["strategy"]].append((float(r["amount"]), float(r["drop_mean"])))

        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.5, 4.8))
        for method in sorted(by_method):
            pts = sorted(by_method[method], key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=method)
        plt.xlabel("Amount pruned")
        plt.ylabel("Accuracy drop under input noise")
        plt.title(f"Noise resistance vs pruning amount (noise_std={args.noise_std})")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
