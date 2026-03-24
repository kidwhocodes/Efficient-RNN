#!/usr/bin/env python3
"""Analyze/plot recurrent functional connectivity vs pruning amount by method.

Connectivity is computed from the post-pruning recurrent matrix W_rec:
  - edge_density: fraction of nonzero directed recurrent edges
  - giant_component_frac: largest weakly-connected component fraction (undirected projection)
  - effective_connectivity: edge_density * mean_abs_weight_nz
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict, deque
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
    p.add_argument(
        "--metric",
        default="effective_connectivity",
        choices=["effective_connectivity", "edge_density", "giant_component_frac", "mean_abs_weight_nz"],
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None, help="Optional override for dataset/eval seed.")
    p.add_argument("--score_batches", type=int, default=20)
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


def _largest_component_fraction(adj_undirected: torch.Tensor) -> float:
    """Largest connected component fraction for boolean undirected adjacency."""
    n = int(adj_undirected.shape[0])
    if n == 0:
        return 0.0
    visited = [False] * n
    best = 0
    for i in range(n):
        if visited[i]:
            continue
        q = deque([i])
        visited[i] = True
        size = 0
        while q:
            u = q.popleft()
            size += 1
            neighbors = torch.nonzero(adj_undirected[u], as_tuple=False).view(-1).tolist()
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        best = max(best, size)
    return float(best / n)


@torch.no_grad()
def functional_connectivity_metrics(model) -> Dict[str, float]:
    w = model.hidden_layer.weight.detach()
    h, wdim = w.shape
    if h != wdim:
        return {
            "edge_density": float("nan"),
            "mean_abs_weight_nz": float("nan"),
            "giant_component_frac": float("nan"),
            "effective_connectivity": float("nan"),
        }

    abs_w = w.abs()
    nz = abs_w > 0
    n = h
    eye_mask = torch.eye(n, dtype=torch.bool, device=nz.device)
    nz_no_diag = nz & (~eye_mask)
    total_edges = n * (n - 1)
    nz_count = int(nz_no_diag.sum().item())
    edge_density = float(nz_count / max(1, total_edges))

    if nz_count > 0:
        mean_abs_nz = float(abs_w[nz_no_diag].mean().item())
    else:
        mean_abs_nz = 0.0

    adj_u = (nz | nz.T)
    giant_frac = _largest_component_fraction(adj_u)
    eff = edge_density * mean_abs_nz
    return {
        "edge_density": edge_density,
        "mean_abs_weight_nz": mean_abs_nz,
        "giant_component_frac": giant_frac,
        "effective_connectivity": eff,
    }


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

    methods = set()
    for m in args.methods.split(","):
        mm = m.strip().lower()
        if mm:
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

    grouped: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    raw_rows: List[Dict[str, str]] = []

    if not args.quiet:
        print(f"task\tstrategy\tamount\t{args.metric}")

    for row in selected:
        strategy = row["strategy"]
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

        m = functional_connectivity_metrics(model)
        value = m[args.metric]
        grouped[(strategy, float(amount))].append(value)
        raw_rows.append(
            {
                "task": task,
                "strategy": strategy,
                "amount": f"{amount:.1f}",
                "edge_density": f"{m['edge_density']:.10f}",
                "mean_abs_weight_nz": f"{m['mean_abs_weight_nz']:.10f}",
                "giant_component_frac": f"{m['giant_component_frac']:.10f}",
                "effective_connectivity": f"{m['effective_connectivity']:.10f}",
            }
        )
        if not args.quiet:
            print(f"{task}\t{strategy}\t{amount:.1f}\t{value:.8f}")

    print("\nsummary (grouped):")
    print(f"strategy\tamount\tn\t{args.metric}_mean\t{args.metric}_std")
    summary_rows: List[Dict[str, str]] = []
    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        strat, amount = key
        vals = grouped[key]
        mean, std = _mean_std(vals)
        print(f"{strat}\t{amount:.1f}\t{len(vals)}\t{mean:.8f}\t{std:.8f}")
        summary_rows.append(
            {
                "strategy": strat,
                "amount": f"{amount:.1f}",
                "n": str(len(vals)),
                f"{args.metric}_mean": f"{mean:.10f}",
                f"{args.metric}_std": f"{std:.10f}",
            }
        )

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["strategy", "amount", "n", f"{args.metric}_mean", f"{args.metric}_std"],
            )
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Wrote {out}")

    if args.plot_out:
        by_method: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for r in summary_rows:
            by_method[r["strategy"]].append((float(r["amount"]), float(r[f"{args.metric}_mean"])))
        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.5, 4.8))
        for method in sorted(by_method):
            pts = sorted(by_method[method], key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=method)
        plt.xlabel("Amount pruned")
        plt.ylabel(args.metric.replace("_", " "))
        plt.title(f"Functional connectivity ({args.metric}) vs pruning amount")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
