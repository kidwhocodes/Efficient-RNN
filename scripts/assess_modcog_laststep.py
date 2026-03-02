#!/usr/bin/env python3
"""Probe whether Mod-Cog tasks are solvable from the final observation only."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.tasks.modcog import list_modcog_tasks, ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import ModCogTrialDM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess last-step solvability for Mod-Cog tasks."
    )
    parser.add_argument("--T", type=int, default=400)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--train_batches", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated Mod-Cog task names or 'all'.",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Optional cap on number of tasks to evaluate.",
    )
    parser.add_argument(
        "--output_csv",
        default="results/modcog_laststep_probe.csv",
        help="Where to write results.",
    )
    return parser.parse_args()


def _parse_tasks(src: str) -> list[str]:
    if src.strip().lower() == "all":
        return list(list_modcog_tasks())
    return [item.strip() for item in src.split(",") if item.strip()]


def _majority_baseline(labels: torch.Tensor, n_classes: int) -> float:
    labels = labels[labels >= 0]
    if labels.numel() == 0:
        return 0.0
    counts = torch.bincount(labels, minlength=n_classes).float()
    if counts.sum().item() == 0:
        return 0.0
    return float(counts.max().item() / counts.sum().item())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tasks = _parse_tasks(args.tasks)
    if args.max_tasks is not None:
        tasks = tasks[: max(1, int(args.max_tasks))]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task",
        "env_id",
        "input_dim",
        "n_classes",
        "train_acc",
        "eval_acc",
        "majority_acc",
        "train_batches",
        "eval_batches",
        "T",
        "B",
        "error",
    ]

    rows = []
    for name in tasks:
        env_id = None
        try:
            env_id = ensure_modcog_env_id(f"modcog:{name}")
            if env_id is None:
                raise ValueError(f"Unknown Mod-Cog task: {name}")
            data = ModCogTrialDM(
                env_id,
                T=args.T,
                B=args.B,
                device=args.device,
                last_only=False,
                seed=args.seed,
                mask_fixation=True,
            )

            n_classes = data.n_classes
            input_dim = data.input_dim
            clf = nn.Linear(input_dim, n_classes).to(args.device)
            opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
            criterion = nn.CrossEntropyLoss(ignore_index=-1)

            # Train on last-step features only.
            clf.train()
            all_labels = []
            for _ in range(args.train_batches):
                x, y = data.sample_batch()
                x_last = x[-1]
                y_last = y[-1]
                logits = clf(x_last)
                loss = criterion(logits, y_last)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                all_labels.append(y_last.detach().cpu())

            # Train accuracy (on fresh batch)
            clf.eval()
            with torch.no_grad():
                x, y = data.sample_batch()
                logits = clf(x[-1])
                pred = logits.argmax(dim=-1)
                valid = y[-1] >= 0
                if valid.any():
                    train_acc = float((pred[valid] == y[-1][valid]).float().mean().item())
                else:
                    train_acc = 0.0

            # Eval accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    x, y = data.sample_batch()
                    logits = clf(x[-1])
                    pred = logits.argmax(dim=-1)
                    valid = y[-1] >= 0
                    correct += int((pred[valid] == y[-1][valid]).sum().item())
                    total += int(valid.sum().item())
            eval_acc = float(correct / max(1, total))

            labels_cat = torch.cat(all_labels, dim=0)
            majority_acc = _majority_baseline(labels_cat, n_classes)

            rows.append(
                {
                    "task": name,
                    "env_id": env_id,
                    "input_dim": input_dim,
                    "n_classes": n_classes,
                    "train_acc": train_acc,
                    "eval_acc": eval_acc,
                    "majority_acc": majority_acc,
                    "train_batches": args.train_batches,
                    "eval_batches": args.eval_batches,
                    "T": args.T,
                    "B": args.B,
                    "error": "",
                }
            )
            print(f"[ok] {name} eval_acc={eval_acc:.3f}")
        except Exception as exc:
            rows.append(
                {
                    "task": name,
                    "env_id": env_id or "",
                    "input_dim": "",
                    "n_classes": "",
                    "train_acc": "",
                    "eval_acc": "",
                    "majority_acc": "",
                    "train_batches": args.train_batches,
                    "eval_batches": args.eval_batches,
                    "T": args.T,
                    "B": args.B,
                    "error": repr(exc),
                }
            )
            print(f"[fail] {name}: {exc}")

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
