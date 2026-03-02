#!/usr/bin/env python3
"""Plot baseline vs ablated accuracy for Mod-Cog checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from pruning_benchmark.tasks.modcog import ensure_modcog_env_id  # noqa: E402
from pruning_benchmark.tasks.neurogym import ModCogTrialDM  # noqa: E402
from pruning_benchmark.training import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise SystemExit("CSV has no rows.")

    tasks = []
    baseline = []
    ablated = []

    for row in rows:
        task = (row.get("task") or "").strip()
        ckpt = (row.get("save_model_path") or "").strip()
        if not task or not ckpt or not Path(ckpt).exists():
            continue

        post_acc_seq = row.get("post_acc_sequence")
        try:
            base_val = float(post_acc_seq)
        except (TypeError, ValueError):
            continue

        env_id = ensure_modcog_env_id(task)
        T = int(row.get("ng_T") or 0) or 32
        B = int(row.get("ng_B") or 64)

        model_state = torch.load(ckpt, map_location=args.device)
        from pruning_benchmark.experiments.runner import fresh_model

        # Build a fresh model from checkpoint metadata
        hidden_size = None
        for key in ("hidden_layer.weight", "gru.weight_hh_l0"):
            if key in model_state:
                hidden_size = model_state[key].shape[0]
                break
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden size from {ckpt}")

        data = ModCogTrialDM(
            env_id,
            T=T,
            B=B,
            device=args.device,
            last_only=False,
            seed=0,
            mask_fixation=True,
        )
        model = fresh_model(
            input_dim=data.input_dim,
            hidden_size=hidden_size,
            output_dim=data.n_classes,
            device=args.device,
            model_type=row.get("model_type", "ctrnn"),
        )
        model.load_state_dict(model_state)
        model.eval()
        with torch.no_grad():
            if hasattr(model, "hidden_layer"):
                model.hidden_layer.weight.zero_()

        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        metrics = evaluate(
            model,
            data,
            args.device,
            criterion,
            steps=args.eval_steps,
            dataset_last_only=False,
            eval_last_only=False,
        )
        ablated_val = metrics.get("acc_sequence", float("nan"))

        tasks.append(task.replace("modcog:", ""))
        baseline.append(base_val)
        ablated.append(float(ablated_val))

    if not tasks:
        raise SystemExit("No valid rows with checkpoints to plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(tasks))
    width = 0.38
    ax.bar([i - width / 2 for i in x], baseline, width, label="baseline")
    ax.bar([i + width / 2 for i in x], ablated, width, label="ablated")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (sequence)")
    ax.set_title("Mod-Cog Baseline vs Ablated Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
