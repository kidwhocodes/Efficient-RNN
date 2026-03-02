#!/usr/bin/env python3
"""Print baseline vs ablated accuracy from a Mod-Cog results CSV."""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import torch

from pruning_benchmark.tasks.modcog import ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import ModCogTrialDM
from pruning_benchmark.training import evaluate
from pruning_benchmark.experiments.runner import fresh_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _infer_hidden_size(state: dict) -> int:
    if "lstm.weight_hh_l0" in state:
        return int(state["lstm.weight_hh_l0"].shape[1])
    if "gru.weight_hh_l0" in state:
        return int(state["gru.weight_hh_l0"].shape[1])
    if "hidden_layer.weight" in state:
        return int(state["hidden_layer.weight"].shape[0])
    raise ValueError("Could not infer hidden size from checkpoint.")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message=r".*env\.new_trial to get variables from other wrappers is deprecated.*",
        category=UserWarning,
    )
    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise SystemExit("CSV has no rows.")

    print("task\tpost_acc_sequence\tablated_acc_sequence\tcheckpoint")
    for row in rows:
        task = (row.get("task") or "").strip()
        ckpt = (row.get("save_model_path") or "").strip()
        if not task or not ckpt or not Path(ckpt).exists():
            print(f"{task or 'NA'}\tMISSING\tMISSING\t{ckpt or 'NA'}")
            continue

        post_acc_seq = row.get("post_acc_sequence")
        try:
            base_val = float(post_acc_seq)
        except (TypeError, ValueError):
            base_val = float("nan")

        env_id = ensure_modcog_env_id(task)
        T = int(row.get("ng_T") or 0) or 32
        B = int(row.get("ng_B") or 64)

        state = torch.load(ckpt, map_location=args.device)
        hidden_size = _infer_hidden_size(state)

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
        model.load_state_dict(state)
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

        print(f"{task}\t{base_val:.6f}\t{ablated_val:.6f}\t{ckpt}")


if __name__ == "__main__":
    main()
