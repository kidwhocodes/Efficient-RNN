#!/usr/bin/env python3
"""Filter Mod-Cog baseline runs by ablation drop ratio."""

import argparse
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument(
        "--ratio_threshold",
        type=float,
        default=0.5,
        help="Keep rows where ablated_acc <= ratio_threshold * post_acc",
    )
    return parser.parse_args()


def _to_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()

    rows = []
    with open(args.input_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("strategy") not in ("none", ""):
                continue
            post_acc = _to_float(row.get("post_acc", ""))
            ablated_acc = _to_float(row.get("ablation_post_acc", ""))
            if post_acc is None or ablated_acc is None:
                continue
            if post_acc <= 0:
                continue
            ratio = ablated_acc / post_acc
            if ablated_acc <= args.ratio_threshold * post_acc:
                rows.append(
                    {
                        "task": row.get("task", ""),
                        "post_acc": post_acc,
                        "ablation_post_acc": ablated_acc,
                        "ratio": ratio,
                    }
                )

    if not rows:
        print("No tasks matched the ablation threshold.")
        return

    rows.sort(key=lambda r: r["ratio"])
    print("task\tpost_acc\tablation_post_acc\tratio")
    for r in rows:
        print(
            f"{r['task']}\t{r['post_acc']:.4f}\t{r['ablation_post_acc']:.4f}\t{r['ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
