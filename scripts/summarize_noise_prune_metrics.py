#!/usr/bin/env python3
"""Summarize noise-prune metrics by pruning amount."""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(csv_path: str) -> List[Dict[str, float]]:
    groups: Dict[float, List[Tuple[float, float]]] = defaultdict(list)

    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("strategy") != "noise_prune":
                continue
            amount = _to_float(row.get("amount", ""))
            if amount is None:
                continue
            alpha_rho = _to_float(row.get("post_alpha_rho", ""))
            rec_mean_nz = _to_float(row.get("post_rec_weight_abs_mean_nz", ""))
            if alpha_rho is None or rec_mean_nz is None:
                continue
            groups[amount].append((alpha_rho, rec_mean_nz))

    out: List[Dict[str, float]] = []
    for amount, values in groups.items():
        if not values:
            continue
        alpha_vals = [v[0] for v in values]
        rec_vals = [v[1] for v in values]
        out.append(
            {
                "amount": amount,
                "post_alpha_rho_mean": sum(alpha_vals) / len(alpha_vals),
                "post_rec_weight_abs_mean_nz_mean": sum(rec_vals) / len(rec_vals),
                "count": float(len(values)),
            }
        )
    out.sort(key=lambda item: item["amount"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to pruning results CSV")
    args = parser.parse_args()

    rows = summarize(args.input_csv)
    if not rows:
        print("No noise_prune rows found or metrics missing.")
        return

    header = ["amount", "post_alpha_rho_mean", "post_rec_weight_abs_mean_nz_mean", "count"]
    print("\t".join(header))
    for row in rows:
        print(
            f"{row['amount']:.2f}\t"
            f"{row['post_alpha_rho_mean']:.6f}\t"
            f"{row['post_rec_weight_abs_mean_nz_mean']:.6f}\t"
            f"{int(row['count'])}"
        )


if __name__ == "__main__":
    main()
