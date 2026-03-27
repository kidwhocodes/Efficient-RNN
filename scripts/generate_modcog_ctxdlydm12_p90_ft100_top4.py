#!/usr/bin/env python3
"""Generate a 90%-prune + 100-step fine-tune suite for four post-pruning methods."""

from __future__ import annotations

import json
from pathlib import Path


SRC_CONFIG = Path("configs/modcog_ctxdlydm12_ctrnn_noise_p90_ft100.json")
DST_CONFIG = Path("configs/modcog_ctxdlydm12_ctrnn_p90_ft100_top4.json")
OUTPUT_CSV = "results/modcog_ctxdlydm12_ctrnn_p90_ft100_top4.csv"
METHODS = (
    "noise_prune",
    "l1_unstructured",
    "obd",
    "random_unstructured",
)


def main() -> None:
    src = json.loads(SRC_CONFIG.read_text())
    out = {
        "output_csv": OUTPUT_CSV,
        "defaults": dict(src["defaults"]),
        "runs": [],
    }
    out["defaults"]["ft_steps"] = 100

    for base_run in src["runs"]:
        task_slug = base_run["task"].replace("modcog:", "")
        for method in METHODS:
            run = {
                "run_id": f"{method}_{task_slug}_p90_ft100",
                "strategy": method,
                "amount": 0.9,
                "seed": base_run["seed"],
                "task": base_run["task"],
                "ng_T": base_run["ng_T"],
                "load_model_path": base_run["load_model_path"],
                "save_model_path": (
                    f"checkpoints/modcog_ctxdlydm12_ctrnn_p90_ft100_top4/"
                    f"{method}_{task_slug}_seed0.pt"
                ),
            }
            out["runs"].append(run)

    DST_CONFIG.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {DST_CONFIG} with {len(out['runs'])} runs")


if __name__ == "__main__":
    main()
