#!/usr/bin/env python3
"""Run the 64-unit ContextDM noise pruning sweep and plot results."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pruning_benchmark.analysis.aggregators import load_experiment_records

# Config paths
CFG_PATH = "configs/pruning_ctxdm64_noise.json"
CSV_PATH = "results/ctxdm64_noise_sweep.csv"
ANALYSIS_DIR = "analysis_ctxdm64_noise"


def main():
    import subprocess

    print("[suite] running ctxdm64 noise-pruning sweep...")
    subprocess.run(
        [
            "python3",
            "-m",
            "pruning_benchmark",
            "--mode",
            "suite",
            "--config",
            CFG_PATH,
        ],
        check=True,
    )

    print("[analysis] generating summary/plots...")
    subprocess.run(
        [
            "python3",
            "scripts/analyze_pruning.py",
            "--csv",
            CSV_PATH,
            "--metrics_dir",
            "results",
            "--out",
            ANALYSIS_DIR,
        ],
        check=True,
    )

    df = load_experiment_records(CSV_PATH)
    print("[completed] rows=", len(df))


if __name__ == "__main__":
    main()
