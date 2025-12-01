import json
import os

import torch

from pruning_benchmark.analysis.plots import plot_metrics
from pruning_benchmark.analysis.summary import summarize_csv
from pruning_benchmark.experiments import run_prune_experiment, run_suite_from_config
from pruning_benchmark.models import CTRNN
from pruning_benchmark.utils import set_global_seed


def test_ctrnn_forward_smoke():
    model = CTRNN(input_dim=3, hidden_size=8, output_dim=2)
    x = torch.randn(5, 4, 3)
    logits, hidden = model(x)
    assert logits.shape == (5, 4, 2)
    assert hidden.shape == (5, 4, 8)


def test_run_prune_experiment_smoke():
    set_global_seed(0)
    row = run_prune_experiment(
        strategy="none",
        amount=0.0,
        train_steps=1,
        ft_steps=0,
        last_only=True,
        seed=0,
        device="cpu",
        movement_batches=1,
        task="synthetic",
        no_prune=True,
        run_id="test_run",
    )
    assert row["run_id"] == "test_run"
    assert abs(row["post_loss"]) >= 0  # smoke check on metric presence


def test_suite_runner_smoke(tmp_path):
    cfg = {
        "run_id": "test_suite",
        "defaults": {
            "train_steps": 1,
            "ft_steps": 0,
            "last_only": True,
            "task": "synthetic",
            "device": "cpu"
        },
        "runs": [
            {"strategy": "none", "amount": 0.0, "no_prune": True, "seed": 0}
        ]
    }
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(cfg))
    csv_path = run_suite_from_config(str(config_path))
    assert os.path.exists(csv_path)
    summaries = summarize_csv(csv_path)
    assert summaries
    out_json = tmp_path / "summary.json"
    summarize_csv(csv_path, output_path=str(out_json))
    assert out_json.exists()
    custom = summarize_csv(
        csv_path,
        group_fields=("strategy",),
        metrics=("post_acc",),
        output_path=None,
    )
    assert custom


def test_plot_metrics(tmp_path):
    csv_path = tmp_path / "results.csv"
    rows = [
        {"strategy": "A", "amount": "0.1", "post_acc": "0.6", "post_loss": "0.5"},
        {"strategy": "A", "amount": "0.2", "post_acc": "0.7", "post_loss": "0.4"},
        {"strategy": "B", "amount": "0.1", "post_acc": "0.55", "post_loss": "0.45"},
    ]
    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "plots"
    files = plot_metrics(str(csv_path), metrics=("post_acc", "post_loss"), output_dir=str(out_dir))
    assert files
    for path in files:
        assert os.path.exists(path)


def test_save_and_load_model(tmp_path):
    checkpoint = tmp_path / "baseline.pt"
    _, _ = run_prune_experiment(
        strategy="none",
        amount=0.0,
        train_steps=2,
        ft_steps=0,
        last_only=True,
        seed=0,
        device="cpu",
        movement_batches=1,
        task="synthetic",
        no_prune=True,
        save_model_path=str(checkpoint),
        return_model=True,
    )
    assert checkpoint.exists()

    row, _ = run_prune_experiment(
        strategy="l1_unstructured",
        amount=0.5,
        train_steps=0,
        ft_steps=0,
        last_only=True,
        seed=0,
        device="cpu",
        movement_batches=1,
        task="synthetic",
        load_model_path=str(checkpoint),
        skip_training=True,
        return_model=True,
    )
    assert abs(row["pre_acc"] - row["pre0_acc"]) < 1e-8


def test_pruning_strategies_smoke():
    set_global_seed(0)
    strategies = ["movement", "snip", "synflow", "fisher", "grasp"]
    for strat in strategies:
        row = run_prune_experiment(
            strategy=strat,
            amount=0.2,
            train_steps=1,
            ft_steps=0,
            last_only=True,
            seed=0,
            device="cpu",
            movement_batches=1,
            task="synthetic",
            run_id=f"smoke_{strat}",
        )
        assert "post_acc" in row
    row_pre = run_prune_experiment(
        strategy="snip",
        amount=0.2,
        train_steps=2,
        ft_steps=0,
        last_only=True,
        seed=0,
        device="cpu",
        movement_batches=1,
        task="synthetic",
        run_id="smoke_snip_pre",
        prune_phase="pre",
    )
    assert row_pre["prune_phase"] == "pre"
