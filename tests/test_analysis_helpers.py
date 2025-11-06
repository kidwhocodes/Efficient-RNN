import pandas as pd

from ctrnn_training.analysis.aggregators import (
    load_experiment_records,
    pairwise_deltas,
    stats_by_strategy,
)

def test_stats_by_strategy_basic(tmp_path):
    csv_path = tmp_path / "results.csv"
    rows = [
        {"task": "synthetic", "strategy": "noise_prune", "amount": 0.1, "post_acc": 0.7, "delta_post_acc": 0.0, "sparsity": 0.2},
        {"task": "synthetic", "strategy": "l1_neuron", "amount": 0.1, "post_acc": 0.65, "delta_post_acc": 0.05, "sparsity": 0.2},
        {"task": "synthetic", "strategy": "l1_neuron", "amount": 0.1, "post_acc": 0.66, "delta_post_acc": 0.04, "sparsity": 0.2},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    df = load_experiment_records(str(csv_path))
    summary = stats_by_strategy(df)
    assert not summary.empty
    assert set(summary.columns) >= {"task", "strategy", "post_acc_mean"}


def test_pairwise_deltas(tmp_path):
    csv_path = tmp_path / "runs.csv"
    rows = [
        {"task": "synthetic", "strategy": "noise_prune", "amount": 0.1, "seed": 0, "post_acc": 0.7},
        {"task": "synthetic", "strategy": "l1_neuron", "amount": 0.1, "seed": 0, "post_acc": 0.65},
        {"task": "synthetic", "strategy": "noise_prune", "amount": 0.1, "seed": 1, "post_acc": 0.71},
        {"task": "synthetic", "strategy": "l1_neuron", "amount": 0.1, "seed": 1, "post_acc": 0.64},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    df = load_experiment_records(str(csv_path))
    deltas = pairwise_deltas(df)
    assert "delta_vs_baseline" in deltas.columns
    assert len(deltas) == 2
