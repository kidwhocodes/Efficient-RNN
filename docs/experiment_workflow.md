# Experiment Workflow Guide

This guide explains the standard workflow for reproducing pruning experiments in the *Efficient RNN Project*.

## 1. Configure Runs

Create a JSON suite config (see `configs/example_suite.json` or `configs/noise_prune_study.json`) with the following fields:

```json
{
  "run_id": "my_suite",
  "output_csv": "results/my_suite.csv",
  "defaults": {
    "train_steps": 600,
    "ft_steps": 200,
    "last_only": true,
    "task": "synthetic",
    "device": "cpu"
  },
  "runs": [
    {"strategy": "none", "amount": 0.0, "no_prune": true, "seed": 0},
    {"strategy": "noise_prune", "amount": 0.5, "seed": 0}
  ]
}
```

- Values in `defaults` apply to every run unless overridden.
- `run_id` is optional; a timestamp will be generated if omitted.

## 2. Execute the Suite

```bash
python -m pruning_benchmark --mode suite --config configs/example_suite.json
# Change-detection benchmark
python -m pruning_benchmark --mode suite --config configs/neurogym_pruning_changedetection.json
# Dual working-memory benchmark
python -m pruning_benchmark --mode suite --config configs/neurogym_pruning_dualdms.json
# Context-dependent decision benchmark
python -m pruning_benchmark --mode suite --config configs/neurogym_pruning_contextdm.json
```

Each run is executed with deterministic seeding, and results are appended to the CSV specified in the config (or `results/<suite_id>.csv` by default).

## 3. Summarise Results

```bash
python -m pruning_benchmark --mode summary --input_csv results/example_suite.csv

# add `--summary_out <path>` to save tables, and `--group_by`/`--metrics` to control aggregation.
```

This prints per-strategy aggregates (`post_acc_mean`, `post_loss_mean`, etc.) to stdout.

## 4. Visualise

```bash
python -m pruning_benchmark --mode plot \
  --input_csv results/example_suite.csv \
  --plot_out plots/example \
  --group_by strategy,amount --metrics post_acc,post_loss
```

This writes PNG plots showing each metric versus the chosen amount field for every strategy.

## 5. Run Smoke Tests

Before committing, run the lightweight test suite:

```bash
pytest tests/test_smoke.py
```

## 6. Recommended Git Workflow

1. Stage files: `git add <modified files>`
2. Commit with a clear message summarising the change.
3. Push to `origin main`.

Following these steps keeps experiment runs reproducible and results easy to interpret.
