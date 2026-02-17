# Experiment Workflow Guide

This guide explains the standard workflow for reproducing pruning experiments in the Efficient RNN Project.

## 1. Configure Runs

Create a JSON suite config with required defaults and a list of runs (see
`configs/prune_synthetic_multirule_10x10_suite.json` for the current reference):

```json
{
  "defaults": {
    "task": "synthetic",
    "hidden_size": 128,
    "train_steps": 2500,
    "ft_steps": 0,
    "last_only": true,
    "device": "cpu",
    "movement_batches": 20,
    "ng_T": 400,
    "ng_B": 64,
    "eval_sample_batches": 32
  },
  "runs": [
    {"strategy": "none", "amount": 0.0, "no_prune": true, "seed": 0},
    {"strategy": "noise_prune", "amount": 0.5, "seed": 0}
  ]
}
```

- Values in `defaults` apply to every run unless overridden.
- `run_id` and `output_csv` are optional; the suite will generate them if omitted.
- `eval_sample_batches` must be > 0 so pre/post pruning metrics use the same evaluation batches.

## 2. Execute the Suite

```bash
python3 -m pruning_benchmark --mode suite --config configs/prune_synthetic_multirule_10x10_suite.json
```

Each run is executed with deterministic seeding, and results are appended to the CSV specified in the config (or `results/<suite_id>.csv` by default).

## 3. Summarise Results

```bash
python3 -m pruning_benchmark --mode summary --input_csv results/example_suite.csv

# add `--summary_out <path>` to save tables, and `--group_by`/`--metrics` to control aggregation.
```

This prints per-strategy aggregates (`post_acc_mean`, `post_loss_mean`, etc.) to stdout.

## 4. Visualise

```bash
python3 -m pruning_benchmark --mode plot \
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
