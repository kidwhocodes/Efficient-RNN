# Rishidev Chaudhuri Method Evaluation

This note outlines how to benchmark the covariance-based `noise_prune` method against key baselines.

## Suite Configuration

Use the current synthetic_multirule suite (`configs/prune_synthetic_multirule_10x10_suite.json`) and
add noise-prune hyperparameter variants in the `runs` list as needed. Keep the baseline checkpoints
fixed so all strategies start from identical weights.

## Running the Suite

```bash
python3 -m pruning_benchmark --mode suite --config configs/noise_prune_study.json
```

Progress logs show run identifiers; results append to `results/noise_prune_study.csv`.

## Summarising Results

```bash
python3 -m pruning_benchmark --mode summary \
  --input_csv results/noise_prune_study.csv \
  --summary_out results/noise_prune_summary.json \
  --group_by strategy,amount,extra_noise_sigma \
  --metrics post_acc,post_loss

python3 -m pruning_benchmark --mode plot \
  --input_csv results/noise_prune_study.csv \
  --plot_out plots/noise_prune \
  --group_by strategy,amount \
  --metrics post_acc,post_loss
```
The summary is grouped by `strategy` and `amount`, reporting mean/stdev of `post_acc` and `post_loss`. The optional `--summary_out` flag writes the same table for later plotting or analysis.
