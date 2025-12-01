# Professor Method Evaluation

This note outlines how to benchmark the biologically inspired `noise_prune` covariance rule against key baselines.

## Suite Configuration

The repository ships with `configs/noise_prune_study.json`, covering:

- Control (`none+no_prune`)
- `l1_unstructured` magnitude pruning
- `noise_prune` across sparsity levels (0.3/0.5/0.7)
- `noise_prune` multi-seed replicates (seeds 0–2)
- Hyper-parameter ablations (`noise_sigma`)
- Structural baselines (`turnover_synapse`, `oja_synapse`)

Feel free to duplicate and modify this JSON to explore additional settings (e.g., `noise_eps`, `noise_matched_diagonal`).

## Running the Suite

```bash
python -m pruning_benchmark --mode suite --config configs/noise_prune_study.json
```

Progress logs show run identifiers; results append to `results/noise_prune_study.csv`.

## Summarising Results

```bash
python -m pruning_benchmark --mode summary \
  --input_csv results/noise_prune_study.csv \
  --summary_out results/noise_prune_summary.json \
  --group_by strategy,amount,extra_noise_sigma \
  --metrics post_acc,post_loss

python -m pruning_benchmark --mode plot \
  --input_csv results/noise_prune_study.csv \
  --plot_out plots/noise_prune \
  --group_by strategy,amount \
  --metrics post_acc,post_loss
```

The summary is grouped by `strategy` and `amount`, reporting mean/stdev of `post_acc` and `post_loss`. The optional `--summary_out` flag writes the same table for later plotting or analysis.

## Alternative Benchmark: Change Detection

```bash
python -m pruning_benchmark --mode suite --config configs/neurogym_pruning_changedetection.json
python -m pruning_benchmark --mode summary --input_csv results/ng_perceptualdm.csv --summary_out results/ng_perceptualdm_summary.json --group_by strategy,amount --metrics post_acc,post_loss
python -m pruning_benchmark --mode plot --input_csv results/ng_perceptualdm.csv --plot_out plots/ng_perceptualdm --group_by strategy,amount --metrics post_acc,post_loss
```

## Suggested Next Steps

- Increase the seed count for statistically robust comparisons.
- Extend `runs` with NeuroGym tasks (e.g., `task: "ng:DelayMatchToSample-v0"`).
- Log additional metrics (`pruning_benchmark/metrics.py`) if your analysis needs more structural detail.
