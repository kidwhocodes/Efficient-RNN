# Efficient RNN Project

This repository explores neuron- and synapse-level pruning strategies for recurrent neural networks (RNNs), with an emphasis on positive controls for the covariance-based `noise_prune` method. The codebase is organised as a minimal research pipeline: experiment configs live under `configs/`, results and plots are written into `results/` and `plots/`, and the pruning logic is modular so additional strategies can be evaluated quickly.

## Package layout

```
ctrnn_training/
├── analysis/        # metrics, summaries, and plotting helpers
├── config/          # experiment dataclasses
├── data/            # synthetic generators and NeuroGym interface
├── experiments/     # single-run driver, suite harness, sweep utility
├── models/          # CTRNN definition
├── pruning/         # pruning strategies and score builders
├── training/        # SGD + evaluation loops
├── utils/           # seeding and run-id helpers
└── __main__.py      # CLI entry point
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Smoke test the installation:
```
python3 -m pytest
```

Run a single synthetic-task pruning experiment and inspect the generated artifacts under `results/<run_id>/`:
```
python3 -m ctrnn_training --strategy l1_neuron --amount 0.3 --train_steps 200 --ft_steps 50 --task synthetic --run_id demo_single
```

Train baseline checkpoints for the NeuroSuite benchmark (skips already-existing files unless `--skip_training` is set):
```
python3 -m ctrnn_training --mode baseline --baseline_config configs/baselines_neurosuite.json
```

Launch a sweep across the built-in pruning baselines (good for tables and plots):
```
python3 -m ctrnn_training --mode sweep --strategies l1_neuron,random_neuron,noise_prune,movement,synflow --amounts 0.1,0.3,0.5 --seeds 0,1 --train_steps 200 --ft_steps 50 --task synthetic --out_csv results/demo_sweep.csv
```

Generate summary statistics and plots from the CSV:
```
python3 -m ctrnn_training --mode summary --input_csv results/demo_sweep.csv --group_by strategy,amount --metrics post_acc,post_loss --summary_out results/demo_summary.json
python3 -m ctrnn_training --mode plot --input_csv results/demo_sweep.csv --group_by strategy,amount --metrics post_acc,post_loss --plot_out plots/demo
```

Run the full NeuroSuite pruning benchmark (assumes baselines have been trained and saved under `checkpoints/neurosuite/`):
```
python3 -m ctrnn_training --mode suite --config configs/pruning_neurosuite.json
```

## Pruning strategies

The refactor reinstates several positive-control strategies alongside `noise_prune`. Each strategy is callable via the CLI (and available through `ctrnn_training.pruning.apply_pruning`):

- `noise_prune`: covariance-guided pruning on the continuous-time operator.
- `l1_unstructured` / `random_unstructured`: synapse-level magnitude or random pruning.
- `l1_neuron` / `random_neuron`: neuron-level pruning via combined row/column norms.
- `movement`: magnitude of the accumulated gradient updates (synapse-level).
- `movement_neuron`: neuron-level version of movement pruning.
- `snip`: single-shot saliency (`grad * weight`).
- `synflow`: data-free sensitivity analysis with positive weights.
- `fisher`: diagonal Fisher-information approximation using sampled batches.

Use these baselines to benchmark `noise_prune` on both synthetic and NeuroGym tasks (see `configs/` for ready-made suites).

## Contributing guidelines

- Keep pruning amounts on 10% increments unless you update `PRUNE_AMOUNT_STEP`.
- Log metrics through `run_prune_experiment` so results end up in `[run_id]/metrics.json`.
- Prefer adding new strategies inside `ctrnn_training/pruning/strategies.py`; tests under `tests/` should exercise new functionality.

For more detailed notes on experiment workflow see `docs/experiment_workflow.md`.
