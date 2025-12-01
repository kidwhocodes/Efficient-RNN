# Efficient RNN Project

This repository explores neuron- and synapse-level pruning strategies for recurrent neural networks (RNNs), with an emphasis on positive controls for the covariance-based `noise_prune` method. The codebase is organised as a minimal research pipeline: experiment configs live under `configs/`, results and plots are written into `results/` and `plots/`, and the pruning logic is modular so additional strategies can be evaluated quickly.

## Package layout

```
pruning_benchmark/
├── analysis/        # metrics, summaries, and plotting helpers
├── config/          # experiment dataclasses
├── tasks/           # synthetic generators and NeuroGym interface
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
python3 -m pruning_benchmark --strategy l1_unstructured --amount 0.3 --train_steps 200 --ft_steps 50 --task synthetic --run_id demo_single
```

Train baseline checkpoints for the NeuroSuite benchmark (skips already-existing files unless `--skip_training` is set):
```
python3 -m pruning_benchmark --mode baseline --baseline_config configs/baselines_neurosuite.json
```

Evaluate those checkpoints to verify the task portfolio difficulty:
```
python3 -m pruning_benchmark --mode portfolio --portfolio_config configs/portfolio_neurosuite.json --out_csv results/neurosuite_portfolio.csv
```

Launch a sweep across the built-in pruning baselines (good for tables and plots):
```
python3 -m pruning_benchmark --mode sweep --strategies l1_unstructured,random_unstructured,noise_prune,movement,synflow --amounts 0.1,0.3,0.5 --seeds 0,1 --train_steps 200 --ft_steps 50 --task synthetic --out_csv results/demo_sweep.csv
```

Generate summary statistics and plots from the CSV:
```
python3 -m pruning_benchmark --mode summary --input_csv results/demo_sweep.csv --group_by strategy,amount --metrics post_acc,post_loss --summary_out results/demo_summary.json
python3 -m pruning_benchmark --mode plot --input_csv results/demo_sweep.csv --group_by strategy,amount --metrics post_acc,post_loss --plot_out plots/demo
```

Run the full NeuroSuite pruning benchmark (assumes baselines have been trained and saved under `checkpoints/neurosuite/`):
```
python3 -m pruning_benchmark --mode suite --config configs/pruning_neurosuite.json
```
This suite now enforces the full pruning pipeline:
- Every strategy/amount reloads the saved checkpoint, so it always starts from the same trained-but-unpruned weights.
- 64 evaluation batches are sampled once per run (using the training seed), then reused before/after pruning so accuracy deltas reflect pruning only.
- `ft_steps` defaults to 0; bump it in the config if you want to study post-pruning fine-tuning.

### Mod_Cog NeuroGym tasks

Recent Mod_Cog environments register themselves with NeuroGym once `Mod_Cog.mod_cog_tasks` is imported. Install them alongside NeuroGym:
```
pip install gym
git clone https://github.com/neurogym/neurogym.git
cd neurogym && pip install -e .
git clone https://github.com/mikailkhona/Mod_Cog.git
cd Mod_Cog && pip install -e .
```

After installation you can invoke Mod_Cog tasks via the new `modcog:` prefix, optionally forwarding any environment keyword arguments through `--ng_kwargs` and Dataset controls through `--ng_dataset_kwargs`:
```
python3 -m pruning_benchmark \
  --task modcog:FlexibleWorkingMemory-v0 \
  --strategy noise_prune \
  --amount 0.3 \
  --ng_T 600 --ng_B 64 \
  --ng_kwargs '{"difficulty": "hard"}' \
  --ng_dataset_kwargs '{"max_batch": 1000}'
```
Under the hood this uses `neurogym.Dataset` so minibatches contain full Mod_Cog trials while still exposing the familiar benchmarking pipeline.

## Pruning strategies

The refactor reinstates several positive-control strategies alongside `noise_prune`. Each strategy is callable via the CLI, and you can access/extend them programmatically through `pruning_benchmark.pruning.get_pruner` / `register_pruner`:

- `random_unstructured`: uniform synapse drop baseline.
- `l1_unstructured`: magnitude-based synapse pruning.
- `movement`: magnitude of the accumulated gradient updates (synapse-level).
- `snip`: single-shot saliency (`grad * weight`).
- `synflow`: data-free sensitivity analysis with positive weights.
- `fisher`: diagonal Fisher-information approximation using sampled batches.
- `grasp`: GraSP curvature-aware saliency using Hessian-vector products.
- `obd`: Optimal Brain Damage using a diagonal Hessian approximation.
- `set`: Sparse Evolutionary Training-inspired rewiring (drop/re-grow edges).
- `noise_prune`: covariance-guided pruning on the continuous-time operator.

Use these baselines to benchmark `noise_prune` on both synthetic and NeuroGym tasks (see `configs/` for ready-made suites).

### Adding your own pruning strategy

The new pruning interface exposes a `BasePruner` class plus a registry so custom methods plug in cleanly:

```python
from pruning_benchmark.pruning import BasePruner, PruneContext, register_pruner

class MyFancyPruner(BasePruner):
    name = "my_fancy"
    requires_batches = True
    default_batch_count = 10

    def prepare(self, context: PruneContext):
        # compute scores from context.batches and return them
        return {"scores": ...}

    def apply(self, context: PruneContext, state):
        # mutate context.model using context.amount + state["scores"]
        return {"extra_metric": 0.123}

register_pruner(MyFancyPruner())
```

Once registered you can reference `--strategy my_fancy` in configs/CLI runs. This makes it easy to benchmark external pruners against the built-in suite without modifying the evaluation pipeline.

### Pruning phases

Strategies that are traditionally applied at initialization (e.g., `snip`, `synflow`, `grasp`) can operate before training by setting `"prune_phase": "pre"` in a suite run (or `--prune_phase pre` on the CLI). In this mode the mask is applied to the random initialization, then the sparse model trains for the standard `train_steps`. Leave the value at `"post"` (default) to prune a trained checkpoint instead. Metrics include the `prune_phase` and a `pruned_pretraining` flag so downstream analysis can differentiate the two regimes.

## Contributing guidelines

- Keep pruning amounts on 10% increments unless you update `PRUNE_AMOUNT_STEP`.
- Log metrics through `run_prune_experiment` so results end up in `[run_id]/metrics.json`.
- Prefer adding new strategies inside `pruning_benchmark/pruning/strategies.py`; tests under `tests/` should exercise new functionality.

For more detailed notes on experiment workflow see `docs/experiment_workflow.md`.
