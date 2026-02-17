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

Run the current synthetic_multirule pruning suite (10 baselines × 10 trials):
```
python3 scripts/generate_multirule_10x10_suite.py \
  --output configs/prune_synthetic_multirule_10x10_suite.json
python3 -m pruning_benchmark --mode suite --config configs/prune_synthetic_multirule_10x10_suite.json
```

Plot the results:
```
MPLCONFIGDIR=/tmp python3 - <<'PY'
from pruning_benchmark.analysis.plots import plot_metrics

plot_metrics(
    "results/prune_synthetic_multirule_10x10_suite.csv",
    metrics=["post_acc"],
    output_dir="plots/prune_synthetic_multirule_10x10_suite",
)
PY
```

Older suites/configs and their outputs are archived under `archives/`.

### Mod_Cog NeuroGym tasks (optional setup)

Recent Mod_Cog environments register themselves with NeuroGym once `Mod_Cog.mod_cog_tasks` is imported. Install them alongside NeuroGym:
```
pip install gym
git clone https://github.com/neurogym/neurogym.git
cd neurogym && pip install -e .
git clone https://github.com/mikailkhona/Mod_Cog.git
cd Mod_Cog && pip install -e .
```

After installation you can invoke Mod_Cog tasks via the new `modcog:` prefix. Pass the builder name from `Mod_Cog.mod_cog_tasks` (e.g., `go`, `dlygointr`, `ctxdlydm1intseq`) and optionally forward environment kwargs through `--ng_kwargs` plus Dataset controls through `--ng_dataset_kwargs`:
```
python3 -m pruning_benchmark \
  --task modcog:go \
  --strategy noise_prune \
  --amount 0.3 \
  --ng_T 600 --ng_B 64 \
  --ng_kwargs '{"difficulty": "hard"}' \
  --ng_dataset_kwargs '{"max_batch": 1000}'
```
Under the hood this uses `neurogym.Dataset` so minibatches contain full Mod_Cog trials while still exposing the familiar benchmarking pipeline.

If you need to programmatically inspect the 132-task battery, use the helper provided under `pruning_benchmark.tasks.modcog`:
```python
from pruning_benchmark.tasks.modcog import list_modcog_tasks

print(list_modcog_tasks()[:10])
# -> ('anti', 'antiseql', 'antiseqr', 'ctxdm1', 'ctxdm1intr', ...)
```

## Pruning strategies

The benchmark supports multiple pruning strategies and uses the same training/evaluation
pipeline for all of them. Strategy names and categories are listed in
`docs/pruning_methods.md`. See `configs/` for ready-made suites.

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

Documentation index: `docs/README.md`.
