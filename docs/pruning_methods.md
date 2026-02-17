# Pruning Strategy Reference

This document lists the strategy names used in suite configs and provides high-level
categorization for reproducibility. It intentionally avoids method-level derivations.

## Strategy Names (as used in configs/CSV)

Synapse-level (post-training unless noted):
- `random_unstructured`
- `l1_unstructured`
- `noise_prune`
- `movement` (uses gradient batches)
- `fisher` (uses gradient batches)
- `snip` (pre-training)
- `synflow` (pre-training)
- `grasp` (pre-training)
- `obd`
- `obs`
- `woodfisher`
- `set`

Neuron-level:
- `causal`
- `lstm_wpr`

## What changes between strategies

All strategies share the same training and evaluation pipeline. The only differences
are (1) the pruning rule, (2) whether batches are required for scoring, and (3) whether
the rule is applied before or after training. See `docs/experiment_workflow.md` for
the pipeline and required suite fields.

## Adding a new strategy

Register a new `BasePruner` implementation in `pruning_benchmark/pruning/strategies.py`
and ensure the name appears in this list so other scientists can reproduce runs.
