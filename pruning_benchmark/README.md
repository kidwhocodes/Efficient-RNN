# Pruning Benchmark

Self-contained benchmark for comparing pruning strategies on cognitive RNN tasks.

## Contents
- `models/`: core network architectures (CTRNN variants)
- `tasks/`: synthetic cognitive tasks + NeuroGym interface
- `pruning/`: pruning rules and utilities
- `experiments/`: training + pruning evaluation loop and suite harness
- `configs/`: JSON suites describing experiments

Run suites with:
```
python3 -m pruning_benchmark --mode suite --config configs/prune_synthetic_multirule_10x10_suite.json
```

For a full workflow and reproducibility notes, see `docs/experiment_workflow.md`.
