# Pruning Benchmark

Clean, self-contained benchmark for comparing pruning strategies on cognitive RNN tasks.

## Contents
- `models/`: core network architectures (CTRNN, Dale variant)
- `tasks/`: synthetic cognitive tasks (multi-rule, hier-context, n-back)
- `pruning/`: pruning rules and utilities
- `pipeline/`: training + pruning evaluation loop
- `configs/`: JSON suites describing experiments

Run suites with `python -m pipeline.evaluator --config configs/example_suite.json` (CLI coming soon).
