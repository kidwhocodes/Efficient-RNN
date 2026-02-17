#!/usr/bin/env bash
set -euo pipefail

MPLCONFIGDIR=${MPLCONFIGDIR:-.matplotlib_cache}

python3 -m pruning_benchmark --mode suite --config configs/modcog_dlygo_basic_pruners_1seed.json
python3 -m pruning_benchmark --mode plot \
  --input_csv results/modcog_dlygo_prune_suite_1seed.csv \
  --group_by strategy,amount \
  --metrics post_acc,post_loss \
  --plot_out plots/modcog_dlygo_prune_suite_1seed
