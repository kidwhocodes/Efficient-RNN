#!/usr/bin/env bash
set -euo pipefail

MPLCONFIGDIR=${MPLCONFIGDIR:-.matplotlib_cache}

python3 -m pruning_benchmark --mode suite --config configs/modcog_rtgo_all_pruners_3seed.json
python3 -m pruning_benchmark --mode plot \
  --input_csv results/modcog_rtgo_prune_suite_3seed.csv \
  --group_by strategy,amount \
  --metrics post_acc,post_loss \
  --plot_out plots/modcog_rtgo_prune_suite_3seed
