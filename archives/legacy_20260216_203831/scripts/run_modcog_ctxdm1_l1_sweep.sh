#!/usr/bin/env bash
set -euo pipefail

MPLCONFIGDIR=${MPLCONFIGDIR:-.matplotlib_cache}

python3 -m pruning_benchmark --mode suite --config configs/modcog_ctxdm1_l1_sweep_1seed.json
python3 -m pruning_benchmark --mode plot \
  --input_csv results/modcog_ctxdm1_l1_sweep_1seed.csv \
  --group_by strategy,amount \
  --metrics post_acc,post_loss \
  --plot_out plots/modcog_ctxdm1_l1_sweep_1seed
