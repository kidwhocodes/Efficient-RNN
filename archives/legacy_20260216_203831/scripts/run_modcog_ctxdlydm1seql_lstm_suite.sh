#!/usr/bin/env bash
set -euo pipefail

MPLCONFIGDIR=${MPLCONFIGDIR:-.matplotlib_cache}

python3 -m pruning_benchmark --mode baseline --baseline_config configs/modcog_ctxdlydm1seql_lstm_baseline.json
python3 -m pruning_benchmark --mode suite --config configs/modcog_ctxdlydm1seql_lstm_post_methods_3seed.json
python3 -m pruning_benchmark --mode plot \
  --input_csv results/modcog_ctxdlydm1seql_lstm_post_methods_3seed.csv \
  --group_by strategy,amount \
  --metrics post_acc,post_loss \
  --plot_out plots/modcog_ctxdlydm1seql_lstm_post_methods_3seed
