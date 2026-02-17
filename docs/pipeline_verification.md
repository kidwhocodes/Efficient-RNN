# Pipeline Verification (Synthetic + Mod-Cog + Model Types)

This checklist records the verification steps for the full pipeline. It is designed
to be repeatable and short enough for other scientists to audit.

## Scope
- Tasks: synthetic (including `synthetic_multirule`) and Mod-Cog (via `modcog:`).
- Models: CTRNN, GRU, LSTM, vanilla RNN.
- Pruning: post-training strategies that operate on recurrent weights.

## Verification checklist

1) **Dataset generation**
   - Synthetic tasks use decision delays and mask inputs after the decision time.
   - Mod-Cog tasks are loaded through `pruning_benchmark.tasks.modcog` and `neurogym`.
   - Reference: `pruning_benchmark/tasks/synthetic.py`, `pruning_benchmark/tasks/modcog.py`.

2) **Model construction**
   - `fresh_model` builds CTRNN/GRU/LSTM/RNN from the same task dimensions.
   - GRU/LSTM/RNN expose `input_layer`/`hidden_layer` proxies so pruning targets the
     true recurrent weights.
   - Reference: `pruning_benchmark/experiments/runner.py`,
     `pruning_benchmark/models/{ctrnn,gru,lstm,rnn}.py`.

3) **Evaluation consistency**
   - Suites sample fixed evaluation batches (`eval_sample_batches`) and reuse them
     pre/post pruning.
   - Ablation checks zero the recurrent weights and re-evaluate on the same batches.
   - Reference: `pruning_benchmark/experiments/runner.py`.

4) **Pruning application**
   - Pruning masks are applied to the recurrent weight matrices; sparsity metrics
     read masks when present.
   - `noise_prune` uses a target density and then enforces exact sparsity via a mask.
   - Reference: `pruning_benchmark/pruning/strategies.py`,
     `pruning_benchmark/analysis/metrics.py`.

5) **Suite harness**
   - Config-required defaults ensure deterministic evaluation and consistent runs.
   - Baseline checkpoint manifests are validated before pruning runs proceed.
   - Reference: `pruning_benchmark/experiments/harness.py`.

## Notes for model-specific behavior
- `noise_prune` requires a square recurrent matrix. For LSTM weights, the strategy
  falls back to `l1_unstructured` (documented in `NoisePruneStrategy`).
- GRU pruning uses the GRU recurrent matrix (`weight_hh_l0`) via a proxy wrapper.

## Re-run instruction (example)
```bash
python3 -m pruning_benchmark --mode suite --config configs/prune_synthetic_multirule_10x10_suite.json
```
