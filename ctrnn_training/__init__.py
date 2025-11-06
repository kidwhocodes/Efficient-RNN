"""Top-level package exports for the Efficient RNN project."""

from .analysis import (
    compile_run_metrics,
    count_nonzero_and_total,
    ctrnn_stability_proxy,
    neuron_keep_fraction,
    neuron_pruning_stats,
    recurrent_sparsity,
    save_metrics,
    snapshot_model,
)
from .config import ExperimentConfig
from .data import NeuroGymDM, SynthCfg, SynthContextCfg, SyntheticContextDM, SyntheticDM
from .experiments import run_prune_experiment, run_suite_from_config, run_sweep
from .models import CTRNN
from .pruning import (
    PRUNE_AMOUNT_STEP,
    apply_pruning,
    available_pruning_strategies,
    enforce_constraints,
    finalize_pruning,
    noise_prune_recurrent,
    prune_l1_unstructured,
    prune_neurons_l1,
    prune_neurons_random,
    prune_random_unstructured,
    validate_prune_fraction,
)
from .training import evaluate, train_epoch
from .utils import make_run_id, set_global_seed

__all__ = [
    "CTRNN",
    "ExperimentConfig",
    "NeuroGymDM",
    "SynthCfg",
    "SynthContextCfg",
    "SyntheticContextDM",
    "SyntheticDM",
    "compile_run_metrics",
    "count_nonzero_and_total",
    "ctrnn_stability_proxy",
    "enforce_constraints",
    "evaluate",
    "finalize_pruning",
    "make_run_id",
    "neuron_keep_fraction",
    "neuron_pruning_stats",
    "noise_prune_recurrent",
    "prune_l1_unstructured",
    "prune_neurons_l1",
    "prune_neurons_random",
    "prune_random_unstructured",
    "recurrent_sparsity",
    "run_prune_experiment",
    "run_suite_from_config",
    "run_sweep",
    "save_metrics",
    "set_global_seed",
    "snapshot_model",
    "train_epoch",
    "validate_prune_fraction",
    "PRUNE_AMOUNT_STEP",
    "available_pruning_strategies",
]
