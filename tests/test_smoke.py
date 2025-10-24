import torch

from ctrnn_training.core import CTRNN
from ctrnn_training.experiments import run_prune_experiment
from ctrnn_training.utils import set_global_seed


def test_ctrnn_forward_smoke():
    model = CTRNN(input_dim=3, hidden_size=8, output_dim=2)
    x = torch.randn(5, 4, 3)
    logits, hidden = model(x)
    assert logits.shape == (5, 4, 2)
    assert hidden.shape == (5, 4, 8)


def test_run_prune_experiment_smoke():
    set_global_seed(0)
    row = run_prune_experiment(
        strategy="none",
        amount=0.0,
        train_steps=1,
        ft_steps=0,
        last_only=True,
        seed=0,
        device="cpu",
        movement_batches=1,
        task="synthetic",
        no_prune=True,
        run_id="test_run",
    )
    assert row["run_id"] == "test_run"
    assert abs(row["post_loss"]) >= 0  # smoke check on metric presence
