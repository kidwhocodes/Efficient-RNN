#!/usr/bin/env python3
"""Train 64-unit ContextDM baselines with hard delays for pruning tests."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from ctrnn_training.data import NeuroGymDM
from ctrnn_training.models import CTRNN
from ctrnn_training.training import train_epoch, evaluate


def train_seed(seed: int, checkpoint_path: str) -> None:
    import neurogym as ngym

    env = ngym.make("ContextDecisionMaking-v0", timing={"delay": [300, 700]})
    data = NeuroGymDM(env, T=650, B=64, device="cpu", last_only=False, seed=seed)

    model = CTRNN(input_dim=data.input_dim, hidden_size=64, output_dim=data.n_classes, activation="tanh")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"[ctxdm64] training seed={seed}")
    train_epoch(model, data, "cpu", opt, criterion, steps=1200, last_only=False)

    print(f"[ctxdm64] evaluating seed={seed}")
    loss, acc = evaluate(model, data, "cpu", criterion, steps=200, last_only=False)
    print({"seed": seed, "loss": loss, "acc": acc})

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[ctxdm64] saved checkpoint -> {checkpoint_path}")


def main():
    for seed in (0, 1):
        checkpoint = f"checkpoints/ctxdm64/ng_ContextDecisionMaking_v0_seed{seed}.pt"
        train_seed(seed, checkpoint)


if __name__ == "__main__":
    main()
