#!/usr/bin/env python3
"""Sanity check: harder NeuroGym task with pruning and 30 evaluation trials."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import neurogym as ngym
import torch

from pruning_benchmark.tasks import NeuroGymDM
from pruning_benchmark.models import CTRNN
from pruning_benchmark.pruning import prune_l1_unstructured
from pruning_benchmark.training import evaluate, train_epoch


def main():
    env = ngym.make("ContextDecisionMaking-v0", timing={"delay": [300, 600]})
    data = NeuroGymDM(env, T=600, B=64, device="cpu", last_only=False, seed=0)

    model = CTRNN(input_dim=data.input_dim, hidden_size=64, output_dim=data.n_classes, activation="tanh")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training harder task...")
    train_epoch(model, data, "cpu", opt, criterion, steps=1200, last_only=False)

    print("Evaluating baseline (30 trials)...")
    loss0, acc0 = evaluate(model, data, "cpu", criterion, steps=30, last_only=False)
    print({"baseline_loss": loss0, "baseline_acc": acc0})

    print("Pruning 50% of synapses (l1 magnitude)...")
    prune_l1_unstructured(model, amount=0.5)
    with torch.no_grad():
        weight = model.hidden_layer.weight
        sparsity = (weight == 0).float().mean().item()
    print({"sparsity": sparsity})

    print("Evaluating after pruning (30 trials)...")
    loss1, acc1 = evaluate(model, data, "cpu", criterion, steps=30, last_only=False)
    print({"pruned_loss": loss1, "pruned_acc": acc1})


if __name__ == "__main__":
    main()
