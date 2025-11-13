#!/usr/bin/env python3
"""Quick sanity check: train a CTRNN, prune, and re-evaluate on fixed batches."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from ctrnn_training.data import SynthCfg, SyntheticDM
from ctrnn_training.models import CTRNN
from ctrnn_training.pruning import prune_l1_unstructured
from ctrnn_training.training import evaluate, train_epoch


def main():
    cfg = SynthCfg(T=60, B=64)
    data = SyntheticDM(cfg)
    model = CTRNN(input_dim=cfg.input_dim, hidden_size=64, output_dim=cfg.output_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training...")
    train_epoch(model, data, "cpu", opt, criterion, steps=500, last_only=True)

    print("Evaluating baseline...")
    loss0, acc0 = evaluate(model, data, "cpu", criterion, steps=200, last_only=True)
    print({"baseline_loss": loss0, "baseline_acc": acc0})

    print("Pruning 50% by magnitude...")
    prune_l1_unstructured(model, amount=0.5)
    with torch.no_grad():
        weight = model.hidden_layer.weight
        sparsity = (weight == 0).float().mean().item()
    print({"sparsity": sparsity})

    print("Evaluating after pruning...")
    loss1, acc1 = evaluate(model, data, "cpu", criterion, steps=200, last_only=True)
    print({"pruned_loss": loss1, "pruned_acc": acc1})


if __name__ == "__main__":
    main()
