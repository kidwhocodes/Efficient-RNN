"""Training and evaluation loops shared across experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..pruning import enforce_constraints


def train_epoch(model, data, device, opt, criterion, steps=50, last_only=True, clip=1.0):
    model.train()
    total_loss, total_count = 0.0, 0
    for _ in range(steps):
        x, y = data.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        if last_only:
            loss = criterion(logits[-1], y[-1])
            N = y[-1].numel()
        else:
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            N = y.numel()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        enforce_constraints(model)
        total_loss += loss.item() * N
        total_count += N
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(
    model,
    data,
    device,
    criterion,
    *,
    steps: int = 20,
    dataset_last_only: bool = True,
    eval_last_only: bool | None = None,
) -> dict:
    """
    Evaluate the model, always tracking final-step accuracy to reflect task decisions.

    dataset_last_only: whether the dataset labels are only valid on the final step.
    eval_last_only: when True, both loss and the primary accuracy use only the final step.
    """
    model.eval()
    if eval_last_only is None:
        eval_last_only = dataset_last_only

    total_loss = 0.0
    total_loss_weight = 0
    total_decision_correct = 0
    total_decision_count = 0
    total_seq_correct = 0
    total_seq_count = 0

    for _ in range(steps):
        x, y = data.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)

        decision_logits = logits[-1]
        decision_targets = y[-1]
        decision_loss = criterion(decision_logits, decision_targets)
        decision_N = decision_targets.numel()

        if eval_last_only:
            loss_val = decision_loss
            loss_weight = decision_N
        else:
            seq_logits = logits.view(-1, logits.size(-1))
            seq_targets = y.view(-1)
            loss_val = criterion(seq_logits, seq_targets)
            loss_weight = seq_targets.numel()

        total_loss += loss_val.item() * loss_weight
        total_loss_weight += loss_weight

        decision_correct = (decision_logits.argmax(-1) == decision_targets).sum().item()
        total_decision_correct += decision_correct
        total_decision_count += decision_N

        seq_correct = (logits.argmax(-1) == y).sum().item()
        seq_total = y.numel()
        total_seq_correct += seq_correct
        total_seq_count += seq_total

    mean_loss = total_loss / max(1, total_loss_weight)
    decision_acc = total_decision_correct / max(1, total_decision_count)
    sequence_acc = total_seq_correct / max(1, total_seq_count)

    return {
        "loss": mean_loss,
        "acc": decision_acc,
        "acc_sequence": sequence_acc,
    }
