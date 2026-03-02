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
            valid = y[-1] >= 0
            N = int(valid.sum().item())
            if N == 0:
                continue
            loss = criterion(logits[-1], y[-1])
        else:
            flat_targets = y.view(-1)
            valid = flat_targets >= 0
            N = int(valid.sum().item())
            if N == 0:
                continue
            loss = criterion(logits.view(-1, logits.size(-1)), flat_targets)
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
    response_window_k: int | None = None,
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
    total_response_correct = 0
    total_response_count = 0
    total_response_loss = 0.0
    total_response_loss_weight = 0

    for _ in range(steps):
        x, y = data.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)

        decision_logits = logits[-1]
        decision_targets = y[-1]
        decision_valid = decision_targets >= 0
        decision_N = int(decision_valid.sum().item())
        decision_loss = criterion(decision_logits, decision_targets)

        if eval_last_only:
            loss_val = decision_loss
            loss_weight = decision_N
        else:
            seq_logits = logits.view(-1, logits.size(-1))
            seq_targets = y.view(-1)
            loss_val = criterion(seq_logits, seq_targets)
            loss_weight = int((seq_targets >= 0).sum().item())

        if loss_weight > 0:
            total_loss += loss_val.item() * loss_weight
            total_loss_weight += loss_weight

        decision_pred = decision_logits.argmax(-1)
        if decision_N > 0:
            decision_correct = ((decision_pred == decision_targets) & decision_valid).sum().item()
            total_decision_correct += decision_correct
            total_decision_count += decision_N

        seq_pred = logits.argmax(-1)
        seq_valid = y >= 0
        seq_total = int(seq_valid.sum().item())
        if seq_total > 0:
            seq_correct = ((seq_pred == y) & seq_valid).sum().item()
            total_seq_correct += seq_correct
            total_seq_count += seq_total

        if response_window_k is not None and response_window_k > 0:
            k = min(response_window_k, logits.size(0))
            if k > 0:
                resp_logits = logits[-k:].reshape(-1, logits.size(-1))
                resp_targets = y[-k:].reshape(-1)
                resp_loss = criterion(resp_logits, resp_targets)
                resp_weight = int((resp_targets >= 0).sum().item())
                if resp_weight > 0:
                    total_response_loss += resp_loss.item() * resp_weight
                    total_response_loss_weight += resp_weight
                    resp_pred = resp_logits.argmax(-1)
                    resp_valid = resp_targets >= 0
                    resp_correct = ((resp_pred == resp_targets) & resp_valid).sum().item()
                    total_response_correct += resp_correct
                    total_response_count += resp_weight

    mean_loss = total_loss / max(1, total_loss_weight)
    decision_acc = total_decision_correct / max(1, total_decision_count)
    sequence_acc = total_seq_correct / max(1, total_seq_count)
    response_acc = None
    response_loss = None
    if response_window_k is not None and response_window_k > 0:
        response_acc = total_response_correct / max(1, total_response_count)
        response_loss = total_response_loss / max(1, total_response_loss_weight)

    metrics = {
        "loss": mean_loss,
        "acc": decision_acc,
        "acc_sequence": sequence_acc,
    }
    if response_acc is not None:
        metrics["acc_response_window"] = response_acc
        metrics["loss_response_window"] = response_loss
    return metrics
