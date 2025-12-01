"""Minimal training/pruning loop for the benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from pruning_benchmark.models.ctrnn import CTRNN, CTRNNConfig
from pruning_benchmark.pruning.strategies import PRUNERS
from pruning_benchmark.tasks.synthetic import (
    HierContextCfg,
    HierContextTask,
    MultiRuleCfg,
    MultiRuleTask,
    NBackCfg,
    NBackTask,
)

TASK_BUILDERS = {
    "synthetic_multirule": (MultiRuleCfg, MultiRuleTask),
    "synthetic_hiercontext": (HierContextCfg, HierContextTask),
    "synthetic_nback": (NBackCfg, NBackTask),
}


def build_task(name: str):
    cfg_cls, task_cls = TASK_BUILDERS[name]
    return task_cls(cfg_cls())


def train_baseline(task, cfg: CTRNNConfig, steps: int) -> CTRNN:
    model = CTRNN(cfg)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(steps):
        x, y = task.sample_batch()
        logits, _ = model(x)
        loss = criterion(logits[-1], y[-1])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def evaluate(model: CTRNN, task, batches: int = 40) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = task.sample_batch()
            logits, _ = model(x)
            loss = criterion(logits[-1], y[-1])
            total_loss += loss.item() * y[-1].numel()
            pred = logits[-1].argmax(dim=-1)
            total_correct += (pred == y[-1]).sum().item()
            total += y[-1].numel()
    return total_loss / total, total_correct / total


def run_suite(config_path: str) -> Dict:
    cfg = json.load(open(config_path))
    results = []
    for run in cfg["runs"]:
        task = build_task(run["task"])
        ctrnn_cfg = CTRNNConfig(
            input_dim=task.sample_batch()[0].size(-1),
            hidden_size=run["hidden_size"],
            output_dim=task.sample_batch()[1].max().item() + 1,
            activation=cfg["defaults"].get("activation", "tanh"),
        )
        if run["strategy"] == "none":
            model = train_baseline(task, ctrnn_cfg, run.get("train_steps", 300))
            loss, acc = evaluate(model, task)
        else:
            path = Path(run["load_model_path"])  # assumes baseline already trained
            model = CTRNN(ctrnn_cfg)
            model.load(str(path))
            weight = model.hidden_layer.weight.data
            pruner = PRUNERS[run["strategy"]]
            pruned = pruner(weight, run["amount"])
            model.hidden_layer.weight.data.copy_(pruned)
            loss, acc = evaluate(model, task)
        results.append({"run_id": run["run_id"], "post_loss": loss, "post_acc": acc})
    return {"runs": results}
