"""Synthetic dataset generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class SynthCfg:
    T: int = 60
    B: int = 64
    input_dim: int = 3  # [bias, left, right]
    output_dim: int = 2
    coh_levels: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    stim_std: float = 0.6


class SyntheticDM:
    def __init__(self, cfg: SynthCfg):
        self.cfg = cfg

    def sample_batch(self):
        T, B, I = self.cfg.T, self.cfg.B, self.cfg.input_dim
        X = np.zeros((T, B, I), np.float32)
        Y = np.zeros((T, B), np.int64)
        X[:, :, 0] = 1.0
        for b in range(B):
            side = np.random.randint(0, 2)
            coh = np.random.choice(self.cfg.coh_levels)
            signed = +coh if side == 1 else -coh
            mu_l, mu_r = (-signed, +signed)
            X[:, b, 1] = np.random.normal(mu_l, self.cfg.stim_std, size=T)
            X[:, b, 2] = np.random.normal(mu_r, self.cfg.stim_std, size=T)
            Y[:, b] = side
        return torch.from_numpy(X), torch.from_numpy(Y)


@dataclass
class SynthContextCfg:
    T: int = 40
    B: int = 64
    context_t: int = 5
    input_dim: int = 3  # [context, left, right]
    output_dim: int = 2
    stim_std: float = 0.6


class SyntheticContextDM:
    """
    Context-dependent comparison task.

    Context feature is presented for the first `context_t` steps (binary 0/1).
    Afterwards only noisy left/right evidence appears; on the final step the
    network must report which stream is larger given the remembered context.
    """

    def __init__(self, cfg: SynthContextCfg):
        self.cfg = cfg

    def sample_batch(self):
        T, B, context_t = self.cfg.T, self.cfg.B, self.cfg.context_t
        X = np.zeros((T, B, self.cfg.input_dim), np.float32)
        Y = np.zeros((T, B), np.int64)

        for b in range(B):
            # context 0 -> choose larger raw evidence
            # context 1 -> choose larger absolute difference (harder rule)
            ctx = np.random.randint(0, 2)
            X[:context_t, b, 0] = float(ctx)

            left = np.random.normal(0.0, self.cfg.stim_std, size=T)
            right = np.random.normal(0.0, self.cfg.stim_std, size=T)

            X[:, b, 1] = left
            X[:, b, 2] = right

            if ctx == 0:
                target = int(left[-1] > right[-1])
            else:
                target = int(abs(left[-1]) > abs(right[-1]))

            Y[:, b] = target

        return torch.from_numpy(X), torch.from_numpy(Y)
