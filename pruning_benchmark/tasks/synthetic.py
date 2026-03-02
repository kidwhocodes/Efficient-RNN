"""Synthetic dataset generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


def _decision_index(T: int, decision_delay: int, *, min_index: int = 0) -> int:
    if decision_delay < 0:
        raise ValueError("decision_delay must be non-negative.")
    t_dec = T - decision_delay - 1
    return max(min_index, t_dec)


def _mask_after_decision(inputs: np.ndarray, t_dec: int) -> None:
    if t_dec < inputs.shape[0] - 1:
        inputs[t_dec + 1 :] = 0.0


@dataclass
class SynthCfg:
    T: int = 60
    B: int = 64
    input_dim: int = 3  # [bias, left, right]
    output_dim: int = 2
    coh_levels: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    stim_std: float = 0.6
    decision_delay: int = 5


class SyntheticDM:
    def __init__(self, cfg: SynthCfg):
        self.cfg = cfg

    def sample_batch(self):
        T, B, I = self.cfg.T, self.cfg.B, self.cfg.input_dim
        t_dec = _decision_index(T, self.cfg.decision_delay)
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
        _mask_after_decision(X, t_dec)
        return torch.from_numpy(X), torch.from_numpy(Y)


@dataclass
class SynthContextCfg:
    T: int = 40
    B: int = 64
    context_t: int = 5
    input_dim: int = 3  # [context, left, right]
    output_dim: int = 2
    stim_std: float = 0.6
    decision_delay: int = 5


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
        t_dec = _decision_index(T, self.cfg.decision_delay)
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
                target = int(left[t_dec] > right[t_dec])
            else:
                target = int(abs(left[t_dec]) > abs(right[t_dec]))

            Y[:, b] = target

        _mask_after_decision(X, t_dec)
        return torch.from_numpy(X), torch.from_numpy(Y)


@dataclass
class SynthMultiRuleCfg:
    T: int = 60
    B: int = 64
    context_steps: int = 5
    input_dim: int = 5  # 3 context channels + 2 evidence streams
    output_classes: int = 6
    evidence_std: float = 0.8
    decision_delay: int = 5


class SyntheticMultiRuleDM:
    """
    Context-dependent multi-rule task with 6 output classes (context/result combos).

    Context 0: decide if mean(left) > mean(right)
    Context 1: decide if |left - right| > threshold
    Context 2: decide if the trend of left evidence is positive vs negative
    Final label = context * 2 + result_bit, so network must track context and rule.
    """

    def __init__(self, cfg: SynthMultiRuleCfg):
        self.cfg = cfg

    def sample_batch(self):
        C = self.cfg
        T, B = C.T, C.B
        t_dec = _decision_index(T, C.decision_delay)
        X = np.zeros((T, B, C.input_dim), np.float32)
        Y = np.zeros((T, B), np.int64)
        thresh = 0.5
        for b in range(B):
            ctx = np.random.randint(0, 3)
            ctx_onehot = np.zeros(3, np.float32)
            ctx_onehot[ctx] = 1.0
            X[: C.context_steps, b, :3] = ctx_onehot

            left = np.random.normal(0.0, C.evidence_std, size=T)
            right = np.random.normal(0.0, C.evidence_std, size=T)
            X[:, b, 3] = left
            X[:, b, 4] = right

            left_slice = left[: t_dec + 1]
            right_slice = right[: t_dec + 1]
            if ctx == 0:
                result = int(left_slice.mean() > right_slice.mean())
            elif ctx == 1:
                result = int(np.abs(left[t_dec] - right[t_dec]) > thresh)
            else:
                result = int(np.polyfit(np.arange(left_slice.size), left_slice, 1)[0] > 0)
            label = ctx * 2 + result
            Y[:, b] = label
        _mask_after_decision(X, t_dec)
        return torch.from_numpy(X), torch.from_numpy(Y)


@dataclass
class SynthHierContextCfg:
    T: int = 70
    B: int = 64
    context_steps: int = 6
    input_dim: int = 6  # 2 context channels + 2 rule channels + 2 evidence streams
    output_dim: int = 4  # family (0/1) * rule (0/1)
    evidence_std: float = 0.7
    diff_thresh: float = 0.4
    decision_delay: int = 5


class SyntheticHierContextDM:
    """
    Hierarchical context task: first cue selects a family (A/B), second cue selects a rule.

    Family A:
      rule 0 -> decide mean(left) > mean(right)
      rule 1 -> decide |left-right| final > threshold
    Family B:
      rule 0 -> decide left variance > right variance
      rule 1 -> decide trend of left is increasing vs decreasing

    Final label encodes family*2 + rule_result (0..3).
    """

    def __init__(self, cfg: SynthHierContextCfg):
        self.cfg = cfg

    def sample_batch(self):
        C = self.cfg
        T, B = C.T, C.B
        t_dec = _decision_index(T, C.decision_delay)
        X = np.zeros((T, B, C.input_dim), np.float32)
        Y = np.zeros((T, B), np.int64)
        for b in range(B):
            family = np.random.randint(0, 2)
            rule = np.random.randint(0, 2)
            # context channels
            fam_onehot = np.zeros(2, np.float32)
            rule_onehot = np.zeros(2, np.float32)
            fam_onehot[family] = 1.0
            rule_onehot[rule] = 1.0
            X[: C.context_steps, b, :2] = fam_onehot
            X[: C.context_steps, b, 2:4] = rule_onehot

            left = np.random.normal(0.0, C.evidence_std, size=T)
            right = np.random.normal(0.0, C.evidence_std, size=T)
            X[:, b, 4] = left
            X[:, b, 5] = right

            left_slice = left[: t_dec + 1]
            right_slice = right[: t_dec + 1]
            if family == 0 and rule == 0:
                result = int(left_slice.mean() > right_slice.mean())
            elif family == 0 and rule == 1:
                result = int(abs(left[t_dec] - right[t_dec]) > C.diff_thresh)
            elif family == 1 and rule == 0:
                result = int(left_slice.var() > right_slice.var())
            else:
                result = int(np.polyfit(np.arange(left_slice.size), left_slice, 1)[0] > 0)
            label = family * 2 + result
            Y[:, b] = label
        _mask_after_decision(X, t_dec)
        return torch.from_numpy(X), torch.from_numpy(Y)


@dataclass
class SynthNBackCfg:
    T: int = 50
    B: int = 64
    alphabet_size: int = 4
    input_dim: int = 1 + 4  # cue channel + one-hot symbol
    output_dim: int = 4  # (n=2 vs 3) x (match vs mismatch)
    n_choices: Tuple[int, ...] = (2, 3)
    match_prob: float = 0.5
    decision_delay: int = 5


class SyntheticNBackDM:
    """
    Variable-n back task.

    Cue indicates whether to perform 2-back or 3-back. Sequence symbols are one-hot.
    Final label indicates match vs no-match on the last item relative to n steps before.
    """

    def __init__(self, cfg: SynthNBackCfg):
        self.cfg = cfg

    def sample_batch(self):
        C = self.cfg
        T, B = C.T, C.B
        t_dec = _decision_index(T, C.decision_delay, min_index=max(C.n_choices))
        X = np.zeros((T, B, C.input_dim), np.float32)
        Y = np.zeros((T, B), np.int64)
        for b in range(B):
            n = int(np.random.choice(C.n_choices))
            cue = np.zeros(1, np.float32)
            cue[0] = float(n) / max(C.n_choices)
            X[:5, b, :1] = cue

            symbols = np.random.randint(0, C.alphabet_size, size=T)
            for t in range(T):
                sym = symbols[t]
                onehot = np.zeros(C.alphabet_size, np.float32)
                onehot[sym] = 1.0
                X[t, b, 1:] = onehot

            match = np.random.rand() < C.match_prob
            if match and t_dec >= n:
                symbols[t_dec] = symbols[t_dec - n]
                sym = symbols[t_dec]
                onehot = np.zeros(C.alphabet_size, np.float32)
                onehot[sym] = 1.0
                X[t_dec, b, 1:] = onehot
            else:
                # ensure mismatch
                candidates = [s for s in range(C.alphabet_size) if s != symbols[t_dec - n]]
                symbols[t_dec] = np.random.choice(candidates)
                for t in range(T):
                    sym = symbols[t]
                    onehot = np.zeros(C.alphabet_size, np.float32)
                    onehot[sym] = 1.0
                    X[t, b, 1:] = onehot
            label = (0 if n == 2 else 2) + int(not match)
            Y[:, b] = label
        _mask_after_decision(X, t_dec)
        return torch.from_numpy(X), torch.from_numpy(Y)
