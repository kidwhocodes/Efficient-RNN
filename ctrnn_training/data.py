from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch

@dataclass
class SynthCfg:
    T: int = 60
    B: int = 64
    input_dim: int = 3      # [bias, left, right]
    output_dim: int = 3     # [left,right,no-go]
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
