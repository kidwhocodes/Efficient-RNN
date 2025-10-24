"""Helpers to sample NeuroGym environments into RNN-ready tensors."""

import torch
import numpy as np
from typing import Tuple

class NeuroGymDM:
    def __init__(self, env, T: int, B: int, device="cpu", last_only: bool = True, seed: int = 0):
        self.env = env
        self.T = T
        self.B = B
        self.device = device
        self.last_only = bool(last_only)
        try:
            # Gym API
            if hasattr(env, "seed"):
                env.seed(seed)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)
        except Exception:
            pass

        obs_space = env.observation_space.shape[0]
        self.input_dim = obs_space
        self.n_classes = env.action_space.n

    def _reset(self) -> Tuple[np.ndarray, dict]:
        out = self.env.reset()
        # Gymnasium returns (obs, info), Gym returns obs
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, info

    def _step(self, action) -> Tuple[np.ndarray, dict, bool]:
        out = self.env.step(action)
        # Gymnasium: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, _, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            # Gym: (obs, reward, done, info)
            obs, _, done, info = out
        if done:  # keep rolling episodes to fill T steps
            obs, _ = self._reset()
        return obs, info, done

    def sample_batch(self):
        T, B, I = self.T, self.B, self.input_dim
        X = torch.zeros(T, B, I, device=self.device)
        Y = torch.zeros(T, B, dtype=torch.long, device=self.device)
        for b in range(B):
            obs, info = self._reset()
            last_gt = int(info.get("gt", 0))
            for t in range(T):
                # convert obs to torch (Gym/Ngym gives np.ndarray)
                x_t = torch.from_numpy(np.asarray(obs)).float().to(self.device)
                X[t, b] = x_t

                # random policy keeps dataset generation unbiased
                action = self.env.action_space.sample()
                obs, info, _ = self._step(action)
                if "gt" in info:
                    last_gt = int(info["gt"])

                # label each step OR only final step
                if not self.last_only:
                    Y[t, b] = last_gt
            Y[T - 1, b] = last_gt  # always label the end
        return X, Y
