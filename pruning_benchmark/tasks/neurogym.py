"""Helpers to sample NeuroGym environments into RNN-ready tensors."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
from neurogym import Dataset


class NeuroGymDM:
    def __init__(self, env, T: int, B: int, device="cpu", last_only: bool = True, seed: int = 0):
        self.env = env
        self.T = T
        self.B = B
        self.device = device
        self.last_only = bool(last_only)
        try:
            env.reset(seed=seed)
        except TypeError:
            try:
                if hasattr(env, "seed"):
                    env.seed(seed)
                else:
                    env.reset()
            except Exception:
                pass
        if hasattr(env.action_space, "seed"):
            try:
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


def _ensure_final_labels(targets: torch.Tensor) -> torch.Tensor:
    """Make sure the final timestep carries a valid class label."""
    if targets.ndim < 2:
        return targets
    last = targets[-1]
    invalid = last < 0
    if not torch.any(invalid):
        return targets
    for idx in invalid.nonzero(as_tuple=False).view(-1):
        column = targets[:, idx]
        valid = torch.nonzero(column >= 0, as_tuple=False).view(-1)
        if valid.numel():
            targets[-1, idx] = column[valid[-1]]
        else:
            targets[-1, idx] = 0
    return targets


class NeuroGymDatasetDM:
    """Wrapper around neurogym.Dataset for Mod_Cog and other dataset-driven tasks."""

    def __init__(
        self,
        env: str | Any,
        T: int,
        B: int,
        *,
        device: str = "cpu",
        last_only: bool = True,
        seed: int = 0,
        env_kwargs: dict | None = None,
        dataset_kwargs: dict | None = None,
    ):
        env_kwargs = dict(env_kwargs or {})
        dataset_kwargs = dict(dataset_kwargs or {})
        dataset_kwargs.setdefault("batch_first", False)
        dataset_kwargs.setdefault("max_batch", np.inf)
        dataset_kwargs.setdefault("cache_len", None)
        dataset_kwargs["seq_len"] = T
        dataset_kwargs["batch_size"] = B
        self.dataset = Dataset(env, env_kwargs=env_kwargs, **dataset_kwargs)
        self.dataset.seed(seed)
        self.device = device
        self.last_only = bool(last_only)
        env = self.dataset.env
        obs_shape = getattr(env.observation_space, "shape", None)
        if not obs_shape:
            raise ValueError(f"Environment {env} has no observation shape.")
        self.input_dim = int(np.prod(obs_shape))
        action_space = env.action_space
        if hasattr(action_space, "n"):
            self.n_classes = int(action_space.n)
        else:
            shape = getattr(action_space, "shape", None)
            if not shape:
                raise ValueError(f"Environment {env} has no action shape.")
            self.n_classes = int(np.prod(shape))

    def sample_batch(self):
        inputs, targets = next(self.dataset)
        X = torch.from_numpy(np.asarray(inputs)).float().to(self.device)
        Y = torch.from_numpy(np.asarray(targets)).clone().to(torch.float32)
        if Y.ndim == 3 and Y.size(-1) == 1:
            Y = Y.squeeze(-1)
        elif Y.ndim == 3:
            Y = Y.argmax(dim=-1)
        Y = torch.nan_to_num(Y, nan=0.0).to(torch.long)
        Y = _ensure_final_labels(Y)
        if not self.last_only:
            return X, Y
        # zero out unused intermediate labels while keeping the final decision
        if Y.ndim == 2:
            mask = torch.ones_like(Y)
            mask[:-1] = 0
            Y = Y * mask
        return X, Y
