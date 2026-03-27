"""Helpers to sample NeuroGym environments into RNN-ready tensors."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
from neurogym import Dataset
import copy

try:
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover - gym fallback
    try:
        import gym  # type: ignore
    except Exception:  # pragma: no cover
        gym = None


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
            targets[-1, idx] = -1
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
        mask_fixation: bool = False,
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
        self.mask_fixation = bool(mask_fixation)
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
        target_array = np.asarray(targets)
        invalid_mask = None
        if target_array.ndim == 3:
            if target_array.shape[-1] == 1:
                invalid_mask = np.isnan(target_array[..., 0])
                target_array = target_array[..., 0]
            else:
                invalid_mask = np.isnan(target_array).any(axis=-1)
                safe = np.where(np.isnan(target_array), -np.inf, target_array)
                target_array = np.argmax(safe, axis=-1)
        else:
            if np.issubdtype(target_array.dtype, np.floating):
                invalid_mask = np.isnan(target_array)
        if self.mask_fixation:
            zero_mask = target_array == 0
            invalid_mask = zero_mask if invalid_mask is None else (invalid_mask | zero_mask)
        Y = torch.from_numpy(target_array).clone()
        if invalid_mask is not None:
            Y = Y.to(torch.float32)
            Y[torch.from_numpy(invalid_mask)] = -1.0
        Y = Y.to(torch.long)
        if not self.mask_fixation:
            Y = _ensure_final_labels(Y)
        if not self.last_only:
            return X, Y
        # zero out unused intermediate labels while keeping the final decision
        if Y.ndim == 2:
            Y[:-1] = -1
        return X, Y


def _seed_env(env, seed: int) -> None:
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
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        try:
            env.action_space.seed(seed)
        except Exception:
            pass


def _labels_from_targets(targets: np.ndarray) -> np.ndarray:
    arr = np.asarray(targets)
    if arr.ndim >= 2 and arr.shape[-1] > 1:
        nan_mask = np.isnan(arr).any(axis=-1) if np.issubdtype(arr.dtype, np.floating) else None
        safe = np.where(np.isnan(arr), -np.inf, arr) if nan_mask is not None else arr
        labels = np.argmax(safe, axis=-1)
        if nan_mask is not None:
            labels = labels.astype(np.float32, copy=False)
            labels[nan_mask] = -1.0
        return labels.astype(np.int64, copy=False)
    if arr.ndim >= 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=False)
        arr[np.isnan(arr)] = -1.0
    return arr.astype(np.int64, copy=False)


class ModCogTrialDM:
    """Trial-aligned Mod-Cog dataset to avoid slicing mid-trial."""

    def __init__(
        self,
        env: str | Any,
        T: int,
        B: int,
        *,
        device: str = "cpu",
        last_only: bool = False,
        seed: int = 0,
        env_kwargs: dict | None = None,
        mask_fixation: bool = True,
    ):
        if last_only:
            raise ValueError("Mod-Cog trial datasets do not support last_only=True.")
        if gym is None and isinstance(env, str):
            raise RuntimeError("Gym is required to instantiate Mod-Cog environments.")
        self.T = int(T)
        self.B = int(B)
        self.device = device
        self.last_only = bool(last_only)
        self.mask_fixation = bool(mask_fixation)
        self._trial_kwargs = dict(env_kwargs or {})

        if isinstance(env, str):
            self.envs = [gym.make(env, **(env_kwargs or {})) for _ in range(self.B)]
        else:
            self.envs = [copy.deepcopy(env) for _ in range(self.B)]

        for idx, env_i in enumerate(self.envs):
            _seed_env(env_i, seed + idx)

        obs_shape = getattr(self.envs[0].observation_space, "shape", None)
        if not obs_shape:
            raise ValueError(f"Environment {env} has no observation shape.")
        self.input_dim = int(np.prod(obs_shape))
        action_space = self.envs[0].action_space
        if hasattr(action_space, "n"):
            self.n_classes = int(action_space.n)
        else:
            shape = getattr(action_space, "shape", None)
            if not shape:
                raise ValueError(f"Environment {env} has no action shape.")
            self.n_classes = int(np.prod(shape))

    def _sample_single(self, env) -> Tuple[np.ndarray, np.ndarray]:
        try:
            new_trial_fn = env.get_wrapper_attr("new_trial")
        except Exception:
            new_trial_fn = getattr(getattr(env, "unwrapped", env), "new_trial")
        try:
            new_trial_fn(**self._trial_kwargs)
        except TypeError:
            new_trial_fn()
        try:
            ob = env.get_wrapper_attr("ob")
        except Exception:
            ob = getattr(getattr(env, "unwrapped", env), "ob")
        ob = np.asarray(ob)
        try:
            gt = env.get_wrapper_attr("gt")
        except Exception:
            gt = getattr(getattr(env, "unwrapped", env), "gt")
        gt = np.asarray(gt)
        ob = ob.reshape(ob.shape[0], -1)
        labels = _labels_from_targets(gt)
        if labels.ndim > 1:
            labels = labels.reshape(labels.shape[0], -1)
            labels = labels[:, 0]
        if self.mask_fixation:
            labels = labels.astype(np.int64, copy=False)
            labels[labels == 0] = -1
        return ob, labels

    def sample_batch(self):
        T, B, I = self.T, self.B, self.input_dim
        X = torch.zeros(T, B, I, device=self.device)
        Y = torch.full((T, B), -1, dtype=torch.long, device=self.device)
        for b, env in enumerate(self.envs):
            ob, labels = self._sample_single(env)
            t_len = min(T, ob.shape[0])
            if t_len > 0:
                X[:t_len, b] = torch.from_numpy(ob[:t_len]).float().to(self.device)
                Y[:t_len, b] = torch.from_numpy(labels[:t_len]).to(torch.long).to(self.device)
        return X, Y
