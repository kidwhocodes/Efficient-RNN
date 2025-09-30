# neurogym_data.py
import torch
import numpy as np

class NeuroGymDM:
    def __init__(self, env, T: int, B: int, device="cpu"):
        self.env = env
        self.T = T
        self.B = B
        self.device = device
        obs_space = env.observation_space.shape[0]
        self.input_dim = obs_space
        self.n_classes = env.action_space.n

    def sample_batch(self):
        X = torch.zeros(self.T, self.B, self.input_dim, device=self.device)
        Y = torch.zeros(self.T, self.B, dtype=torch.long, device=self.device)
        # neurogym_data.py (inside sample_batch)
        for b in range(self.B):
            obs = self.env.reset()
            last_gt = 0
            for t in range(self.T):
                X[t, b] = torch.from_numpy(obs).float().to(self.device)
                action = self.env.action_space.sample()
                obs, _, _, info = self.env.step(action)
                if "gt" in info:
                    last_gt = int(info["gt"])
            Y[self.T - 1, b] = last_gt   # label only at the end

        return X, Y
