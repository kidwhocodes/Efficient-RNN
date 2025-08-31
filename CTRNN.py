# ctrnn_oneline.py
# A one-line friendly CTRNN + tiny trainer + pruning demo (no extra deps beyond torch/numpy).

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Core: one-line CTRNN
# -----------------------------
class CTRNN(nn.Module):
    """
    Continuous-time RNN (rate-based) with Euler step, optional noise and constraints.

    Update per step:
      v_t = (1 - alpha) * v_{t-1} + alpha * ( W_in x_t + W_rec fr_{t-1} + b )
      fr_t = act(v_t) + post_noise
      (optionally add pre-activation noise before act)

    One-line constructor example:
      model = CTRNN(input_dim=3, hidden_size=64, output_dim=3,
                    dt=100, tau=100, activation="relu",
                    preact_noise=0.0, postact_noise=0.0,
                    use_dale=False, no_self_connections=True, scaling=1.0)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 64,
        output_dim: int = 1,
        *,
        dt: float = 100.0,
        tau: float = 100.0,
        activation: Literal["relu", "tanh", "softplus"] = "relu",
        preact_noise: float = 0.0,
        postact_noise: float = 0.0,
        use_dale: bool = False,
        ei_ratio: float = 0.8,                 # fraction excitatory if Dale is used
        no_self_connections: bool = True,      # zero diagonal of W_rec
        scaling: float = 1.0,                  # scale recurrent init
        bias: bool = True,
    ):
        super().__init__()
        self.I, self.H, self.O = input_dim, hidden_size, output_dim
        self.alpha = float(dt) / float(tau)
        self.oneminusalpha = 1.0 - self.alpha
        self._activation_name = activation
        self.preact_noise = float(preact_noise)
        self.postact_noise = float(postact_noise)
        self.use_dale = bool(use_dale)
        self.no_self_connections = bool(no_self_connections)

        # Layers
        self.input_layer = nn.Linear(self.I, self.H, bias=bias)
        self.hidden_layer = nn.Linear(self.H, self.H, bias=bias)
        self.readout_layer = nn.Linear(self.H, self.O, bias=bias)

        # Inits
        nn.init.kaiming_uniform_(self.input_layer.weight, a=0.0)
        nn.init.zeros_(self.input_layer.bias)
        # Recurrent scaled orthogonal-ish init for stability
        nn.init.kaiming_uniform_(self.hidden_layer.weight, a=0.0)
        self.hidden_layer.weight.data *= scaling
        nn.init.zeros_(self.hidden_layer.bias)
        nn.init.kaiming_uniform_(self.readout_layer.weight, a=0.0)
        nn.init.zeros_(self.readout_layer.bias)

        # Dale's law mask (sign constraints) if requested
        if self.use_dale:
            n_exc = int(round(ei_ratio * self.H))
            sign = torch.cat([torch.ones(n_exc), -torch.ones(self.H - n_exc)])  # (H,)
            self.register_buffer("dale_sign", sign.view(1, -1))  # broadcast across rows
            # project current W_rec to Dale-compliant by taking abs and applying signs column-wise
            with torch.no_grad():
                W = self.hidden_layer.weight.data
                self.hidden_layer.weight.data = W.abs() * self.dale_sign  # columns excit/inhib

        # Remove self connections if requested (diagonal -> 0)
        if self.no_self_connections:
            with torch.no_grad():
                self.hidden_layer.weight.data.fill_diagonal_(0.0)

        # Buffers for noise gating (train(): on, eval(): off)
        self.register_buffer("_noise_enabled", torch.tensor(1, dtype=torch.uint8))

    # --- utilities ---
    def act(self, x: torch.Tensor) -> torch.Tensor:
        if self._activation_name == "relu":
            return F.relu(x)
        if self._activation_name == "tanh":
            return torch.tanh(x)
        if self._activation_name == "softplus":
            return F.softplus(x)
        raise ValueError(f"unknown activation {self._activation_name}")

    def enable_noise(self, enabled: bool = True):
        # _noise_enabled is a 0-dim tensor; use fill_() instead of slicing
        self._noise_enabled.fill_(1 if enabled else 0)

    def train(self, mode: bool = True):
        # keep Module's normal behavior
        super().train(mode)
        # noise/constraints ON in train, OFF in eval (bool(mode))
        self.enable_noise(mode)
        return self

    def eval(self):
        # call parent .eval() (which internally calls .train(False))
        super().eval()
        # be explicit and set noise off
        self.enable_noise(False)
        return self


    # --- core recurrence ---
    def init_state(self, B: int, device=None):
        v0 = torch.zeros(B, self.H, device=device)
        fr0 = self.act(v0)
        return fr0, v0

    def step(self, fr_t: torch.Tensor, v_t: torch.Tensor, u_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Affine input
        w_in_u = self.input_layer(u_t)              # (B,H)
        w_h_fr = self.hidden_layer(fr_t)            # (B,H)

        # Dale/self-connection enforcement every step (cheap & explicit)
        if self.use_dale:
            with torch.no_grad():
                W = self.hidden_layer.weight.data
                self.hidden_layer.weight.data = W.abs() * self.dale_sign
        if self.no_self_connections:
            with torch.no_grad():
                self.hidden_layer.weight.data.fill_diagonal_(0.0)

        # Euler membrane update
        v_t = self.oneminusalpha * v_t + self.alpha * (w_in_u + w_h_fr)

        # pre-activation noise
        if self.preact_noise > 0.0 and bool(self._noise_enabled.item()):
            v_t = v_t + self.alpha * torch.randn_like(v_t) * self.preact_noise

        fr_t = self.act(v_t)

        # post-activation noise
        if self.postact_noise > 0.0 and bool(self._noise_enabled.item()):
            fr_t = fr_t + torch.randn_like(fr_t) * self.postact_noise

        return fr_t, v_t

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inputs: (T,B,I)  -> returns (logits: (T,B,O), hidden_states: (T,B,H))
        """
        T, B, _ = inputs.shape
        device = inputs.device
        fr, v = self.init_state(B, device)
        hs = []
        for t in range(T):
            fr, v = self.step(fr, v, inputs[t])
            hs.append(fr)
        hidden_seq = torch.stack(hs, dim=0)         # (T,B,H)
        logits = self.readout_layer(hidden_seq)     # (T,B,O)
        return logits, hidden_seq

    # --- Save/Load like the article showed ---
    def save(self, path: str):
        torch.save({"state": self.state_dict(),
                    "meta": {"I": self.I, "H": self.H, "O": self.O,
                             "act": self._activation_name}}, path)

    def load(self, path: str, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location or "cpu")
        self.load_state_dict(ckpt["state"])


# -----------------------------
# Tiny synthetic task (2AFC)
# -----------------------------
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


# -----------------------------
# Train / Eval helpers
# -----------------------------
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(loss) * N
        total_count += N
    return total_loss / max(1, total_count)

@torch.no_grad()
def evaluate(model, data, device, criterion, steps=20, last_only=True):
    model.eval()  # disables noise/constraints as per article convention
    total_loss, total_correct, total_count = 0.0, 0, 0
    for _ in range(steps):
        x, y = data.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        if last_only:
            loss = criterion(logits[-1], y[-1])
            pred = logits[-1].argmax(-1)
            N = y[-1].numel()
            correct = (pred == y[-1]).sum().item()
        else:
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            pred = logits.argmax(-1)
            N = y.numel()
            correct = (pred == y).sum().item()
        total_loss += float(loss) * N
        total_correct += correct
        total_count += N
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


# -----------------------------
# Pruning utility (magnitude)
# -----------------------------
def prune_recurrent_unstructured(model: CTRNN, amount: float = 0.5):
    """
    Prune a fraction `amount` of the smallest-magnitude weights in the recurrent matrix.
    """
    import torch.nn.utils.prune as P
    P.l1_unstructured(model.hidden_layer, name="weight", amount=amount)
    # Optional: make pruning permanent
    # P.remove(model.hidden_layer, "weight")


# -----------------------------
# Demo run (train + prune + eval)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build CTRNN in *one line* (inspired by the article)
    model = CTRNN(
        input_dim=3, hidden_size=64, output_dim=3,
        dt=100, tau=100, activation="relu",
        preact_noise=0.05, postact_noise=0.0,     # noise ON in train(), OFF in eval()
        use_dale=False, no_self_connections=True, scaling=1.0
    ).to(device)

    # Data/task
    cfg = SynthCfg(T=60, B=64, coh_levels=(0.0, 0.05, 0.1, 0.2), stim_std=0.6)
    data = SyntheticDM(cfg)

    # Train
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    print(model)
    # quick baseline before training
    va0_loss, va0_acc = evaluate(model, data, device, criterion, steps=50, last_only=True)
    print(f"before training | valid loss {va0_loss:.4f} | acc {va0_acc*100:.1f}%")

    for ep in range(1, 11):
        tr_loss = train_epoch(model, data, device, opt, criterion, steps=60, last_only=True)
        va_loss, va_acc = evaluate(model, data, device, criterion, steps=50, last_only=True)
        print(f"epoch {ep:02d} | train {tr_loss:.4f} | valid {va_loss:.4f} | acc {va_acc*100:.1f}%")

    # Save / Load demo (optional)
    model.save("my_ctrnn.pth")
    # (To demonstrate load:)
    _tmp = CTRNN(input_dim=3, hidden_size=64, output_dim=3)
    _tmp.load("my_ctrnn.pth")

    # ---- Pruning experiment ----
    va_pre_loss, va_pre_acc = evaluate(model, data, device, criterion, steps=100, last_only=True)
    print(f"[pre-prune]  valid loss {va_pre_loss:.4f} | acc {va_pre_acc*100:.1f}%")

    prune_recurrent_unstructured(model, amount=0.5)  # prune 50% of recurrent synapses
    # (optional) brief finetune after pruning
    for _ in range(3):
        _ = train_epoch(model, data, device, opt, criterion, steps=20, last_only=True)

    va_post_loss, va_post_acc = evaluate(model, data, device, criterion, steps=100, last_only=True)
    print(f"[post-prune] valid loss {va_post_loss:.4f} | acc {va_post_acc*100:.1f}%")
