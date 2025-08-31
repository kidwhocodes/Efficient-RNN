# ctrnn_train.py
# Train a Continuous-Time RNN (CTRNN) on a synthetic perceptual decision task.
# No external deps besides torch and numpy.
#
# Usage:
#   pip install torch numpy
#   python ctrnn_train.py --epochs 10
#
# Optional knobs:
#   python ctrnn_train.py --hidden 64 --T 100 --B 64 --lr 1e-2 --dt 100 --tau 100

import argparse
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# -----------------------------
# Config & seed
# -----------------------------
@dataclass
class TrainConfig:
    T: int = 100                 # time steps
    B: int = 64                  # batch size
    input_size: int = 3          # [bias, left, right]
    output_size: int = 3         # [left, right, no-go] (we'll use {0,1})
    hidden_size: int = 64
    lr: float = 1e-2
    epochs: int = 10
    steps_per_epoch: int = 100   # synthetic stream; how many batches per epoch
    dt: float = 100.0            # ms; Euler step
    tau: float = 100.0           # ms; neuron time constant
    stim_std: float = 0.5
    coh_levels: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "ctrnn.pt"

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Synthetic decision-making task
# -----------------------------
class SyntheticDM:
    """
    2AFC: choose LEFT (0) or RIGHT (1). Each step gives noisy evidence.
    Input per step: [1.0, left_t, right_t]
    Labels Y[t, b] are constant over time for that trial; we train on the last step.
    """
    def __init__(self, T, B, input_size, output_size, coh_levels, stim_std):
        assert input_size == 3 and output_size == 3
        self.T, self.B = T, B
        self.I, self.O = input_size, output_size
        self.coh_levels, self.stim_std = coh_levels, stim_std

    def sample_batch(self):
        T, B, I = self.T, self.B, self.I
        X = np.zeros((T, B, I), dtype=np.float32)
        Y = np.zeros((T, B), dtype=np.int64)
        X[:, :, 0] = 1.0  # bias channel

        for b in range(B):
            side = np.random.randint(0, 2)                 # 0=left, 1=right
            coh = np.random.choice(self.coh_levels)
            signed = +coh if side == 1 else -coh            # positive favors right
            mu_left, mu_right = (-signed, +signed)

            left  = np.random.normal(mu_left,  self.stim_std, size=T).astype(np.float32)
            right = np.random.normal(mu_right, self.stim_std, size=T).astype(np.float32)
            X[:, b, 1] = left
            X[:, b, 2] = right
            Y[:, b] = side

        return torch.from_numpy(X), torch.from_numpy(Y)


# -----------------------------
# CTRNN core (fused affine)
# -----------------------------
class CTRNN(nn.Module):
    """
    Continuous-time rate RNN (Euler discretization, ReLU rates).

    Update:
      h_t = ReLU( (1 - alpha) * h_{t-1} + alpha * (W @ [x_t, h_{t-1}] + b) )
    where alpha = dt / tau.

    Shapes:
      x: (T, B, I)   h: (B, H)   output: (T, B, H), (B, H)
    """
    def __init__(self, input_size, hidden_size, dt=100.0, tau=100.0):
        super().__init__()
        self.I, self.H = input_size, hidden_size
        self.alpha = float(dt) / float(tau)
        self.oneminusalpha = 1.0 - self.alpha

        # one affine on concatenated [x, h] for fewer kernel launches
        self.affine = nn.Linear(self.I + self.H, self.H, bias=True)
        nn.init.kaiming_uniform_(self.affine.weight, a=0.0)
        nn.init.zeros_(self.affine.bias)

    def init_hidden(self, B, device=None):
        h0 = torch.zeros(B, self.H)
        return h0.to(device) if device is not None else h0

    def forward(self, x, h0=None):
        T, B, _ = x.shape
        device = x.device
        h = self.init_hidden(B, device) if h0 is None else h0
        outs = []
        for t in range(T):
            z = torch.cat([x[t], h], dim=-1)   # (B, I+H)
            pre = self.affine(z)               # (B, H)
            h = F.relu(self.oneminusalpha * h + self.alpha * pre)
            outs.append(h)
        y = torch.stack(outs, dim=0)           # (T, B, H)
        return y, h


# -----------------------------
# Model = CTRNN + linear readout
# -----------------------------
class CTRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=100.0, tau=100.0):
        super().__init__()
        self.core = CTRNN(input_size, hidden_size, dt=dt, tau=tau)
        self.head = nn.Linear(hidden_size, output_size)
        nn.init.kaiming_uniform_(self.head.weight, a=0.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h_seq, _ = self.core(x)        # (T,B,H)
        logits = self.head(h_seq)      # (T,B,O)
        return logits, h_seq


# -----------------------------
# Training / Evaluation
# -----------------------------
def train_epoch(model, data, device, optimizer, criterion, steps, last_step_only=True, clip=1.0):
    model.train()
    total_loss, total_count = 0.0, 0
    for _ in range(steps):
        x, y = data.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        if last_step_only:
            loss = criterion(logits[-1], y[-1])             # (B,O) vs (B,)
            count = y[-1].numel()
        else:
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            count = y.numel()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()

        total_loss += float(loss) * count
        total_count += count
    return total_loss / max(1, total_count)

@torch.no_grad()
def evaluate(model, data, device, criterion, steps=20, last_step_only=True):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for _ in range(steps):
        x, y = data.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        if last_step_only:
            loss = criterion(logits[-1], y[-1])
            pred = logits[-1].argmax(dim=-1)
            correct = (pred == y[-1]).sum().item()
            count = y[-1].numel()
        else:
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            pred = logits.argmax(dim=-1)
            correct = (pred == y).sum().item()
            count = y.numel()

        total_loss += float(loss) * count
        total_correct += correct
        total_count += count
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=TrainConfig.T)
    parser.add_argument("--B", type=int, default=TrainConfig.B)
    parser.add_argument("--hidden", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--steps_per_epoch", type=int, default=TrainConfig.steps_per_epoch)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--dt", type=float, default=TrainConfig.dt)
    parser.add_argument("--tau", type=float, default=TrainConfig.tau)
    parser.add_argument("--save_path", type=str, default=TrainConfig.save_path)
    args = parser.parse_args()

    set_seed(0)
    cfg = TrainConfig(T=args.T, B=args.B, hidden_size=args.hidden,
                      epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                      lr=args.lr, dt=args.dt, tau=args.tau, save_path=args.save_path)

    device = cfg.device
    data = SyntheticDM(cfg.T, cfg.B, cfg.input_size, cfg.output_size,
                       cfg.coh_levels, cfg.stim_std)

    model = CTRNNClassifier(cfg.input_size, cfg.hidden_size, cfg.output_size,
                            dt=cfg.dt, tau=cfg.tau).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    print(model)
    for ep in range(1, cfg.epochs + 1):
        tr_loss = train_epoch(model, data, device, opt, criterion,
                              steps=cfg.steps_per_epoch, last_step_only=True, clip=1.0)
        va_loss, va_acc = evaluate(model, data, device, criterion,
                                   steps=20, last_step_only=True)
        print(f"epoch {ep:02d} | train loss {tr_loss:.4f} | valid loss {va_loss:.4f} | acc {va_acc*100:.1f}%")

    # Save checkpoint
    torch.save({"model_state": model.state_dict(),
                "cfg": cfg.__dict__}, cfg.save_path)
    print(f"saved checkpoint to {cfg.save_path}")

    # Quick sanity: show a tiny batch of final-step probabilities
    x, y = data.sample_batch()
    x = x.to(device)
    with torch.no_grad():
        logits, _ = model(x)
        probs = logits[-1].softmax(dim=-1).cpu().numpy()   # (B, 3)
    print("final-step class probs for first 5 trials (cols: left,right,no-go):")
    print(np.round(probs[:5], 3))

if __name__ == "__main__":
    main()
