import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Literal

class CTRNN(nn.Module):
    """
    Continuous-time RNN (rate-based) with Euler step, optional noise and constraints.

    Update:
        v_t = (1 - alpha) * v_{t-1} + alpha * ( W_in x_t + W_rec f_{t-1} + b )
        f_t = act(v_t) + post_noise
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
        ei_ratio: float = 0.8,
        no_self_connections: bool = True,
        scaling: float = 1.0,
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

        # layers
        self.input_layer = nn.Linear(self.I, self.H, bias=bias)
        self.hidden_layer = nn.Linear(self.H, self.H, bias=bias)
        self.readout_layer = nn.Linear(self.H, self.O, bias=bias)

        # inits
        nn.init.kaiming_uniform_(self.input_layer.weight, a=0.0)
        nn.init.zeros_(self.input_layer.bias)

        nn.init.kaiming_uniform_(self.hidden_layer.weight, a=0.0)
        self.hidden_layer.weight.data *= scaling
        nn.init.zeros_(self.hidden_layer.bias)

        nn.init.kaiming_uniform_(self.readout_layer.weight, a=0.0)
        nn.init.zeros_(self.readout_layer.bias)

        # Dale's Law
        if self.use_dale:
            n_exc = int(round(ei_ratio * self.H))
            sign = torch.cat([torch.ones(n_exc), -torch.ones(self.H - n_exc)]).view(1, -1)
            self.register_buffer("dale_sign", sign)
            with torch.no_grad():
                W = self.hidden_layer.weight.data
                self.hidden_layer.weight.data = W.abs() * self.dale_sign

        # remove self-connections
        if self.no_self_connections:
            with torch.no_grad():
                self.hidden_layer.weight.data.fill_diagonal_(0.0)

        # noise gate
        self.register_buffer("_noise_enabled", torch.tensor(1, dtype=torch.uint8))

    # utils
    def act(self, x: torch.Tensor) -> torch.Tensor:
        if self._activation_name == "relu":
            return F.relu(x)
        if self._activation_name == "tanh":
            return torch.tanh(x)
        if self._activation_name == "softplus":
            return F.softplus(x)
        raise ValueError(f"unknown activation {self._activation_name}")

    def enable_noise(self, enabled: bool = True):
        self._noise_enabled.fill_(1 if enabled else 0)

    def train(self, mode: bool = True):
        super().train(mode)
        self.enable_noise(mode)
        return self

    def eval(self):
        super().eval()
        self.enable_noise(False)
        return self

    # core
    def init_state(self, B: int, device=None):
        v0 = torch.zeros(B, self.H, device=device)
        fr0 = self.act(v0)
        return fr0, v0

    def step(self, fr_t: torch.Tensor, v_t: torch.Tensor, u_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # affine transforms
        w_in_u = self.input_layer(u_t)
        w_h_fr = self.hidden_layer(fr_t)

        # continuous-time Euler update
        v_t = self.oneminusalpha * v_t + self.alpha * (w_in_u + w_h_fr)

        # optional pre-activation noise
        if self.preact_noise > 0.0 and bool(self._noise_enabled.item()):
            v_t = v_t + self.alpha * torch.randn_like(v_t) * self.preact_noise

        # nonlinearity
        fr = self.act(v_t)

        # optional post-activation noise
        if self.postact_noise > 0.0 and bool(self._noise_enabled.item()):
            fr = fr + torch.randn_like(fr) * self.postact_noise

        return fr, v_t

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inputs: (T,B,I) -> returns (logits: (T,B,O), hidden_states: (T,B,H))
        """
        T, B, _ = inputs.shape
        device = inputs.device
        fr, v = self.init_state(B, device)
        hs = []
        for t in range(T):
            fr, v = self.step(fr, v, inputs[t])
            hs.append(fr)
        hidden_seq = torch.stack(hs, dim=0)
        logits = self.readout_layer(hidden_seq)
        return logits, hidden_seq

    def forward_sequence(self, x):
        logits, _ = self.forward(x)
        return logits

    def hidden_sequence(self, x):
        _, h = self.forward(x)
        return h

    # save/load
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str = "cpu"):
        state = torch.load(path, map_location=map_location)  # removed weights_only
        self.load_state_dict(state)
