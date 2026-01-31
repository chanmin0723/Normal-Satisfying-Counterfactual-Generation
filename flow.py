import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU(inplace=True))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RealNVP(nn.Module):
    """
    Minimal RealNVP density model for vector embeddings.

    x: [B, D]
    log_prob(x) returns [B]
    """
    def __init__(self, dim: int, n_coupling: int = 8, hidden: int = 256):
        super().__init__()
        self.dim = int(dim)
        self.n_coupling = int(n_coupling)

        masks = []
        for i in range(self.n_coupling):
            m = torch.zeros(self.dim)
            if i % 2 == 0:
                m[::2] = 1.0
            else:
                m[1::2] = 1.0
            masks.append(m)
        self.register_buffer("masks", torch.stack(masks, dim=0))

        self.s_nets = nn.ModuleList([MLP(self.dim, self.dim, hidden=hidden, depth=2) for _ in range(self.n_coupling)])
        self.t_nets = nn.ModuleList([MLP(self.dim, self.dim, hidden=hidden, depth=2) for _ in range(self.n_coupling)])

    def forward(self, x: torch.Tensor):
        logdet = torch.zeros(x.shape[0], device=x.device)
        z = x
        for k in range(self.n_coupling):
            m = self.masks[k].unsqueeze(0)
            z_m = z * m
            s = self.s_nets[k](z_m) * (1 - m)
            t = self.t_nets[k](z_m) * (1 - m)
            s = torch.tanh(s)
            z = z_m + (1 - m) * (z * torch.exp(s) + t)
            logdet = logdet + torch.sum(s, dim=1)
        return z, logdet

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, logdet = self.forward(x)
        logpz = -0.5 * (z * z + math.log(2 * math.pi)).sum(dim=1)
        return logpz + logdet

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x)
