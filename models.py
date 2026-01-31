import numpy as np
import torch
import torch.nn as nn


class SmallResNet1D(nn.Module):
    def __init__(self, in_ch: int = 12, base: int = 64, multihead: bool = True):
        super().__init__()
        self.multihead = bool(multihead)

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=7, padding=3),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.b1 = self._block(base)
        self.b2 = self._block(base)
        self.skip = nn.Conv1d(base, base, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flat = nn.Flatten()

        if self.multihead:
            self.head_mi = nn.Linear(base, 1)
            self.head_norm = nn.Linear(base, 1)
        else:
            self.head = nn.Linear(base, 1)

    def _block(self, ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        r = x
        x = self.b1(x) + self.skip(r)
        x = torch.relu(x)

        r = x
        x = self.b2(x) + self.skip(r)
        x = torch.relu(x)

        x = self.pool(x)
        x = self.flat(x)
        return x

    def forward(self, x: torch.Tensor):
        feat = self.forward_features(x)
        if self.multihead:
            return {
                "mi": self.head_mi(feat).squeeze(1),
                "norm": self.head_norm(feat).squeeze(1),
            }
        return self.head(feat).squeeze(1)


class CondEncoder(nn.Module):
    """
    Conditional encoder (ECG -> latent feature map).

    x: [B, 12, T]
    y: [B]  (integer class id)
    returns: (mu, logvar)
    """
    def __init__(self, in_ch: int = 12, z_ch: int = 64, y_dim: int = 2, y_emb: int = 16):
        super().__init__()
        self.y_emb = nn.Embedding(y_dim, y_emb)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch + y_emb, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.to_mu = nn.Conv1d(256, z_ch, kernel_size=1)
        self.to_lv = nn.Conv1d(256, z_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        emb = self.y_emb(y).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        h = torch.cat([x, emb], dim=1)
        h = self.net(h)
        return self.to_mu(h), self.to_lv(h)


class CondDecoder(nn.Module):
    """
    Conditional decoder (latent feature map -> ECG).

    z: [B, z_ch, Tz]
    y: [B]
    returns: x_hat [B, 12, T]
    """
    def __init__(self, out_ch: int = 12, z_ch: int = 64, y_dim: int = 2, y_emb: int = 16):
        super().__init__()
        self.y_emb = nn.Embedding(y_dim, y_emb)
        self.net = nn.Sequential(
            nn.Conv1d(z_ch + y_emb, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_ch, kernel_size=7, padding=3),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        emb = self.y_emb(y).unsqueeze(-1).repeat(1, 1, z.shape[-1])
        h = torch.cat([z, emb], dim=1)
        return self.net(h)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu * mu - 1.0 - logvar)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    acp = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - (acp[1:] / acp[:-1])
    return torch.clip(betas, 1e-5, 0.999)


class TimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(0, half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return self.mlp(emb)


class SimpleUNet1D(nn.Module):
    """
    Tiny UNet-like network used in latent diffusion (denoise epsilon).

    zt:    [B, z_ch, Lz]
    t:     [B]
    y:     [B]
    rmask: [B, 1, T] or None
    """
    def __init__(self, z_ch: int = 64, y_dim: int = 2, y_emb: int = 32, time_dim: int = 128):
        super().__init__()
        self.t_emb = TimeEmbed(time_dim)
        self.y_emb = nn.Embedding(y_dim, y_emb)

        self.in_conv = nn.Conv1d(z_ch + y_emb + 1, 128, kernel_size=3, padding=1)
        self.mid = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.out_conv = nn.Conv1d(128, z_ch, kernel_size=3, padding=1)

        self.to_scale = nn.Linear(time_dim, 128)
        self.to_shift = nn.Linear(time_dim, 128)

    def forward(self, zt: torch.Tensor, t: torch.Tensor, y: torch.Tensor, rmask: torch.Tensor | None):
        B, _, Lz = zt.shape
        te = self.t_emb(t)
        scale = self.to_scale(te).unsqueeze(-1)
        shift = self.to_shift(te).unsqueeze(-1)

        ye = self.y_emb(y).unsqueeze(-1).repeat(1, 1, Lz)

        if rmask is None:
            rm = torch.zeros((B, 1, Lz), device=zt.device)
        else:
            rm = torch.nn.functional.adaptive_avg_pool1d(rmask, Lz)

        h = torch.cat([zt, ye, rm], dim=1)
        h = self.in_conv(h)
        h = h * (1 + scale) + shift
        h = self.mid(h)
        return self.out_conv(h)