import torch
import torch.nn.functional as F


def per_lead_mean_std(x: torch.Tensor):
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True) + 1e-6
    return mu, sd


def zscore(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor):
    return (x - mu) / sd


def smooth_moving_average(x: torch.Tensor, k: int):
    """
    Moving average smoothing along the time axis.
    x: [B, C, T]
    """
    if k <= 1:
        return x
    pad = k // 2
    C = x.shape[1]
    w = torch.ones((C, 1, k), device=x.device, dtype=x.dtype) / float(k)
    return F.conv1d(F.pad(x, (pad, pad), mode="reflect"), w, groups=C)


def build_wave_input(x_raw: torch.Tensor, smooth_k: int):
    """
    Build 24-channel input by concatenating:
      - per-lead zscore
      - smoothed signal scaled by per-lead std
    """
    mu, sd = per_lead_mean_std(x_raw)
    xz = zscore(x_raw, mu, sd)
    xs = smooth_moving_average(x_raw, smooth_k) / sd
    return torch.cat([xz, xs], dim=1)


def pooled_latent_mu(enc, x12: torch.Tensor, y: torch.Tensor):
    """
    enc: CondEncoder
    x12: [B,12,T] normalized
    y:   [B]
    returns: [B, z_ch]
    """
    mu, _ = enc(x12, y)
    return mu.mean(dim=-1)
