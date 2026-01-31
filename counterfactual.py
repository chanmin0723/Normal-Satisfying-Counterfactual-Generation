import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pooled_latent_mu


def optimize_latent_counterfactual(
    x12_norm: torch.Tensor,
    y_enc: torch.Tensor,
    enc,
    dec,
    clf_mi,
    clf_norm,
    flow_norm,
    steps: int = 200,
    lr: float = 1e-2,
    lam_prox: float = 0.1,
    lam_flow: float = 0.1,
    target: str = "norm",
):
    """
    Latent-space counterfactual optimization (minimal public version).

    Inputs
      - x12_norm: [B,12,T] (already normalized)
      - y_enc:    [B] encoder condition ids
      - enc/dec:  conditional VAE modules
      - clf_mi/clf_norm: classifier heads that accept waveform-space features
      - flow_norm: RealNVP density model for "normal" embeddings (vector space)
      - target: "norm" or "mi"

    Returns
      - x_cf: [B,12,T] reconstructed counterfactual
      - z_cf: [B,z_ch,Lz] optimized latent map
    """
    enc.eval()
    dec.eval()
    z0, _ = enc(x12_norm, y_enc)
    z = z0.detach().clone()
    z.requires_grad_(True)

    opt = torch.optim.Adam([z], lr=lr)

    for _ in range(steps):
        x_hat = dec(z, y_enc)

        logits = clf_mi(x_hat)
        if isinstance(logits, dict):
            mi_logit = logits["mi"]
            norm_logit = logits["norm"]
        else:
            mi_logit = logits
            norm_logit = torch.zeros_like(mi_logit)

        if target == "norm":
            loss_target = F.softplus(-norm_logit).mean()
        else:
            loss_target = F.softplus(-mi_logit).mean()

        z_vec = z.mean(dim=-1)
        flow_nll = flow_norm.nll(z_vec).mean()

        loss_prox = F.mse_loss(z, z0)

        loss = loss_target + lam_flow * flow_nll + lam_prox * loss_prox

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    x_cf = dec(z.detach(), y_enc)
    return x_cf.detach(), z.detach()
