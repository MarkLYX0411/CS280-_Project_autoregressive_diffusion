import torch
import torch.nn.functional as F

def local_window_mask(T: int,
                      window: int = 2,
                      extra_tokens: int = 0) -> torch.Tensor:
    """
    Allow each frame‑token to attend to ±window frames.
    Returns a Boolean mask, True = *block*.
    Shape: (T+extra, T+extra)
    """
    mask = torch.ones(T, T, dtype=torch.bool)
    for i in range(T):
        lo = max(0, i - window)
        hi = min(T, i + window + 1)
        mask[i, lo:hi] = False
    if extra_tokens:
        extra = torch.zeros(T, extra_tokens, dtype=torch.bool)
        mask = torch.cat([mask, extra], dim=1)
        extra_row = torch.zeros(extra_tokens, T + extra_tokens, dtype=torch.bool)
        mask = torch.cat([mask, extra_row], dim=0)
    return mask     # (T+E, T+E)

def get_grad_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute the total norm of gradients across all parameters in `model`.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            # p.grad is a tensor of the same shape as p
            param_norm = p.grad.data.norm(norm_type)      # :contentReference[oaicite:0]{index=0}
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

def focal_mse(pred, target, beta=10.0, gamma=2.0, eps=1e-8):
    # |target| is proxy for importance (non‑zero on strokes)
    weight = 1.0 - torch.exp(-beta * target.abs())
    # optional: clamp to [0,1]
    weight = weight.clamp(max=1.0)
    if gamma != 1.0:
        weight = weight.pow(gamma)
    loss = weight * (pred - target).pow(2)
    return loss.mean()          # mean over all pixels / latents

