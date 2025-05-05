import torch

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
