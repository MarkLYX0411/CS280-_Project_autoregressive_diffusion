import torch, torch.nn as nn
import math
from layers    import FMTBlock
from layers    import Attention   # (exported for later, optional)
from utils     import local_window_mask
from embedding import FrameEmbed, ClassEmbed, TimestepEmbed

class FlowMatchingTransformer(nn.Module):
    def __init__(self,
                 chunk_frames: int = 12,
                 prev_frames:  int = 4,
                 dim_latent:   int = 64,     # VAE latent depth
                 hidden_dim:   int = 256,
                 depth:        int = 6,
                 heads:        int = 4,
                 mlp_ratio:    float = 4.0,
                 num_class:    int = 10,
                 attn_window:  int = 2,
                 device:       str = 'cuda'):
        super().__init__()
        self.device = device
        self.T_cur   = chunk_frames
        self.T_prev  = prev_frames
        self.T_total = self.T_prev + self.T_cur

        # positional encoding (fixed sinusoid)
        pos = torch.arange(self.T_total, dtype=torch.float32)
        inv = torch.exp(-torch.arange(0, hidden_dim, 2) * math.log(10000)/hidden_dim)
        pe  = torch.zeros(self.T_total, hidden_dim)
        pe[:,0::2] = torch.sin(pos[:,None]*inv)
        pe[:,1::2] = torch.cos(pos[:,None]*inv)
        self.register_buffer('pos_emb', pe.unsqueeze(0))      # (1,T,D)

        # embedders
        self.frame_embed = nn.Linear(dim_latent, hidden_dim)
        self.time_embed  = TimestepEmbed(hidden_dim)
        self.class_embed = ClassEmbed(num_class, hidden_dim)

        # transformer blocks
        self.blocks = nn.ModuleList([
            FMTBlock(hidden_dim, heads, mlp_ratio) for _ in range(depth)
        ])

        # decoder: hidden → latent dim
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.linear   = nn.Linear(hidden_dim, dim_latent)

        # causal/local mask (same for every batch)
        mask = local_window_mask(self.T_total, attn_window)
        self.register_buffer('attn_mask', mask, persistent=False)

    # ---------- core forward ----------
    def forward(self,
                t:      torch.Tensor,      # (B,) scalar in [0,1]
                x_cur:  torch.Tensor,      # (B, T_cur, dim_latent)
                c_onehot: torch.Tensor,    # (B, num_class)
                x_prev: torch.Tensor = None   # (B, T_prev, dim_latent)
                ):
        """
        Returns velocity field v_theta((1‑t)x0 + t x1, cond)
        Shape → (B, T_cur, dim_latent)
        """
        B = x_cur.size(0)
        # history concat
        if x_prev is None:
            x_prev = torch.zeros(B, self.T_prev, x_cur.size(-1),
                                 device=x_cur.device, dtype=x_cur.dtype)
        x_seq = torch.cat([x_prev, x_cur], dim=1)       # (B,T_total,D_latent)

        # token embed + position
        h = self.frame_embed(x_seq) + self.pos_emb      # (B,T_total,hidden)

        # condition embedding (broadcast across frames)
        cond = self.time_embed(t) + self.class_embed(c_onehot)
        cond = cond.unsqueeze(1)                        # (B,1,hidden)

        # run transformer
        for blk in self.blocks:
            h = blk(h, cond, self.attn_mask)

        # decode & slice out prev frames
        h = self.linear(self.norm_out(h))               # (B,T_total,dim_latent)
        v_cur = h[:, self.T_prev:]                      # (B,T_cur,dim_latent)
        return v_cur                                    # velocity field