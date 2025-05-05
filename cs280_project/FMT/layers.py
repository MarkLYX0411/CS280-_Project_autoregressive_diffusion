import math, torch, torch.nn as nn, torch.nn.functional as F
from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp


# ---------- Self‑attention (with optional fused SDPA) ----------
class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, attn_drop: float = 0.):
        super().__init__()
        self.heads  = heads
        self.scale  = (dim // heads) ** -0.5
        self.fused  = use_fused_attn()

        self.qkv    = nn.Linear(dim, dim * 3, bias=True)
        self.proj   = nn.Linear(dim, dim)
        self.drop   = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2,0,3,1,4)          # 3 × (B, H, N, D)
        if self.fused:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = ~mask if mask is not None else None,
                dropout_p = self.drop.p if self.training else 0.
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if mask is not None: attn.masked_fill_(mask[:,None], -1e9)
            attn = attn.softmax(-1)
            out  = attn @ v
        out = out.transpose(1,2).reshape(B, N, C) # (B,N,C)
        return self.proj(out)


# ---------- AdaLN‑guided Transformer block ----------
class FMTBlock(nn.Module):
    """
    DiT‑style block with AdaLayerNorm & gating.
    """
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn  = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp   = Mlp(in_features=dim,
                         hidden_features=int(dim*mlp_ratio),
                         act_layer=nn.GELU)
        # AdaLN modulation → [shift_msa, scale_msa, gate_msa,
        #                     shift_mlp, scale_mlp, gate_mlp]
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6*dim)
        )

    @staticmethod
    def modulate(x, shift, scale):       # AdaLN re‑scaling
        return x * (1 + scale) + shift

    def forward(self, x, cond, mask):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.mod(cond).chunk(6,dim=-1)
        x = x + g_msa * self.attn(
                self.modulate(self.norm1(x), s_msa, sc_msa), mask=mask)
        x = x + g_mlp * self.mlp(
                self.modulate(self.norm2(x), s_mlp, sc_mlp))
        return x
