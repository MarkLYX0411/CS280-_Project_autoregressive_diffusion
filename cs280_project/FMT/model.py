from fmt import FlowMatchingTransformer
from vae import VAE
import torch
import numpy as np
from config.AutoregressiveFMTConfig import AutoregressiveFMTConfig
from typing import Optional
from utils import *

class AutoregressiveFMT(torch.nn.Module):
    """
    Wrapper = VAE (encoder / decoder)  +  Flow‑Matching Transformer
    """

    def __init__(self, config: AutoregressiveFMTConfig):
        super().__init__()
        self.config = config
        self.device = config.general.device
        torch.manual_seed(config.general.seed)

        # ---------- VAE ----------
        self.vae = VAE(z_ch=config.VAEConfig.z_ch,
                       kl_beta=config.VAEConfig.kl_beta)
        if config.VAEConfig.pretrained_path:
            ckpt = torch.load(config.VAEConfig.pretrained_path,
                              map_location=self.device)
            self.vae.load_state_dict(ckpt)
            print(f"[AutoregressiveFMT] loaded VAE weights "
                  f"from {config.VAEConfig.pretrained_path}")
        self.vae.to(self.device).eval()

        if config.VAEConfig.freeze:
            for p in self.vae.parameters():
                p.requires_grad = False

        # ---------- FMT ----------
        f = config.FMTConfig
        self.fmt = FlowMatchingTransformer(
            chunk_frames=f.chunk_frames,
            prev_frames=f.prev_frames,
            dim_latent=f.dim_latent,
            hidden_dim=f.dim_hidden,
            depth=f.depth,
            heads=f.heads,
            mlp_ratio=f.mlp_ratio,
            num_class=config.general.num_class,
            attn_window=f.attn_window,
            device=self.device,
        )
        self.fmt.to(self.device)

    # -----------------------------------------------------------
    #  Training forward  (Flow‑Matching loss in latent space)
    # -----------------------------------------------------------
    def forward(self, x_clean, c_onehot, x_prev=None):
        """
        x_clean : (B, T_cur, 1, 256, 256)  – current video chunk (clean)
        x_prev  : (B, T_prev, 1, 256, 256) – previous chunk (optional)
        c_onehot: (B, num_class)           – class conditioning vector
        returns : scalar FM loss (MSE)
        """
        self.fmt.train()
        self.vae.eval() if self.config.VAEConfig.freeze else self.vae.train()
        B = x_clean.size(0)
        x_prev = self.pad_prev_chunk(x_prev, self.config.FMTConfig.prev_frames)
        x_clean = self.pad_cur_chunk(x_clean, self.config.FMTConfig.chunk_frames)
        z_clean = self.vae.encode_frames(x_clean)                     # (B,T_cur,Dz)
        z_prev  = self.vae.encode_frames(x_prev) if x_prev is not None else None

        # Flow‑Matching latent mix
        z_noise  = torch.randn_like(z_clean)
        t        = torch.rand(B, device=self.device)                  # (B,)
        z_t      = (1.0 - t.view(B, 1, 1)) * z_noise + t.view(B, 1, 1) * z_clean
        v_target = z_clean - z_noise

        # classifier‑free dropout mask
        keep_mask = (torch.rand(B, device=self.device)
                     > self.config.train.p_uncond).float().unsqueeze(1)

        # predict velocity
        v_pred = self.fmt(t, z_t, c_onehot, z_prev, keep_mask)

        #loss = torch.nn.functional.mse_loss(v_pred, v_target, reduction='mean') #/ (B * self.config.FMTConfig.chunk_frames)
        loss = focal_mse(v_pred, v_target, beta=10.0, gamma=2.0)
        return loss

    # -----------------------------------------------------------
    #  Latent ODE sampling with Euler steps
    # -----------------------------------------------------------
    @torch.no_grad()
    def sample(self,
               c_onehot: torch.Tensor,
               num_steps: int = 100,
               x_prev: Optional[torch.Tensor] = None,
               guidance: Optional[float] = None,
               decode_output: bool = True,
               ):
        """
        Generate a latent chunk and decode to frames.

        Args
        ----
        c_onehot  : (B, num_class) – class condition
        num_steps : number of Euler steps on t ∈ [0,1]
        x_prev    : (optional) previous video chunk (frames tensor)
        guidance  : CFG scale (≥ 0). 1.0 → no guidance; 0 → unconditional.
        decode_output: output decoded image using vae
        """
        self.fmt.eval();  self.vae.eval()

        if guidance is None:
            guidance = getattr(self.config.eval, "guidance_scale", 1.0)


        B = c_onehot.size(0)
        T_cur = self.config.FMTConfig.chunk_frames
        Dz    = self.config.FMTConfig.dim_latent

        z_prev = self.vae.encode_frames(x_prev) if x_prev is not None else None
        z = torch.randn(B, T_cur, Dz, device=self.device)

        # time grid (forward 0 ⟶ 1)
        t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)

        # Euler integration
        for idx in range(num_steps):
            t0, t1 = t_grid[idx], t_grid[idx + 1]
            dt = t1 - t0
            t_mid = (t0 + t1) / 2                                  # midpoint for Heun‑like stability

            # ----- unconditional velocity -----
            uncond_mask = torch.zeros(B, 1, device=self.device, dtype=torch.float32)
            v_uncond = self.fmt(t_mid.expand(B), z, c_onehot, z_prev, uncond_mask)

            # ----- conditional velocity -----
            cond_mask = torch.ones(B, 1, device=self.device, dtype=torch.float32)
            v_cond = self.fmt(t_mid.expand(B), z, c_onehot, z_prev, cond_mask)

            # cfg
            v = v_uncond + guidance * (v_cond - v_uncond)
            z = z + v * dt

        #decode latent sequence to frames
        if decode_output:
            x_recon = self.vae.decode_latents_sigmoid(z) #(B, T_cur, 1, 256, 256)
        else:
            x_recon = z #(B, T_cur, 16*64*64)
        return x_recon
    
    @torch.no_grad()
    def autoregressive_sample(self,
                              c_onehot: torch.Tensor,
                              num_chunks: int = 5,
                              num_steps: int = 100,
                              prev_chunk: Optional[torch.Tensor] = None,
                              guidance: Optional[float] = None,
                              seed: Optional[int] = None,
                              ):
        """
        Generate an entire sequence of `num_chunks * T_cur` frames by
        rolling the model forward chunk‑by‑chunk.

        Returns
        -------
        frames : (B, num_chunks * T_cur, 1, 256, 256)
        """
        self.fmt.eval();  self.vae.eval()

        if guidance is None:
            guidance = getattr(self.config.eval, "guidance_scale", 1.0)
        
        if seed is not None:
            torch.manual_seed(seed)

        B          = c_onehot.size(0)
        T_cur      = self.config.FMTConfig.chunk_frames
        T_prev     = self.config.FMTConfig.prev_frames

        chunks     = [] if prev_chunk is None else [prev_chunk]            # list of decoded frame tensors

        for _ in range(num_chunks):
            prev_chunk = self.pad_prev_chunk(prev_chunk, T_prev)
            # ----- generate current chunk -----
            cur_chunk = self.sample(
                c_onehot=c_onehot,
                num_steps=num_steps,
                x_prev=prev_chunk,
                guidance=guidance,
                decode_output=True          # always decode for chaining
            )                                # (B, T_cur, 1, 256, 256)
            chunks.append(cur_chunk)

            # ----- update history -----
            prev_chunk = cur_chunk[:, -T_prev:].detach()  # last T_prev frames
        # concat along the time dimension
        return torch.cat(chunks, dim=1)                   # (B, N_total, 1, 256, 256)
    
    @staticmethod
    def pad_prev_chunk(prev_chunk, T_prev):
        if prev_chunk is not None:
            B, L, C, H, W = prev_chunk.shape
            if L < T_prev:
                pad_len = T_prev - L
                pad = torch.zeros(
                    B, pad_len, C, H, W,
                    device=prev_chunk.device,
                    dtype=prev_chunk.dtype
                )
                prev_chunk = torch.cat([pad, prev_chunk], dim=1)
            prev_chunk = prev_chunk[:, -T_prev:, ...]
        return prev_chunk
    
    @staticmethod
    def pad_cur_chunk(cur_chunk, T_cur):
        B, L, C, H, W = cur_chunk.shape
        if L < T_cur:
            pad_len = T_cur - L
            pad = torch.zeros(
                B, pad_len, C, H, W,
                device=cur_chunk.device,
                dtype=cur_chunk.dtype
            )
            cur_chunk = torch.cat([cur_chunk, pad], dim=1)
        return cur_chunk

## Helper
def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == '__main__':
    config = AutoregressiveFMTConfig()
    model = AutoregressiveFMT(config)
    print(count_parameters(model))