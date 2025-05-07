class AutoregressiveFMTConfig:
    class general:
        device: str = 'cuda'
        seed: int = 3036702335
        img_height: int = 256
        img_width: int = 256
        img_channel: int = 1
        num_class: int = 2
    class train:
        p_uncond: float = 0.15
    class eval:
        guidance_scale: float = 5.0
    class FMTConfig:
        chunk_frames: int = 4
        prev_frames: int = 12
        dim_hidden: int = 256
        depth: int = 6
        heads: int = 4
        mlp_ratio: float = 4.0
        attn_window: int = 2
        dim_latent = 64 * 16 * 16
    class VAEConfig:
        z_ch: int = 64
        kl_beta: float = 1e-2
        pretrained_path: str = '/home/haoyuexiao/Documents/CS280/final_proj/CS280-_Project_autoregressive_diffusuon/cs280_project/scripts/checkpoints/vae_50/best.pth'
        freeze = True
