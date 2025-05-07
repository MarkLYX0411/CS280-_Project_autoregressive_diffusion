# models/vae.py
import torch, torch.nn as nn, torch.nn.functional as F

# ----------   building blocks   ----------
def conv(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )

def deconv(in_ch, out_ch, k=4, s=2, p=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, k, s, p, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )

# ----------   encoder   ----------
class Encoder(nn.Module):
    """256² → (μ, logσ²) of shape (B,64,16,16)."""
    def __init__(self, z_ch=64):
        super().__init__()
        self.enc = nn.Sequential(
            conv(1,   64, 4, 2, 1),   # 128²
            conv(64, 128, 4, 2, 1),   # 64²
            conv(128,256, 4, 2, 1),   # 32²
            conv(256,256, 4, 2, 1),   # 16²
        )
        self.mu     = nn.Conv2d(256, z_ch, 1)
        self.logvar = nn.Conv2d(256, z_ch, 1)

    def forward(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

# ----------   decoder   ----------
class Decoder(nn.Module):
    """z_map (B,64,16,16) → reconstructed 1‑channel 256²."""
    def __init__(self, z_ch=64):
        super().__init__()
        self.dec = nn.Sequential(
            deconv(z_ch, 256),        # 32²
            deconv(256,128),          # 64²
            deconv(128,64),           # 128²
            deconv(64, 32),           # 256²
            nn.Conv2d(32, 1, 3, 1, 1),
            # nn.Sigmoid(),             # pixel range [0,1]
        )

    def forward(self, z):
        return self.dec(z)

# ----------   VAE wrapper   ----------
class VAE(nn.Module):
    def __init__(self, z_ch=64, kl_beta=1e-4):
        super().__init__()
        self.enc = Encoder(z_ch)
        self.dec = Decoder(z_ch)
        self.kl_beta = kl_beta

    @staticmethod
    def reparameterise(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):                               # x: (B,1,256,256)
        mu, logvar = self.enc(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.dec(z)
        # losses
        recon_bce = F.binary_cross_entropy_with_logits(x_hat, x, reduction='mean')
        kl = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1)
        loss = recon_bce + self.kl_beta * kl
        return {'x_hat': x_hat, 'mu': mu, 'logvar': logvar,
                'recon_bce': recon_bce, 'kl': kl, 'loss': loss}

    # handy helpers for downstream modules
    @torch.no_grad()
    def encode_frames(self, frames):                    # (B,T,1,256,256)
        B,T = frames.shape[:2]
        mu, _ = self.enc(frames.reshape(B*T,1,256,256))
        return mu.view(B, T, -1)                        # (B,T,64*16*16)

    @torch.no_grad()
    def decode_latents(self, z_seq):                    # (B,T,64*16*16)
        B,T,_ = z_seq.shape
        imgs = self.dec(z_seq.reshape(B*T,64,16,16))
        return imgs.view(B, T, 1, 256, 256)
    
    @torch.no_grad()
    def decode_latents_sigmoid(self, z_seq):
        decoded_seq = self.decode_latents(z_seq)
        return torch.nn.functional.sigmoid(decoded_seq) # (B, T, 1, 256, 256)

# ----------   demo sanity check   ----------
if __name__ == "__main__":
    vae = VAE()
    x = torch.rand(8,1,256,256)
    out = vae(x)
    print({k: v.shape if torch.is_tensor(v) else v for k,v in out.items()})
