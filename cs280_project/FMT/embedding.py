import torch, torch.nn as nn
import math

class ClassEmbed(nn.Module):
    """
    One‑hot → hidden dim (with optional dropout for classifier‑free guidance).
    """
    def __init__(self, num_class: int, dim_out: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(num_class, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, c):          # (B, num_class)
        return self.ff(c)          # (B, dim_out)

# ---------- Time embedding ----------
class TimestepEmbed(nn.Module):
	"""
	Embeds scalar timesteps into vector representations.
	"""
	def __init__(self, hidden_size, frequency_embedding_size = 256):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(frequency_embedding_size, hidden_size, bias=True),
			nn.SiLU(),
			nn.Linear(hidden_size, hidden_size, bias=True),
		)
		self.frequency_embedding_size = frequency_embedding_size

	@staticmethod
	def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
		"""
		Create sinusoidal timestep embeddings.
		:param t: a 1-D Tensor of N indices, one per batch element.
						  These may be fractional.
		:param dim: the dimension of the output.
		:param max_period: controls the minimum frequency of the embeddings.
		:return: an (N, D) Tensor of positional embeddings.
		"""
		# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
		half = dim // 2
		freqs = torch.exp(
			-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
		).to(device=t.device)
		args = t[:, None].float() * freqs[None]
		embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
		if dim % 2:
			embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
		return embedding

	def forward(self, t: torch.Tensor) -> torch.Tensor:
		t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
		t_emb = self.mlp(t_freq)
		return t_emb