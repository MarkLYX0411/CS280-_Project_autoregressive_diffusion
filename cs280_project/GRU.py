import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """
    CNN backbone that ingests a stack of T grayscale frames
    (B, T, H, W)  →  (B, feat_dim)
    """
    def __init__(self, in_frames: int, feat_dim: int = 512):
        super().__init__()
        ch = [in_frames, 64, 128, 256, feat_dim]

        blocks = []
        for c_in, c_out in zip(ch[:-1], ch[1:]):
            blocks += [
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, c_out),
                nn.SiLU()
            ]
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)            # global avg-pool

    def forward(self, x):                              # (B, T, H, W)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)                    # (B, feat_dim)
        return x
    
class HistoryEncoder(nn.Module):
    """
    One GRU update **per chunk**.
    forward() returns:
        h_vec   – (B, hidden)  the new hidden state (last layer)
        h_next  – tuple of all GRU layers’ states (to carry forward)
    """
    def __init__(self,
                 in_frames:     int,      # chunk size (e.g. 12 or 24)
                 feat_dim:      int = 512,
                 hidden_dim:    int = 512,
                 n_gru_layers:  int = 2):
        super().__init__()
        self.cnn = ConvEncoder(in_frames, feat_dim)
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.gru  = nn.GRU(hidden_dim,
                           hidden_dim,
                           n_gru_layers,
                           batch_first=True)

    @torch.jit.ignore
    def init_state(self, batch: int, device=None):
        """Return zero-initialised hidden state."""
        return torch.zeros(self.gru.num_layers,
                           batch,
                           self.gru.hidden_size,
                           device=device)

    def forward(self,
                chunk:   torch.Tensor,    # (B, T, H, W)
                h_prev:  torch.Tensor):   # (n_layers, B, hidden)
        x = self.cnn(chunk)               # (B, feat_dim)
        x = self.proj(x).unsqueeze(1)     # (B, 1, hidden)
        out, h_next = self.gru(x, h_prev) # sequence len = 1
        h_vec = out.squeeze(1)            # (B, hidden)
        return h_vec, h_next

if __name__ == "__main__":
    NN = HistoryEncoder(12)
    out = NN(torch.rand(16, 12, 256, 256), NN.init_state(16))
    print(out[0].shape, out[1].shape)
