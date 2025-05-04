import torch
from Unet import *
from typing import *
from GRU import *

def fm_forward(
    unet: UNet,
    GRU: HistoryEncoder,
    x_1: torch.Tensor,
    x_p: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
    p_uncond: float,
) -> torch.Tensor:
    """
    Args:
        unet: TimeConditionalUNet
        x_1: (B, C * num_Step, H, W) ground truth tensor.
        x_p: (B, C * num_Step, H, W) past trejectory
        h: (GRU_layer, B, GRU_hidden) GRU hidden state
        c: (B, num_class)
        p: dropout frequency
    Returns:
        (,) loss.
    """
    unet.train()
    GRU.train()
    device = next(unet.parameters()).device
    c_masked = c.clone()
    noise = torch.randn_like(x_1).to(device)
    time_sample = torch.rand((x_1.size(0), 1)).to(device)
    time = time_sample.reshape(-1, 1, 1, 1)
    x_t = (1 - time) * noise + time * x_1

    random_number = torch.rand(()).item()
    if random_number < p_uncond:
      mask = torch.zeros_like(c)
      c_masked = c_masked * mask

    h_vec, h_next = GRU(x_p, h)
    predicted_v = unet(x_t, h_vec, time_sample, c_masked)
    loss = nn.functional.mse_loss(predicted_v, (x_1 - noise).detach()) #detach the ground velo because it's a label

    return loss, h_next

def fm_forward_fixed_GRU(
    unet: UNet,
    x_1: torch.Tensor,
    h_vec: torch.Tensor,
    c: torch.Tensor,
    p_uncond: float,
) -> torch.Tensor:
    """
    Args:
        unet: TimeConditionalUNet
        x_1: (B, C * num_Step, H, W) ground truth tensor.
        x_p: (B, C * num_Step, H, W) past trejectory
        n: int, number of times to run the forward pass
        h_vec: (GRU_layer, B, GRU_hidden) GRU Feature
        c: (B, num_class)
        p: dropout frequency
    Returns:
        (,) loss.
    """
    unet.train()
    device = next(unet.parameters()).device
    c_masked = c.clone()
    noise = torch.randn_like(x_1).to(device)
    time_sample = torch.rand((x_1.size(0), 1)).to(device)
    time = time_sample.reshape(-1, 1, 1, 1)
    x_t = (1 - time) * noise + time * x_1

    random_number = torch.rand(()).item()
    if random_number < p_uncond:
      mask = torch.zeros_like(c)
      c_masked = c_masked * mask

    predicted_v = unet(x_t, h_vec, time_sample, c_masked)
    loss = nn.functional.mse_loss(predicted_v, (x_1 - noise).detach()) #detach the ground velo because it's a label

    return loss


@torch.inference_mode()
def fm_sample(
    unet: UNet,
    GRU: HistoryEncoder,
    c: torch.Tensor,
    x_p: torch.Tensor,
    h: torch.Tensor,
    img_wh: Tuple[int, int],
    num_ts: int,
    time_Step: int = 12,
    guidance_scale: float = 5.0,
    seed: int = 0,
    channel_in: int = 1,
) -> torch.Tensor:
    """
    Args:
        unet: ClassConditionalUNet
        c: (N, num_class) int64 condition tensor. Only for class-conditional
        x_p: (N, C * Time_step, H, W), past trejectory
        h: (num_layer_GRU, N, num_hidden_GRU) GRU hidden state
        img_wh: (H, W) output image width and height.
        num_ts: int, number of timesteps.
        guidance_scale: float, CFG scale.
        seed: int, random seed.
        channel_in: channel of image

    Returns:
        (N, C, H, W) final sample.
        (N, T_animation, C, H, W) caches.
    """
    unet.eval()
    GRU.eval()
    device = next(unet.parameters()).device
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    H, W = img_wh
    N = c.size(0)
    noise = torch.randn(N, channel_in * time_Step, H, W).to(device)
    out = noise
    time = torch.full((N, 1), 0.0).to(device)
    mask = torch.zeros_like(c)
    h_vec, h_new = GRU(x_p, h)

    for _ in range(num_ts):
      u_cond = unet(out, h_vec, time, c)
      u_uncond = unet(out, h_vec, time, mask)
      u = u_uncond + guidance_scale * (u_cond - u_uncond)
      out = out + 1/num_ts * u
      time += 1/num_ts

    return out, h_new

class FlowMatching(nn.Module):
    def __init__(
        self,
        unet: UNet,
        GRU: HistoryEncoder,
        time_step: int = 12,
        num_ts: int = 300,
        p_uncond: float = 0.1,
        channel_in: int = 1
    ):
        super().__init__()
        self.unet = unet
        self.GRU = GRU
        self.num_ts = num_ts
        self.p_uncond = p_uncond
        self.time_step = time_step
        self.channel_in = channel_in
        self.device = next(unet.parameters()).device

    def forward(self, x: torch.Tensor, x_p: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C * time_step, H, W) input tensor.
            x_p: (N, C * time_step, H, W) past trejectory
            h: (num_layer_GRU, N, num_hidden_GRU) GRU hidden state
            c: (N,) int64 condition tensor.

        Returns:
            (,) loss.
            (num_layer_GRU, N, num_hidden_GRU) h_next.
        """
        return fm_forward(
            self.unet, self.GRU, x, x_p, h, c, self.p_uncond
        )
    
    def forward_fixed_GRU(self, x: torch.Tensor, h_vec: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return fm_forward_fixed_GRU(
            self.unet, x, h_vec, c, self.p_uncond
        )
    
    def get_GRU_feature(self, x: torch.Tensor, x_p: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.GRU.train()
        h_vec, h_next = self.GRU(x_p, h)
        return h_vec, h_next

    @torch.inference_mode()
    def sample(
        self,
        c: torch.Tensor,
        x_p: torch.Tensor,
        h: torch.Tensor,
        img_wh: Tuple[int, int],
        guidance_scale: float = 5.0,
        seed: int = 0,
    ):
        return fm_sample(
            self.unet, self.GRU, c, x_p, h, img_wh, self.num_ts, self.time_step, guidance_scale, seed, self.channel_in
        )
    
    @torch.inference_mode()
    def autoregressive_sample(self, 
                            c: torch.Tensor, 
                            autoregressive_steps: int,
                            img_wh: Tuple[int, int], 
                            x_p: torch.Tensor = None,
                            guidance_scale: float = 5.0, 
                            seed: int = 0):
        """
        Args:
            c: (N, num_class) int64 condition tensor.
            autoregressive_steps: int, number of autoregressive steps.
            img_wh: (H, W) output image width and height.
            x_p: (N, C * time_step, H, W) the starting chunk of the video
            guidance_scale: float, CFG scale.
            seed: int, random seed.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        B, H, W   = c.size(0), *img_wh
        Cin       = self.channel_in            # usually 1
        T         = self.time_step             # 12
        device    = self.device
        # rolling buffer that always holds *last 12 frames* as channels
        buf = torch.zeros(B, self.channel_in * self.time_step, H, W, device=device) if x_p is None else x_p
        video = []
        # initialize GRU hidden state
        h_state = self.GRU.init_state(batch=B, device=device)
        for _ in range(autoregressive_steps):
            chunk, h_state = self.sample(c, buf, h_state, img_wh, guidance_scale=guidance_scale, seed=seed)
            chunk = chunk.view(B, self.time_step, self.channel_in, H, W)  # (B, 12, 1, H, W)
            video.append(chunk)
            buf = chunk.reshape(B, Cin*T, H, W)  
        video = torch.cat(video, dim=1)
        return video
        
        
    

if __name__ == "__main__":
    model = FlowMatching(UNet(1, 64, 64), HistoryEncoder(12))
    out_1 = model.forward(torch.rand(16, 12, 256, 256), torch.rand(16, 12, 256, 256), torch.rand(2, 16, 512), torch.rand(16, 10))
    print(out_1[0].item(), out_1[1].shape)
    out_2 = model.cuda().sample(torch.rand(16, 10).cuda(), torch.rand(16, 12, 256, 256).cuda(), torch.rand(2, 16, 512).cuda(), (256, 256))
    print(out_2[0].shape, out_2[1].shape)