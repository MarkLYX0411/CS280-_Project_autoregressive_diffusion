import torch
from model import *
from typing import *


def fm_forward(
    unet: UNet,
    x_1: torch.Tensor,
    x_p: torch.Tensor,
    c: torch.Tensor,
    p_uncond: float,
) -> torch.Tensor:
    """
    Args:
        unet: TimeConditionalUNet
        x_1: (B, C, H, W) ground truth tensor.
        x_p: (B, C * num_Step, H, W) past trejectory
        c: (B, num_class)
        p: dropout frequency
    Returns:
        (,) loss.
    """
    unet.train()
    device = next(unet.parameters()).device
    loss_f = nn.MSELoss()

    noise = torch.randn_like(x_1).to(device)
    time_sample = torch.rand((x_1.size(0), 1)).to(device)
    time = time_sample.reshape(-1, 1, 1, 1)
    x_t = (1 - time) * noise + time * x_1

    random_number = torch.rand(()).item()
    if random_number < p_uncond:
      mask = torch.zeros_like(c)
      c = c * mask

    predicted_v = unet(x_t, x_p, time_sample, c)
    loss = loss_f(predicted_v, (x_1 - noise))

    return loss


@torch.inference_mode()
def fm_sample(
    unet: UNet,
    c: torch.Tensor,
    x_p: torch.Tensor,
    img_wh: Tuple[int, int],
    num_ts: int,
    guidance_scale: float = 5.0,
    seed: int = 0,
    channel_in: int = 1
) -> torch.Tensor:
    """
    Args:
        unet: ClassConditionalUNet
        c: (N, num_class) int64 condition tensor. Only for class-conditional
        x_p: (N, C * Time_step, H, W), past trejectory
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
    device = next(unet.parameters()).device
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    H, W = img_wh
    N = c.size(0)
    noise = torch.randn(N, channel_in, H, W).to(device)
    out = noise
    time = torch.full((N, 1), 0.0).to(device)
    mask = torch.zeros_like(c)
    for _ in range(num_ts):
      u_cond = unet(out, x_p, time, c)
      u_uncond = unet(out, x_p, time, mask)
      u = u_uncond + guidance_scale * (u_cond - u_uncond)
      out = out + 1/num_ts * u
      time += 1/num_ts

    return out

class FlowMatching(nn.Module):
    def __init__(
        self,
        unet: UNet,
        num_ts: int = 300,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.unet = unet
        self.num_ts = num_ts
        self.p_uncond = p_uncond

    def forward(self, x: torch.Tensor, x_p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            x_p: (N, C * time_step, H, W) past trejectory
            c: (N,) int64 condition tensor.

        Returns:
            (,) loss.
        """
        return fm_forward(
            self.unet, x, x_p, c, self.p_uncond
        )

    @torch.inference_mode()
    def sample(
        self,
        c: torch.Tensor,
        x_p: torch.Tensor,
        img_wh: Tuple[int, int],
        guidance_scale: float = 5.0,
        seed: int = 0,
    ):
        return fm_sample(
            self.unet, c, x_p, img_wh, self.num_ts, guidance_scale, seed
        )
    

if __name__ == "__main__":
    model = FlowMatching(UNet(1, 64, 64, 10))
    out_1 = model.forward(torch.rand(16, 1, 224, 224), torch.rand(16, 4, 224, 224), torch.rand(16, 10))
    print(out_1.item())
    out_2 = model.cuda().sample(torch.rand(16, 10).cuda(), torch.rand(16, 4, 224, 224).cuda(), (224, 224))
    print(out_2.shape)