import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.l_n = nn.LayerNorm(out_channels)
        self.b_n = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #out = self.activation(self.l_n(self.conv_layer(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        out = self.activation(self.b_n(self.conv_layer(x)))
        return out


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.l_n = nn.LayerNorm(out_channels)
        self.b_n = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x_1 = self.conv_layer(x).permute(0, 2, 3, 1)
        #out = self.activation((self.l_n(x_1)).permute(0, 3, 1, 2))
        out = self.activation(self.b_n(self.conv_layer(x)))
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.activation = nn.GELU()
        self.l_n = nn.LayerNorm(out_channels)
        self.b_n = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #out = self.activation(self.l_n(self.conv_layer(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        out = self.activation(self.b_n(self.conv_layer(x)))
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = Conv(in_channels, out_channels)
        self.conv_2 = Conv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_2(self.conv_1(x))
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = DownConv(in_channels, out_channels)
        self.conv_2 = Conv(out_channels, out_channels)
        self.conv_3 = Conv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_3(self.conv_2(self.conv_1(x)))
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = UpConv(in_channels, out_channels)
        self.conv_2 = Conv(in_channels, out_channels)
        self.conv_3 = Conv(out_channels, out_channels)

    def forward(self, x: torch.Tensor, x_cat: torch.Tensor, x_cat_next: torch.Tensor) -> torch.Tensor:
        out = self.conv_3(self.conv_2(torch.cat((self.conv_1(x), x_cat_next, x_cat), dim = 1)))
        return out

class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear_2(self.activation(self.linear_1(x)))
        return out

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_hiddens_next: int,
        num_class, 
        past_time_step: int = 4
    ):
        super().__init__()

        ### Encoder block for past trejectory
        self.conv_1 = ConvBlock(in_channels * past_time_step, num_hiddens) #(b, hidden, h, w)
        self.down_1 = DownBlock(num_hiddens, 2 * num_hiddens) #(b, 2 * hidden, h//2, w//2)
        self.down_2 = DownBlock(2 * num_hiddens, 4 * num_hiddens) #(b, 4 * hidden, h//4, w//4)
        self.down_3 = DownBlock(4 * num_hiddens, 8 * num_hiddens) #(b, 8 * hidden, h//8, w//8)
        self.down_4 = DownBlock(8 * num_hiddens, 16 * num_hiddens) #(b, 16 * hidden, h//16, w//16)
        ### 

        ### Ecoder block for next step 
        self.conv_1_next = ConvBlock(in_channels, num_hiddens_next) #(b, hidden_next, h, w)
        self.down_1_next = DownBlock(num_hiddens_next, 2 * num_hiddens_next) #(b, 2 * hidden_next, h//2, w//2)
        self.down_2_next = DownBlock(2 * num_hiddens_next, 4 * num_hiddens_next) #(b, 4 * hidden_next, h//4, w//4)
        self.down_3_next = DownBlock(4 * num_hiddens_next, 8 * num_hiddens_next) #(b, 8 * hidden_next, h//8, w//8)
        self.down_4_next = DownBlock(8 * num_hiddens_next, 16 * num_hiddens_next) #(b, 16 * hidden_next, h//16, w//16)
        ###

        ### Decoder
        dim_out = num_hiddens +  num_hiddens_next
        self.up_1 = UpBlock(16 * dim_out, 8 * dim_out) #(b, 8 * hidden_out, h//8, w//8)
        self.up_2 = UpBlock(8 * dim_out, 4 * dim_out) # (b, 4 * hidden_out, h//4, w//4)
        self.up_3 = UpBlock(4 * dim_out, 2 * dim_out) # (b, 2 * hidden_out, h//2, w//2)
        self.up_4 = UpBlock(2 * dim_out, dim_out) # (b, hidden_out, h, w)
        self.final_conv = nn.Sequential(nn.Conv2d(dim_out, num_hiddens_next, 3, stride=1, padding=1), 
                        nn.Conv2d(num_hiddens_next, in_channels, 3, stride=1, padding=1)) # (b, in_channel, h, w)
        ###

        ### Time and class embedding
        self.timeembed_1 = FCBlock(1, 8 * dim_out)
        self.timeembed_2 = FCBlock(1, 4 * dim_out)
        self.classembed_1 = FCBlock(num_class, 8 * dim_out)
        self.classembed_2 = FCBlock(num_class, 4 * dim_out)

    def forward(self, x: torch.Tensor, x_past: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ 
        args: 
            x: (b, channel_in, h, w), 
            x_past: (b, channel_in * time_step, h, w), 
            t: (b, 1), 
            c:(b, num_class)

        return: 
            out: (b, channel_in, h, w)
        """

        ### Encode past trejectory
        x_layer1 = self.conv_1(x_past) #(b, hidden, h, w)
        x_down1 = self.down_1(x_layer1) #(b, 2 * hidden, h//2, w//2)
        x_down2 = self.down_2(x_down1) #(b, 4 * hidden, h//4, w//4)
        x_down3 = self.down_3(x_down2) #(b, 8 * hidden, h//8, w//8)
        x_bottleneck = self.down_4(x_down3) #(b, 16 * hidden, h//16, w//16)
        ###

        ### Encode next step
        x_layer1_next = self.conv_1_next(x) #(b, hidden_next, h, w)
        x_down1_next = self.down_1_next(x_layer1_next) #(b_next, 2 * hidden_next, h//2, w//2)
        x_down2_next = self.down_2_next(x_down1_next) #(b_next, 4 * hidden_next, h//4, w//4)
        x_down3_next = self.down_3_next(x_down2_next) #(b, 8 * hidden_next, h//8, w//8)
        x_bottleneck_next = self.down_4_next(x_down3_next) #(b, 16 * hidden_next, h//16, w//16)
        ###

        ### Decode 
        x_up1 = self.up_1(torch.cat((x_bottleneck_next, x_bottleneck), dim = 1), x_down3, 
            x_down3_next) * (self.classembed_1(c)[..., None, None]) +  (self.timeembed_1(t)[..., None, None]) #(b, 8 * hidden_out, h//8, w//8)
        x_up2 = self.up_2(x_up1, x_down2, x_down2_next) * (
            self.classembed_2(c)[..., None, None]) +  (self.timeembed_2(t)[..., None, None]) #(b, 4 * hidden_out, h//4, w//4)
        x_up3 = self.up_3(x_up2, x_down1, x_down1_next) # (b, 2 * hidden_out, h//2, w//2)
        x_up4 = self.up_4(x_up3, x_layer1, x_layer1_next)  # (b, hidden_out, h, w)
        out = self.final_conv(x_up4)
        ###

        return out
    
### Count model parameter
def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == "__main__":
    NN = UNet(1, 64, 64, 10)
    out = NN(torch.rand(16, 1, 224, 224), torch.rand(16, 4, 224, 224), torch.rand(16, 1), torch.rand(16, 10))
    print(out.shape)
