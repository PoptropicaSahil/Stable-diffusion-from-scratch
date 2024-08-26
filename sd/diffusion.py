import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):

    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear = nn.Linear(4 * n_embd, 4 * n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)

        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        # (1, 1280) i.e. 4* 320
        return x

class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_feature = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(in_features=n_time, out_features=out_channels)

        self.groupnorm_merged = nn.G


class Upsample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (Batch_Size, features, Height, Width)

        # (Batch_Size, features, Height, Width) -> (Batch_Size, features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode = 'nearest')
        return self.conv(x)



class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                # UNET_AttentionBlock computes crossattention between latents (x) and prompt (context)
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                # UNET_ResidualBlock matches latent with timestamp
                x = layer(x, time)
            else:
                x = layer(x)
            
        return x
            



class UNET(nn.Module):
    
    def __init__(self):
        super().__init__()
        # We model unet as encoder(reduce size, increase features) - decoder (opposite)

        self.encoders = nn.ModuleList([
            # SwitchSequential is equivalent to Sequential only
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size = 3, padding = 1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size = 3, padding = 1, stride = 2)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size = 3, padding = 1, stride = 2)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size = 3, padding = 1, stride = 2)),

            # Just ResidualBlock does not affect shape
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            # Output of Residual Block is 1280 (goes directly to decoder)
            # Output of encoder is also 1280 (skip connection to decoder)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), 
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)), 

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(940, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(940, 640), UNET_AttentionBlock(8, 160), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(940, 320), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)),
            # output dimension (Batch, 320, Height / 8, Width / 8)
            # Check forward of Diffusion
        ])


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8) 

        x = self.groupnorm(x)
        x = F.silu(x)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8) 
        x = self.conv(x)

        return x


class Diffusion(nn.Module):
    """U-net only, just the dimensions were modified by the stable diffusion authors"""

    def __init__(self):
        self.time_embedding = TimeEmbedding(n_embd = 320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(in_channels = 320, out_channels = 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        """Foward pass of UNET

        Args:
            latent (torch.Tensor): Shape (Batch_Size, 4, Height / 8, Width / 8) -- z i.e. output of VAE_Encoder
            context (torch.Tensor): Shape (Batch_Size, Seq_Len, Dim) -- from the text embedding
            time (torch.Tensor): Shape (1, 320) -- time (as vector) at which latent was noisified
        """

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)

        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)

        # (Batch, 4, Height / 8, Width / 8)
        return output
 
