import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups = 32, num_channels=channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, channels, Height, Width)
        # channels <==> features

        residue = x
        
        n, c, h, w = x.shape

        # (Batch_Size, channels, Height, Width) -> (Batch_Size, channels, Height * Width)
        x = x.view(n, c, h*w)

        # (Batch_Size, channels, Height * Width) -> (Batch_Size, Height * Width, channels)
        x = x.transpose(-1, -2)

        # (Batch_Size, Height * Width, channels) -> (Batch_Size, channels, Height, Width)
        x = x.view((n, c, h, w))






class VAE_ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(num_groups=32, out_channels=out_channels)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # to ensure same shape while adding
            self.residual_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, in_channels, Height, Width)

        residue = x

        # None of the operations change shape of x
        # L_out = [(L_in−K+2P)/S]+1  --> [(H - 3 + 2*1)/1] + 1 --> H
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        x = x + self.residual_layer(residue) 
        return x