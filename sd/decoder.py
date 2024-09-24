import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups = 32, num_channels=channels)
        self.selfattention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, channels, Height, Width)
        # channels <==> features

        residue = x

        # (Batch_Size, channels, Height, Width) -> (Batch_Size, channels, Height, Width)
        x = self.groupnorm(x)
        
        n, c, h, w = x.shape

        # (Batch_Size, channels, Height, Width) -> (Batch_Size, channels, Height * Width)
        x = x.view(n, c, h*w)

        # (Batch_Size, channels, Height * Width) -> (Batch_Size, Height * Width, channels)
        # Usual attention is like calculating attention b/w each tokens (now pixels h*w)
        # Each token has its own embedding (now channels <==> features)
        # So we are relating pixels to each other
        x = x.transpose(-1, -2)

        # Self attention without mask
        # Self attention because Q, K, V all come from same input
        # (Batch_Size, Height * Width, channels) -> (Batch_Size, Height * Width, channels)
        x = self.selfattention(x)

        # (Batch_Size, Height * Width, channels) -> (Batch_Size, channels, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, channels, Height * Width) -> (Batch_Size, channels, Height, Width)
        x = x.view((n, c, h, w))

        # (Batch_Size, channels, Height, Width)
        x += residue

        return x




class VAE_ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # to ensure same shape while adding the residue
            self.residual_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, in_channels, Height, Width)

        residue = x

        # None of the operations change shape of x
        # L_out = [(L_in − K + 2P) / S] + 1  --> [(H - 3 + 2*1)/1] + 1 --> H
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        x = x + self.residual_layer(residue) 
        return x
    

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 1, padding=0),
            
            nn.Conv2d(in_channels = 4, out_channels = 512, kernel_size = 3, padding = 1),

            VAE_ResidualBlock(in_channels = 512, out_channels = 512),
            
            VAE_AttentionBlock(channels = 512),

            VAE_ResidualBlock(in_channels = 512, out_channels = 512),
            VAE_ResidualBlock(in_channels = 512, out_channels = 512),
            VAE_ResidualBlock(in_channels = 512, out_channels = 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8) 
            VAE_ResidualBlock(in_channels = 512, out_channels = 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor = 2),

            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1),

            VAE_ResidualBlock(in_channels = 512, out_channels = 512),
            VAE_ResidualBlock(in_channels = 512, out_channels = 512),
            VAE_ResidualBlock(in_channels = 512, out_channels = 512),
  
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor = 2),     

            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1),
            
            VAE_ResidualBlock(in_channels = 512, out_channels = 256),
            VAE_ResidualBlock(in_channels = 256, out_channels = 256),
            VAE_ResidualBlock(in_channels = 256, out_channels = 256),
   
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor = 2),  

            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1),

            VAE_ResidualBlock(in_channels = 256, out_channels = 128),
            VAE_ResidualBlock(in_channels = 128, out_channels = 128),
            VAE_ResidualBlock(in_channels = 128, out_channels = 128),
   
            # Divide the 128 features (channels) in groups of 32
            nn.GroupNorm(num_groups = 32, num_channels = 128),

            nn.SiLU(),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(in_channels = 128, out_channels = 3, kernel_size = 3, padding = 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input to decoder is the latent representation
        # x: (Batch_Size, 4, Height / 8, Width / 8)

        # Reverse the scaling
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width)
        return x