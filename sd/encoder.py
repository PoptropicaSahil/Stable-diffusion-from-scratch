import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAEAttentionBlock, VAEResidualBlack


class VAR_Encoder(nn.Sequential):
    def __init__(self):
        """
        Idea is to compress the image i.e. reduce the shape, but keep adding more features. 
        So each pixel carries more information with more features
        """

        super().__init__(

            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width/2)
            # L_out = [(L_in−K+2P)/S]+1 (L is the length/width/height)
            # [(H - 3 + 0) / 2] + 1 --> H/2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 128, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),

            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/4, Width/4)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 256, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),

            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Width/8)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512), # same shape
            VAE_ResidualBlock(512, 512), # same shape

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAEAttentionBlock(512, ),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.SiLU(),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=1, padding=1),

            # (Batch_Size, 8, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0)

        )

    def forward(self, x:torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, out_Channels, Height/8, Width/8)

        for module in self.modules:
            if getattr(module, 'stride', None) == (2, 2):
                # (Padding_left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

