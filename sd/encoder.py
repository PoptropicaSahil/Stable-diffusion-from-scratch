import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        """
        Idea is to compress the image i.e. reduce the shape, but keep adding more features/channels. 
        So each pixel carries more information with more features.
        This variational autoencoder not only compresses the image, but also 
        LEARNS A LATENT SPACE i.e. learns a DISTRIBUTION. 

        Learn the mu, sigma of the latent space.
        We can then sample from this distribution.
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
            VAE_AttentionBlock(512, ),

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
        # Shape of noise is same as output of encoder


        # run all the modules sequentially
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # For the conv2d modules having stride = 2, we have already put 0 padding
                # because default padding is symmetrical in all directions
                # Instead we (asymmetrically) apply it only to right and bottom

                # (Padding_left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            
            # Apply all the modules
            x = module(x)

        # Output of the Variational Autoencoder is mean and log variance
        # (Batch_Size, 8 , Height, Height / 8, Width / 8) 
        # -> two tensors of shape (Batch_Size, 4, Height / 8, Height / 8)
        mean, log_variance = torch.chunk(inputs = x, chunks = 2, dim = 1)

        # Convert/ Clamp it to this range (useful if it is too small or too large)
        # (Batch_Size, 4, Height / 8, Height / 8) -> (Batch_Size, 4, Height / 8, Height / 8)
        log_variance = torch.clamp(input = log_variance, min = -30, max = 20)

        # (Batch_Size, 4, Height / 8, Height / 8)
        variance = log_variance.exp()
        
        # (Batch_Size, 4, Height / 8, Height / 8)
        std_dev = variance.sqrt()

        # Given noise - N(0, 1) i.e. Z
        # Given data - N(mean, std_dev) i.e x
        # Z = (x - mean) / std_dev
        x = mean + std_dev * noise

        # Scale the output by a constant (as given in the paper and repo)
        # (Batch_Size, 4, Height / 8, Height / 8)
        x *= 0.18215

        return x

