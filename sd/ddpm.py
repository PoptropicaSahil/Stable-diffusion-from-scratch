import torch
import numpy as np


class DDPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        # The default params mean that at 1000 steps, a beta value of beta_end makes the image as complete noise
        # We take sqrt, then do **2, as given in the official repo
        # Called scaled linear scheduler in huggingface implementation
        self.betas = (
            torch.linspace(
                start=beta_start**0.5,
                end=beta_end**0.5,
                steps=num_training_steps,
                dtype=torch.float32,
            )
            ** 2
        )

        self.alphas = 1.0 - self.betas  # as given in paper formula 4
        self.alpha_cumprod = torch.cumprod(
            self.alphas, dim=0
        )  # like [a0, a0 * a1, a0 * a1 * a2]
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        # timesteps in decreasing order, because we want to remove noise (more noise to less noise)
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timestamps(self, num_inference_steps=50):
        # default num_training_steps 1000
        # given num_inference_steps 50
        # so we make a range of 1000 till 0 at steps of 1000/50 
        self.num_inference_steps = num_inference_steps
        stop_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange())


