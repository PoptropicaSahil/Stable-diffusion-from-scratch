import numpy as np
import torch


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
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    
    def _get_previous_timestep(self, timestep: int) -> int:
        # like doing timestep - step_ratio to get previous timestep
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t


    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # code taken from the huggingface implementation of the diffusers library
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t





    def add_noise(
        self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        # Refer to add noise heading in readme
        # At timestep 1 --> not very noisy
        # At timestep 1000 --> very noisy
        alpha_cumprod = self.alpha_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        # keep adding dimensions to match shape
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # (1-alpha_cumprod[timesteps]) is the variance,  we want stddev
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        # keep adding dimensions to match shape
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # sample noise
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        # According to the equation (4) of paper
        # Same, X - mu / stddev ~ Z
        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples
