from email import generator
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
        # Paper equation (2) calls beta as variance scheduler wrt the forward process
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
        )  # like [a0, a0 * a1, a0 * a1 * a2], cumprod is cumulative product
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        
        # timesteps in decreasing order, because we want to remove noise (more noise to less noise)
        # this is detault values, will be updated if user does set_inference_timesteps
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
        # timesteps have been redefined

    def set_strength(self, strength = 1):
        """Like how much noise to add to latent. IF say strength = 0.8, then we don't add full noise, 
        to feed into UNET i.e. don't start from pure noise. 
        We skip 20% of inference steps i.e. we start from 80% noisy image.
        Remember if we add less noise, UNET is given less freedom
        Starting from full noise means full freedom"""

        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:] # skipping timesteps
        self.start_step = start_step 

    
    def _get_previous_timestep(self, timestep: int) -> int:
        # like doing timestep - step_ratio to get previous timestep
        # maybe not setting self.step ratio because set_inference_timestamps may be optional. 
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t


    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """Each 'step' is to remove the predicted noise i.e. reverse process
        Check the last line in for loop when we enumerate over timesteps in pipeline.
        Remember timestep 1000 (end) is pure noise, so we want to go till lesser timesteps
        """

        t = timestep
        prev_t = self._get_previous_timestep(t)

        # code taken from the huggingface implementation of the diffusers library
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t


        ### Calculate the mean ### 
        # NOTE: Check README section on Removing noise from image (reverse process)
        # compute original sample according to Formula (15). 
        # x_t = latents
        # sqrt(1 - alpha_t) = sqrt(beta_prod_t)
        # Epsilon thing is mode output
        pred_original_sample = (latents - (beta_prod_t ** 0.5 * model_output)) / alpha_prod_t ** 0.5


        # Compute the coefficients for pred_original_sample and current sample x_t
        # i.e. Equation (7) for calculating mean, get the two coeffs of x_0 and x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 + beta_prod_t_prev / beta_prod_t

        # Compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents


        ### Now calculate the variance ###
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev  # NOTE: THIS IS SUS. WHY CHANGED?
        variance_calculated = (1-alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance_calculated = torch.clamp(variance_calculated, min=1e-20)

        variance = 0
        if t > 0:
            noise = torch.randn(model_output.shape, generator=generator, device=model_output.device, dtype= model_output.dtype)
            variance = (variance_calculated ** 0.5) * noise
        
        # Z ~ N(0, 1)
        # Z ~ X - mu / sigma # sigma  is std_dev
        # X = mu + sigma * Z
        # mu == pred_prev_sample
        # sigma * Z  == std_dev * noise as done above
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample




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

        # (1-alpha_cumprod[timesteps]) is the variance, we want stddev
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        # keep adding dimensions to match shape
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # sample noise that we can add
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        # According to the equation (4) of paper
        # Same, X - mu / stddev ~ Z
        # NOTE: We are adding noise to the original samples
        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples
