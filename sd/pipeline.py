import torch
import numpy as np
from sd.ddpm import DDPMSampler
from tqdm import tqdm
import numpy as np

# Stable diffusion can only images of size 512 x 512
WIDTH = 512
HEIGHT = 512

# Remember VAE_Encoder converts input image to Height / 8, Width / 8
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
# LATENTS_CHANNELS = 4  # always fixed 4 channels


def generate(
    prompt: str,
    uncond_prompt: str, # Negative prompt or empty string
    input_image: None,
    strength = 0.8, # Between 0 to 1 - how much noise to add. Controls how much attention paid to input image to generate output image. More strength -> more noise added to latent -> more creative model
    do_cfg = True, # classifier free guidance
    cfg_scale = 7.5, # ranges between 1 to 14 (check!)
    sampler_name = "ddpm",
    n_inference_steps = 50, # usually about 50 steps are good enough for ddpm sampler
    models = {},
    seed = None,
    device = None,
    idle_device = None,
    tokenizer = None
):
    # uncond_prompt is the negative prompt - from which we want to 'go away'

    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle= lambda x: x.to(idle_device)
        else:
            to_idle= lambda x: x
        
        # generator is as good as random number generator
        generator = torch.Generator(device = device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # we have to pass through the model twice, with and without prompt (context)

            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids

            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype = torch.long, device = device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens) 

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding = "max_length", max_length = 77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)  # (Batch_Size, Seq_Len, Dim)

            # Just concat the two contexts (TEXTS)
            # Batch_Size is 1 only!
            # (2, Seq_Len, Dim) == (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # no classifier free guidance, so no uncond_prompt
            # Convert it to list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length =77).input_ids
            tokens = torch.tensor(tokens, dtype = torch.long, device=device)

            context = clip(tokens) # (Batch_Size, Seq_Len, Dim) = (1, 77, 768)
        
        to_idle(clip)

        if sampler_name.lower().trim() == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}, please choose ddpm only")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

    

    if input_image: 
        # for image to image case
        # we pass through encoder, get latent, and add noise to it
        # then scheduler will 'remove' noise

        encoder = models["encoder"]
        encoder.to(device)

        input_image_tensor = input_image.resize((WIDTH, HEIGHT))
        input_image_tensor = np.array(input_image_tensor)

        # (HEIGHT, WIDTH, Channels = 3)
        input_image_tensor = torch.tensor(input_image_tensor, dtype = torch.float32)
        input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

        # (Height, Width, Channels) -> (Batch_Size = 1, Height, Width, Channels) 
        input_image_tensor = input_image_tensor.unsqueeze(0)

        # Convert to shape as taken by encoder
        # (Batch_Size = 1, Height, Width, Channels) -> (Batch_Size = 1, Channels, Height, Width) 
        input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

        # Sample some noise (encoder needs noise in its forward pass)
        encoder_noise = torch.randn(size = latents_shape, generator=generator, device=device)

        # Run through VAE_Encoder 
        latents = encoder(input_image_tensor, encoder_noise)

        # ************** IMP ***************
        # Now we have to add noise TO THIS latent
        sampler.set_strength(strength = strength)
        latents = sampler.add_noise(latents, sampler.timesteps[0])

        # Encoder's part is done, we now shift it to the idle device
        # OP only!!
        to_idle(encoder)
    else:
        # if no input image is passed, it is text to image. 
        # We start with random noise then. N(0, 1)
        latents = torch.randn(size = latents_shape, generator=generator, device=device)

    diffusion = models["diffusion"]
    diffusion.to(device)

    # Remember forward process of Diffusion class takes latent, context, time - all torch.Tensor

    timesteps = tqdm(sampler.timesteps)
    for i, timestep in enumerate(timesteps):
        # for each timestep, we donoise the image continuously
        # UNET sees the *latent* and predicts how much noise is present
        # Scheduler removes that noise and UNET predicts noise in the remaining image
        # We do this 'sampler.timesteps' times
        
        # get_time_embedding converts timestep number to vector
        # (1, 320)
        time_embedding = get_time_embedding(timestep).to(device)

        # (Batch_Size, 4, Height_Latent, Width_Latent)
        model_input = latents


        if do_cfg:
            # we have to pass through the model twice, with and without prompt (context), so we repeat
            # (Batch_Size, 4, Height_Latent, Width_Latent) -> (2 * Batch_Size, 4, Height_Latent, Width_Latent)
            model_input = model_input.repeat(2, 1, 1, 1)

        # model_output is predicted noise by the UNET
        model_output = diffusion(model_input, context, time_embedding)

        if do_cfg:
            # get the two outputs separated
            output_cond, output_uncond = model_output.chunk(2)

            # Apply formula as given for cfg (check readme)
            model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

        # remove the noise predicted by the UNET
        latents = sampler.step(timestep, latents, model_output)

    # work of diffusion model is done
    to_idle(diffusion)

    decoder = models["decoder"]
    decoder.to(device)

    images = decoder(latents)
    to_idle(decoder)

    # Invert the rescaling of image done earlier
    images = rescale(images, (-1, 1), (0, 255), clamp = True)

    # To see image on CPU, Channel dimension should be last apparently
    # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
    images = images.permute(0, 2, 3, 1)
    images = images.to("cpu", torch.uint8).numpy()[0]

    return images


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    # Same as temperature scale conversions
    # (A - a_min)/(a_max - a_min) = (B - b_min)/(b_max - b_min)
    # B = (A - a_min) * (b_max - b_min) / (a_max - a_min) + b_min
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    # Convert number to vector (embedding)
    # wrt readme diagram, timestep <==> pos

    # Negative sign because we multiply be reciprocal instead of dividing
    # (160, )
    freqs = torch.pow(10000, -torch.range(start=0, end=160, dtype=torch.float32) / 160)

    # like unsequeeze dimension
    x = torch.tensor([timestep], dtype = torch.float32)[:, None] * freqs[None]

    # (1, 320)
    output = torch.cat([torch.cos(x), torch.sin(x)], dim = -1)

    return output





