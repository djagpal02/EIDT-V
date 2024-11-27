from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents

import torch
import PIL

# adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img
def prepare_latents(
    pipe, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
):
    timestep = timestep.repeat(batch_size * num_images_per_prompt)

    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    latents_mean = latents_std = None
    if hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None:
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None:
        latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1)

    # Offload text encoder if `enable_model_cpu_offload` was enabled
    if hasattr(pipe, "final_offload_hook") and pipe.final_offload_hook is not None:
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        init_latents = image

    else:
        if pipe.vae.config.force_upcast:
            image = image.float()
            pipe.vae.to(dtype=torch.float32)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        elif isinstance(generator, list):
            if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
            elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                )

            init_latents = [
                retrieve_latents(pipe.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(pipe.vae.encode(image), generator=generator)

        if pipe.vae.config.force_upcast:
            pipe.vae.to(dtype)

        init_latents = init_latents.to(dtype)
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=device, dtype=dtype)
            latents_std = latents_std.to(device=device, dtype=dtype)
            init_latents = (init_latents - latents_mean) * pipe.vae.config.scaling_factor / latents_std
        else:
            init_latents = pipe.vae.config.scaling_factor * init_latents

    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        # expand init_latents for batch_size
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)

    if add_noise:
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        init_latents = pipe.scheduler.add_noise(init_latents, noise, timestep)

    latents = init_latents

    return latents

def add_noise(pipe, latents, timestep, batch_size, num_images_per_prompt, device, dtype):
    timestep = timestep.repeat(batch_size * num_images_per_prompt)

    shape = latents.shape
    noise = randn_tensor(shape, device=device, dtype=dtype)
    try:
        latents = pipe.scheduler.add_noise(latents, noise, timestep)
    except:
        # If SD3 different function
        latents = pipe.scheduler.scale_noise(latents, timestep, noise)
    return latents