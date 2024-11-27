import torch
import numpy as np
import torch.nn.functional as F
from diffusers.callbacks import PipelineCallback
from intersection_mask.attention_to_mask import get_intersection_mask
from PIL import ImageDraw, ImageFont, Image
import os

from callbacks.utils import prepare_latents, add_noise



class GridPromptChangeCallback(PipelineCallback):
    def __init__(self, attention_maps, original_image=None, original_latents=None, path=None, falloff=2):        
        self.attention_maps = attention_maps
        self.original_image = original_image
        self.original_latents = original_latents
        self.path = path # Path to save the grid visualization
        # Adjust exponent to control the fall-off - we take normalised values ^ falloff to control boundaries
        self.falloff = falloff

        if original_image is None and original_latents is None:
            raise ValueError("Either `original_image` or `original_latents` must be provided.")
        
    @property
    def tensor_inputs(self):
        return ["latents"]

    def callback_fn(self, pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents", None)

        if step == 0:
            self.switch_time_grid = get_intersection_mask(self.attention_maps, latents.shape, exponent=self.falloff)
            # Convert the switch_time_grid to a tensor
            self.switch_time_grid = torch.tensor(self.switch_time_grid, dtype=torch.float32, device=latents.device)
            self.save_switch_grid_visualization()

        shape = latents.shape
        original_dtype = latents.dtype
        device = pipe._execution_device
        total_timesteps = pipe.scheduler.config.num_train_timesteps

        all_timesteps = pipe.scheduler.timesteps

        # Since we are at the end of the current step, we need to replace with noise for the next step
        noise_timestep = all_timesteps[step + 1] if step + 1 < len(all_timesteps) else timestep

        original_latents = self.get_latents(pipe, step, noise_timestep, original_dtype, device, all_timesteps)
        
        # Create a mask based on the switch times
        switch_mask = (timestep <= total_timesteps * self.switch_time_grid).float()

        # Replace sections of the latents based on the switch mask
        latents = switch_mask * latents + (1 - switch_mask) * original_latents

        # Ensure latents are of the original dtype
        latents = latents.to(original_dtype)

        return {"latents": latents}


    def get_latents(self, pipe, step, noise_timestep, original_dtype, device, all_timesteps):
        if self.original_latents is not None:
            original_latents = add_noise(pipe, self.original_latents, noise_timestep, batch_size=1, num_images_per_prompt=1, device=device, dtype=original_dtype)
        
        else:
            if step==0:
                self.preprocessed_image = pipe.preprocess_image(self.original_image)
            
            original_latents = prepare_latents(
                pipe,
                self.preprocessed_image,
                noise_timestep,
                batch_size=1,
                num_images_per_prompt=1,
                dtype=original_dtype,
                device=device,
                add_noise=step + 1 < len(all_timesteps)  # Add noise only if we are not at the last step
            )

        return original_latents
    

    def save_switch_grid_visualization(self):
        if self.path is None or self.original_image is None:
            return

        # Create directory if it doesn't exist
        os.makedirs(self.path, exist_ok=True)

        # Ensure the switch_time_grid is compatible with the desired output size
        if len(self.switch_time_grid.shape) == 3:
            self.switch_time_grid = self.switch_time_grid.unsqueeze(0)

        # Resize switch_time_grid to the size of the original image
        switch_grid_resized = F.interpolate(self.switch_time_grid, size=self.original_image.size[::-1], mode='bilinear').squeeze()

        def save_attention_overlay(image, attention_map, prefix):
            """Helper function to create and save attention overlay"""
            if torch.is_tensor(attention_map):
                attention_np = attention_map.cpu().numpy()
            else:
                attention_np = attention_map
            
            # Handle different dimension cases
            if attention_np.ndim == 4:  # (batch, channel, height, width)
                attention_np = attention_np[0].mean(axis=0)  # Take first batch and average channels
            elif attention_np.ndim == 3:  # (channel, height, width)
                attention_np = attention_np.mean(axis=0)  # Average channels
            
            # Normalize to [0, 255]
            attention_np = ((attention_np - attention_np.min()) / (attention_np.max() - attention_np.min()) * 255).astype(np.uint8)
            
            # Resize attention map to match image dimensions
            attention_image = Image.fromarray(attention_np)
            attention_resized = attention_image.resize(image.size, Image.Resampling.BILINEAR)
            attention_np = np.array(attention_resized)
            
            # Create heatmap (red channel)
            heatmap = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
            heatmap[:, :, 0] = attention_np
            
            # Blend with original image
            image_np = np.array(image.convert('RGB'))
            blended = (0.7 * image_np + 0.3 * heatmap).astype(np.uint8)
            
            # Save blended image
            Image.fromarray(blended).save(os.path.join(self.path, f"{prefix}_attention_overlay.png"))

        def save_grid_overlay(image, grid, prefix):
            """Helper function to create and save grid overlay with values"""
            if torch.is_tensor(grid):
                grid_np = grid.cpu().numpy()
            else:
                grid_np = grid

            # Handle different dimension cases
            if grid_np.ndim == 4:  # (batch, channel, height, width)
                grid_np = grid_np[0].mean(axis=0)  # Take first batch and average channels
            elif grid_np.ndim == 3:  # (channel, height, width)
                grid_np = grid_np.mean(axis=0)  # Average channels
            
            # Normalize to [0, 255]
            grid_np = ((grid_np - grid_np.min()) / (grid_np.max() - grid_np.min()) * 255).astype(np.uint8)
            
            # Resize grid to match image dimensions
            grid_image = Image.fromarray(grid_np)
            grid_resized = grid_image.resize(image.size, Image.Resampling.BILINEAR)
            grid_np = np.array(grid_resized)

            # Create grid visualization
            heatmap = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
            heatmap[:, :, 0] = grid_np
            
            image_np = np.array(image.convert('RGB'))
            blended = (0.7 * image_np + 0.3 * heatmap).astype(np.uint8)
            blended_image = Image.fromarray(blended)
            
            # Add numerical values
            draw = ImageDraw.Draw(blended_image)
            font = ImageFont.load_default()
            
            step_size_y = max(1, grid_np.shape[0] // 10)
            step_size_x = max(1, grid_np.shape[1] // 10)
            
            for y in range(0, grid_np.shape[0], step_size_y):
                for x in range(0, grid_np.shape[1], step_size_x):
                    value = float(grid_np[y, x]) / 255.0  # Convert back to 0-1 range for display
                    draw_x = min(x, blended_image.width - 10)
                    draw_y = min(y, blended_image.height - 10)
                    draw.text((draw_x, draw_y), f"{value:.2f}", fill=(255, 255, 255), font=font)
            
            blended_image.save(os.path.join(self.path, f"{prefix}_grid_overlay.png"))

        # Save original image
        self.original_image.save(os.path.join(self.path, "original.png"))

        # For each attention map
        for i, attention_map in enumerate(self.attention_maps):
            frame_dir = os.path.join(self.path, f"frame_{i}")
            os.makedirs(frame_dir, exist_ok=True)
            
            # Save raw attention map
            attention_np = attention_map
            if torch.is_tensor(attention_np):
                attention_np = attention_np.cpu().numpy()
            
            # Normalize to [0, 255] range
            attention_np = ((attention_np - attention_np.min()) / (attention_np.max() - attention_np.min()) * 255)
            attention_np = attention_np.astype(np.uint8)
            
            # Squeeze out singleton dimensions and ensure 2D
            while attention_np.ndim > 2:
                attention_np = attention_np.squeeze()
                if attention_np.ndim > 2:  # If still more than 2D, take mean of first dimension
                    attention_np = attention_np.mean(axis=0)
                
            attention_image = Image.fromarray(attention_np, mode='L')  # Use 'L' mode for grayscale
            attention_image.save(os.path.join(frame_dir, "attention_map.png"))
            
            # Save attention overlay on original image
            save_attention_overlay(self.original_image, attention_map, f"frame_{i}")
            
            # Save grid overlay for this frame
            save_grid_overlay(self.original_image, switch_grid_resized, f"frame_{i}")

            # If we have latents, save latent visualizations
            if self.original_latents is not None:
                # Convert latents to image space for visualization
                latent_image = self.latents_to_image(self.original_latents[i] if len(self.original_latents.shape) > 3 else self.original_latents)
                latent_image.save(os.path.join(frame_dir, "latent.png"))
                
                # Save attention overlay on latent
                save_attention_overlay(latent_image, attention_map, f"frame_{i}_latent")

    def latents_to_image(self, latents):
        """Helper function to convert latents to image for visualization"""
        if torch.is_tensor(latents):
            latents = latents.cpu().numpy()
        
        # Scale and normalize latents
        latents = (latents - latents.min()) / (latents.max() - latents.min())
        latents = (latents * 255).astype(np.uint8)
        
        # If latents have multiple channels, average them
        if latents.ndim == 3:
            latents = latents.mean(axis=0)
        
        return Image.fromarray(latents, mode='L')