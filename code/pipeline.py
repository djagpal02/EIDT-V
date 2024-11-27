# Python standard libraries
import gc
from typing import Optional
import time
import csv
import os
import logging

# Other libraries
import torch
from tqdm import tqdm
from diffusers.callbacks import MultiPipelineCallbacks

# Local imports
from utils.load import (
    load_sd,
    load_flux, 
    load_sd3, 
    load_sdxl, 
    load_ip_adapter, 
    load_clip_segmentation
)
from callbacks.grid_prompt_change import GridPromptChangeCallback
from callbacks.prompt_change import PromptChangeCallback
from callbacks.save_latents import SaveFinalLatentsCallback
from intersection_mask.clip_attention import get_attention_map
from details import VideoDetails
from utils.utils import save_frames_as_gif


class Pipeline:
    def __init__(self, device, model='SDXL', fp16=True, run_history_path='../output/pipeline_run_history.csv'):
        self.device = device
        self.fp16 = fp16
        self.model = model
        self.set_attention_modules()
        self.run_history = []  # New attribute to store run history
        self.run_history_path = run_history_path
        logging.debug("Pipeline initialized with device: %s, model: %s, fp16: %s", device, model, fp16)

    def set_model(self):
        # Load the appropriate model based on self.model
        model_loaders = {
            'SD': load_sd,
            'SDXL': load_sdxl,
            'SD3': load_sd3,
            'FLUX': load_flux
        }
        self.pipe = model_loaders[self.model](self.device, self.fp16)
        self.pipe.set_progress_bar_config(disable=True)

    def clear_model(self):
        # Clear the model from memory
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def set_attention_modules(self):
        # Load CLIP models for attention
        self.clip_processor, self.clip_model = load_clip_segmentation(self.device)

    
    def multi_prompt(self, pipe_args, prompt, details, i, strategy):
        # Handle multi-prompt strategies for SDXL, SD3, and FLUX models
        if self.model in ['SDXL', 'SD3', 'FLUX']:
            if strategy in ['PreviousFrame', 'PreviousFrameWithThird']:
                pipe_args['prompt_2'] = [details.reduced[f'frame_{i-1}']] * len(pipe_args['prompt'])
            elif strategy in ['BaseFrame', 'BaseFrameWithThird']:
                pipe_args['prompt_2'] = [details.reduced['frame_0']] * len(pipe_args['prompt'])
            elif strategy == 'VideoText':
                pipe_args['prompt_2'] = [prompt] * len(pipe_args['prompt'])
            
            if strategy in ['PreviousFrameWithThird', 'BaseFrameWithThird'] and self.model in ['SD3', 'FLUX']:
                pipe_args['prompt_3'] = [details.prompts[f'frame_{i}']] * len(pipe_args['prompt'])

        return pipe_args
    
    def create_run_info(self, call_params, total_run_time, details, first_frame_time):
        run_info = {
            'device': self.device,
            'fp16': self.fp16,
            'model': self.model,
            'prompt': call_params.get('prompt', None),
            'num_frames': call_params.get('num_frames', None),
            'list_of_prompts': call_params.get('list_of_prompts', None),
            'batch_size': call_params.get('batch_size', None),
            'save_intermediate': call_params.get('save_intermediate', None),
            'guidance': call_params.get('guidance', None),
            'ip_scale': call_params.get('ip_scale', None),
            'multi_prompt_strategy': call_params.get('multi_prompt_strategy', None),
            'intersection': call_params.get('intersection', None),
            'save_name': call_params.get('save_name', None),
            'display': call_params.get('display', None),
            'use_reducer': call_params.get('use_reducer', None),
            'total_run_time': total_run_time,
            'first_frame_generation_time': first_frame_time,
            'get_prompts_time': details.time_cost['get_prompts'],
            'get_reduction_time': details.time_cost['get_reduction'],
            'get_differences_time': details.time_cost['get_differences']
        }
        return run_info

    def __call__(self, prompt=None, num_frames=10, list_of_prompts=None, batch_size=1,
                 save_intermediate=False, guidance=7.0, ip_scale=0.0,
                 multi_prompt_strategy='none', intersection='previous',
                 save_name=None, details=None, display=False, use_reducer=False, 
                 method='grid', falloff=2):
        
        logging.debug("Pipeline __call__ started with prompt: %s, num_frames: %d", prompt, num_frames)
        start_time = time.time()

        # Clear the model to allow space on GPU VRAM
        self.clear_model()

        # Get frame-wise prompts
        details = details or VideoDetails(prompt, num_frames, list_of_prompts, display=display, use_reducer=use_reducer)
        num_frames = len(details)

        frames = []
        self.set_model()

        # Handle IP adapter if used
        if ip_scale > 0:
            base_image = self.pipe(details.reduced['frame_0']).images[0]
            self.pipe = load_ip_adapter(self.pipe, self.model, scale=ip_scale)

        # Log the initial inputs
        logging.debug(f"Initial inputs: prompt={prompt}, num_frames={num_frames}, list_of_prompts={list_of_prompts}, "
                      f"batch_size={batch_size}, save_intermediate={save_intermediate}, guidance={guidance}, "
                      f"ip_scale={ip_scale}, multi_prompt_strategy={multi_prompt_strategy}, intersection={intersection}, "
                      f"save_name={save_name}, details={details}, display={display}, use_reducer={use_reducer}")

        # Ensure all prompts are of the same length by padding them
        max_prompt_length = max(len(details.reduced[f"frame_{j}"]) for j in range(len(details)))
        for j in range(len(details)):
            prompt = details.reduced[f"frame_{j}"]
            if len(prompt) < max_prompt_length:
                prompt += ' ' * (max_prompt_length - len(prompt))  # Pad with spaces
            details.reduced[f"frame_{j}"] = prompt

        # Log the processed prompts
        logging.debug(f"Processed prompts: {details.reduced}")

        with tqdm(total=num_frames, desc="Generating frames") as pbar:
            # Generate the first frame
            callback_save = SaveFinalLatentsCallback()
            pipe_args = {
                'prompt': details.reduced['frame_0'],
                'callback_on_step_end': callback_save,
                'callback_on_step_end_tensor_inputs': callback_save.tensor_inputs,
                'guidance_scale': guidance
            }
            if ip_scale > 0:
                pipe_args['ip_adapter_image'] = base_image
            
            first_frame_start = time.time()
            original_image = self.pipe(**pipe_args).images[0]
            first_frame_time = time.time() - first_frame_start
            
            frames.append(original_image)
            pbar.update(1)

            if save_intermediate:
                original_image.save("../output/frame_0.png")

            # Generate remaining frames in batches
            for i in range(1, len(details), batch_size):
                # Get attention maps for current original image
                attention_maps = [
                    get_attention_map(self.clip_processor, self.clip_model, original_image, details.differences[f"frame_{j}"])
                    for j in range(i, min(i + batch_size, len(details)))
                ]

                original_latents = callback_save.latent_values if callback_save.latent_values.ndim == 3 else callback_save.latent_values[-1]

                if method == 'grid':    
                    callback_grid = GridPromptChangeCallback(
                        attention_maps, 
                        original_image=original_image if save_intermediate else None,
                        original_latents=original_latents,
                        path=f"../output", 
                        falloff=falloff
                        )
                    callbacks = [callback_grid]
                elif method == 'prompt':
                    callback_prompt = PromptChangeCallback(
                        new_prompt=details.reduced[f"frame_{i}"],
                        switch_time=0.5
                    )
                    callbacks = [callback_prompt]
                else:
                    callbacks=[]

                # Set up callbacks
                if intersection == 'previous':
                    callback_save = SaveFinalLatentsCallback()
                    callbacks.append(callback_save)

                callback = MultiPipelineCallbacks(callbacks)

                # Generate the images for the current batch
                prompts = [details.reduced[f"frame_{j}"] for j in range(i, min(i + batch_size, len(details)))]
                pipe_args = {
                    'prompt': prompts,
                    'callback_on_step_end': callback,
                    'callback_on_step_end_tensor_inputs': callback.tensor_inputs,
                    'guidance_scale': guidance
                }
                if ip_scale > 0:
                    pipe_args['ip_adapter_image'] = base_image
                pipe_args = self.multi_prompt(pipe_args, prompt, details, i, multi_prompt_strategy)
                images = self.pipe(**pipe_args).images
                
                for image in images:
                    frames.append(image)
                    original_image = image
                    pbar.update(1)

        total_run_time = time.time() - start_time

        # Save the generated frames as a GIF
        if save_name:
            save_frames_as_gif(frames, save_name)
            
        # Create run info dictionary
        call_params = locals()
        del call_params['self']  # Remove 'self' from the dictionary
        run_info = self.create_run_info(call_params, total_run_time, details, first_frame_time)

        # Append run_info to run_history
        self.run_history.append(run_info)

        logging.debug("Pipeline __call__ completed with total_run_time: %f seconds", total_run_time)

        return frames

    def log_run_history_to_csv(self):
        os.makedirs(os.path.dirname(self.run_history_path), exist_ok=True)
        
        file_exists = os.path.isfile(self.run_history_path)
        
        with open(self.run_history_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.run_history[0].keys())
            
            if not file_exists:
                writer.writeheader()
            
            for run_info in self.run_history:
                writer.writerow(run_info)
        
        # Clear the run history after logging
        self.run_history.clear()