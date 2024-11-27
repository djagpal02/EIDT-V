# Python standard libraries
import os
import time
import contextlib
import io
import sys
import re

# Other libraries
from tqdm import tqdm

# Local imports
from utils.load import load_sdxl, load_ip_adapter, load_clip_segmentation, load_data, load_llama
from callbacks.prompt_change import PromptChangeCallback
from callbacks.grid_prompt_change import GridPromptChangeCallback
from callbacks.save_latents import SaveFinalLatentsCallback
from intersection_mask.clip_attention import get_attention_map
from utils.metrics import Metrics
from utils.utils import prompt_to_name
from details import ImageDetails

def run_image_test():
    device = 'cuda'
    input_json = load_data('../data/tests/image.json')
  
    IM = ImageManipulation(device)
    IM(input_json, save_img=True, save_stats=True)
    

class ImageManipulation:
    def __init__(self, device, save_path='../output/tests/image'):
        self.device = device
        self.save_path = save_path
        self.inference_steps = 30
        self.metrics = Metrics()
        self.pipe = None
        self.pbar = None

    # Loading and Setup
    def reload_sdxl(self, scale=0.0):
        self.pipe = None
        # Suppress all output
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), open(os.devnull, 'w') as fnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = fnull
            sys.stderr = fnull
            try:
                if scale != 0.0:
                    self.pipe = load_sdxl(self.device)
                    self.pipe = load_ip_adapter(self.pipe, 'SDXL', scale)
                else:
                    self.pipe = load_sdxl(self.device)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        self.pipe.set_progress_bar_config(disable=True)

    def generate_base_images(self, input_dict):
        base_images = {}
        for initial_prompt, values in input_dict.items():
            modified_prompt = values['after']
            folder_name = prompt_to_name(initial_prompt)
            folder_path = os.path.join(self.save_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Generate base image
            self.reload_sdxl()
            callback = SaveFinalLatentsCallback()
            base_image = self.pipe(
                initial_prompt,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=callback.tensor_inputs,
                num_inference_steps=self.inference_steps
            ).images[0]
            base_image.save(os.path.join(folder_path, 'base_image.png'))

            base_images[initial_prompt] = {
                'modified_prompt': modified_prompt,
                'folder_path': folder_path,
                'base_image': base_image,
                'latents': callback.latent_values
            }
        return base_images

    def extract_details(self, base_images):
        self.pipe = None
        preprocessed_data = {}
        processor, model = load_clip_segmentation(device=self.device)
        tokenizer, llama_model = load_llama(device=self.device)

        for initial_prompt, data in base_images.items():
            modified_prompt = data['modified_prompt']
            folder_path = data['folder_path']
            base_image = data['base_image']

            # Get prompt differences and attention map
            details = ImageDetails(initial_prompt, modified_prompt, tokenizer=tokenizer, model=llama_model)
            prompt_differences = details.differences['frame_1']
            attention_maps = [get_attention_map(processor, model, base_image, prompt_differences)]

            preprocessed_data[initial_prompt] = {
                'modified_prompt': modified_prompt,
                'folder_path': folder_path,
                'base_image': base_image,
                'latents': data['latents'],
                'attention_maps': attention_maps,
                'group': data.get('group', ''),  # Ensure 'group' key is present
                'test_type': data.get('type', '')  # Ensure 'test_type' key is present
            }

        return preprocessed_data

    def preprocess_prompts(self, input_dict):
        base_images = self.generate_base_images(input_dict)
        return self.extract_details(base_images)

    # Image Manipulation
    def ip_adapter(self, prompt, base_image):
        return self.pipe(prompt=prompt, ip_adapter_image=base_image, num_inference_steps=self.inference_steps).images[0]
    
    def prompt_switch(self, initial_prompt, modified_prompt, switch_time):
        callback = PromptChangeCallback(modified_prompt, switch_time)
        return self.pipe(prompt=initial_prompt, callback_on_step_end=callback, callback_on_step_end_tensor_inputs=callback.tensor_inputs, num_inference_steps=self.inference_steps).images[0]
    
    def grid_prompt_switch(self, initial_prompt, attention_maps, latents, falloff):
        callback = GridPromptChangeCallback(attention_maps, original_latents=latents, falloff=falloff)
        return self.pipe(
            prompt=initial_prompt, 
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=callback.tensor_inputs, 
            num_inference_steps=self.inference_steps
        ).images[0]

    # Stats Collection
    def get_stats(self, out_img, base_image, initial_prompt, modified_prompt):
        ssim = self.metrics.compute_ssim(out_img, base_image)
        lpips = self.metrics.lpips(out_img, base_image)
        clip_score_old = self.metrics.clip_score(out_img, initial_prompt)
        clip_score_new = self.metrics.clip_score(out_img, modified_prompt)
        return ssim, lpips, clip_score_old, clip_score_new

    # Main Function
    def __call__(self, input_dict, save_img=False, save_stats=False):
        preprocessed_data = self.preprocess_prompts(input_dict)
        stats_data = []
        total_generations = len(preprocessed_data) * (4 + 4 + 4)  # IP Adapter + Prompt Switch + Grid Prompt Switch
        self.pbar = tqdm(total=total_generations, desc="Overall Progress", ncols=100)

        for initial_prompt, prompt_data in preprocessed_data.items():
            modified_prompt = prompt_data['modified_prompt']
            group = prompt_data['group']
            test_type = prompt_data['test_type']

            # IP Adapter
            self.reload_sdxl(scale=0.8)  # Load once with the highest scale
            for scale in [0.2, 0.4, 0.6, 0.8]:
                self.pipe.set_ip_adapter_scale(scale)  # Adjust scale without reloading
                out_img = self.ip_adapter(modified_prompt, prompt_data['base_image'])
                self.save_and_collect_stats('ip_adapter', scale, out_img, prompt_data, initial_prompt, modified_prompt, save_img, save_stats, stats_data, group, test_type)

            # Prompt Switch
            self.reload_sdxl()
            for switch_time in [0.2, 0.4, 0.6, 0.8]:
                out_img = self.prompt_switch(initial_prompt, modified_prompt, switch_time)
                self.save_and_collect_stats('prompt_switch', switch_time, out_img, prompt_data, initial_prompt, modified_prompt, save_img, save_stats, stats_data, group, test_type)

            # Grid Prompt Switch
            for falloff in [0.5, 1, 2, 3]:
                out_img = self.grid_prompt_switch(initial_prompt, prompt_data['attention_maps'], prompt_data['latents'], falloff)
                self.save_and_collect_stats('grid_prompt_switch', falloff, out_img, prompt_data, initial_prompt, modified_prompt, save_img, save_stats, stats_data, group, test_type)

        if save_stats:
            self.write_stats_to_csv(stats_data)

        self.pbar.close()

    def save_and_collect_stats(self, model_type, param_value, out_img, prompt_data, initial_prompt, modified_prompt, save_img, save_stats, stats_data, group, test_type):
        if save_img:
            out_img.save(os.path.join(prompt_data['folder_path'], f"{model_type}_{param_value}.png"))
        if save_stats:
            ssim, lpips, clip_score_old, clip_score_new = self.get_stats(out_img, prompt_data['base_image'], initial_prompt, modified_prompt)
            stats_data.append({
                'group': group,
                'test_type': test_type,
                'original_prompt': initial_prompt,
                'modified_prompt': modified_prompt,
                'model_type': model_type,
                'model_param_name': 'scale' if model_type == 'ip_adapter' else 'switch_time' if model_type == 'prompt_switch' else 'falloff',
                'model_parameter': param_value,
                'ssim': ssim,
                'lpips': lpips,
                'clip_score_old': clip_score_old,
                'clip_score_new': clip_score_new
            })
        self.pbar.update(1)

    def write_stats_to_csv(self, stats_data):
        import csv
        with open(os.path.join(self.save_path, "stats.csv"), 'w', newline='') as csvfile:
            fieldnames = ['group', 'test_type', 'original_prompt', 'modified_prompt', 'model_type', 'model_param_name', 'model_parameter', 'ssim', 'lpips', 'clip_score_old', 'clip_score_new']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in stats_data:
                writer.writerow(row)