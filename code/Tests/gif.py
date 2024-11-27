import os
import csv
import time

from tqdm import tqdm
from pipeline import Pipeline
from utils.metrics import Metrics
from utils.utils import prompt_to_name
import numpy as np
from itertools import product

from utils.load import load_data, load_llama
from details import VideoDetails

def run_gif_test():
    device = 'cuda'
    test_options = {
        'grid_search': '../data/tests/grid_search.json',
        'prev_work_comparison': {
            'SD': '../data/tests/prev_work_comparison.json',
            'SDXL': '../data/tests/sdxl.json',
            'SD3': '../data/tests/sd3.json'
        },
        'ablation_study': '../data/tests/ablation.json',
        'sd3_final_batch': '../data/tests/sd3_final_batch.json',
        'sdxl_final_batch': '../data/tests/sdxl_final_batch.json'
    }

    print("Available test options:")
    for option in test_options.keys():
        print(f"- {option}")

    test_name = input("Please enter the test name you want to run: ")

    if test_name not in test_options:
        raise ValueError(f"Unknown test name: {test_name}")

    if test_name == 'grid_search':
        input_json = load_data(test_options[test_name])
        GG = GifGeneration(device)
        GG(input_json, save_path='../output/tests/gif_grid_search', save_gif=True, save_stats=True)

    elif test_name == 'prev_work_comparison':
        for model, path in test_options[test_name].items():
            input_json = load_data(path)
            GG = GifGeneration(device, save_path='../output/tests/gif_prev_work_comparison', model=model)
            GG(input_json, save_gif=True, save_stats=True)

    elif test_name == 'ablation_study':
        input_json = load_data(test_options[test_name])
        framewise_prompts = load_data('../data/tests/chatgpt_framewise.json')

        # Our frames with grid
        GG = GifGeneration(device, save_path='../output/tests/gif_ablation_study/our_frames_grid', model='SD')
        GG(input_json, save_gif=True, save_stats=True)

        # Our frames without grid
        input_json['method'] = 'none'
        GG = GifGeneration(device, save_path='../output/tests/gif_ablation_study/our_frames_no_grid', model='SD')
        GG(input_json, save_gif=True, save_stats=True)

        # ChatGPT frames without grid
        input_json['prompt'] = framewise_prompts
        GG = GifGeneration(device, save_path='../output/tests/gif_ablation_study/chatgpt_frames_no_grid', model='SD')
        GG(input_json, save_gif=True, save_stats=True)

        # ChatGPT frames with grid
        input_json['method'] = 'grid'
        GG = GifGeneration(device, save_path='../output/tests/gif_ablation_study/chatgpt_frames_grid', model='SD')
        GG(input_json, save_gif=True, save_stats=True)

    elif test_name == 'sd3_final_batch':
        input_json = load_data(test_options[test_name])
        GG = GifGeneration(device, save_path='../output/tests/gif_sd3_final_batch', model='SD3')
        GG(input_json, save_gif=True, save_stats=True)

    elif test_name == 'sdxl_final_batch':
        input_json = load_data(test_options[test_name])
        GG = GifGeneration(device, save_path='../output/tests/gif_sdxl_final_batch', model='SDXL')
        GG(input_json, save_gif=True, save_stats=True)

class GifGeneration:
    def __init__(self, device, save_path='../output/tests/gif', model='SDXL'):
        self.device = device
        self.save_path = os.path.join(save_path, model)
        self.metrics = Metrics(device)
        self.model = model
        
    def list_of_tests(self,input_dict):
        if 'prompt' not in input_dict or not isinstance(input_dict['prompt'], dict):
            raise ValueError("input_dict must contain a 'prompt' key with a dictionary value")
        
        prompt_dict = input_dict['prompt']
        input_dict['prompt'] = list(prompt_dict.keys())

        # Get all keys and values from the input dictionary
        keys = list(input_dict.keys())
        values = [input_dict[key] if isinstance(input_dict[key], list) else [input_dict[key]] for key in keys]
        
        test_setups = []
        for combination in product(*values):
            test_setup = dict(zip(keys, combination))
            test_setups.append(test_setup)
          
        return test_setups, prompt_dict
    
    def get_details(self, test_setups, prompt_dict):
        # Compute unique prompts and num frames
        unique_prompts_and_num_frames = {(setup['prompt'], setup['num_frames']) for setup in test_setups}
        tokenizer, model = load_llama(device=self.device)

        # Create details dictionary
        details = {}
        for prompt, num_frames in unique_prompts_and_num_frames:
            frames = prompt_dict.get(prompt, [])
            if len(frames) != num_frames:
                frames = None
            details[(prompt, num_frames)] = VideoDetails(video_text=prompt, num_frames=num_frames, list_of_prompts=frames, tokenizer=tokenizer, model=model )

        # Add details to test setups
        for setup in test_setups:
            setup['details'] = details[(setup['prompt'], setup['num_frames'])]
        
        print("Details generated and added to test setups")

        del tokenizer, model
        return test_setups

    def get_stats(self, test_params, frames, metrics):
        if not frames:
            raise ValueError("Frames cannot be empty or None")
        start_time = time.time()
        stats = {}

        if 'clip' in metrics:
            prompt = test_params['prompt']
            stats['clip'] = self.metrics.clip_score_frames(frames, prompt)   

        if 'ms_ssim' in metrics:
            stats['ms_ssim'] = self.metrics.ms_ssim_frames(frames)    

        if 'lpips' in metrics:
            stats['lpips'] = self.metrics.lpips_frames(frames)

        if 'temporal_consistency_loss' in metrics:
            losses = self.metrics.temporal_consistency_loss(frames)
            stats['temporal_consistency_loss'] = losses[0]
            stats['temporal_consistency_loss_warp'] = losses[1]
            stats['temporal_consistency_loss_smooth'] = losses[2]

        end_time = time.time()
        print(f"Time taken to get stats: {end_time - start_time:.2f} seconds")

        return stats
    
    def create_detailed_filename(self, params):
        if 'prompt' not in params:
            raise ValueError("params must contain a 'prompt' key")

        filename_parts = []

        prompt_name = prompt_to_name(params['prompt'])[:30]
        filename_parts.append(prompt_name)

        param_mapping = {
            'num_frames': 'f',
            'batch_size': 'b',
            'guidance': 'g',
            'ip_scale': 'ip',
            'multi_prompt_strategy': '',
            'intersection': ''
        }

        for param, prefix in param_mapping.items():
            if param in params:
                value = params[param]
                if isinstance(value, float):
                    formatted_value = f"{value:.1f}"
                else:
                    formatted_value = str(value)
                filename_parts.append(f"{prefix}{formatted_value}")

        # Join all parts and add .gif extension
        filename = "_".join(filename_parts) + ".gif"

        # Replace any characters that might be invalid in filenames
        filename = "".join(c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in filename)
        return filename
    
    def save_stats(self, stats_data):
        if not stats_data:
            raise ValueError("stats_data cannot be empty")
        os.makedirs(self.save_path, exist_ok=True)
        csv_path = os.path.join(self.save_path, "gif_stats.csv")
        
        # Check if the file already exists
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:  # Open in append mode
            fieldnames = ['prompt', 'num_frames', 'batch_size', 'guidance', 'ip_scale', 
                          'multi_prompt_strategy', 'intersection', 'clip', 'ms_ssim', 'lpips', 
                          'temporal_consistency_loss', 'temporal_consistency_loss_warp', 'temporal_consistency_loss_smooth']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()  # Write header only if file does not exist
            
            for stat in stats_data:
                writer.writerow(stat)
        
        print(f"Stats appended to {csv_path}")

    def estimated_time(self, test_setups, prompt_dict):
        total_seconds = sum((setup['num_frames'] * 13 + 15) for setup in test_setups)
        total_seconds += len(prompt_dict) * 30
        days, remainder = divmod(total_seconds, 86400)
        hours, seconds = divmod(remainder, 3600)
        minutes, seconds = divmod(seconds, 60)
        return days, hours, minutes, seconds

    def __call__(self, input_dict, save_gif=False, save_stats=False, metrics=['clip', 'ms_ssim', 'lpips', 'temporal_consistency_loss']):
        if not isinstance(input_dict, dict):
            raise ValueError("input_dict must be a dictionary")
        pipe = Pipeline(self.device, self.model)
        stats_data = []

        # get details added to each setup
        test_setups, prompt_dict = self.list_of_tests(input_dict)
        
        if self.model == 'SDXL':    
            days, hours, minutes, seconds = self.estimated_time(test_setups, prompt_dict)
            # Display estimated time in seconds, hours and days
            print(f"\n Running {len(test_setups)} tests, estimated time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds\n")
           
        
        test_setups = self.get_details(test_setups, prompt_dict)

        for test_params in tqdm(test_setups, desc="Running tests", unit="test"):  # Add tqdm here
            # Create a detailed filename
            filename = self.create_detailed_filename(test_params)
            
            if save_gif:
                test_params['save_name'] = os.path.join(self.save_path, filename)

            # Call the pipeline with the test parameters
            # Remove user_prompt from test_params before calling pipeline
            pipeline_params = {k: v for k, v in test_params.items() if k != 'user_prompt'}
            frames = pipe(**pipeline_params)

            if save_stats:
                stats = self.get_stats(test_params, frames, metrics)
                stat_entry = {
                    'prompt': test_params['prompt'],
                    'num_frames': test_params['num_frames'],
                    'batch_size': test_params['batch_size'],
                    'guidance': test_params['guidance'],
                    'ip_scale': test_params['ip_scale'],
                    'multi_prompt_strategy': test_params['multi_prompt_strategy'],
                    'intersection': test_params['intersection'],
                    'clip': stats.get('clip', np.nan),
                    'ms_ssim': stats.get('ms_ssim', np.nan),
                    'lpips': stats.get('lpips', np.nan),
                    'temporal_consistency_loss': stats.get('temporal_consistency_loss', np.nan),
                    'temporal_consistency_loss_warp': stats.get('temporal_consistency_loss_warp', np.nan),
                    'temporal_consistency_loss_smooth': stats.get('temporal_consistency_loss_smooth', np.nan)
                }
                stats_data.append(stat_entry)
                
        # Log run history to CSV
        pipe.log_run_history_to_csv()

        if save_stats:
            self.save_stats(stats_data)

        return stats_data


