import os
import re
import json
import numpy as np

from tqdm import tqdm

from Llama.framewise_prompts import LlamaFramewisePrompts
from utils.load import load_llama, load_data
from utils.metrics import Metrics



def run_text_test():
    device = 'cuda'
    input_json = load_data('../data/tests/text.json')

    FG = FrameWiseTextGeneration(device)
    FG(input_json, save_text=True, save_stats=True)
    

class FrameWiseTextGeneration:
    def __init__(self, device, save_path='../output/tests/text'):
        self.device = device
        self.save_path = save_path
        self.metrics = Metrics(device)
        self.tokenizer, self.model = load_llama(device=self.device)

    def process_frames(self, user_text, frames, frame_type):
        all_texts = [user_text] + frames
        similarity_matrix = self.metrics.text_cosine_similarity(all_texts)
        
        # Extract cosine similarities between frames
        frame_similarities = similarity_matrix[1:, 1:]
        # Get the upper triangular matrix to avoid repeating comparisons
        cosine_similarities = frame_similarities[np.triu_indices(len(frames), k=1)]
        
        # Extract base similarities (between user_text and each frame)
        base_similarities = similarity_matrix[0, 1:]

        avg_cosine_similarity = np.mean(cosine_similarities)
        avg_base_similarity = np.mean(base_similarities)

        return {
            'User Text': user_text,
            'Frame Type': frame_type,
            'Num Frames': len(frames),
            'Avg Cosine Similarity': avg_cosine_similarity,
            'Avg Base Cosine Similarity': avg_base_similarity,
        }

    def __call__(self, input_dict, save_text=False, save_stats=False):
        LFP = LlamaFramewisePrompts(self.tokenizer, self.model)
        save_log = {}
        stats_data = []
        
        for user_text, values in tqdm(input_dict.items(), desc="Processing prompts"):
            # Process given frames
            given_frames = values['frames']
            given_stats = self.process_frames(user_text, given_frames, values['text_source'])
            stats_data.append(given_stats)

            # Generate and process new frames
            generated_frames = LFP(user_text, values['num_frames'])
            generated_stats = self.process_frames(user_text, generated_frames, "generated")
            stats_data.append(generated_stats)
            
            if save_text:
                save_log[user_text] = {
                    values['text_source']: given_frames,
                    "generated_frames": generated_frames
                }

        if save_text:
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, 'log.json'), 'w') as f:
                json.dump(save_log, f, indent=2)

        if save_stats:
            import csv
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, "stats.csv"), 'w', newline='') as csvfile:
                fieldnames = ['User Text', 'Frame Type', 'Num Frames', 'Avg Cosine Similarity', 'Avg Base Cosine Similarity']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in stats_data:
                    writer.writerow(row)


