import os
import json

from utils.load import load_llama, load_data
from utils.metrics import Metrics
from details import VideoDetails


def run_difference_test():
    device = 'cuda'
    input_json = load_data('../data/tests/text.json')
    prompt_diff_gen = PromptDifferenceGeneration(device)
    result_dict = prompt_diff_gen(input_json, save_text=True)
    
    

class PromptDifferenceGeneration:
    def __init__(self, device, save_path='../output/tests/difference'):
        self.device = device
        self.save_path = save_path
        self.metrics = Metrics(device)
        self.tokenizer, self.model = load_llama(device=self.device)


    def __call__(self, input_dict, save_text=False):
        result_dict = {}
        for key, value in input_dict.items():
            VD = VideoDetails(key, value['num_frames'], value['frames'], tokenizer=self.tokenizer, model=self.model)
            result_dict[key] = {
                'frames': value['frames'],
                'differences': VD.differences
            }

        if save_text:
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, 'log.json'), 'w') as f:
                json.dump(result_dict, f, indent=2)
        
        return result_dict

