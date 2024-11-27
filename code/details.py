# Python standard libraries
import gc
import time

# Other libraries
import torch
from tqdm import tqdm

# Local imports
from Llama.difference_detector import LlamaDifferenceDetector
from Llama.prompt_reducer import LlamaPromptReducer
from Llama.framewise_prompts import LlamaFramewisePrompts
from utils.load import load_llama

class VideoDetails:
    def __init__(self, video_text, num_frames, list_of_prompts=None, display=False, use_reducer=False, tokenizer=None, model=None):
        self.video_text = video_text
        self.num_frames = num_frames
        self.prompts = {}
        self.differences = {}
        self.reduced = {}
        self.time_cost = {
            'get_prompts': 0,
            'get_reduction': 0,
            'get_differences': 0
        }  

        if tokenizer is None or model is None:
            self.tokenizer, self.model = load_llama(device="cuda")
        else:
            self.tokenizer = tokenizer
            self.model = model

        # Fix mismatch incase there is one
        if list_of_prompts:
            self.num_frames = len(list_of_prompts)

        self.use_reducer = use_reducer

        self.get_prompts(list_of_prompts, display=display)
        self.get_differences(display=display)
        if self.use_reducer:
            self.get_reduction(display=display)
        else:
            self.reduced = self.prompts.copy()  # Use original prompts if reducer is not used

        self.clear_models()

    def __len__(self):
        return self.num_frames

    def clear_models(self):
        del self.tokenizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def get_prompts(self, list_of_prompts=None, display=False):
        start_time = time.time()
        if list_of_prompts:
            self.prompts = {f"frame_{i}": prompt for i, prompt in enumerate(list_of_prompts)}
        else:
            print("Generating frame-wise prompts...")
            llama = LlamaFramewisePrompts(self.tokenizer, self.model)
            self.prompts = {f"frame_{i}": prompt for i, prompt in enumerate(llama(self.video_text, self.num_frames))}

            del llama
            gc.collect()
            torch.cuda.empty_cache()
        
        end_time = time.time()
        self.time_cost['get_prompts'] = end_time - start_time
        
        if display:
            for i in range(self.num_frames):
                print(f"Frame {i}: {self.prompts[f'frame_{i}']}")

        return self.prompts

    def get_reduction(self, display=False):
        start_time = time.time()
        if not self.use_reducer:
            print("Prompt reducer is disabled. Using original prompts.")
            return self.prompts.copy()

        # Instantiate the LlamaPromptReducer
        llama = LlamaPromptReducer(self.video_text, self.tokenizer, self.model)

        for i in tqdm(range(self.num_frames), desc="Reducing prompts"):
            prompt = self.prompts[f"frame_{i}"]

            # Tokenize the prompt to get the number of tokens
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", truncation=False)

            # Check if the prompt exceeds the 75 token limit
            if tokenized_prompt.input_ids.shape[-1] > 75:
                # Reduce the prompt if it exceeds 75 tokens
                self.reduced[f"frame_{i}"] = llama(prompt)
            else:
                # If the prompt is within the limit, no reduction is necessary
                self.reduced[f"frame_{i}"] = prompt

        end_time = time.time()
        self.time_cost['get_reduction'] = end_time - start_time

        # Delete the reducer object and clear GPU memory
        del llama
        gc.collect()
        torch.cuda.empty_cache()

        if display:
            for i in range(self.num_frames):
                print(f"Frame {i}: {self.reduced[f'frame_{i}']}")

        return self.reduced

    def get_differences(self, display=False):
        start_time = time.time()
        llama = LlamaDifferenceDetector(self.video_text, self.tokenizer, self.model)
        for i in tqdm(range(self.num_frames - 1), desc="Calculating differences"):
            prompt1 = self.prompts[f"frame_{i}"]
            prompt2 = self.prompts[f"frame_{i+1}"]
            self.differences[f"frame_{i+1}"] = llama(prompt1, prompt2)

        end_time = time.time()
        self.time_cost['get_differences'] = end_time - start_time

        # Delete the llama object and clear GPU memory
        del llama
        gc.collect()
        torch.cuda.empty_cache()

        if display:
            for i in range(1, self.num_frames):
                print(f"Frame {i}: {self.differences[f'frame_{i}']}")

        return self.differences
    

class ImageDetails(VideoDetails):
    def __init__(self, prompt1, prompt2, use_reducer=False, tokenizer=None, model=None):
        super().__init__(prompt1, num_frames=2, list_of_prompts=[prompt1, prompt2], use_reducer=use_reducer, 
                         tokenizer=tokenizer, model=model)
