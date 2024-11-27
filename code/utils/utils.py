import torch
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from accelerate import PartialState
from compel import Compel, ReturnedEmbeddingsType
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os
import imageio

def prompt_to_name(prompt):
    # Convert prompt to lowercase and replace spaces with underscores
    folder_name = re.sub(r'[^\w\s-]', '', prompt.lower())
    folder_name = re.sub(r'[-\s]+', '_', folder_name)
    return folder_name


def save_frames_as_gif(frames, save_name):
    # Make directory if it doesn't exist
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    # Save frames as GIF with each frame displayed for 0.125 seconds (8 fps) and loop indefinitely
    imageio.mimsave(save_name, frames, fps=8, loop=0)




def prompt_reweighting(pipe, prompt, neg_prompt=None):
    # reweighting the prompt to allow for longer prompts
    compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True]
    )

    conditioning, pooled = compel(prompt)
    if neg_prompt is not None:
        negative_conditioning, negative_pooled = compel(neg_prompt)
    else:
        negative_conditioning = None
        negative_pooled = None
    
    return {
        "prompt_embeds": conditioning,
        "negative_prompt_embeds": negative_conditioning,
        "pooled_prompt_embeds": pooled,
        "negative_pooled_prompt_embeds": negative_pooled
    }


    
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)