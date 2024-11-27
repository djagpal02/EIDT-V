# 1. Python standard libraries
import json

# 2. Other libraries
import torch
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPProcessor, CLIPModel
import lpips

# 3. Local imports
from Llama.login import *



def load_sd(device, fp16=True):
    if fp16:
        pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
    else:
        pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    return pipe.to(device)

def load_sdxl(device, fp16=True):
    if fp16:
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    else:
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    return pipe.to(device)

def load_sd3(device, fp16=True):
    # FP16 only here for consistency with other models - will always be fp16
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)    
    return pipe.to(device)

def load_flux(device, fp16=True):
    # Only fp16 is supported for Flux for vram reasons
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
    return pipe.to(device)

def load_ip_adapter(pipeline, model_type="SDXL", scale=0.6):
    # Determine the appropriate subfolder and weight name based on the model type
    if model_type.upper() == "SDXL":
        subfolder = "sdxl_models"
        weight_name = "ip-adapter_sdxl.bin"
    elif model_type.upper() == "SD":
        subfolder = "models"
        weight_name = "ip-adapter_sd15.bin"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load the IP-Adapter with the specified subfolder and weight name
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder=subfolder, weight_name=weight_name)

    # Set the scale for the IP-Adapter
    pipeline.set_ip_adapter_scale(scale)
    return pipeline


def load_llama(device):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, pad_token='<pad>')
    tokenizer.pad_token = tokenizer.eos_token  # Explicitly setting pad_token to eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use half precision for faster computation
        device_map=device,
    )
    return tokenizer, model

def load_clip(device="cuda"):
    # Score Model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

def load_clip_segmentation(device):
    # Segmentation Model
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    return processor, model

def load_lpips():
    lpips_model = lpips.LPIPS(net="vgg")
    return lpips_model

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data
