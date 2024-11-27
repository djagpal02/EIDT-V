import torch
from diffusers.callbacks import PipelineCallback

class PromptChangeCallback(PipelineCallback):
    def __init__(self, new_prompt, switch_time):        
        self.new_prompt = new_prompt
        self.switch_time = switch_time # 0 = image/latent, 1 = noise
        self.switched = False

    @property
    def tensor_inputs(self):
        return []
    
    def callback_fn(self, pipe, step, timestep, callback_kwargs):
        total_timesteps = pipe.scheduler.config.num_train_timesteps

        if self.switch_time * total_timesteps >= timestep and self.switched == False:
            self.switched = True
            return self.get_updated_params(pipe, callback_kwargs)
        return {}


    def get_updated_params(self, pipe, callback_kwargs):
        device = pipe._execution_device

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=self.new_prompt,
            device=device,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            clip_skip=pipe.clip_skip,
        )

        add_text_embeds = pooled_prompt_embeds

        if pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)

 
        return {
                    "prompt_embeds": prompt_embeds,
                    "add_text_embeds": add_text_embeds,
                }
    