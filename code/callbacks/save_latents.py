from diffusers.callbacks import PipelineCallback

class SaveLatentsCallback(PipelineCallback):
    def __init__(self):
        self.latent_values = []

    @property
    def tensor_inputs(self):
        return ["latents"]
    
    def callback_fn(self, pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents", None)
        if latents is not None:
            # Convert the latents tensor to a numpy array and save it
            self.latent_values.append(latents.cpu().detach())
        return callback_kwargs
    

class SaveFinalLatentsCallback(PipelineCallback):
    def __init__(self, to_cpu=False):
        self.latent_values = None
        self.to_cpu = to_cpu

    @property
    def tensor_inputs(self):
        return ["latents"]

    def callback_fn(self, pipe, step_index, timestep, callback_kwargs):    
        latents = callback_kwargs.get("latents", None)
        if latents is not None and step_index == pipe._num_timesteps - 1:
            # Convert the latents tensor to a numpy array and save it
            self.latent_values = latents.detach()
            if self.to_cpu:
                self.latent_values = self.latent_values.cpu()

        return callback_kwargs