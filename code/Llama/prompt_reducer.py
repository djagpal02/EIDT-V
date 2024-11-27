import torch

class LlamaPromptReducer:
    def __init__(self, video_context, tokenizer, model):
        self.device = model.device
        self.video_context = video_context
        self.tokenizer, self.model = tokenizer, model

        # Initial system message
        self.initial_message = [
            {"role": "system", "content": (
                "You are an expert in prompt optimization for video generation. "
                "Your task is to reduce the provided prompt while keeping essential details intact, "
                "ensuring coherence between frames."
            )}
        ]

        self.max_tokens = 75
        self.reset_history()

    def __call__(self, prompt):
        messages = self.user_template(prompt)
        response = self.generate_response(messages)
        reduced_prompt = self.format_response(response)
        self.update_history(reduced_prompt)
        return reduced_prompt

    # Function to generate response
    def generate_response(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        attention_mask = torch.ones(input_ids.shape, device=self.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Enable automatic mixed precision
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=256,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
        response_ids = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
    
        return response_text

    def user_template(self, prompt):
        reduce_instruction = (
            f"You will be given a single prompt intended for video generation.\n\n"
            f"The video context is: '{self.video_context}'.\n\n"
            f"Your task is to reduce the current prompt to fewer than {self.max_tokens} tokens "
            f"while retaining as much essential detail as possible. Focus on maintaining the key elements, "
            f"such as actions, descriptive details, and important objects or scenery. "
            f"Ensure that the prompt is concise but still conveys the full meaning and imagery of the original.\n\n"
            f"Original prompt: {prompt}\n\n"
            f"Do not provide any extra text, explanations, other details, or instructions.\n\n"
            f"Please respond with the reduced prompt only."
        )
        return self.initial_message + [{"role": "user", "content": reduce_instruction}]



    def format_response(self, response_text):
        tokens = self.tokenizer.tokenize(response_text)
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
        return self.tokenizer.convert_tokens_to_string(tokens)

    def reset_history(self):
        self.history = []

    def update_history(self, reduced_prompt):
        if reduced_prompt:
            self.history.append(reduced_prompt)

    def history_to_text(self):
        return " | ".join(self.history)