import torch

class LlamaDifferenceDetector:
    def __init__(self, video_context, tokenizer, model):
        self.device = model.device
        self.video_context = video_context
        self.tokenizer, self.model = tokenizer, model

        # Initial system message
        self.initial_message = [
            {"role": "system", "content": (
                "You are an expert video director."
            )}
        ]

        self.heading = "List of changed objects:"

        # Get possible changes based on the video context using the LLM
        self.possible_changes = self.get_possible_changes(video_context)

        self.reset_history()

    def get_possible_changes(self, context):
        prompt = (
            f"Based on the context of the video '{context}', list the possible objects or elements that "
            f"might be in motion or changing throughout the video. Provide a concise list of these possible changes."
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.generate_response(messages)
        return response

    def __call__(self, previous_frame, current_frame):
        messages = self.user_template(previous_frame, current_frame)
        response = self.generate_response(messages)
        differences = self.format_response(response)
        differences = self.ensure_valid_differences(differences)
        self.update_history(differences)
        return differences

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
                    temperature=0.5,
                    top_p=0.7,
                )
        response_ids = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
    
        return response_text

    def user_template(self, previous_frame, current_frame):
        compare_instruction = (
            f"You will be given the description of two consecutive frames from a video.\n\n"
            f"Overall context of the video: {self.video_context}\n\n"
            f"Possible changes: {self.possible_changes}\n\n"
            f"Previous frame differences within this video: {self.history_to_text()}\n\n"
            f"Compare the following two frames and identify the objects or elements that have changed between them. "
            f"Use the context of the video, the history of previous differences, and the list of possible changes to ensure your response is consistent "
            f"with the natural flow of the video. Ensure the changes are logical for what could occur in a natural video. "
            f"Provide a concise list of these changes. "
            f"If there are no changes, please respond with whatever may be in motion. "
            f"If the new frame is completely different from the previous frame, please respond with 'everything'. "
            f"Format your response as follows:\n"
            f"- Start the list with '{self.heading}'\n"
            f"- Each item should be a 1-3 word bullet point\n"
            f"- Avoid any additional explanations or details\n\n"
            f"Previous frame: {previous_frame}\n\n"
            f"Current frame: {current_frame}"
        )
        return self.initial_message + [{"role": "user", "content": compare_instruction}]

    def format_response(self, response_text):
        heading_index = response_text.lower().find(self.heading.lower())
        if heading_index == -1:
            return []  # Return an empty list if the heading is not found
        
        changed_objects_section = response_text[heading_index + len(self.heading):].strip()
        lines = changed_objects_section.split('\n')
        changed_objects = []

        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                item = line.lstrip('- *• ').strip()
                # Remove any details in brackets
                item = item.split(' (')[0].strip()
                changed_objects.append(item)

        return changed_objects

    def ensure_valid_differences(self, differences):
        valid_differences = [diff for diff in differences if diff.lower() != "none" and diff.lower() != "everything"]
        if not valid_differences:
            return ["Motion detected"]  # A placeholder to indicate some motion if no valid differences are found
        return valid_differences

    def reset_history(self):
        self.history = []

    def update_history(self, differences):
        if differences:
            self.history.append(", ".join(differences))

    def history_to_text(self):
        return " , ".join(self.history)