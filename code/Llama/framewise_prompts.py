import torch
import re

from utils.load import load_data

class LlamaFramewisePrompts:
    def __init__(self, tokenizer, model, examples_path='../data/example.json'):
        self.device = model.device
        self.tokenizer, self.model = tokenizer, model
        self.examples_path = examples_path
        self.examples = load_data(self.examples_path)

        self.initial_message = [
            {"role": "system", "content": "You are an expert video director."}
        ]

    def format_examples(self):
        formatted_examples = []
        for video_text, text in self.examples.items():
            formatted_examples.append({
                'role': 'user',
                'content': self.user_template(video_text, len(text["dynamic part"]))[0]['content']
            })
            formatted_examples.append({
                'role': 'assistant',
                'content': self.format_example_output(text)
            })

        return formatted_examples

    def format_example_output(self, example_text):
        fixed_part = example_text['fixed part']
        dynamic_part = example_text['dynamic part']
        formatted_output = f"Fixed Part:\n{fixed_part}\n\nDynamic Part:\n"
        for frame in dynamic_part:
            formatted_output += f"{frame}\n"
        return formatted_output

    def __call__(self, video_context, num_frames):
        attempts = 0

        while attempts < 50:
            messages = self.initial_message + self.format_examples() + self.user_template(video_context, num_frames)
            response = self.generate_response(messages)
            try:
                frames = self.extract_frames(response, num_frames)
                return frames
            
            except Exception as e:
                print(f"Attempt {attempts + 1} failed: {e}. Retrying...")
                attempts += 1
                continue
        raise ValueError("Failed to generate the correct number of frames after 50 attempts.")

    def generate_response(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        attention_mask = torch.ones(input_ids.shape, device=self.device)
       
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=2048,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.9,  # Increased temperature for more variety
                    top_p=0.95,      # Increased top_p for more variety
                    repetition_penalty=1.2,  # Added to prevent repetitive outputs
                )
        response_ids = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    def user_template(self, video_context, num_frames):
        instructions = (
            f"You will be given a single prompt intended for video generation.\n"
            f"Your task is to create a video plan that includes a fixed part and a dynamic part.\n"
            f"- The **fixed part** should be about 60 tokens and include all the consistent features of the video.\n"
            f"- The **dynamic part** should be about 15 tokens per frame and briefly describe what is changing in each frame.\n\n"
            f"Format your response as follows:\n\n"
            f"Fixed Part:\n"
            f"[Fixed part description]\n\n"
            f"Dynamic Part:\n"
            f"[Description of changes in frame 1]\n"
            f"[Description of changes in frame 2]\n"
            f"...\n"
            f"[Description of changes in frame {num_frames}]\n\n"
            f"Ensure the fixed part sets all consistent features clearly.\n"
            f"For the dynamic part, make brief but clear descriptions of what is changing in each frame.\n"
            f"Provide **exactly {num_frames} descriptions** in the dynamic part, one per line.\n"
            f"Do not include any extra text, explanations, or instructions beyond the frame descriptions.\n"
            f"**Video Context**: '{video_context}'\n\n"
            f"**Number of Frames**: '{num_frames}'\n\n"
            f"**Ensure your response follows the video context and includes exactly {num_frames} frame descriptions.**"
        )
        return [{"role": "user", "content": instructions}]

    def extract_frames(self, response, expected_num_frames):
        # Split response into fixed and dynamic parts
        try:
            response_sections = response.strip().split("Dynamic Part:")
            if len(response_sections) != 2:
                raise ValueError("Response format incorrect: Missing 'Dynamic Part:' section")
            
            fixed_part_section = response_sections[0].replace("Fixed Part:", "").strip()
            if not fixed_part_section:
                raise ValueError("Empty fixed part section")
            
            dynamic_part_section = response_sections[1].strip()
            if not dynamic_part_section:
                raise ValueError("Empty dynamic part section")

            # Extract frame descriptions and filter out empty lines and numbering
            frames = []
            for line in dynamic_part_section.splitlines():
                line = line.strip()
                # Remove frame numbers if they exist (e.g., "1.", "[1]", "Frame 1:")
                line = re.sub(r'^(?:\d+\.|\[\d+\]|Frame \d+:|\d+\))\s*', '', line)
                if line:  # Only add non-empty lines
                    frames.append(line)

            # Allow for up to 10% more frames than expected and truncate if needed
            max_allowed_frames = int(expected_num_frames * 1.1)
            least_allowed_frames = int(expected_num_frames * 0.9)
            if len(frames) == expected_num_frames:
                pass
            elif expected_num_frames < len(frames) < max_allowed_frames:
                frames = frames[:expected_num_frames]
            elif least_allowed_frames < len(frames) < expected_num_frames:
                while len(frames) < expected_num_frames:
                    frames.append(frames[-1])
            else:
                raise ValueError(
                    f"Incorrect number of frames. Expected {expected_num_frames}, but got {len(frames)}."
                )

            # Merge fixed and dynamic parts
            merged_frames = [fixed_part_section + " Action: " + frame for frame in frames]
            return merged_frames

        except Exception as e:
            raise ValueError(f"Frame extraction failed: {str(e)}")
