# headers: text_prompt, video_url_1, video_url_2, video_url_3, video_url_4
# URL:
# https://video-evaluation-study.s3.eu-north-1.amazonaws.com/UserStudy/{model name}/{file name}
# Model names: [DirecT2V, FreeBloom, T2VZero, EIDTV]
import json
import re
import csv
import random

def prompt_to_name(prompt):
    # Convert prompt to lowercase and replace spaces with underscores
    folder_name = re.sub(r'[^\w\s-]', '', prompt.lower())
    folder_name = re.sub(r'[-\s]+', '_', folder_name)
    return folder_name

def file_name(prompt):
    return prompt_to_name(prompt)[:30] + ".gif"


def get_prompts():
    with open('data/tests/prev_work_comparison.json', 'r') as file:
        data_dict = json.load(file)

    return list(data_dict['prompt'].keys())

def generate_csv(prompts):
    model_names = ["DirecT2V", "FreeBloom", "T2VZero", "EIDTV"]
    csv_data = []
    mapping_data = []

    for prompt in prompts:
        random.shuffle(model_names)
        video_urls = [
            f"https://video-evaluation-study.s3.eu-north-1.amazonaws.com/UserStudy/{model}/{file_name(prompt)}"
            for model in model_names
        ]
        csv_data.append([prompt] + video_urls)
        mapping_data.append([prompt] + model_names)

    with open('UserStudy/Web/data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text_prompt', 'video_url_1', 'video_url_2', 'video_url_3', 'video_url_4'])
        writer.writerows(csv_data)

    with open('UserStudy/Web/mapping.csv', 'w', newline='') as mappingfile:
        writer = csv.writer(mappingfile)
        writer.writerow(['text_prompt', 'model_url_1', 'model_url_2', 'model_url_3', 'model_url_4'])
        writer.writerows(mapping_data)

def main():
    prompts = get_prompts()
    generate_csv(prompts)

if __name__ == "__main__":
    main()