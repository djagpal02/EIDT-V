import torch


def get_attention_map(processor, model, original_image, prompts):
    device = model.device

    # Process the image and prompts
    inputs = processor(text=prompts, images=[original_image] * len(prompts), padding=True, truncation=True, return_tensors="pt").to(device)
            
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    # Get the attention map
    attention_map = preds.detach().cpu().numpy()

    return attention_map