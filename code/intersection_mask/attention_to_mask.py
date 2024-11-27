import cv2
import numpy as np

def normalise(attention_map, exponent=2):
    min_val = np.min(attention_map, axis=(1, 2), keepdims=True)
    max_val = np.max(attention_map, axis=(1, 2), keepdims=True)
    normalized_map = (attention_map - min_val) / (max_val - min_val)
    return np.power(normalized_map, exponent)

def resize_mask(attention_map, target_shape):
    # Use list comprehension for batch processing
    # Ensure that we resize only 2D maps, hence the shape should be [height, width]
    attention_map_resized = np.array([cv2.resize(att, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR) 
                                      for att in attention_map])
    return attention_map_resized

def combine_maps(attention_maps):
    # Combine the maps using np.max along the batch dimension (axis=0)
    combined_map = np.max(attention_maps, axis=0)
    return combined_map


def get_intersection_mask(attention_map_batch, shape, exponent=2):
    # List to hold the resulting masks for each item in the batch
    batch_masks = []

    for attention_map in attention_map_batch:
        # Process each attention map individually
        normalized_maps = normalise(attention_map, exponent=exponent)
        combined_map = combine_maps(normalized_maps)

        # Remove the singleton dimension if it exists
        combined_map = combined_map.squeeze(0)  # This will remove the dimension with size 1 if it's there

        # Resize the combined map to match the spatial dimensions of the latents
        combined_map_resized = resize_mask(combined_map[np.newaxis, :, :], shape[2:])[0]  # shape[2:] is (height, width)
        batch_masks.append(combined_map_resized)

    # Stack the batch of masks into a single array with shape [batch_size, height, width]
    batch_masks = np.stack(batch_masks, axis=0)
    
    # Expand dimensions of the masks to match [batch_size, 1, height, width]
    batch_masks = np.expand_dims(batch_masks, axis=1)  # Shape [batch_size, 1, height, width]
    
    # Ensure that the expanded dimensions match the latents' shape
    if batch_masks.shape[1] != shape[1]:
        # Replicate the mask along the channel dimension
        batch_masks = np.repeat(batch_masks, shape[1], axis=1)  # Shape [batch_size, channels, height, width]
    
    # At this point, batch_masks should have the shape [batch_size, channels, height, width]
    return batch_masks