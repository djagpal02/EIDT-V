from PIL import Image
import os
import pathlib

def force_gif_loop(input_path):
    """Force a GIF to loop by modifying it in place"""
    try:
        with Image.open(input_path) as img:
            # Get all frames from the GIF
            frames = []
            try:
                while True:
                    frames.append(img.copy())
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            # Save back to the same file with loop enabled
            frames[0].save(
                input_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                loop=0  # 0 means infinite loop
            )
            return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_directory(directory_path):
    """Recursively process all GIFs in the given directory and its subdirectories"""
    # Convert the path to a Path object for easier handling
    directory = pathlib.Path(directory_path)
    
    # Counter for processed files
    processed = 0
    failed = 0
    
    # Walk through all files in directory and subdirectories
    for filepath in directory.rglob("*.gif"):
        print(f"Processing: {filepath}")
        if force_gif_loop(filepath):
            processed += 1
        else:
            failed += 1
    
    return processed, failed

if __name__ == "__main__":
    # Example usage
    folder_path = "_Assets_"  # Replace with your folder path
    
    print(f"Starting to process GIFs in: {folder_path}")
    processed, failed = process_directory(folder_path)
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed} GIFs")
    print(f"Failed to process: {failed} GIFs")