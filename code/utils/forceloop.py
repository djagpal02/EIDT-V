from PIL import Image
import os
import pathlib

def force_gif_loop(input_path):
    """Force a GIF to loop by modifying it in place"""
    # Validate input path
    if not os.path.exists(input_path):
        print(f"Error: File does not exist: {input_path}")
        return False
    if not os.path.isfile(input_path):
        print(f"Error: Path is not a file: {input_path}")
        return False
    
    try:
        with Image.open(input_path) as img:
            # Validate that it's actually a GIF
            if img.format != 'GIF':
                print(f"Error: File is not a GIF: {input_path}")
                return False
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
    # Validate directory path
    if not os.path.exists(directory_path):
        print(f"Error: Directory does not exist: {directory_path}")
        return 0, 0
    if not os.path.isdir(directory_path):
        print(f"Error: Path is not a directory: {directory_path}")
        return 0, 0

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
    folder_path = "_Assets_"
    
    # Validate and convert to absolute path for clarity
    folder_path = os.path.abspath(folder_path)

    print(f"Starting to process GIFs in: {folder_path}")
    processed, failed = process_directory(folder_path)
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed} GIFs")
    print(f"Failed to process: {failed} GIFs")