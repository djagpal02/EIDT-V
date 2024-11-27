import logging
import datetime

from pipeline import Pipeline
from utils.load import load_data
from Tests.text import run_text_test
from Tests.image import run_image_test
from Tests.gif import run_gif_test
from Tests.difference import run_difference_test

# Configure logging
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.DEBUG, filename=f'../output/pipeline_debug_{current_time}.txt', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def _setup_pipeline():
    """Helper function to setup common pipeline parameters"""
    # Load parameters from sd3.json
    params = load_data('../data/tests/sd3.json')
    
    # Extract parameters, excluding the prompts dictionary
    generation_params = {k: v for k, v in params.items() if k != 'prompt'}
    
    pipe = Pipeline('cuda', model='SD3')
    return pipe, generation_params

def run_single_prompt():
    pipe, generation_params = _setup_pipeline()
    prompt = input("Enter a prompt: ")
    save_name = f"../output/{prompt.replace(' ', '_')}.gif"
    
    pipe(prompt, 
         display=True, 
         save_name=save_name,
         save_intermediate=True,
         **generation_params)

    pipe.log_run_history_to_csv()

def run_examples(path='../data/example.json'):
    pipe, generation_params = _setup_pipeline()
    data = load_data(path)
    
    for key, value in data.items():
        updated_frames = [f"{value['fixed part']} Action:{frame}" for frame in value['dynamic part']]
        save_name = f"../output/examples/{key.replace(' ', '_')}.gif"
        pipe(list_of_prompts=updated_frames, 
             display=True, 
             save_name=save_name,
             **generation_params)

    pipe.log_run_history_to_csv()

def menu():
    options = {
        "1": ("Run single prompt", run_single_prompt),
        "2": ("Run examples", run_examples),
        "3": ("Run text test", lambda: run_text_test()),
        "4": ("Run image test", lambda: run_image_test()),
        "5": ("Run gif test", run_gif_test),
        "6": ("Run difference test", run_difference_test),
        "Q": ("Quit", None)
    }

    while True:
        print("\n=== Image Generation Pipeline ===")
        for key, (description, _) in options.items():
            print(f"{key}. {description}")
        
        choice = input("\nEnter your choice: ").strip().upper()
        
        if choice == 'Q':
            print("Goodbye!")
            break
        elif choice in options:
            options[choice][1]()
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    menu()
