# EIDT-V

**EIDT-V**: *Exploiting Intersections in Diffusion Trajectories for Model-Agnostic, Zero-Shot, Training-Free Text-to-Video Generation*

This repository provides a implementation of **EIDT-V**, leveraging diffusion models and advanced methodologies for zero-shot text-to-video generation.

---
## Usage

### Prerequisites

- Docker installed on your machine.
- Access to a compatible GPU (NVIDIA recommended).
- HuggingFace access token for Llama 3 (to be put in `Code/Llama/login.py`)

### Running the Program

To build the Docker container and run the program, use:

```bash
./run.sh <gpu_id>
```

- Replace `<gpu_id>` with the GPU ID (e.g., `0` for the default GPU).
- **Note:** Only a single GPU is required for this low-cost model.
- The first run may take longer as it installs all necessary dependencies.

---

## Customizing Prompts and Configuration

Running the script opens a menu where you can easily enter text prompts that will be used to generate gifs. This feature uses our best configuration with the SD3 Medium diffusion model for best results.

---

## Project Overview

### Core Functionality

- **Diffusion Models**: The methodology is integrated via callbacks in diffusion models from the `diffusers` library to apply attention-based logic for frame-to-frame transformations through only the latent space.
- **Intersection Masks**: The logic for mask creation resides in the `intersection_mask` subfolder. Masks are generated through attention between previous frame image and next frame text.
- **LLM Support**: The `llama` subfolder contains logic for two Llama-based models:
  - Framewise generation.
  - Text difference detection.  
  Both models are implemented using Llama 3 via the `transformers` library.

---

## Repository Structure

- **`code/`**: Main code folder, containing:
  - `callbacks/`: Implements callbacks used for integration with the `diffusers` library.
  - `intersection_mask/`: Logic for attention-based intersection masks.
- **`llama/`**: Logic for Llama-based models for framewise generation and text difference detection.
- **`tests/`**: Contains various tests used for building and validating the model. Select tests from the menu in `main.py`.
- **`utils/`**: Utility scripts for auxiliary tasks.
- **`pipeline.py`**: Defines the main pipeline.
- **`details.py`**: Contains a class for passing configuration settings across different components of the model.
- **`data/`**: Includes:
  - `example.json`: Examples provided to the Llama model for in-context learning.
  - Other JSON files for various tests.
- **`results/`**: Contains Jupyter notebooks used for analysis and generating figures.
- **`user_study/`**: Resources for the user study, including:
  - HTML for the web application.
  - Methods for randomizing the order of presented videos.
  - Response data and analysis scripts.

---