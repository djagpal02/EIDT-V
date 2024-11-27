FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y gcc

# Working directory inside the container
WORKDIR /app

# Upgrade pip (optional but recommended)
RUN pip install --upgrade pip

# Install dependencies, prioritizing HF_HOME environment variable
ENV HF_HOME="/app/cache"
ENV CC=/usr/bin/gcc

# Copy requirements and install in one step (more efficient)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Files
COPY code /app/code

# Working directory to code folder
WORKDIR /app/code

# Run the main script
CMD ["python", "main.py"]
