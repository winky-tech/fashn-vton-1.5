# Use a base image with CUDA 11.8 or higher for the RTX 30/40 series GPUs
FROM runpod/base:0.4.0-cuda11.8.0

# Install system dependencies needed for git and image processing
RUN apt-get update && apt-get install -y git python3-pip libgl1-mesa-glx libglib2.0-0

# Set the working directory
WORKDIR /

# Copy all files from your repo (including rp_handler.py) into the container
COPY . .

# Install the fashn_vton package and the runpod library
RUN python3 -m pip install --no-cache-dir . runpod

# Pre-download the weights so the worker doesn't download them on every start
# This saves about 2-3 minutes of "Cold Start" time for your users
RUN python3 scripts/download_weights.py --weights-dir ./weights || (echo "Download failed. Dumping pip and error info:" && python3 -m pip list && exit 1)

# Run the handler script when the container starts
CMD [ "python3", "-u", "/rp_handler.py" ]