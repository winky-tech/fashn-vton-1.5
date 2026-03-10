import base64
import io
import urllib.request

import runpod
import torch
from PIL import Image

from fashn_vton import TryOnPipeline

# Initialize the pipeline OUTSIDE the handler so it remains loaded in memory
# between requests. This drastically reduces execution time.
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = TryOnPipeline(weights_dir="./weights", device=device)

def load_image(image_data):
    """Loads an image from either a base64 string or a URL."""
    if image_data.startswith("http://") or image_data.startswith("https://"):
        req = urllib.request.Request(image_data, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    else:
        # Assume base64
        # Strip potential data URI prefix e.g., "data:image/jpeg;base64,"
        if "," in image_data:
            image_data = image_data.split(",")[1]
        decoded_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(decoded_bytes)).convert("RGB")

def handler(job):
    """
    This is the handler function that is called for every RunPod serverless request.
    """
    job_input = job['input']

    # Required parameters
    if 'person_image' not in job_input:
        return {"error": "Missing required parameter 'person_image'."}
    if 'garment_image' not in job_input:
        return {"error": "Missing required parameter 'garment_image'."}
    if 'category' not in job_input:
        return {"error": "Missing required parameter 'category'."}

    try:
        person_image = load_image(job_input['person_image'])
        garment_image = load_image(job_input['garment_image'])
    except Exception as e:
        return {"error": f"Failed to load images: {str(e)}"}

    category = job_input['category']
    if category not in ["tops", "bottoms", "one-pieces"]:
        return {"error": "Invalid 'category'. Must be one of ['tops', 'bottoms', 'one-pieces']."}

    # Optional parameters with their defaults
    garment_photo_type = job_input.get("garment_photo_type", "model")
    num_samples = job_input.get("num_samples", 1)
    num_timesteps = job_input.get("num_timesteps", 30)
    guidance_scale = job_input.get("guidance_scale", 1.5)
    seed = job_input.get("seed", 42)
    segmentation_free = job_input.get("segmentation_free", True)

    try:
        # Run the FASHN VTON pipeline
        result = pipeline(
            person_image=person_image,
            garment_image=garment_image,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=num_samples,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
            segmentation_free=segmentation_free,
        )

        # Encode resulting images to Base64 strings to send back in HTTP response
        output_images = []
        for img in result.images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            output_images.append(img_str)

        return {
            "status": "success",
            "images": output_images
        }

    except Exception as e:
        return {"error": f"Inference pipeline failed: {str(e)}"}

# Start the RunPod Serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
