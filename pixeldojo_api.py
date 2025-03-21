"""
PixelDojo ComfyUI Node
A ComfyUI extension for using PixelDojo's Flux API to generate high-quality images directly within ComfyUI workflows.

Repository: https://github.com/blovett80/ComfyUI-PixelDojo
License: MIT
"""

import os
import json
import requests
from PIL import Image
import io
import folder_paths
import torch
import numpy as np

# PixelDojo ComfyUI Node
class PixelDojoAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["flux-pro", "flux-1.1-pro", "flux-1.1-pro-ultra", "flux-dev-single-lora"], {"default": "flux-pro"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"], {"default": "1:1"}),
                "num_outputs": ("INT", {"default": 1, "min": 1, "max": 4}),
                "api_key_override": ("STRING", {"default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "lora_weights": ("STRING", {"default": ""}),
                "lora_scale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "PixelDojo"

    def get_api_key(self, api_key_override):
        # Check for API key override first
        if api_key_override:
            return api_key_override
            
        # Try to get from environment variable
        env_key = os.environ.get("PIXELDOJO_API_KEY")
        if env_key:
            return env_key
            
        # Try to read from a text file
        key_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pixeldojo_api_key.txt")
        if os.path.exists(key_file_path):
            with open(key_file_path, "r") as f:
                return f.read().strip()
                
        return None

    def generate(self, prompt, model, aspect_ratio, num_outputs=1, seed=None, api_key_override="", lora_weights="", lora_scale=0.7):
        api_key = self.get_api_key(api_key_override)
        if not api_key:
            raise ValueError("PixelDojo API key not found. Please set the PIXELDOJO_API_KEY environment variable, create a pixeldojo_api_key.txt file, or provide the key directly.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "num_outputs": num_outputs,
            "output_format": "png",
            "output_quality": 100,
        }

        # Add optional parameters if provided
        if seed is not None and seed > 0:
            payload["seed"] = seed

        # Handle LoRA when using flux-dev-single-lora model
        if model == "flux-dev-single-lora" and lora_weights:
            payload["lora_weights"] = lora_weights
            payload["lora_scale"] = lora_scale

        def extract_image_url(img_data):
            """Safely extract image URL from various response formats"""
            if isinstance(img_data, str):
                return img_data
            elif isinstance(img_data, dict):
                if 'url' in img_data:
                    return img_data['url']
                # Try to find any key that might contain a URL
                for key, value in img_data.items():
                    if isinstance(value, str) and (value.startswith('http://') or value.startswith('https://')):
                        return value
            # If we can't extract a URL, return the original data for error reporting
            return str(img_data)

        try:
            response = requests.post(
                "https://pixeldojo.ai/api/v1/flux",
                headers=headers,
                json=payload,
                timeout=600  # 10 minutes timeout
            )

            if response.status_code != 200:
                error_message = f"API Error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data and "message" in error_data["error"]:
                        error_message = f"API Error: {error_data['error']['message']}"
                except:
                    pass
                raise ValueError(error_message)

            # Parse response
            data = response.json()
            
            # Debug the response structure if needed
            print(f"API Response: {json.dumps(data, indent=2)}")
            
            if "images" not in data or not data["images"]:
                raise ValueError(f"No images found in API response: {json.dumps(data, indent=2)}")
            
            # Process all returned images
            images = []
            for img_data in data.get("images", []):
                try:
                    img_url = extract_image_url(img_data)
                    
                    # Print the URL for debugging
                    print(f"Downloading image from: {img_url}")
                    
                    # Handle case where img_url might still be a dictionary string representation
                    if img_url.startswith("{") and img_url.endswith("}"):
                        try:
                            # Try to parse it as JSON and extract the URL
                            url_dict = json.loads(img_url.replace("'", "\""))
                            if isinstance(url_dict, dict) and 'url' in url_dict:
                                img_url = url_dict['url']
                                print(f"Extracted URL from dictionary string: {img_url}")
                        except json.JSONDecodeError:
                            # If it can't be parsed as JSON, we'll try to use it as is
                            print(f"Could not parse as JSON: {img_url}")
                    
                    img_response = requests.get(img_url)
                    if img_response.status_code != 200:
                        raise ValueError(f"Failed to download image from {img_url}, status code: {img_response.status_code}")
                    
                    image = Image.open(io.BytesIO(img_response.content))
                    
                    # Convert to RGB if image has alpha channel
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                        
                    # Convert PIL image to torch tensor
                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None,]
                    images.append(image_tensor)
                except Exception as e:
                    raise ValueError(f"Error processing image {img_data}: {str(e)}")
            
            # If we have images, combine them into a batch
            if images:
                return (torch.cat(images, dim=0),)
            else:
                raise ValueError("No images returned from API")
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error: {str(e)}")

# Add the node class to ComfyUI
NODE_CLASS_MAPPINGS = {
    "PixelDojoAPI": PixelDojoAPI
}

# Add descriptions
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelDojoAPI": "PixelDojo Image Generator"
} 