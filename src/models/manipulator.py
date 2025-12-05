"""
Text-guided Image Manipulation Project

A modern implementation of text-guided image manipulation using diffusion models.
This project demonstrates how to modify existing images based on textual descriptions
using state-of-the-art diffusion models and CLIP guidance.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import random
from dataclasses import dataclass
from omegaconf import OmegaConf
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ManipulationConfig:
    """Configuration for text-guided image manipulation."""
    
    # Model settings
    model_name: str = "runwayml/stable-diffusion-v1-5"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    
    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.8  # How much to modify the original image
    
    # Device settings
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Image settings
    image_size: int = 512
    batch_size: int = 1
    
    # Random seed
    seed: Optional[int] = None


class DeviceManager:
    """Manages device selection and deterministic seeding."""
    
    def __init__(self, device: str = "auto", seed: Optional[int] = None):
        self.device = self._get_device(device)
        self.set_seed(seed)
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Automatically select the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def set_seed(self, seed: Optional[int] = None):
        """Set deterministic seeds for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            logger.info(f"Set random seed to: {seed}")


class TextGuidedImageManipulator:
    """
    Main class for text-guided image manipulation using diffusion models.
    
    This class combines CLIP for text-image alignment with diffusion models
    for high-quality image generation and manipulation.
    """
    
    def __init__(self, config: ManipulationConfig):
        self.config = config
        self.device_manager = DeviceManager(config.device, config.seed)
        self.device = self.device_manager.device
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load the diffusion model and CLIP model."""
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info("Loading Stable Diffusion model...")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to(self.device)
            
            logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained(self.config.clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
            self.clip_model = self.clip_model.to(self.device)
            
            logger.info("Models loaded successfully!")
            
        except ImportError as e:
            logger.error(f"Failed to import required models: {e}")
            raise ImportError("Please install required dependencies: pip install diffusers transformers")
    
    def manipulate_image(
        self,
        image: Union[str, Path, Image.Image],
        text_prompt: str,
        negative_prompt: str = "",
        **kwargs
    ) -> Image.Image:
        """
        Manipulate an image based on a text prompt.
        
        Args:
            image: Input image (path, PIL Image, or tensor)
            text_prompt: Text description of desired changes
            negative_prompt: Text description of what to avoid
            **kwargs: Additional generation parameters
            
        Returns:
            Manipulated PIL Image
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            # Convert tensor to PIL
            import torchvision.transforms as transforms
            if image.dim() == 4:
                image = image.squeeze(0)
            image = transforms.ToPILImage()(image)
        
        # Resize image if needed
        if image.size != (self.config.image_size, self.config.image_size):
            image = image.resize((self.config.image_size, self.config.image_size))
        
        # Generate parameters
        generation_params = {
            "prompt": text_prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "strength": kwargs.get("strength", self.config.strength),
            "num_inference_steps": kwargs.get("num_inference_steps", self.config.num_inference_steps),
            "guidance_scale": kwargs.get("guidance_scale", self.config.guidance_scale),
        }
        
        logger.info(f"Manipulating image with prompt: '{text_prompt}'")
        
        # Generate manipulated image
        with torch.autocast(self.device.type if self.device.type == "cuda" else "cpu"):
            result = self.pipe(**generation_params)
        
        return result.images[0]
    
    def evaluate_similarity(
        self,
        image: Union[str, Path, Image.Image],
        text_prompt: str
    ) -> Dict[str, float]:
        """
        Evaluate the similarity between an image and text using CLIP.
        
        Args:
            image: Input image
            text_prompt: Text description
            
        Returns:
            Dictionary with similarity scores
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Process with CLIP
        inputs = self.clip_processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get CLIP embeddings
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
        
        return {
            "image_text_similarity": logits_per_image.item(),
            "text_image_similarity": logits_per_text.item(),
            "average_similarity": (logits_per_image.item() + logits_per_text.item()) / 2
        }


def create_sample_dataset(output_dir: Path, num_samples: int = 10) -> None:
    """
    Create a sample dataset for testing the manipulation pipeline.
    
    Args:
        output_dir: Directory to save sample images
        num_samples: Number of sample images to generate
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample prompts for manipulation
    sample_prompts = [
        "Add a beautiful sunset in the background",
        "Change the sky to be cloudy and dramatic",
        "Add snow falling in the scene",
        "Make the colors more vibrant and saturated",
        "Add a rainbow in the sky",
        "Change the lighting to golden hour",
        "Add autumn leaves falling",
        "Make it look like a painting",
        "Add fog and mist",
        "Change the season to winter"
    ]
    
    # Create simple test images (colored rectangles for demo)
    for i in range(num_samples):
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color=(100 + i*20, 150 + i*10, 200 - i*15))
        
        # Add some simple shapes
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 400, 400], fill=(255, 255, 255, 100))
        draw.ellipse([200, 200, 300, 300], fill=(255, 0, 0, 100))
        
        # Save image
        img_path = output_dir / f"sample_{i:03d}.png"
        img.save(img_path)
        
        # Save corresponding prompt
        prompt_path = output_dir / f"sample_{i:03d}_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(sample_prompts[i % len(sample_prompts)])


def main():
    """Main function to demonstrate text-guided image manipulation."""
    # Create configuration
    config = ManipulationConfig(
        seed=42,
        num_inference_steps=20,  # Faster for demo
        strength=0.7
    )
    
    # Initialize manipulator
    manipulator = TextGuidedImageManipulator(config)
    
    # Create sample dataset
    sample_dir = Path("assets/samples")
    create_sample_dataset(sample_dir, num_samples=3)
    
    # Process sample images
    for img_path in sample_dir.glob("sample_*.png"):
        prompt_path = img_path.with_suffix("").with_suffix("_prompt.txt")
        
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
            
            logger.info(f"Processing {img_path.name} with prompt: '{prompt}'")
            
            # Manipulate image
            manipulated_img = manipulator.manipulate_image(img_path, prompt)
            
            # Evaluate similarity
            similarity_scores = manipulator.evaluate_similarity(manipulated_img, prompt)
            logger.info(f"Similarity scores: {similarity_scores}")
            
            # Save result
            output_path = sample_dir / f"manipulated_{img_path.stem}.png"
            manipulated_img.save(output_path)
            
            logger.info(f"Saved manipulated image to: {output_path}")


if __name__ == "__main__":
    main()
