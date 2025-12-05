#!/usr/bin/env python3
"""
Sample script for text-guided image manipulation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.manipulator import TextGuidedImageManipulator, ManipulationConfig
from utils.sampling import SamplingUtils, SamplingConfig
from utils.visualization import VisualizationUtils
from PIL import Image

def main():
    """Main sampling function."""
    
    # Configuration
    manip_config = ManipulationConfig(
        seed=42,
        num_inference_steps=20,  # Faster for demo
        strength=0.7
    )
    
    sampling_config = SamplingConfig(
        num_samples=4,
        seed=42,
        output_dir="assets/samples"
    )
    
    # Initialize components
    manipulator = TextGuidedImageManipulator(manip_config)
    sampling_utils = SamplingUtils(manipulator, sampling_config)
    viz_utils = VisualizationUtils("assets/visualizations")
    
    # Create a sample image
    sample_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
    
    # Sample prompts
    prompts = [
        "Add a beautiful sunset in the background",
        "Change the sky to be cloudy and dramatic",
        "Add snow falling in the scene",
        "Make the colors more vibrant and saturated"
    ]
    
    # Generate samples
    print("Generating samples...")
    samples_dict = sampling_utils.sample_from_multiple_prompts(sample_image, prompts)
    
    # Save samples
    saved_paths = sampling_utils.save_samples(samples_dict, "demo_sample")
    print(f"Saved {len(saved_paths)} files")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Create comparison grid
    all_samples = []
    all_prompts = []
    for prompt, samples in samples_dict.items():
        all_samples.extend(samples)
        all_prompts.extend([prompt] * len(samples))
    
    grid_img = sampling_utils.create_comparison_grid(sample_image, all_samples, all_prompts)
    grid_img.save("assets/samples/comparison_grid.png")
    
    # Create before/after comparison for first sample
    if all_samples:
        comparison_img = sampling_utils.create_before_after_comparison(
            sample_image, all_samples[0], all_prompts[0]
        )
        comparison_img.save("assets/samples/before_after.png")
    
    print("Sampling completed successfully!")

if __name__ == "__main__":
    main()
