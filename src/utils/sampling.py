"""
Sampling utilities and visualization tools for text-guided image manipulation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import logging
import json
from dataclasses import dataclass
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for sampling and visualization."""
    
    # Sampling parameters
    num_samples: int = 4
    seed: Optional[int] = None
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.8
    
    # Visualization parameters
    image_size: int = 512
    grid_cols: int = 2
    grid_rows: int = 2
    save_individual: bool = True
    save_grid: bool = True
    
    # Output settings
    output_dir: str = "assets/samples"
    format: str = "png"
    quality: int = 95


class SamplingUtils:
    """
    Utilities for sampling and generating images with text-guided manipulation.
    """
    
    def __init__(self, manipulator, config: SamplingConfig):
        self.manipulator = manipulator
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed if provided
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
    
    def sample_from_prompt(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: str = "",
        num_samples: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate multiple samples from a single prompt.
        
        Args:
            image: Input image
            prompt: Text prompt for manipulation
            negative_prompt: Negative prompt
            num_samples: Number of samples to generate
            
        Returns:
            List of generated images
        """
        num_samples = num_samples or self.config.num_samples
        samples = []
        
        logger.info(f"Generating {num_samples} samples for prompt: '{prompt}'")
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Set different seed for each sample
            if self.config.seed is not None:
                torch.manual_seed(self.config.seed + i)
            
            # Generate sample
            sample = self.manipulator.manipulate_image(
                image=image,
                text_prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                strength=self.config.strength
            )
            
            samples.append(sample)
        
        return samples
    
    def sample_from_multiple_prompts(
        self,
        image: Union[str, Path, Image.Image],
        prompts: List[str],
        negative_prompts: Optional[List[str]] = None
    ) -> Dict[str, List[Image.Image]]:
        """
        Generate samples for multiple prompts.
        
        Args:
            image: Input image
            prompts: List of text prompts
            negative_prompts: List of negative prompts
            
        Returns:
            Dictionary mapping prompts to generated images
        """
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)
        
        results = {}
        
        for prompt, neg_prompt in zip(prompts, negative_prompts):
            logger.info(f"Processing prompt: '{prompt}'")
            samples = self.sample_from_prompt(image, prompt, neg_prompt)
            results[prompt] = samples
        
        return results
    
    def create_comparison_grid(
        self,
        original_image: Image.Image,
        manipulated_images: List[Image.Image],
        prompts: List[str],
        title: str = "Text-Guided Image Manipulation"
    ) -> Image.Image:
        """
        Create a comparison grid showing original and manipulated images.
        
        Args:
            original_image: Original input image
            manipulated_images: List of manipulated images
            prompts: List of prompts used
            title: Title for the grid
            
        Returns:
            PIL Image containing the comparison grid
        """
        num_images = len(manipulated_images)
        cols = self.config.grid_cols
        rows = (num_images + cols - 1) // cols
        
        # Calculate grid size
        img_size = self.config.image_size
        grid_width = cols * img_size
        grid_height = rows * img_size
        
        # Add space for titles
        title_height = 50
        total_height = grid_height + title_height
        
        # Create grid image
        grid_img = Image.new('RGB', (grid_width, total_height), color=(255, 255, 255))
        
        # Draw title
        draw = ImageDraw.Draw(grid_img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Center title
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (grid_width - title_width) // 2
        draw.text((title_x, 10), title, fill=(0, 0, 0), font=font)
        
        # Place images in grid
        for i, (img, prompt) in enumerate(zip(manipulated_images, prompts)):
            row = i // cols
            col = i % cols
            
            x = col * img_size
            y = row * img_size + title_height
            
            # Resize image if needed
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size))
            
            # Paste image
            grid_img.paste(img, (x, y))
            
            # Add prompt text
            prompt_text = prompt[:30] + "..." if len(prompt) > 30 else prompt
            draw.text((x + 5, y + img_size - 25), prompt_text, fill=(255, 255, 255), font=font)
        
        return grid_img
    
    def create_before_after_comparison(
        self,
        original_image: Image.Image,
        manipulated_image: Image.Image,
        prompt: str,
        similarity_score: Optional[float] = None
    ) -> Image.Image:
        """
        Create a before/after comparison image.
        
        Args:
            original_image: Original image
            manipulated_image: Manipulated image
            prompt: Text prompt used
            similarity_score: CLIP similarity score
            
        Returns:
            PIL Image with before/after comparison
        """
        img_size = self.config.image_size
        
        # Resize images
        original_img = original_image.resize((img_size, img_size))
        manipulated_img = manipulated_image.resize((img_size, img_size))
        
        # Create comparison image
        comparison_width = img_size * 2 + 20  # 20px gap
        comparison_height = img_size + 100    # Space for labels
        
        comparison_img = Image.new('RGB', (comparison_width, comparison_height), color=(255, 255, 255))
        
        # Paste images
        comparison_img.paste(original_img, (0, 50))
        comparison_img.paste(manipulated_img, (img_size + 20, 50))
        
        # Add labels
        draw = ImageDraw.Draw(comparison_img)
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw labels
        draw.text((10, 10), "Original", fill=(0, 0, 0), font=font_large)
        draw.text((img_size + 30, 10), "Manipulated", fill=(0, 0, 0), font=font_large)
        
        # Draw prompt
        prompt_text = f"Prompt: {prompt}"
        draw.text((10, comparison_height - 30), prompt_text, fill=(0, 0, 0), font=font_small)
        
        # Draw similarity score if provided
        if similarity_score is not None:
            score_text = f"CLIP Score: {similarity_score:.3f}"
            draw.text((10, comparison_height - 15), score_text, fill=(0, 0, 0), font=font_small)
        
        return comparison_img
    
    def save_samples(
        self,
        samples: Union[List[Image.Image], Dict[str, List[Image.Image]]],
        filename_prefix: str = "sample",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """
        Save generated samples to disk.
        
        Args:
            samples: Generated samples (list or dict)
            filename_prefix: Prefix for saved files
            metadata: Additional metadata to save
            
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        if isinstance(samples, dict):
            # Multiple prompts
            for prompt, prompt_samples in samples.items():
                prompt_safe = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
                prompt_safe = prompt_safe.replace(' ', '_')[:50]  # Limit length
                
                for i, sample in enumerate(prompt_samples):
                    filename = f"{filename_prefix}_{prompt_safe}_{i:03d}.{self.config.format}"
                    filepath = self.output_dir / filename
                    sample.save(filepath, quality=self.config.quality)
                    saved_paths.append(filepath)
        else:
            # Single prompt
            for i, sample in enumerate(samples):
                filename = f"{filename_prefix}_{i:03d}.{self.config.format}"
                filepath = self.output_dir / filename
                sample.save(filepath, quality=self.config.quality)
                saved_paths.append(filepath)
        
        # Save metadata if provided
        if metadata:
            metadata_path = self.output_dir / f"{filename_prefix}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_paths.append(metadata_path)
        
        logger.info(f"Saved {len(saved_paths)} files to {self.output_dir}")
        return saved_paths


class VisualizationUtils:
    """
    Utilities for creating visualizations and analysis plots.
    """
    
    def __init__(self, output_dir: str = "assets/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_metrics_comparison(
        self,
        metrics_data: Dict[str, Dict[str, float]],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create a comparison plot of different metrics.
        
        Args:
            metrics_data: Dictionary with model names and their metrics
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Text-Guided Image Manipulation Metrics Comparison', fontsize=16)
        
        # Extract data
        models = list(metrics_data.keys())
        metrics = ['clip_score', 'lpips_distance', 'fid_score', 'mean_color_change']
        metric_labels = ['CLIP Score', 'LPIPS Distance', 'FID Score', 'Color Change']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i // 2, i % 2]
            
            values = [metrics_data[model].get(metric, 0) for model in models]
            
            bars = ax.bar(models, values, alpha=0.7)
            ax.set_title(label)
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison plot saved to: {save_path}")
        
        plt.show()
    
    def plot_similarity_distribution(
        self,
        similarity_scores: List[float],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot distribution of similarity scores.
        
        Args:
            similarity_scores: List of similarity scores
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.hist(similarity_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(similarity_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(similarity_scores):.3f}')
        plt.axvline(np.median(similarity_scores), color='green', linestyle='--', 
                   label=f'Median: {np.median(similarity_scores):.3f}')
        
        plt.xlabel('CLIP Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of CLIP Similarity Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Similarity distribution plot saved to: {save_path}")
        
        plt.show()
    
    def create_manipulation_analysis(
        self,
        original_images: List[Image.Image],
        manipulated_images: List[Image.Image],
        prompts: List[str],
        metrics: Dict[str, float],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create a comprehensive analysis visualization.
        
        Args:
            original_images: Original images
            manipulated_images: Manipulated images
            prompts: Text prompts
            metrics: Evaluation metrics
            save_path: Path to save the analysis
        """
        num_samples = min(len(original_images), 6)  # Show max 6 samples
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, num_samples, height_ratios=[1, 1, 0.5])
        
        # Plot images
        for i in range(num_samples):
            # Original image
            ax_orig = fig.add_subplot(gs[0, i])
            ax_orig.imshow(original_images[i])
            ax_orig.set_title(f'Original {i+1}', fontsize=12)
            ax_orig.axis('off')
            
            # Manipulated image
            ax_manip = fig.add_subplot(gs[1, i])
            ax_manip.imshow(manipulated_images[i])
            ax_manip.set_title(f'Manipulated {i+1}', fontsize=12)
            ax_manip.axis('off')
            
            # Prompt text
            ax_text = fig.add_subplot(gs[2, i])
            ax_text.text(0.5, 0.5, prompts[i][:50] + '...' if len(prompts[i]) > 50 else prompts[i],
                        ha='center', va='center', fontsize=10, wrap=True)
            ax_text.axis('off')
        
        # Add metrics text
        metrics_text = f"""
        Evaluation Metrics:
        CLIP Score: {metrics.get('clip_score', 'N/A'):.3f}
        LPIPS Distance: {metrics.get('lpips_distance', 'N/A'):.3f}
        FID Score: {metrics.get('fid_score', 'N/A'):.3f}
        Color Change: {metrics.get('mean_color_change', 'N/A'):.3f}
        """
        
        fig.text(0.02, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle('Text-Guided Image Manipulation Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Manipulation analysis saved to: {save_path}")
        
        plt.show()
    
    def create_prompt_effectiveness_plot(
        self,
        prompts: List[str],
        clip_scores: List[float],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create a plot showing effectiveness of different prompts.
        
        Args:
            prompts: List of prompts
            clip_scores: Corresponding CLIP scores
            save_path: Path to save the plot
        """
        # Sort by CLIP score
        sorted_data = sorted(zip(prompts, clip_scores), key=lambda x: x[1], reverse=True)
        sorted_prompts, sorted_scores = zip(*sorted_data)
        
        plt.figure(figsize=(12, 8))
        
        # Truncate long prompts for display
        display_prompts = [p[:30] + '...' if len(p) > 30 else p for p in sorted_prompts]
        
        bars = plt.barh(range(len(display_prompts)), sorted_scores, alpha=0.7)
        plt.yticks(range(len(display_prompts)), display_prompts)
        plt.xlabel('CLIP Score')
        plt.title('Prompt Effectiveness (CLIP Scores)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prompt effectiveness plot saved to: {save_path}")
        
        plt.show()


def create_sample_script():
    """Create a sample script demonstrating the sampling utilities."""
    
    script_content = '''#!/usr/bin/env python3
"""
Sample script for text-guided image manipulation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

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
'''
    
    return script_content


if __name__ == "__main__":
    # Create sample script
    script_content = create_sample_script()
    
    with open("scripts/sample.py", "w") as f:
        f.write(script_content)
    
    logger.info("Sample script created at scripts/sample.py")
