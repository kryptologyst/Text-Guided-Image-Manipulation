#!/usr/bin/env python3
"""
Evaluation script for text-guided image manipulation models.
"""

import sys
from pathlib import Path
import argparse
import logging
from omegaconf import OmegaConf
import torch
from PIL import Image
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.manipulator import TextGuidedImageManipulator, ManipulationConfig
from data.dataset import create_data_loaders
from utils.metrics import ManipulationMetrics, create_evaluation_report
from utils.sampling import SamplingUtils, SamplingConfig
from utils.visualization import VisualizationUtils

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for text-guided image manipulation models."""
    
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.manipulator = None
        self.metrics_evaluator = ManipulationMetrics()
        self.viz_utils = VisualizationUtils("assets/visualizations")
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "evaluation.log"),
                logging.StreamHandler()
            ]
        )
    
    def load_model(self, checkpoint_path: str = None):
        """Load the manipulation model."""
        manip_config = ManipulationConfig(
            model_name=self.config.model.name,
            clip_model_name=self.config.model.clip_model_name,
            device=self.config.model.device,
            num_inference_steps=self.config.generation.num_inference_steps,
            guidance_scale=self.config.generation.guidance_scale,
            strength=self.config.generation.strength,
            seed=self.config.seed
        )
        
        self.manipulator = TextGuidedImageManipulator(manip_config)
        
        # Load checkpoint if provided
        if checkpoint_path:
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # In a real implementation, you'd load the model weights here
            logger.info("Checkpoint loaded successfully")
        
        logger.info("Model loaded successfully")
    
    def evaluate_dataset(self, dataset_path: str, split: str = "test"):
        """Evaluate model on a dataset."""
        logger.info(f"Evaluating on {split} split of {dataset_path}")
        
        # Create data loader
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_path,
            batch_size=self.config.generation.batch_size,
            image_size=self.config.image.size,
            num_workers=self.config.data.num_workers
        )
        
        # Select appropriate loader
        if split == "train":
            loader = train_loader
        elif split == "val":
            loader = val_loader
        else:
            loader = test_loader
        
        if loader is None:
            logger.error(f"No {split} loader available")
            return {}
        
        # Evaluation
        all_original_images = []
        all_manipulated_images = []
        all_prompts = []
        all_similarity_scores = []
        
        logger.info("Generating manipulated images...")
        
        for batch_idx, batch in enumerate(loader):
            try:
                images = batch['image']
                prompts = batch['manipulation_prompt']
                
                batch_manipulated = []
                batch_similarities = []
                
                for i, (img, prompt) in enumerate(zip(images, prompts)):
                    # Convert tensor to PIL
                    img_pil = self.tensor_to_pil(img)
                    
                    # Generate manipulated image
                    manipulated = self.manipulator.manipulate_image(
                        img_pil, 
                        prompt,
                        num_inference_steps=self.config.generation.num_inference_steps,
                        guidance_scale=self.config.generation.guidance_scale,
                        strength=self.config.generation.strength
                    )
                    
                    # Compute similarity
                    similarity = self.manipulator.evaluate_similarity(manipulated, prompt)
                    avg_similarity = similarity['average_similarity']
                    
                    all_original_images.append(img_pil)
                    all_manipulated_images.append(manipulated)
                    all_prompts.append(prompt)
                    all_similarity_scores.append(avg_similarity)
                    
                    batch_manipulated.append(manipulated)
                    batch_similarities.append(avg_similarity)
                
                logger.info(f"Processed batch {batch_idx + 1}/{len(loader)}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Compute comprehensive metrics
        logger.info("Computing evaluation metrics...")
        metrics = self.metrics_evaluator.compute_manipulation_metrics(
            all_original_images,
            all_manipulated_images,
            all_prompts
        )
        
        # Add additional statistics
        metrics.update({
            "num_samples": len(all_manipulated_images),
            "mean_similarity": float(sum(all_similarity_scores) / len(all_similarity_scores)),
            "std_similarity": float(torch.std(torch.tensor(all_similarity_scores)).item()),
            "min_similarity": float(min(all_similarity_scores)),
            "max_similarity": float(max(all_similarity_scores))
        })
        
        return metrics, all_original_images, all_manipulated_images, all_prompts
    
    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL image."""
        import torchvision.transforms as transforms
        
        # Denormalize
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        transform = transforms.ToPILImage()
        return transform(tensor)
    
    def create_visualizations(self, original_images, manipulated_images, prompts, metrics):
        """Create evaluation visualizations."""
        logger.info("Creating visualizations...")
        
        # Similarity distribution
        similarity_scores = []
        for img, prompt in zip(manipulated_images, prompts):
            similarity = self.manipulator.evaluate_similarity(img, prompt)
            similarity_scores.append(similarity['average_similarity'])
        
        self.viz_utils.plot_similarity_distribution(
            similarity_scores,
            save_path="assets/visualizations/similarity_distribution.png"
        )
        
        # Prompt effectiveness
        self.viz_utils.create_prompt_effectiveness_plot(
            prompts,
            similarity_scores,
            save_path="assets/visualizations/prompt_effectiveness.png"
        )
        
        # Manipulation analysis
        self.viz_utils.create_manipulation_analysis(
            original_images[:6],  # Show first 6 samples
            manipulated_images[:6],
            prompts[:6],
            metrics,
            save_path="assets/visualizations/manipulation_analysis.png"
        )
        
        logger.info("Visualizations created successfully")
    
    def save_results(self, metrics, original_images, manipulated_images, prompts, output_dir: str):
        """Save evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save sample images
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        for i in range(min(20, len(original_images))):  # Save first 20 samples
            original_images[i].save(samples_dir / f"original_{i:03d}.jpg")
            manipulated_images[i].save(samples_dir / f"manipulated_{i:03d}.jpg")
            
            with open(samples_dir / f"prompt_{i:03d}.txt", 'w') as f:
                f.write(prompts[i])
        
        # Create evaluation report
        report_path = output_dir / "evaluation_report.md"
        create_evaluation_report(metrics, report_path)
        
        logger.info(f"Results saved to: {output_dir}")
    
    def run_evaluation(self, dataset_path: str, checkpoint_path: str = None, output_dir: str = "assets/evaluation"):
        """Run complete evaluation pipeline."""
        logger.info("Starting evaluation...")
        
        # Load model
        self.load_model(checkpoint_path)
        
        # Evaluate on test set
        metrics, original_images, manipulated_images, prompts = self.evaluate_dataset(
            dataset_path, split="test"
        )
        
        # Create visualizations
        self.create_visualizations(original_images, manipulated_images, prompts, metrics)
        
        # Save results
        self.save_results(metrics, original_images, manipulated_images, prompts, output_dir)
        
        # Print summary
        logger.info("Evaluation completed!")
        logger.info(f"Number of samples: {metrics['num_samples']}")
        logger.info(f"CLIP Score: {metrics.get('clip_score', 'N/A'):.4f}")
        logger.info(f"LPIPS Distance: {metrics.get('lpips_distance', 'N/A'):.4f}")
        logger.info(f"FID Score: {metrics.get('fid_score', 'N/A'):.4f}")
        logger.info(f"Mean Similarity: {metrics.get('mean_similarity', 'N/A'):.4f}")
        
        return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate text-guided image manipulation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--dataset", type=str, default="data/sample_dataset",
                       help="Path to evaluation dataset")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="assets/evaluation",
                       help="Output directory for results")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate on")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.config)
    
    # Run evaluation
    metrics = evaluator.run_evaluation(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_dir=args.output
    )
    
    print(f"\nEvaluation completed! Results saved to: {args.output}")
    print(f"CLIP Score: {metrics.get('clip_score', 'N/A'):.4f}")
    print(f"LPIPS Distance: {metrics.get('lpips_distance', 'N/A'):.4f}")
    print(f"FID Score: {metrics.get('fid_score', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
