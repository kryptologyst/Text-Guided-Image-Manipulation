#!/usr/bin/env python3
"""
Training script for text-guided image manipulation models.
"""

import sys
from pathlib import Path
import argparse
import logging
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.manipulator import TextGuidedImageManipulator, ManipulationConfig
from data.dataset import create_data_loaders, create_sample_dataset_structure
from utils.metrics import ManipulationMetrics, create_evaluation_report
from utils.sampling import SamplingUtils, SamplingConfig

logger = logging.getLogger(__name__)


class ManipulationTrainer:
    """Trainer for text-guided image manipulation models."""
    
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.manipulator = None
        self.metrics_evaluator = ManipulationMetrics()
        self.sampling_utils = None
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        
        # Setup logging
        self.setup_logging()
        
        # Setup wandb
        if self.config.logging.get("wandb_project"):
            wandb.init(
                project=self.config.logging.wandb_project,
                config=OmegaConf.to_container(self.config, resolve=True),
                name=f"manipulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
    
    def prepare_data(self):
        """Prepare data loaders."""
        data_path = Path(self.config.data.dataset_path)
        
        # Create sample dataset if it doesn't exist
        if not data_path.exists():
            logger.info(f"Creating sample dataset at {data_path}")
            create_sample_dataset_structure(data_path, num_samples=200)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            data_path,
            batch_size=self.config.training.batch_size,
            image_size=self.config.image.size,
            num_workers=self.config.data.num_workers
        )
        
        logger.info(f"Created data loaders: {len(self.train_loader)} train batches")
    
    def initialize_model(self):
        """Initialize the manipulation model."""
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
        
        # Initialize sampling utils
        sampling_config = SamplingConfig(
            num_samples=4,
            seed=self.config.seed,
            output_dir="assets/samples"
        )
        self.sampling_utils = SamplingUtils(self.manipulator, sampling_config)
        
        logger.info("Model initialized successfully")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.manipulator.pipe.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Get batch data
                images = batch['image'].to(self.device)
                prompts = batch['manipulation_prompt']
                
                # For this demo, we'll use a simple loss based on CLIP similarity
                # In a real implementation, you'd train a custom model
                loss = self.compute_training_loss(images, prompts)
                
                # Backward pass (simplified for demo)
                if hasattr(self.manipulator.pipe, 'unet'):
                    # This is a placeholder - real training would involve proper loss computation
                    pass
                
                total_loss += loss.item() if hasattr(loss, 'item') else loss
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
                
                # Log to wandb
                if wandb.run:
                    wandb.log({
                        'train/loss': total_loss / num_batches,
                        'epoch': self.current_epoch,
                        'batch': batch_idx
                    })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {self.current_epoch} - Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def compute_training_loss(self, images, prompts):
        """Compute training loss (placeholder implementation)."""
        # This is a simplified loss computation
        # In a real implementation, you'd compute proper losses for your specific model
        
        # For demonstration, we'll use a dummy loss
        dummy_loss = torch.tensor(0.1, requires_grad=True, device=self.device)
        return dummy_loss
    
    def evaluate(self):
        """Evaluate the model on validation set."""
        if self.val_loader is None:
            logger.warning("No validation loader available")
            return {}
        
        self.manipulator.pipe.eval()
        
        all_original_images = []
        all_manipulated_images = []
        all_prompts = []
        
        logger.info("Evaluating model...")
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                try:
                    images = batch['image']
                    prompts = batch['manipulation_prompt']
                    
                    # Generate manipulated images
                    batch_manipulated = []
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
                        
                        all_original_images.append(img_pil)
                        all_manipulated_images.append(manipulated)
                        all_prompts.append(prompt)
                        
                        batch_manipulated.append(manipulated)
                
                except Exception as e:
                    logger.error(f"Error in evaluation batch: {e}")
                    continue
        
        # Compute metrics
        if all_manipulated_images:
            metrics = self.metrics_evaluator.compute_manipulation_metrics(
                all_original_images,
                all_manipulated_images,
                all_prompts
            )
            
            # Log metrics
            logger.info(f"Validation Metrics: {metrics}")
            
            if wandb.run:
                wandb.log({f"val/{k}": v for k, v in metrics.items()})
            
            return metrics
        
        return {}
    
    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL image."""
        import torchvision.transforms as transforms
        
        # Denormalize
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        transform = transforms.ToPILImage()
        return transform(tensor)
    
    def save_checkpoint(self, metrics):
        """Save model checkpoint."""
        checkpoint_dir = Path("assets/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'config': OmegaConf.to_container(self.config, resolve=True),
            'metrics': metrics,
            'best_metrics': self.best_metrics
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if not self.best_metrics or metrics.get('clip_score', 0) > self.best_metrics.get('clip_score', 0):
            self.best_metrics = metrics
            best_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best checkpoint saved: {best_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Prepare data and model
        self.prepare_data()
        self.initialize_model()
        
        # Training loop
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Evaluate
            if epoch % self.config.training.eval_interval == 0:
                metrics = self.evaluate()
                
                # Save checkpoint
                if epoch % self.config.training.save_interval == 0:
                    self.save_checkpoint(metrics)
            
            logger.info(f"Epoch {epoch} completed")
        
        # Final evaluation
        logger.info("Training completed. Running final evaluation...")
        final_metrics = self.evaluate()
        
        # Create evaluation report
        report_path = Path("assets/evaluation_report.md")
        create_evaluation_report(final_metrics, report_path)
        
        logger.info(f"Final metrics: {final_metrics}")
        logger.info(f"Evaluation report saved to: {report_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train text-guided image manipulation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ManipulationTrainer(args.config)
    
    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_metrics = checkpoint.get('best_metrics', {})
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
