"""
Evaluation metrics for text-guided image manipulation.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import json

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logging.warning("LPIPS not available. Install with: pip install lpips")

try:
    from clean_fid import fid
    CLEAN_FID_AVAILABLE = True
except ImportError:
    CLEAN_FID_AVAILABLE = False
    logging.warning("Clean-FID not available. Install with: pip install clean-fid")

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available. Install with: pip install transformers")

logger = logging.getLogger(__name__)


class ManipulationMetrics:
    """
    Comprehensive evaluation metrics for text-guided image manipulation.
    """
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.lpips_model = None
        self.clip_model = None
        self.clip_processor = None
        
        # Initialize models
        self._load_models()
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_models(self):
        """Load evaluation models."""
        # Load LPIPS model
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            logger.info("LPIPS model loaded")
        
        # Load CLIP model
        if CLIP_AVAILABLE:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)
            logger.info("CLIP model loaded")
    
    def compute_clip_score(
        self,
        images: List[Image.Image],
        texts: List[str]
    ) -> Dict[str, float]:
        """
        Compute CLIP score between images and texts.
        
        Args:
            images: List of PIL Images
            texts: List of text descriptions
            
        Returns:
            Dictionary with CLIP scores
        """
        if not CLIP_AVAILABLE or self.clip_model is None:
            logger.warning("CLIP not available, returning dummy scores")
            return {"clip_score": 0.0, "clip_score_std": 0.0}
        
        scores = []
        
        for image, text in zip(images, texts):
            # Process with CLIP
            inputs = self.clip_processor(
                text=[text],
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
                score = logits_per_image.item()
                scores.append(score)
        
        scores = np.array(scores)
        return {
            "clip_score": float(np.mean(scores)),
            "clip_score_std": float(np.std(scores))
        }
    
    def compute_lpips_distance(
        self,
        images1: List[Image.Image],
        images2: List[Image.Image]
    ) -> Dict[str, float]:
        """
        Compute LPIPS perceptual distance between two sets of images.
        
        Args:
            images1: First set of images
            images2: Second set of images
            
        Returns:
            Dictionary with LPIPS distances
        """
        if not LPIPS_AVAILABLE or self.lpips_model is None:
            logger.warning("LPIPS not available, returning dummy distances")
            return {"lpips_distance": 0.0, "lpips_distance_std": 0.0}
        
        distances = []
        
        for img1, img2 in zip(images1, images2):
            # Convert PIL to tensor
            img1_tensor = self._pil_to_tensor(img1).to(self.device)
            img2_tensor = self._pil_to_tensor(img2).to(self.device)
            
            # Compute LPIPS distance
            with torch.no_grad():
                distance = self.lpips_model(img1_tensor, img2_tensor)
                distances.append(distance.item())
        
        distances = np.array(distances)
        return {
            "lpips_distance": float(np.mean(distances)),
            "lpips_distance_std": float(np.std(distances))
        }
    
    def compute_fid_score(
        self,
        real_images_dir: Union[str, Path],
        generated_images_dir: Union[str, Path]
    ) -> Dict[str, float]:
        """
        Compute FID score between real and generated images.
        
        Args:
            real_images_dir: Directory containing real images
            generated_images_dir: Directory containing generated images
            
        Returns:
            Dictionary with FID score
        """
        if not CLEAN_FID_AVAILABLE:
            logger.warning("Clean-FID not available, returning dummy FID")
            return {"fid_score": 0.0}
        
        try:
            fid_score = fid.compute_fid(
                str(real_images_dir),
                str(generated_images_dir),
                device=self.device
            )
            return {"fid_score": float(fid_score)}
        except Exception as e:
            logger.error(f"Failed to compute FID: {e}")
            return {"fid_score": 0.0}
    
    def compute_manipulation_metrics(
        self,
        original_images: List[Image.Image],
        manipulated_images: List[Image.Image],
        prompts: List[str],
        real_images_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for text-guided image manipulation.
        
        Args:
            original_images: Original input images
            manipulated_images: Manipulated output images
            prompts: Text prompts used for manipulation
            real_images_dir: Directory with real images for FID computation
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # CLIP scores
        clip_scores = self.compute_clip_score(manipulated_images, prompts)
        metrics.update(clip_scores)
        
        # LPIPS distance (how much the image changed)
        lpips_distances = self.compute_lpips_distance(original_images, manipulated_images)
        metrics.update(lpips_distances)
        
        # FID score (if real images provided)
        if real_images_dir:
            # Save manipulated images temporarily for FID computation
            temp_dir = Path("temp_fid_eval")
            temp_dir.mkdir(exist_ok=True)
            
            for i, img in enumerate(manipulated_images):
                img.save(temp_dir / f"generated_{i:04d}.jpg")
            
            fid_scores = self.compute_fid_score(real_images_dir, temp_dir)
            metrics.update(fid_scores)
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)
        
        # Additional metrics
        metrics.update(self._compute_additional_metrics(original_images, manipulated_images))
        
        return metrics
    
    def _compute_additional_metrics(
        self,
        original_images: List[Image.Image],
        manipulated_images: List[Image.Image]
    ) -> Dict[str, float]:
        """Compute additional metrics like color statistics."""
        metrics = {}
        
        # Color statistics
        original_colors = []
        manipulated_colors = []
        
        for orig_img, manip_img in zip(original_images, manipulated_images):
            orig_array = np.array(orig_img)
            manip_array = np.array(manip_img)
            
            # Mean color values
            original_colors.append(np.mean(orig_array, axis=(0, 1)))
            manipulated_colors.append(np.mean(manip_array, axis=(0, 1)))
        
        original_colors = np.array(original_colors)
        manipulated_colors = np.array(manipulated_colors)
        
        # Color change magnitude
        color_change = np.linalg.norm(manipulated_colors - original_colors, axis=1)
        metrics["mean_color_change"] = float(np.mean(color_change))
        metrics["std_color_change"] = float(np.std(color_change))
        
        return metrics
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor for LPIPS."""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transform(image).unsqueeze(0)
    
    def evaluate_model(
        self,
        model,
        test_loader,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a manipulation model on a test dataset.
        
        Args:
            model: The manipulation model to evaluate
            test_loader: DataLoader with test data
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        model.eval()
        
        all_original_images = []
        all_manipulated_images = []
        all_prompts = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Get batch data
                original_images = batch['image']
                prompts = batch['manipulation_prompt']
                
                # Generate manipulated images
                manipulated_images = model.generate(original_images, prompts)
                
                # Convert tensors to PIL images
                for i in range(len(original_images)):
                    orig_img = self._tensor_to_pil(original_images[i])
                    manip_img = self._tensor_to_pil(manipulated_images[i])
                    
                    all_original_images.append(orig_img)
                    all_manipulated_images.append(manip_img)
                    all_prompts.append(prompts[i])
        
        # Compute metrics
        metrics = self.compute_manipulation_metrics(
            all_original_images,
            all_manipulated_images,
            all_prompts
        )
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            with open(output_dir / "evaluation_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save sample images
            sample_dir = output_dir / "samples"
            sample_dir.mkdir(exist_ok=True)
            
            for i in range(min(10, len(all_original_images))):
                all_original_images[i].save(sample_dir / f"original_{i:03d}.jpg")
                all_manipulated_images[i].save(sample_dir / f"manipulated_{i:03d}.jpg")
                
                with open(sample_dir / f"prompt_{i:03d}.txt", 'w') as f:
                    f.write(all_prompts[i])
        
        return metrics
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL image."""
        import torchvision.transforms as transforms
        
        # Denormalize
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        transform = transforms.ToPILImage()
        return transform(tensor)


def create_evaluation_report(
    metrics: Dict[str, Any],
    output_path: Union[str, Path]
) -> None:
    """
    Create a comprehensive evaluation report.
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_path: Path to save the report
    """
    output_path = Path(output_path)
    
    report = f"""
# Text-Guided Image Manipulation Evaluation Report

## Overview
This report presents the evaluation results for text-guided image manipulation models.

## Metrics Summary

### CLIP Score
- **Mean CLIP Score**: {metrics.get('clip_score', 'N/A'):.4f}
- **CLIP Score Std**: {metrics.get('clip_score_std', 'N/A'):.4f}

### Perceptual Distance (LPIPS)
- **Mean LPIPS Distance**: {metrics.get('lpips_distance', 'N/A'):.4f}
- **LPIPS Distance Std**: {metrics.get('lpips_distance_std', 'N/A'):.4f}

### Fréchet Inception Distance (FID)
- **FID Score**: {metrics.get('fid_score', 'N/A'):.4f}

### Color Statistics
- **Mean Color Change**: {metrics.get('mean_color_change', 'N/A'):.4f}
- **Std Color Change**: {metrics.get('std_color_change', 'N/A'):.4f}

## Interpretation

### CLIP Score
Higher CLIP scores indicate better alignment between the manipulated images and the text prompts.
- **Good**: > 0.25
- **Excellent**: > 0.30

### LPIPS Distance
LPIPS distance measures perceptual similarity between original and manipulated images.
- **Low change**: < 0.1
- **Moderate change**: 0.1 - 0.3
- **High change**: > 0.3

### FID Score
Lower FID scores indicate better quality and realism of generated images.
- **Excellent**: < 10
- **Good**: 10 - 50
- **Fair**: 50 - 100
- **Poor**: > 100

## Recommendations

Based on the evaluation results:
1. **CLIP Score**: {'Consider improving text-image alignment' if metrics.get('clip_score', 0) < 0.25 else 'Good text-image alignment achieved'}
2. **LPIPS Distance**: {'Appropriate level of manipulation' if 0.1 <= metrics.get('lpips_distance', 0) <= 0.3 else 'Consider adjusting manipulation strength'}
3. **FID Score**: {'Focus on improving image quality' if metrics.get('fid_score', 0) > 50 else 'Good image quality achieved'}

---
*Report generated on {Path().cwd()}*
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation report saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    metrics_evaluator = ManipulationMetrics()
    
    # Create dummy data for testing
    dummy_images = [Image.new('RGB', (512, 512), color=(255, 0, 0)) for _ in range(5)]
    dummy_prompts = ["Add a blue sky"] * 5
    
    # Test CLIP score
    clip_scores = metrics_evaluator.compute_clip_score(dummy_images, dummy_prompts)
    print(f"CLIP Scores: {clip_scores}")
    
    # Test LPIPS distance
    lpips_distances = metrics_evaluator.compute_lpips_distance(dummy_images, dummy_images)
    print(f"LPIPS Distances: {lpips_distances}")
    
    logger.info("Evaluation metrics test completed!")
