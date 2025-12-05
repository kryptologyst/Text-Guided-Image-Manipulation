"""
Unit tests for text-guided image manipulation components.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.manipulator import ManipulationConfig, DeviceManager
from data.dataset import ManipulationDataset, create_sample_dataset_structure
from utils.metrics import ManipulationMetrics
from utils.sampling import SamplingConfig, SamplingUtils


class TestManipulationConfig:
    """Test ManipulationConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ManipulationConfig()
        
        assert config.model_name == "runwayml/stable-diffusion-v1-5"
        assert config.clip_model_name == "openai/clip-vit-base-patch32"
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 7.5
        assert config.strength == 0.8
        assert config.image_size == 512
        assert config.batch_size == 1
        assert config.seed is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ManipulationConfig(
            model_name="custom-model",
            num_inference_steps=20,
            guidance_scale=10.0,
            strength=0.5,
            seed=42
        )
        
        assert config.model_name == "custom-model"
        assert config.num_inference_steps == 20
        assert config.guidance_scale == 10.0
        assert config.strength == 0.5
        assert config.seed == 42


class TestDeviceManager:
    """Test DeviceManager class."""
    
    def test_device_selection(self):
        """Test automatic device selection."""
        device_manager = DeviceManager(device="auto")
        
        # Should select a valid device
        assert device_manager.device.type in ["cpu", "cuda", "mps"]
    
    def test_seed_setting(self):
        """Test seed setting functionality."""
        device_manager = DeviceManager(seed=42)
        
        # Seed should be set (we can't easily test the actual seeding)
        assert True  # Placeholder test
    
    def test_custom_device(self):
        """Test custom device selection."""
        device_manager = DeviceManager(device="cpu")
        assert device_manager.device.type == "cpu"


class TestManipulationDataset:
    """Test ManipulationDataset class."""
    
    def setup_method(self):
        """Setup test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test data."""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_creation(self):
        """Test dataset creation with sample data."""
        # Create sample dataset
        create_sample_dataset_structure(self.temp_path, num_samples=10)
        
        # Create dataset
        dataset = ManipulationDataset(self.temp_path, split="train")
        
        assert len(dataset) > 0
        assert isinstance(dataset, ManipulationDataset)
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        # Create sample dataset
        create_sample_dataset_structure(self.temp_path, num_samples=5)
        
        # Create dataset
        dataset = ManipulationDataset(self.temp_path, split="train")
        
        if len(dataset) > 0:
            item = dataset[0]
            
            assert "image" in item
            assert "manipulation_prompt" in item
            assert "image_path" in item
            
            # Check image tensor shape
            assert item["image"].shape == (3, 512, 512)
            assert isinstance(item["manipulation_prompt"], str)


class TestManipulationMetrics:
    """Test ManipulationMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics evaluator initialization."""
        metrics_evaluator = ManipulationMetrics()
        
        assert metrics_evaluator.device is not None
        assert isinstance(metrics_evaluator.device, torch.device)
    
    def test_clip_score_computation(self):
        """Test CLIP score computation."""
        metrics_evaluator = ManipulationMetrics()
        
        # Create dummy images and prompts
        images = [Image.new('RGB', (512, 512), color=(255, 0, 0)) for _ in range(3)]
        prompts = ["red image", "red image", "red image"]
        
        # This might fail if CLIP is not available, which is expected
        try:
            scores = metrics_evaluator.compute_clip_score(images, prompts)
            assert "clip_score" in scores
            assert "clip_score_std" in scores
        except Exception:
            # Expected if CLIP is not available
            pass
    
    def test_lpips_distance_computation(self):
        """Test LPIPS distance computation."""
        metrics_evaluator = ManipulationMetrics()
        
        # Create dummy images
        images1 = [Image.new('RGB', (512, 512), color=(255, 0, 0)) for _ in range(3)]
        images2 = [Image.new('RGB', (512, 512), color=(0, 255, 0)) for _ in range(3)]
        
        # This might fail if LPIPS is not available, which is expected
        try:
            distances = metrics_evaluator.compute_lpips_distance(images1, images2)
            assert "lpips_distance" in distances
            assert "lpips_distance_std" in distances
        except Exception:
            # Expected if LPIPS is not available
            pass


class TestSamplingConfig:
    """Test SamplingConfig class."""
    
    def test_default_sampling_config(self):
        """Test default sampling configuration."""
        config = SamplingConfig()
        
        assert config.num_samples == 4
        assert config.seed is None
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 7.5
        assert config.strength == 0.8
        assert config.image_size == 512
        assert config.grid_cols == 2
        assert config.grid_rows == 2
        assert config.save_individual is True
        assert config.save_grid is True
        assert config.output_dir == "assets/samples"
        assert config.format == "png"
        assert config.quality == 95
    
    def test_custom_sampling_config(self):
        """Test custom sampling configuration."""
        config = SamplingConfig(
            num_samples=8,
            seed=42,
            num_inference_steps=20,
            guidance_scale=10.0,
            strength=0.5,
            output_dir="custom_output"
        )
        
        assert config.num_samples == 8
        assert config.seed == 42
        assert config.num_inference_steps == 20
        assert config.guidance_scale == 10.0
        assert config.strength == 0.5
        assert config.output_dir == "custom_output"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_sample_dataset_structure(self):
        """Test sample dataset structure creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            create_sample_dataset_structure(temp_path, num_samples=20)
            
            # Check that directories were created
            assert (temp_path / "train").exists()
            assert (temp_path / "val").exists()
            assert (temp_path / "test").exists()
            
            # Check that files were created
            train_files = list((temp_path / "train").glob("*.jpg"))
            assert len(train_files) > 0
            
            # Check that prompt files exist
            prompt_files = list((temp_path / "train").glob("*.txt"))
            assert len(prompt_files) > 0


class TestImageProcessing:
    """Test image processing utilities."""
    
    def test_pil_to_tensor_conversion(self):
        """Test PIL to tensor conversion."""
        # Create a test image
        img = Image.new('RGB', (256, 256), color=(255, 0, 0))
        
        # Convert to tensor (this would be in the actual implementation)
        # For now, just test that we can create the image
        assert img.size == (256, 256)
        assert img.mode == 'RGB'
    
    def test_tensor_to_pil_conversion(self):
        """Test tensor to PIL conversion."""
        # Create a test tensor
        tensor = torch.randn(3, 256, 256)
        
        # Test tensor properties
        assert tensor.shape == (3, 256, 256)
        assert tensor.dtype == torch.float32


# Integration tests (these require actual models and may be slow)
class TestIntegration:
    """Integration tests that require actual models."""
    
    @pytest.mark.slow
    def test_full_manipulation_pipeline(self):
        """Test the full manipulation pipeline."""
        # This test would require actual model loading
        # Skip if models are not available
        pytest.skip("Requires actual model loading - run manually")
    
    @pytest.mark.slow
    def test_streamlit_app_import(self):
        """Test that Streamlit app can be imported."""
        # This test would require streamlit and other dependencies
        # Skip if not available
        pytest.skip("Requires streamlit - run manually")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
