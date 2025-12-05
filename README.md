# Text-Guided Image Manipulation

A production-ready implementation of text-guided image manipulation using diffusion models. This project demonstrates how to modify existing images based on textual descriptions using state-of-the-art diffusion models and CLIP guidance.

## Features

- **Advanced Text-Guided Manipulation**: Uses Stable Diffusion for high-quality image manipulation
- **CLIP Integration**: Leverages CLIP for text-image alignment and evaluation
- **Comprehensive Evaluation**: Multiple metrics including CLIP Score, LPIPS, and FID
- **Interactive Demo**: Streamlit-based web interface for easy experimentation
- **Production Ready**: Clean code structure, type hints, and comprehensive documentation
- **Device Support**: Automatic device detection (CUDA, MPS, CPU)
- **Reproducible**: Deterministic seeding and configuration management

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Text-Guided-Image-Manipulation.git
cd Text-Guided-Image-Manipulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive demo:
```bash
streamlit run demo/streamlit_app.py
```

### Basic Usage

```python
from src.models.manipulator import TextGuidedImageManipulator, ManipulationConfig
from PIL import Image

# Initialize the manipulator
config = ManipulationConfig(
    seed=42,
    num_inference_steps=20,
    strength=0.7
)
manipulator = TextGuidedImageManipulator(config)

# Load an image
image = Image.open("path/to/your/image.jpg")

# Manipulate the image
manipulated_image = manipulator.manipulate_image(
    image=image,
    text_prompt="Add a beautiful sunset in the background"
)

# Save the result
manipulated_image.save("manipulated_image.jpg")
```

## Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   └── manipulator.py        # Main manipulation model
│   ├── data/                     # Data handling
│   │   └── dataset.py            # Dataset classes and loaders
│   └── utils/                    # Utilities
│       ├── metrics.py            # Evaluation metrics
│       └── sampling.py            # Sampling and visualization
├── configs/                      # Configuration files
│   └── default.yaml             # Default configuration
├── scripts/                      # Training and evaluation scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── sample.py                 # Sampling script
├── demo/                         # Demo applications
│   └── streamlit_app.py         # Streamlit web interface
├── tests/                        # Unit tests
├── assets/                       # Generated assets
│   ├── samples/                  # Generated samples
│   ├── checkpoints/              # Model checkpoints
│   └── visualizations/           # Evaluation plots
├── data/                         # Dataset storage
├── logs/                         # Training logs
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Configuration

The project uses YAML configuration files for easy customization. Key configuration options:

```yaml
# Model settings
model:
  name: "runwayml/stable-diffusion-v1-5"
  clip_model_name: "openai/clip-vit-base-patch32"
  device: "auto"

# Generation settings
generation:
  num_inference_steps: 50
  guidance_scale: 7.5
  strength: 0.8

# Image settings
image:
  size: 512
  format: "png"
  quality: 95
```

## Training

To train a custom model:

```bash
python scripts/train.py --config configs/default.yaml
```

The training script supports:
- Custom dataset loading
- Checkpoint saving and resuming
- Wandb integration for experiment tracking
- Comprehensive evaluation metrics

## Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint assets/checkpoints/best_checkpoint.pt
```

Evaluation includes:
- CLIP Score (text-image alignment)
- LPIPS Distance (perceptual similarity)
- FID Score (image quality)
- Color change analysis
- Comprehensive visualizations

## Interactive Demo

The Streamlit demo provides an intuitive interface for:
- Image upload and manipulation
- Real-time parameter adjustment
- Multiple sample generation
- Metrics visualization
- Result download

Launch the demo:
```bash
streamlit run demo/streamlit_app.py
```

## Evaluation Metrics

### CLIP Score
Measures how well the generated image matches the text prompt.
- **Good**: > 0.25
- **Excellent**: > 0.30

### LPIPS Distance
Measures perceptual similarity between original and manipulated images.
- **Low change**: < 0.1
- **Moderate change**: 0.1 - 0.3
- **High change**: > 0.3

### FID Score
Measures the quality and realism of generated images.
- **Excellent**: < 10
- **Good**: 10 - 50
- **Fair**: 50 - 100
- **Poor**: > 100

## Dataset

The project includes utilities for creating and managing datasets:

```python
from src.data.dataset import create_sample_dataset_structure

# Create a sample dataset
create_sample_dataset_structure("data/my_dataset", num_samples=100)
```

Supported dataset formats:
- JSON with structured metadata
- Text files with image paths and prompts
- Directory-based organization

## Advanced Usage

### Custom Sampling

```python
from src.utils.sampling import SamplingUtils, SamplingConfig

# Configure sampling
sampling_config = SamplingConfig(
    num_samples=8,
    seed=42,
    output_dir="my_samples"
)

sampling_utils = SamplingUtils(manipulator, sampling_config)

# Generate multiple samples
samples = sampling_utils.sample_from_prompt(
    image=my_image,
    prompt="Add dramatic storm clouds",
    num_samples=4
)
```

### Batch Processing

```python
# Process multiple images
prompts = ["Add sunset", "Add snow", "Change to night"]
results = sampling_utils.sample_from_multiple_prompts(my_image, prompts)
```

### Custom Evaluation

```python
from src.utils.metrics import ManipulationMetrics

metrics_evaluator = ManipulationMetrics()

# Evaluate custom images
metrics = metrics_evaluator.compute_manipulation_metrics(
    original_images=[img1, img2],
    manipulated_images=[manip1, manip2],
    prompts=["prompt1", "prompt2"]
)
```

## Model Card

### Intended Use
This model is designed for research and educational purposes in text-guided image manipulation. It can be used to:
- Modify existing images based on textual descriptions
- Explore the relationship between text and visual content
- Generate creative variations of images

### Limitations
- Performance depends on the quality of input images and prompts
- May not handle complex spatial relationships perfectly
- Generated images may contain artifacts or inconsistencies
- Requires significant computational resources

### Bias and Safety
- The model inherits biases from the training data
- May generate inappropriate content based on prompts
- Users should review outputs before use
- Not recommended for generating misleading or harmful content

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{text_guided_manipulation,
  title={Text-Guided Image Manipulation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Text-Guided-Image-Manipulation}
}
```

## Acknowledgments

- Stable Diffusion team for the base model
- OpenAI for CLIP
- Hugging Face for the transformers library
- The open-source community for various tools and libraries

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the example notebooks

---

**Note**: This project is for educational and research purposes. Please ensure responsible use of the technology and respect for intellectual property rights.
# Text-Guided-Image-Manipulation
