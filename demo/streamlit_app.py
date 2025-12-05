"""
Streamlit demo interface for text-guided image manipulation.
"""

import streamlit as st
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import time
import json
from typing import Optional, List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.manipulator import TextGuidedImageManipulator, ManipulationConfig
from utils.sampling import SamplingUtils, SamplingConfig
from utils.metrics import ManipulationMetrics
from utils.visualization import VisualizationUtils

# Page configuration
st.set_page_config(
    page_title="Text-Guided Image Manipulation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #5a6fd8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'manipulator' not in st.session_state:
    st.session_state.manipulator = None
if 'metrics_evaluator' not in st.session_state:
    st.session_state.metrics_evaluator = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}


@st.cache_resource
def load_models():
    """Load models with caching."""
    try:
        config = ManipulationConfig(
            seed=42,
            num_inference_steps=20,  # Faster for demo
            strength=0.7
        )
        
        manipulator = TextGuidedImageManipulator(config)
        metrics_evaluator = ManipulationMetrics()
        
        return manipulator, metrics_evaluator
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None


def initialize_models():
    """Initialize models if not already loaded."""
    if st.session_state.manipulator is None:
        with st.spinner("Loading models... This may take a few minutes on first run."):
            manipulator, metrics_evaluator = load_models()
            st.session_state.manipulator = manipulator
            st.session_state.metrics_evaluator = metrics_evaluator


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🎨 Text-Guided Image Manipulation</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        num_inference_steps = st.slider("Inference Steps", 10, 50, 20, help="More steps = better quality but slower")
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5, help="Higher = more adherence to prompt")
        strength = st.slider("Manipulation Strength", 0.1, 1.0, 0.7, 0.1, help="How much to modify the original image")
        
        # Generation settings
        st.subheader("Generation Settings")
        num_samples = st.slider("Number of Samples", 1, 8, 4, help="Number of variations to generate")
        seed = st.number_input("Random Seed", value=42, help="For reproducible results")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            negative_prompt = st.text_area(
                "Negative Prompt", 
                value="blurry, low quality, distorted, ugly",
                help="What to avoid in the generated image"
            )
            
            device = st.selectbox(
                "Device",
                ["auto", "cpu", "cuda", "mps"],
                help="Device to run inference on"
            )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Input Image")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to manipulate"
        )
        
        # Sample images
        st.subheader("Or try a sample image:")
        sample_images = {
            "Colorful Rectangle": "assets/samples/sample_000.png",
            "Simple Pattern": "assets/samples/sample_001.png",
            "Geometric Shape": "assets/samples/sample_002.png"
        }
        
        selected_sample = st.selectbox("Choose sample image", list(sample_images.keys()))
        
        # Display input image
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert('RGB')
            st.image(input_image, caption="Uploaded Image", use_column_width=True)
        else:
            sample_path = Path(sample_images[selected_sample])
            if sample_path.exists():
                input_image = Image.open(sample_path).convert('RGB')
                st.image(input_image, caption=f"Sample: {selected_sample}", use_column_width=True)
            else:
                # Create a default sample image
                input_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
                st.image(input_image, caption="Default Sample Image", use_column_width=True)
    
    with col2:
        st.header("✍️ Text Prompt")
        
        # Prompt input
        prompt = st.text_area(
            "Enter your manipulation prompt:",
            value="Add a beautiful sunset in the background",
            height=100,
            help="Describe how you want to modify the image"
        )
        
        # Preset prompts
        st.subheader("Quick Prompts:")
        preset_prompts = [
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
        
        selected_preset = st.selectbox("Choose a preset prompt", ["Custom"] + preset_prompts)
        if selected_preset != "Custom":
            prompt = selected_preset
            st.text_area("Current prompt:", value=prompt, height=100, disabled=True)
        
        # Generate button
        generate_button = st.button("🎨 Generate Manipulated Images", type="primary")
    
    # Generation and results
    if generate_button:
        if not prompt.strip():
            st.error("Please enter a text prompt!")
            return
        
        # Initialize models
        initialize_models()
        
        if st.session_state.manipulator is None:
            st.error("Failed to load models. Please check the error messages above.")
            return
        
        # Update configuration
        config = ManipulationConfig(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
            device=device
        )
        
        # Generate images
        with st.spinner("Generating manipulated images..."):
            try:
                # Generate samples
                sampling_config = SamplingConfig(
                    num_samples=num_samples,
                    seed=seed,
                    output_dir="assets/samples"
                )
                sampling_utils = SamplingUtils(st.session_state.manipulator, sampling_config)
                
                generated_images = sampling_utils.sample_from_prompt(
                    input_image, 
                    prompt, 
                    negative_prompt,
                    num_samples
                )
                
                st.session_state.generated_images = generated_images
                
                # Compute metrics
                if st.session_state.metrics_evaluator:
                    metrics = st.session_state.metrics_evaluator.compute_manipulation_metrics(
                        [input_image] * len(generated_images),
                        generated_images,
                        [prompt] * len(generated_images)
                    )
                    st.session_state.metrics = metrics
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
                return
    
    # Display results
    if st.session_state.generated_images:
        st.header("🎨 Generated Results")
        
        # Display images in grid
        cols = st.columns(min(len(st.session_state.generated_images), 4))
        for i, img in enumerate(st.session_state.generated_images):
            with cols[i % 4]:
                st.image(img, caption=f"Sample {i+1}", use_column_width=True)
        
        # Metrics
        if st.session_state.metrics:
            st.header("📊 Evaluation Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "CLIP Score",
                    f"{st.session_state.metrics.get('clip_score', 0):.3f}",
                    help="Higher = better text-image alignment"
                )
            
            with col2:
                st.metric(
                    "LPIPS Distance",
                    f"{st.session_state.metrics.get('lpips_distance', 0):.3f}",
                    help="Lower = more similar to original"
                )
            
            with col3:
                st.metric(
                    "FID Score",
                    f"{st.session_state.metrics.get('fid_score', 0):.1f}",
                    help="Lower = better quality"
                )
            
            with col4:
                st.metric(
                    "Color Change",
                    f"{st.session_state.metrics.get('mean_color_change', 0):.1f}",
                    help="Magnitude of color changes"
                )
        
        # Download options
        st.header("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download individual images
            for i, img in enumerate(st.session_state.generated_images):
                img_bytes = img.tobytes()
                st.download_button(
                    label=f"Download Sample {i+1}",
                    data=img_bytes,
                    file_name=f"manipulated_sample_{i+1}.png",
                    mime="image/png"
                )
        
        with col2:
            # Download metadata
            if st.session_state.metrics:
                metadata = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "config": {
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "strength": strength,
                        "seed": seed
                    },
                    "metrics": st.session_state.metrics
                }
                
                st.download_button(
                    label="Download Metadata",
                    data=json.dumps(metadata, indent=2),
                    file_name="generation_metadata.json",
                    mime="application/json"
                )
    
    # Information section
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        This demo showcases text-guided image manipulation using diffusion models.
        
        **How it works:**
        1. Upload an image or select a sample
        2. Enter a text prompt describing the desired changes
        3. Adjust generation parameters in the sidebar
        4. Click "Generate" to create manipulated images
        
        **Key Features:**
        - Multiple sample generation
        - Real-time evaluation metrics
        - Adjustable generation parameters
        - Download results and metadata
        
        **Metrics Explained:**
        - **CLIP Score**: Measures how well the generated image matches the text prompt
        - **LPIPS Distance**: Measures perceptual similarity to the original image
        - **FID Score**: Measures the quality and realism of generated images
        - **Color Change**: Measures the magnitude of color modifications
        
        **Tips for better results:**
        - Use descriptive, specific prompts
        - Adjust the strength parameter to control manipulation intensity
        - Higher guidance scale = more adherence to prompt
        - More inference steps = better quality but slower generation
        """)


if __name__ == "__main__":
    main()
