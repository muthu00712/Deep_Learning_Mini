"""
PawBreed AI — Dog Breed Classifier
Streamlit Web Application for MobileNetV2-based Dog Breed Classification
Phase 7 - Final Web Application Deployment
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------------------------------
# Page Configuration & Modern Aesthetics
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PawBreed AI — Dog Breed Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern card UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
        margin-bottom: 0.2rem;
        text-align: left;
    }

    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    .winner-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 2px solid #3B82F6;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.15);
        margin-bottom: 1.2rem;
    }

    .winner-title {
        color: #1E40AF;
        font-size: 0.9rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    .winner-breed {
        color: #0F172A;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        line-height: 1.2;
    }

    .winner-confidence {
        font-size: 1.25rem;
        font-weight: 700;
        color: #2563EB;
    }

    .prob-bar-container {
        margin-bottom: 0.85rem;
    }

    .prob-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
        font-size: 0.95rem;
    }

    .breed-label {
        font-weight: 600;
        color: #1E293B;
    }

    .prob-val {
        font-weight: 700;
        color: #3B82F6;
    }

    .bar-bg {
        background-color: #E2E8F0;
        border-radius: 9999px;
        height: 10px;
        overflow: hidden;
    }

    .bar-fill {
        background: linear-gradient(90deg, #3B82F6, #06B6D4);
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s ease;
    }

    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        color: #92400E;
        font-size: 0.88rem;
        margin-top: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Path Resolution & Resource Loaders
# ---------------------------------------------------------------------------
def resolve_project_paths() -> Tuple[Path, Path, Path]:
    """
    Locates the models directory, final keras model, and class_names.json
    whether run from project root, dog_breed_classification/, or app/.
    """
    curr = Path(__file__).resolve().parent
    search_dirs = [
        curr.parent,                  # dog_breed_classification
        curr,                         # app
        Path.cwd() / "dog_breed_classification",
        Path.cwd(),
        Path.cwd().parent / "dog_breed_classification",
    ]

    for base in search_dirs:
        model_p = base / "models" / "dog_breed_classifier_final.keras"
        class_p = base / "models" / "class_names.json"
        if model_p.is_file() and class_p.is_file():
            return base, model_p, class_p

    # Fallback to relative models/
    return Path.cwd(), Path("models/dog_breed_classifier_final.keras"), Path("models/class_names.json")

PROJECT_DIR, MODEL_PATH, CLASS_NAMES_PATH = resolve_project_paths()


@st.cache_resource(show_spinner="Loading trained MobileNetV2 model into memory...")
def load_classification_model(model_path_str: str):
    """
    Loads and caches the final trained Keras model once using Streamlit resource caching.
    """
    model_file = Path(model_path_str)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Trained model not found at '{model_file}'. "
            "Please ensure models/dog_breed_classifier_final.keras exists."
        )
    model = tf.keras.models.load_model(str(model_file))
    return model


@st.cache_data(show_spinner=False)
def load_class_names(class_names_path_str: str) -> List[str]:
    """
    Loads and caches class names list from class_names.json.
    """
    class_file = Path(class_names_path_str)
    if not class_file.is_file():
        raise FileNotFoundError(
            f"Class mapping file not found at '{class_file}'. "
            "Please ensure models/class_names.json exists."
        )
    with open(class_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "class_names" in data:
        return data["class_names"]
    elif "classes" in data:
        return data["classes"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid format in class_names.json")


def format_breed_name(breed_raw: str) -> str:
    """Formats raw identifier like 'standard_poodle' into 'Standard Poodle'."""
    return breed_raw.replace("_", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# Preprocessing & Inference Logic
# ---------------------------------------------------------------------------
def preprocess_image_for_model(
    img: Image.Image,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Strictly applies MobileNetV2 preprocessing used during training:
    1. Converts image to RGB
    2. Resizes to 224x224 (Bilinear)
    3. Converts to float32 NumPy array
    4. Expands batch dimension: (1, 224, 224, 3)
    5. Applies tf.keras.applications.mobilenet_v2.preprocess_input (scales to [-1, 1])
    """
    # 1. RGB conversion
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 2. Resize
    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)

    # 3. Array conversion
    img_array = np.array(img_resized, dtype=np.float32)

    # 4. Batch dimension
    img_batch = np.expand_dims(img_array, axis=0)

    # 5. MobileNetV2 preprocessing (DO NOT manually divide by 255)
    tensor = preprocess_input(img_batch)
    return tensor


def predict_dog_breed(
    model: tf.keras.Model,
    class_names: List[str],
    img: Image.Image,
    top_k: int = 5
) -> List[Dict]:
    """
    Runs model inference and returns top-k sorted predictions with names and probabilities.
    """
    input_tensor = preprocess_image_for_model(img)
    raw_probs = model.predict(input_tensor, verbose=0)[0]

    # Verify output shape matches class count
    if len(raw_probs) != len(class_names):
        raise ValueError(
            f"Model output dimension ({len(raw_probs)}) does not match class count ({len(class_names)})."
        )

    # Top-K sorting descending
    top_indices = np.argsort(raw_probs)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        breed_id = class_names[idx]
        prob = float(raw_probs[idx])
        results.append({
            "rank": rank,
            "breed_id": breed_id,
            "breed_name": format_breed_name(breed_id),
            "probability": prob,
            "percentage": prob * 100.0,
            "class_index": int(idx)
        })

    return results


# ---------------------------------------------------------------------------
# Sidebar: System Metadata, Model Stats, and Quick Sample Loader
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🐾 PawBreed AI")
    st.caption("Deep Learning Dog Breed Classifier")
    st.markdown("---")
    st.markdown("#### ⚡ Quick Test Samples")
    st.caption("Don't have a dog photo handy? Pick one of our curated test samples:")

    # Look for sample test images in dataset directory
    sample_options = {
        "None (Upload My Own)": None,
    }
    
    # Try finding test samples in dataset/train
    train_dir = PROJECT_DIR / "dataset" / "train"
    if train_dir.is_dir():
        # Curated sample mapping with known labels from labels.csv
        curated_samples = [
            ("Samoyed (Arctic Spiz)", "8452a26d7243a299ea782a7ba4036f1f.jpg"),
            ("Scottish Deerhound", "bf7004de0f825049194139051f7a8025.jpg"),
            ("Weimaraner (Grey Ghost)", "7e1b9c2578655329d460cc9bebe697e6.jpg"),
            ("Lakeland Terrier", "2e6db85e0641e14e0025fe7f0853fa89.jpg"),
            ("Ibizan Hound", "7baa672b0451bb6f944c7d969edb61b6.jpg")
        ]
        for name, fname in curated_samples:
            sample_file = train_dir / fname
            if sample_file.is_file():
                sample_options[name] = sample_file

    selected_sample = st.selectbox("Choose a sample:", list(sample_options.keys()))

    st.markdown("---")
    st.markdown("#### 💡 Tips for Best Accuracy")
    st.markdown("""
    1. Use a clear, well-lit photo of a dog.
    2. Ensure the dog is centered in the frame.
    3. Avoid photos where the dog is too far away or heavily occluded.
    """)


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">PawBreed AI — Dog Breed Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Classify dog breeds instantly across <b>120 distinct breeds</b> using an optimized MobileNetV2 Transfer Learning neural network.'
    '</div>',
    unsafe_allow_html=True
)

# Load model & class mapping with friendly error handling
try:
    model = load_classification_model(str(MODEL_PATH))
    class_names = load_class_names(str(CLASS_NAMES_PATH))
except Exception as e:
    st.error(f"❌ **Failed to load model or class mapping**: {e}")
    st.info("Please verify that `models/dog_breed_classifier_final.keras` and `models/class_names.json` exist.")
    st.stop()

# Layout: Two Columns (Left: Input & Image, Right: Results Dashboard)
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 📷 Upload Dog Image")
    st.write("Upload a JPG, JPEG, or PNG photograph of a dog to identify its breed.")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, and PNG. Max file size: 200MB."
    )

    # Determine image source: uploaded file or selected sample
    active_image: Optional[Image.Image] = None
    image_source_label = ""

    if uploaded_file is not None:
        try:
            active_image = Image.open(uploaded_file)
            image_source_label = f"Uploaded File: `{uploaded_file.name}`"
        except Exception as e:
            st.error(f"⚠️ **Invalid image file**: Could not open image. Error: {e}")
            active_image = None
    elif selected_sample and sample_options.get(selected_sample) is not None:
        sample_path = sample_options[selected_sample]
        try:
            active_image = Image.open(sample_path)
            image_source_label = f"Test Sample: `{selected_sample}`"
        except Exception as e:
            st.error(f"⚠️ Could not load sample image: {e}")
            active_image = None

    if active_image is not None:
        st.markdown(f"**Image Preview** ({image_source_label})")
        st.image(
            active_image,
            use_container_width=True,
            caption=f"Dimensions: {active_image.width} × {active_image.height} px | Mode: {active_image.mode}"
        )

        identify_button = st.button(
            "🔍 Identify Dog Breed",
            type="primary",
            use_container_width=True
        )
    else:
        st.info("👆 Please upload an image or select a sample from the sidebar to begin.")
        identify_button = False

with col_results:
    st.markdown("### 🏆 Prediction Dashboard")

    if active_image is not None and identify_button:
        with st.spinner("🧠 Analyzing facial features and coat patterns with MobileNetV2..."):
            try:
                predictions = predict_dog_breed(model, class_names, active_image, top_k=5)
                top_winner = predictions[0]

                # Top-1 Winner Card
                st.markdown(f"""
                <div class="winner-card">
                    <div class="winner-title">Primary Breed Prediction</div>
                    <div class="winner-breed">{top_winner['breed_name']}</div>
                    <div class="winner-confidence">Confidence: {top_winner['percentage']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📊 Top 5 Candidate Breeds")
                st.write("Ranked by model softmax probability distribution:")

                # Probability Bars for Top 5
                for pred in predictions:
                    pct = pred["percentage"]
                    b_name = pred["breed_name"]
                    rank = pred["rank"]
                    st.markdown(f"""
                    <div class="prob-bar-container">
                        <div class="prob-header">
                            <span class="breed-label">#{rank} {b_name}</span>
                            <span class="prob-val">{pct:.2f}%</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width: {max(pct, 2.0):.1f}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Expandable detailed breakdown table
                with st.expander("🔍 View Detailed Metrics & Raw Indices", expanded=False):
                    df_preds = pd.DataFrame([
                        {
                            "Rank": p["rank"],
                            "Breed Name": p["breed_name"],
                            "Raw Identifier": p["breed_id"],
                            "Class Index": p["class_index"],
                            "Probability": f"{p['probability']:.4f}",
                            "Confidence": f"{p['percentage']:.2f}%"
                        }
                        for p in predictions
                    ])
                    st.dataframe(df_preds, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ **Inference Error**: An unexpected error occurred during prediction: {e}")
                st.exception(e)
    elif active_image is not None and not identify_button:
        st.info("Ready! Click **'Identify Dog Breed'** on the left to run inference.")
    else:
        st.markdown("""
        <div class="metric-card">
            <h4>Awaiting Input</h4>
            <p style="color: #64748B; font-size: 0.95rem;">
                Prediction results, confidence percentages, and top-5 probability distributions will appear here once an image is uploaded and analyzed.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Important Non-Dog Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Important Scope & Disclaimer:</strong><br>
        PawBreed AI is trained exclusively on <strong>120 recognized dog breeds</strong>.
        Predictions are intended specifically for dog photographs. If a non-dog photo (e.g., cat, person, car)
        is provided, the model will map the image to the nearest visual dog breed pattern.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "PawBreed AI • MobileNetV2-based Transfer Learning • "
    "120 Breeds • Preprocessed with official MobileNetV2 normalization [-1, 1]"
)
