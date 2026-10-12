"""
MobileNetV2 Model Architecture and Training Setup Module
Phase 3 - Dog Breed Classification Using MobileNetV2-Based Transfer Learning

This module provides:
1. MobileNetV2 pretrained base model loading (ImageNet weights, include_top=False)
2. Freezing the base feature extractor (base_model.trainable = False)
3. Dynamic NUM_CLASSES resolution from Phase 2 class definitions
4. Custom classification head (GlobalAveragePooling2D -> Dense(256, ReLU) -> Dropout(0.3) -> Dense(NUM_CLASSES, Softmax))
5. Compilation with Adam optimizer (lr=0.001), SparseCategoricalCrossentropy loss, and metrics
6. Model parameter verification (Trainable vs Non-trainable)
7. Textual summary export to outputs/model_summary.txt
8. Visual architecture diagram generation to outputs/mobilenetv2_architecture.png
9. Forward-pass sanity check with zero full training
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Default constants
DEFAULT_INPUT_SHAPE = (224, 224, 3)
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_DENSE_UNITS = 256


def resolve_paths(
    class_names_path_arg: Optional[str] = None,
    output_dir_arg: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Dynamically locates class_names.json and the outputs directory
    without hardcoding absolute paths.
    """
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent

    # Candidate paths for class_names.json
    candidates = [
        class_names_path_arg,
        proj_root / "outputs" / "class_names.json",
        Path("dog_breed_classification/outputs/class_names.json"),
        Path("outputs/class_names.json"),
        Path("../outputs/class_names.json"),
        Path("../../outputs/class_names.json")
    ]

    resolved_class_names = None
    for cand in candidates:
        if cand is not None:
            p = Path(cand)
            if p.is_file():
                resolved_class_names = p
                break

    if output_dir_arg:
        resolved_output_dir = Path(output_dir_arg)
    elif resolved_class_names:
        resolved_output_dir = resolved_class_names.parent
    elif (proj_root / "outputs").is_dir() or proj_root.exists():
        resolved_output_dir = proj_root / "outputs"
    else:
        resolved_output_dir = Path("outputs")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return resolved_class_names, resolved_output_dir


def get_num_classes(class_names_path: Optional[Path] = None) -> int:
    """
    Determines the number of classes dynamically from class_names.json.
    Falls back to inspecting breed names or raising an error.
    """
    if class_names_path and class_names_path.is_file():
        with open(class_names_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            num_classes = data.get("num_classes", len(data.get("class_names", [])))
            if num_classes > 0:
                logger.info(f"Loaded {num_classes} classes dynamically from {class_names_path.name}.")
                return num_classes

    # Fallback search
    class_path, _ = resolve_paths()
    if class_path and class_path.is_file():
        with open(class_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("num_classes", len(data.get("class_names", [])))

    raise FileNotFoundError(
        "Could not determine NUM_CLASSES dynamically. Please ensure Phase 2 (outputs/class_names.json) is complete."
    )


def build_model(
    num_classes: Optional[int] = None,
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    include_augmentation: bool = True,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs and compiles the MobileNetV2 transfer learning model.

    Architecture:
      Input (224, 224, 3)
         ↓
      Data Augmentation (Flip, Rotation, Zoom, Translation, Contrast; training-only)
         ↓
      MobileNetV2 Base (Pretrained on ImageNet, Frozen / trainable=False)
         ↓
      GlobalAveragePooling2D
         ↓
      Dense(256, activation="relu")
         ↓
      Dropout(0.3)
         ↓
      Dense(NUM_CLASSES, activation="softmax")

    Returns:
      (model, base_model)
    """
    if num_classes is None:
        num_classes = get_num_classes()

    logger.info(f"Building MobileNetV2 Transfer Learning Model with NUM_CLASSES = {num_classes}")

    # 1. Input Layer
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # 2. Data Augmentation (active only during training)
    if include_augmentation:
        x = tf.keras.layers.RandomFlip("horizontal", name="random_flip_h")(inputs)
        x = tf.keras.layers.RandomRotation(0.10, fill_mode="reflect", name="random_rotation")(x)
        x = tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", name="random_zoom")(x)
        x = tf.keras.layers.RandomTranslation(0.10, 0.10, fill_mode="reflect", name="random_translation")(x)
        x = tf.keras.layers.RandomContrast(0.10, name="random_contrast")(x)
    else:
        x = inputs

    # 3. MobileNetV2 Base Model (ImageNet weights, include_top=False)
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model weights for initial transfer learning stage
    base_model.trainable = False

    # Inference mode for base model to preserve BatchNormalization statistics
    x = base_model(x, training=False)

    # 4. Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name=f"dense_{dense_units}")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{dropout_rate}")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="dog_breed_predictions"
    )(x)

    # 5. Assemble Functional Model
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="DogBreed_MobileNetV2"
    )

    # 6. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    logger.info("Model successfully built and compiled.")
    return model, base_model


def verify_trainable_parameters(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Computes and verifies total, trainable, and non-trainable parameters.
    Confirms base model weights are non-trainable while head weights are trainable.
    """
    total_params = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable_params = sum(tf.size(w).numpy() for w in model.non_trainable_weights)

    logger.info(f"Total Parameters:         {total_params:,}")
    logger.info(f"Trainable Parameters:     {trainable_params:,} (Classification Head)")
    logger.info(f"Non-Trainable Parameters: {non_trainable_params:,} (Frozen MobileNetV2 Base)")

    assert trainable_params > 0, "Error: No trainable parameters found in classification head!"
    assert non_trainable_params > 0, "Error: Base model appears to not be frozen!"
    assert trainable_params + non_trainable_params == total_params, "Error: Parameter counts do not sum to total!"

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "trainable_ratio_pct": round(trainable_params / total_params * 100, 2)
    }


def save_model_summary(model: tf.keras.Model, output_path: Path, param_info: Dict[str, Any]) -> None:
    """Saves complete model summary to outputs/model_summary.txt."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    summary_text = "\n".join(summary_lines)

    sep = "=" * 80
    header = (
        f"{sep}\n"
        "PHASE 3 — MOBILENETV2 MODEL ARCHITECTURE SUMMARY\n"
        "Dog Breed Classification Using MobileNetV2-Based Transfer Learning\n"
        f"{sep}\n\n"
        f"Input Shape:              {model.input_shape}\n"
        f"Output Shape:             {model.output_shape}\n"
        f"Total Parameters:         {param_info['total_params']:,}\n"
        f"Trainable Parameters:     {param_info['trainable_params']:,} ({param_info['trainable_ratio_pct']}%)\n"
        f"Non-Trainable Parameters: {param_info['non_trainable_params']:,}\n"
        f"Base Model Status:        Frozen (trainable=False)\n"
        f"Optimizer:                Adam (lr={DEFAULT_LEARNING_RATE})\n"
        f"Loss Function:            SparseCategoricalCrossentropy\n"
        f"Metrics:                  accuracy, top_5_accuracy\n\n"
        f"{sep}\n"
        "KERAS MODEL SUMMARY\n"
        f"{sep}\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + summary_text + "\n" + "=" * 80 + "\n")

    logger.info(f"Saved model summary to: {output_path}")


def plot_architecture_diagram(
    output_path: Path,
    num_classes: int,
    param_info: Dict[str, Any]
) -> None:
    """
    Renders an elegant, publication-quality architecture diagram
    highlighting each stage, shapes, parameter counts, and frozen/trainable states.
    Saved to outputs/mobilenetv2_architecture.png.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 14), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title Banner
    ax.text(
        50, 96,
        "MobileNetV2 Transfer Learning Architecture",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1a2530"
    )
    ax.text(
        50, 93,
        f"Multi-Class Dog Breed Classification ({num_classes} Breeds)",
        ha="center", va="center", fontsize=12, color="#556575"
    )

    # Box specifications: [y_center, title, subtitle, shape, color, tag, tag_color]
    layers_data = [
        (
            85,
            "Input Image",
            "RGB Normalized Tensor (MobileNetV2 scaled: [-1, 1])",
            "Shape: (batch, 224, 224, 3)",
            "#E8F4F8", "#2B7A78",
            "Input", "#17A2B8"
        ),
        (
            73,
            "Data Augmentation (Sequential)",
            "RandomFlip, Rotation(0.10), Zoom(0.10), Translation(0.10), Contrast(0.10)",
            "Shape: (batch, 224, 224, 3) | Training-Only Active",
            "#FFF8E7", "#D48806",
            "Regularization", "#E6A23C"
        ),
        (
            59,
            "MobileNetV2 Base (ImageNet Pretrained)",
            "154 Inverted Residual Convolutional Layers (Bottlenecks)",
            "Output: (batch, 7, 7, 1280) | 2,257,984 Params",
            "#F0F0F0", "#595959",
            "[FROZEN] Non-Trainable", "#8C8C8C"
        ),
        (
            45,
            "GlobalAveragePooling2D",
            "Spatial Average Reduction: (7, 7) → 1 vector per feature map",
            "Output: (batch, 1280) | 0 Params",
            "#E6F7FF", "#1890FF",
            "Pooling", "#40A9FF"
        ),
        (
            33,
            "Dense (Fully Connected)",
            "256 Hidden Units with ReLU Activation",
            "Output: (batch, 256) | 327,936 Params",
            "#F6FFED", "#389E0D",
            "[TRAINABLE] Head", "#52C41A"
        ),
        (
            22,
            "Dropout (rate = 0.3)",
            "Regularization to prevent overfitting during transfer learning",
            "Output: (batch, 256) | 0 Params",
            "#FFF0F6", "#EB2F96",
            "Regularization", "#F759AB"
        ),
        (
            10,
            f"Dense Output (Softmax, {num_classes} Classes)",
            "Categorical Probability Distribution across 120 Dog Breeds",
            f"Output: (batch, {num_classes}) | 30,840 Params",
            "#E6FFFB", "#08979C",
            "[TRAINABLE] Head", "#13C2C2"
        ),
    ]

    box_width = 76
    box_height = 8.5
    left_x = (100 - box_width) / 2

    for i, (y, title, subtitle, shape, bg_color, border_color, tag, tag_color) in enumerate(layers_data):
        bottom_y = y - box_height / 2

        # Draw main card
        rect = patches.FancyBboxPatch(
            (left_x, bottom_y), box_width, box_height,
            boxstyle="round,pad=0.6,rounding_size=1.5",
            linewidth=2, edgecolor=border_color, facecolor=bg_color
        )
        ax.add_patch(rect)

        # Title
        ax.text(
            left_x + 3, y + 1.8, title,
            fontsize=12, fontweight="bold", color="#1f2d3d", va="center"
        )

        # Subtitle
        ax.text(
            left_x + 3, y - 0.3, subtitle,
            fontsize=9.5, color="#506172", va="center"
        )

        # Shape / details
        ax.text(
            left_x + 3, y - 2.3, shape,
            fontsize=9, fontweight="bold", color=border_color, va="center"
        )

        # Badge tag
        tag_width = 24
        tag_height = 3.2
        tag_x = left_x + box_width - tag_width - 2
        tag_y = y + 0.8
        tag_rect = patches.FancyBboxPatch(
            (tag_x, tag_y), tag_width, tag_height,
            boxstyle="round,pad=0.2,rounding_size=0.8",
            facecolor=tag_color, edgecolor="none"
        )
        ax.add_patch(tag_rect)
        ax.text(
            tag_x + tag_width / 2, tag_y + tag_height / 2, tag,
            fontsize=8.5, fontweight="bold", color="white", ha="center", va="center"
        )

        # Connector arrow to next block
        if i < len(layers_data) - 1:
            next_y = layers_data[i + 1][0]
            arrow_start_y = bottom_y
            arrow_end_y = next_y + box_height / 2
            ax.annotate(
                "",
                xy=(50, arrow_end_y),
                xytext=(50, arrow_start_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#4A5568",
                    lw=2,
                    mutation_scale=15
                )
            )

    # Footer parameter summary
    footer_text = (
        f"Parameters Summary:   Total: {param_info['total_params']:,}   |   "
        f"Trainable (Head): {param_info['trainable_params']:,}   |   "
        f"Non-Trainable (Base): {param_info['non_trainable_params']:,}"
    )
    ax.text(
        50, 2.5, footer_text,
        ha="center", va="center", fontsize=10.5, fontweight="bold",
        color="#2D3748",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EDF2F7", edgecolor="#CBD5E0", lw=1.5)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved architecture diagram to: {output_path}")


def validate_model_forward_pass(
    model: tf.keras.Model,
    num_classes: int,
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Performs forward-pass sanity check with a miniature batch:
    - Verifies shape: (batch_size, num_classes)
    - Verifies softmax sums to ~1.0
    - Confirms zero NaN / Inf outputs
    - Confirms non-negative probabilities
    """
    logger.info(f"Performing forward-pass sanity check with batch_size={batch_size}...")

    # Generate dummy input batch within normalized [-1, 1] range
    dummy_input = tf.random.uniform(
        (batch_size, DEFAULT_INPUT_SHAPE[0], DEFAULT_INPUT_SHAPE[1], DEFAULT_INPUT_SHAPE[2]),
        minval=-1.0, maxval=1.0, dtype=tf.float32
    )

    predictions = model(dummy_input, training=False)
    preds_np = predictions.numpy()

    output_shape = list(preds_np.shape)
    sums_per_sample = np.sum(preds_np, axis=1)
    has_nan = bool(np.isnan(preds_np).any())
    has_inf = bool(np.isinf(preds_np).any())
    min_prob = float(np.min(preds_np))
    max_prob = float(np.max(preds_np))

    logger.info(f"Output Tensor Shape:       {output_shape}")
    logger.info(f"Sum per sample (Softmax):  {sums_per_sample}")
    logger.info(f"Min Probability:           {min_prob:.6f}")
    logger.info(f"Max Probability:           {max_prob:.6f}")
    logger.info(f"NaN Detected:              {has_nan}")
    logger.info(f"Inf Detected:              {has_inf}")

    assert output_shape == [batch_size, num_classes], f"Expected output shape {[batch_size, num_classes]}, got {output_shape}"
    assert not has_nan, "NaN values detected in forward pass output!"
    assert not has_inf, "Inf values detected in forward pass output!"
    assert np.allclose(sums_per_sample, 1.0, atol=1e-5), "Softmax probabilities do not sum to 1.0!"
    assert min_prob >= 0.0, "Negative probability values detected!"

    logger.info("Forward-pass sanity check: PASSED successfully.")

    return {
        "batch_size": batch_size,
        "input_shape": [batch_size] + list(DEFAULT_INPUT_SHAPE),
        "output_shape": output_shape,
        "sums_approx_one": bool(np.allclose(sums_per_sample, 1.0, atol=1e-5)),
        "min_prob": min_prob,
        "max_prob": max_prob,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "sanity_check_passed": True
    }


def run_phase_3(
    class_names_path_arg: Optional[str] = None,
    output_dir_arg: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes all Phase 3 steps:
    1. Dynamic class detection
    2. Model construction and compilation
    3. Parameter verification
    4. Model summary generation and file export
    5. Architecture diagram visualization
    6. Forward pass validation
    """
    class_names_path, output_dir = resolve_paths(class_names_path_arg, output_dir_arg)
    num_classes = get_num_classes(class_names_path)

    logger.info("=== PHASE 3: MOBILENETV2 ARCHITECTURE & TRAINING SETUP ===")
    logger.info(f"Target Dog Breeds (Classes): {num_classes}")
    logger.info(f"Outputs Directory:           {output_dir}")

    # Build model
    model, base_model = build_model(num_classes=num_classes)

    # Verify parameters
    param_info = verify_trainable_parameters(model)

    # Save summary text
    summary_path = output_dir / "model_summary.txt"
    save_model_summary(model, summary_path, param_info)

    # Plot architecture diagram
    diagram_path = output_dir / "mobilenetv2_architecture.png"
    plot_architecture_diagram(diagram_path, num_classes, param_info)

    # Perform forward-pass validation
    validation_info = validate_model_forward_pass(model, num_classes=num_classes)

    # Save Phase 3 metadata JSON
    phase_3_meta = {
        "model_name": model.name,
        "base_model": "MobileNetV2",
        "weights": "imagenet",
        "num_classes": num_classes,
        "input_shape": list(model.input_shape[1:]),
        "output_shape": list(model.output_shape[1:]),
        "parameters": param_info,
        "optimizer": {
            "name": "Adam",
            "learning_rate": DEFAULT_LEARNING_RATE
        },
        "loss": "SparseCategoricalCrossentropy",
        "metrics": ["accuracy", "top_5_accuracy"],
        "base_model_frozen": True,
        "forward_pass_validation": validation_info,
        "status": "READY FOR PHASE 4 TRAINING"
    }

    meta_path = output_dir / "model_architecture_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(phase_3_meta, f, indent=2)
    logger.info(f"Saved model architecture metadata to: {meta_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("PHASE 3 MODEL ARCHITECTURE SETUP COMPLETE")
    print("=" * 50)
    print(f"Model:                  {model.name} (MobileNetV2 Transfer Learning)")
    print(f"Pretrained Weights:     ImageNet")
    print(f"Input Shape:            {model.input_shape}")
    print(f"Output Shape:           {model.output_shape} ({num_classes} Dog Breeds)")
    print(f"Total Parameters:       {param_info['total_params']:,}")
    print(f"Trainable Parameters:   {param_info['trainable_params']:,} (Classification Head)")
    print(f"Non-Trainable:          {param_info['non_trainable_params']:,} (Frozen MobileNetV2 Base)")
    print(f"Optimizer:              Adam (learning_rate={DEFAULT_LEARNING_RATE})")
    print(f"Loss Function:          SparseCategoricalCrossentropy")
    print(f"Metrics:                accuracy, top_5_accuracy")
    print(f"Summary Saved:          {summary_path}")
    print(f"Diagram Saved:          {diagram_path}")
    print(f"Sanity Check:           PASSED (Softmax probabilities valid, 0 NaNs)")
    print("Status:                 READY FOR PHASE 4 TRAINING")
    print("=" * 50 + "\n")

    return phase_3_meta


if __name__ == "__main__":
    run_phase_3()
